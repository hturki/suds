from typing import Union, Tuple, Optional, List, Dict

import torch
import torch.nn.functional as F
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.utils.comms import get_rank, get_world_size
from torch.utils.data import DataLoader, TensorDataset

from suds.data.dataset_utils import get_w2c_and_K
from suds.data.image_metadata import ImageMetadata
from suds.data.suds_dataset import RGB, DEPTH, BACKWARD_FLOW, FORWARD_FLOW, VIDEO_ID
from suds.kmeans import kmeans_predict
from suds.suds_constants import MASK, FEATURES, RAY_INDEX, BACKWARD_NEIGHBOR_TIME_DIFF, FORWARD_NEIGHBOR_TIME_DIFF, \
    BACKWARD_NEIGHBOR_W2C, BACKWARD_NEIGHBOR_K, FORWARD_NEIGHBOR_W2C, FORWARD_NEIGHBOR_K, SKY


class SUDSEvalDataLoader(DataLoader):

    def __init__(
            self,
            all_items: List[ImageMetadata],
            cameras: Cameras,
            load_depth: bool,
            load_features: bool,
            load_flow: bool,
            load_sky: bool,
            feature_clusters: Dict[str, torch.Tensor],
            feature_colors: Dict[str, torch.Tensor],
            image_indices: Optional[Tuple[int]] = None,
            device: Union[torch.device, str] = 'cpu'
    ):
        if image_indices is None:
            self.image_indices = []
            val_items = list(filter(lambda x: x.is_val, all_items))
            for item_index in range(get_rank(), len(val_items), get_world_size()):
                self.image_indices.append(val_items[item_index].image_index)
        else:
            self.image_indices = image_indices

        super().__init__(dataset=TensorDataset(torch.LongTensor(self.image_indices)))

        self.all_items = all_items
        self.cameras = cameras.to(device)
        self.load_depth = load_depth
        self.load_features = load_features
        self.load_flow = load_flow
        self.load_sky = load_sky
        self.feature_clusters = feature_clusters
        self.feature_colors = feature_colors
        self.device = device
        self.count = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self) -> Tuple[RayBundle, Dict]:
        if self.count < len(self.image_indices):
            data = self.get_image_data(self.image_indices[self.count])
            self.count += 1
            return data

        raise StopIteration

    def get_image_data(self, image_index: int) -> Tuple[RayBundle, Dict]:
        metadata_item = self.all_items[image_index]

        batch = {
            RGB: metadata_item.load_image().float().to(self.device) / 255.,
            MASK: metadata_item.load_mask().to(self.device)
        }

        if self.load_depth:
            batch[DEPTH] = metadata_item.load_depth().to(self.device)

        if self.load_features:
            features = metadata_item.load_features(False).to(self.device)
            for key, val in self.feature_clusters.items():
                feature_colors = self.feature_colors[key].to(self.device)[
                    kmeans_predict(features.view(-1, features.shape[-1]), val.to(self.device), device=self.device,
                                   tqdm_flag=False)].view((*features.shape[:-1], 3))

                if feature_colors.shape[0] != metadata_item.H or feature_colors.shape[1] != metadata_item.W:
                    feature_colors = F.interpolate(feature_colors.permute(2, 0, 1).unsqueeze(0),
                                                   size=(metadata_item.H, metadata_item.W)).squeeze().permute(1, 2, 0)

                batch[f'{FEATURES}_{key}'] = feature_colors

        ray_bundle = self.cameras.generate_rays(camera_indices=image_index, keep_shape=True)

        if self.load_flow:
            if metadata_item.backward_neighbor_index is not None:
                batch[BACKWARD_FLOW] = metadata_item.load_backward_flow()[0].to(self.device)
                backward_w2c, backward_K = get_w2c_and_K(self.all_items[metadata_item.backward_neighbor_index])
                ray_bundle.metadata[BACKWARD_NEIGHBOR_W2C] = backward_w2c.to(self.device).reshape(1, 1, 16).expand(
                    metadata_item.H, metadata_item.W, -1)
                ray_bundle.metadata[BACKWARD_NEIGHBOR_K] = backward_K.to(self.device).reshape(1, 1, 9).expand(
                    metadata_item.H, metadata_item.W, -1)
                ray_bundle.metadata[BACKWARD_NEIGHBOR_TIME_DIFF] = \
                    torch.ones_like(ray_bundle.origins[..., 0:1], dtype=torch.long) * \
                    (metadata_item.time - self.all_items[metadata_item.backward_neighbor_index].time)

            if metadata_item.forward_neighbor_index is not None:
                batch[FORWARD_FLOW] = metadata_item.load_forward_flow()[0].to(self.device)
                forward_w2c, forward_K = get_w2c_and_K(self.all_items[metadata_item.forward_neighbor_index])
                ray_bundle.metadata[FORWARD_NEIGHBOR_W2C] = forward_w2c.to(self.device).reshape(1, 1, 16).expand(
                    metadata_item.H, metadata_item.W, -1)
                ray_bundle.metadata[FORWARD_NEIGHBOR_K] = forward_K.to(self.device).reshape(1, 1, 9).expand(
                    metadata_item.H, metadata_item.W, -1)
                ray_bundle.metadata[FORWARD_NEIGHBOR_TIME_DIFF] = \
                    torch.ones_like(ray_bundle.origins[..., 0:1], dtype=torch.long) * \
                    (self.all_items[metadata_item.forward_neighbor_index].time - metadata_item.time)

            pixel_indices = torch.arange(metadata_item.W * metadata_item.H, device=self.device).unsqueeze(-1)
            v = pixel_indices // metadata_item.W
            u = pixel_indices % metadata_item.W
            image_indices = torch.ones_like(pixel_indices) * metadata_item.image_index
            batch[RAY_INDEX] = torch.cat([image_indices, v, u], -1).view(metadata_item.H, metadata_item.W, 3)

        if self.load_sky:
            batch[SKY] = metadata_item.load_sky_mask().to(self.device)

        ray_bundle.times = torch.ones_like(ray_bundle.origins[..., 0:1]) * metadata_item.time
        ray_bundle.metadata[VIDEO_ID] = torch.ones_like(ray_bundle.times, dtype=torch.int32) * metadata_item.video_id
        return ray_bundle, batch
