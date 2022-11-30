import random
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union, Literal, Set

import numpy as np
import torch
import tyro
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.datamanagers.base_datamanager import DataManager
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.comms import get_rank, get_world_size
from rich.console import Console
from torch.nn import Parameter
from torch.utils.data import DistributedSampler, DataLoader

from suds.data.stream_input_dataset import StreamInputDataset
from suds.data.suds_dataparser import SUDSDataParserConfig, ALL_CAMERAS, ALL_ITEMS, POSE_SCALE_FACTOR
from suds.data.suds_dataset import SUDSDataset, RAY_INDEX, VIDEO_ID, Split
from suds.data.suds_eval_dataloader import SUDSEvalDataLoader
from suds.suds_constants import FEATURES, TIME, BACKWARD_NEIGHBOR_TIME_DIFF, FORWARD_NEIGHBOR_TIME_DIFF, \
    BACKWARD_FLOW_VALID, FORWARD_FLOW_VALID, BACKWARD_NEIGHBOR_W2C, BACKWARD_NEIGHBOR_K, FORWARD_NEIGHBOR_W2C, \
    FORWARD_NEIGHBOR_K

CONSOLE = Console(width=120)


@dataclass
class SUDSDataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation; DataManager is in charge of keeping the train/eval dataparsers;
    After instantiation, data manager holds both train/eval datasets and is in charge of returning unpacked
    train/eval data at each iteration
    """

    _target: Type = field(default_factory=lambda: SUDSDataManager)
    """Target class to instantiate."""
    dataparser: SUDSDataParserConfig = SUDSDataParserConfig()
    """Specifies the dataparser used to unpack the data."""
    train_num_rays_per_batch: int = 4096
    """Number of rays per batch to use per training iteration."""
    eval_num_rays_per_batch: int = 8192
    eval_image_indices: Optional[Tuple[int, ...]] = None
    """Specifies the image indices to use during eval; if None, uses all val images."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(
        optimizer=AdamOptimizerConfig(lr=6e-6, eps=1e-15),
        scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-4, max_steps=250000))
    """Specifies the camera pose optimizer used during training. Helpful if poses are noisy, such as for data from
    Record3D."""
    items_per_chunk: int = 12800000
    """Number of entries to load into memory at a time"""
    load_on_demand: List[str] = field(default_factory=lambda: [FEATURES])
    """Field to load when loading a chunk. Fields not included will be cached in memory across the dataset."""
    load_random_subset: bool = False
    """Loads shuffled batches from all images if specified."""
    max_viewer_images: Optional[int] = 500
    """Maximum number of images to show in viewer"""

    load_depth: tyro.conf.Suppress[bool] = True
    load_features: tyro.conf.Suppress[bool] = True
    load_flow: tyro.conf.Suppress[bool] = True
    load_sky: tyro.conf.Suppress[bool] = True
    include_val_items: tyro.conf.Suppress[bool] = False
    feature_clusters: tyro.conf.Suppress[Dict[str, torch.Tensor]] = to_immutable_dict({})
    feature_colors: tyro.conf.Suppress[Dict[str, torch.Tensor]] = to_immutable_dict({})


class SUDSDataManager(DataManager):
    config: SUDSDataManagerConfig

    train_dataset: StreamInputDataset  # Used by the viewer and other things in trainer
    eval_batch_dataset: SUDSDataset

    def __init__(
            self,
            config: SUDSDataManagerConfig,
            device: Union[torch.device, str] = 'cpu',
            test_mode: Literal['test', 'val', 'inference'] = 'val',
            world_size: int = 1,
            local_rank: int = 0
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = 'test' if test_mode in ['test', 'inference'] else 'val'
        self.dataparser = self.config.dataparser.setup()

        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split='train')

        self.camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=len(self.train_dataparser_outputs.metadata[ALL_CAMERAS]), device=self.device
        )
        self.train_ray_generator = RayGenerator(self.train_dataparser_outputs.metadata[ALL_CAMERAS].to(self.device),
                                                self.camera_optimizer)

        train_split = Split.ALL if self.config.dataparser.train_with_val_images else Split.TRAIN
        self.train_batch_dataset = self._create_suds_dataset(train_split)
        self.iter_train_image_dataloader = iter([])

        if len(self.train_dataparser_outputs.image_filenames) > self.config.max_viewer_images:
            indices = set(
                np.linspace(0, len(self.train_dataparser_outputs.metadata[ALL_CAMERAS]), self.config.max_viewer_images,
                            endpoint=False, dtype=np.int32))
            viewer_outputs = self.dataparser.get_dataparser_outputs(split='train', indices=indices)
        else:
            viewer_outputs = self.train_dataparser_outputs

        self.train_dataset = StreamInputDataset(viewer_outputs)

        # Make a separate dataset so that the downsample factor is correct
        # Hack to get around "if self.eval_dataset" check in trainer.py
        # which is initially false as len(self.eval_batch_dataset) == 0 before loading a chunk
        self.eval_dataset = True
        self.eval_batch_dataset = self._create_suds_dataset(Split.VAL)
        self.iter_eval_batch_dataloader = iter([])
        self.dataparser.config.metadata = None  # Clear loaded cached metadata

    @cached_property
    def fixed_indices_eval_dataloader(self) -> SUDSEvalDataLoader:
        eval_dataparser_outputs = self.dataparser.get_dataparser_outputs(split='val')

        return SUDSEvalDataLoader(
            all_items=eval_dataparser_outputs.metadata[ALL_ITEMS],
            cameras=eval_dataparser_outputs.metadata[ALL_CAMERAS],
            load_depth=self.config.load_depth,
            load_features=self.config.load_features,
            image_indices=self.config.eval_image_indices,
            load_flow=self.config.load_flow,
            load_sky=self.config.load_sky,
            feature_clusters=self.config.feature_clusters,
            feature_colors=self.config.feature_colors,
            device=self.device
        )

    def all_indices_eval_dataloader(self, generate_ring_view: bool, video_ids: Optional[Set[int]],
                                    start_frame: Optional[int], end_frame: Optional[int], focal_mult: Optional[float],
                                    pos_shift: Optional[torch.Tensor]) -> SUDSEvalDataLoader:
        eval_dataparser_outputs = self.dataparser.get_dataparser_outputs(split='val')

        image_indices = []
        all_items = eval_dataparser_outputs.metadata[ALL_ITEMS]
        image_chunk = 7 if generate_ring_view else 1

        video_items = all_items if video_ids is None else filter(lambda x: x.video_id in video_ids, all_items)
        if start_frame is not None or end_frame is not None:
            filtered_items = []
            cur_base = None
            cur_video_id = None
            for i, item in enumerate(video_items):
                if cur_video_id != item.video_id:
                    cur_video_id = item.video_id
                    cur_base = i
                video_index = i - cur_base
                if (start_frame is None or start_frame * image_chunk <= video_index) \
                        and (end_frame is None or end_frame * image_chunk > video_index):
                    filtered_items.append(item)
        else:
            filtered_items = video_items

        for i in range(get_rank() * image_chunk, len(filtered_items), image_chunk * get_world_size()):
            for j in range(image_chunk):
                image_indices.append(filtered_items[i + j].image_index)

        cameras = eval_dataparser_outputs.metadata[ALL_CAMERAS]
        if focal_mult is not None:
            cameras = Cameras(
                camera_to_worlds=cameras.camera_to_worlds,
                fx=cameras.fx * focal_mult,
                fy=cameras.fy * focal_mult,
                cx=cameras.cx,
                cy=cameras.cy,
                width=cameras.width,
                height=cameras.height,
                camera_type=cameras.camera_type,
                times=cameras.times
            )

        if pos_shift is not None:
            c2w = cameras.camera_to_worlds.clone()
            c2w[..., 3] += pos_shift.to(c2w) / self.train_dataparser_outputs.metadata[POSE_SCALE_FACTOR]
            cameras = Cameras(
                camera_to_worlds=c2w,
                fx=cameras.fx,
                fy=cameras.fy,
                cx=cameras.cx,
                cy=cameras.cy,
                width=cameras.width,
                height=cameras.height,
                camera_type=cameras.camera_type,
                times=cameras.times
            )

        return SUDSEvalDataLoader(
            all_items=all_items,
            cameras=cameras,
            load_depth=self.config.load_depth,
            load_features=self.config.load_features,
            image_indices=image_indices,
            load_flow=self.config.load_flow,
            load_sky=self.config.load_sky,
            feature_clusters=self.config.feature_clusters,
            feature_colors=self.config.feature_colors,
            device=self.device
        )

    def _create_suds_dataset(self, split: Split) -> SUDSDataset:
        # Use train resolution even for random val batches, or else pixel indices will not line up with
        # downsample when using parquet files
        return SUDSDataset(
            all_items=self.train_dataparser_outputs.metadata[ALL_ITEMS],
            load_depth=self.config.load_depth,
            load_features=self.config.load_features,
            load_flow=self.config.load_flow,
            load_sky=self.config.load_sky,
            load_on_demand=set(self.config.load_on_demand),
            load_random_subset=self.config.load_random_subset,
            items_per_chunk=(self.config.eval_num_rays_per_batch * 10) if split == Split.VAL \
                else self.config.items_per_chunk,
            split=split
        )

    def _set_train_loader(self):
        batch_size = self.config.train_num_rays_per_batch // self.world_size
        if self.world_size > 0:
            self.train_sampler = DistributedSampler(self.train_batch_dataset, self.world_size, self.local_rank)
            assert self.config.train_num_rays_per_batch % self.world_size == 0
            self.train_image_dataloader = DataLoader(self.train_batch_dataset, batch_size=batch_size,
                                                     sampler=self.train_sampler, num_workers=0, pin_memory=True)
        else:
            self.train_image_dataloader = DataLoader(self.train_batch_dataset, batch_size=batch_size, shuffle=True,
                                                     num_workers=0, pin_memory=True)

        self.iter_train_image_dataloader = iter(self.train_image_dataloader)

    def _set_eval_batch_loader(self):
        batch_size = self.config.eval_num_rays_per_batch // self.world_size
        if self.world_size > 0:
            self.eval_sampler = DistributedSampler(self.eval_batch_dataset, self.world_size, self.local_rank)
            assert self.config.eval_num_rays_per_batch % self.world_size == 0
            self.eval_batch_dataloader = DataLoader(self.eval_batch_dataset, batch_size=batch_size,
                                                    sampler=self.eval_sampler,
                                                    num_workers=0, pin_memory=True)
        else:
            self.eval_batch_dataloader = DataLoader(self.eval_batch_dataset, batch_size=batch_size,
                                                    shuffle=True, num_workers=0, pin_memory=True)

        self.iter_eval_batch_dataloader = iter(self.eval_batch_dataloader)

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        batch = next(self.iter_train_image_dataloader, None)
        if batch is None:
            self.train_batch_dataset.load_chunk()
            self._set_train_loader()
            batch = next(self.iter_train_image_dataloader)

        ray_bundle = self.train_ray_generator(batch[RAY_INDEX])
        ray_bundle.times = batch[TIME].to(self.device)
        ray_bundle.metadata[VIDEO_ID] = batch[VIDEO_ID].to(self.device)

        if BACKWARD_NEIGHBOR_TIME_DIFF in batch:
            ray_bundle.metadata[BACKWARD_NEIGHBOR_TIME_DIFF] = batch[BACKWARD_NEIGHBOR_TIME_DIFF].to(self.device)
            ray_bundle.metadata[FORWARD_NEIGHBOR_TIME_DIFF] = batch[FORWARD_NEIGHBOR_TIME_DIFF].to(self.device)

            # Need to flatten these or else metadata slicing won't work
            ray_bundle.metadata[BACKWARD_NEIGHBOR_W2C] = batch[BACKWARD_NEIGHBOR_W2C].flatten(-2).to(self.device)
            ray_bundle.metadata[FORWARD_NEIGHBOR_W2C] = batch[FORWARD_NEIGHBOR_W2C].flatten(-2).to(self.device)
            ray_bundle.metadata[BACKWARD_NEIGHBOR_K] = batch[BACKWARD_NEIGHBOR_K].flatten(-2).to(self.device)
            ray_bundle.metadata[FORWARD_NEIGHBOR_K] = batch[FORWARD_NEIGHBOR_K].flatten(-2).to(self.device)

            ray_bundle.metadata[BACKWARD_FLOW_VALID] = batch[BACKWARD_FLOW_VALID].to(self.device)
            ray_bundle.metadata[FORWARD_FLOW_VALID] = batch[FORWARD_FLOW_VALID].to(self.device)

        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the eval dataloader."""
        self.eval_count += 1
        batch = next(self.iter_eval_batch_dataloader, None)
        if batch is None:
            self.eval_batch_dataset.load_chunk()
            self._set_eval_batch_loader()
            batch = next(self.iter_eval_batch_dataloader)

        ray_bundle = self.train_ray_generator(batch[RAY_INDEX])
        ray_bundle.times = batch[TIME].to(self.device)
        ray_bundle.metadata[VIDEO_ID] = batch[VIDEO_ID].to(self.device)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[int, RayBundle, Dict]:
        image_index = random.choice(self.fixed_indices_eval_dataloader.image_indices)
        ray_bundle, batch = self.fixed_indices_eval_dataloader.get_image_data(image_index)
        return image_index, ray_bundle, batch

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Optional[Path]:
        return Path('datapath')

    def get_param_groups(self) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        camera_opt_params = list(self.camera_optimizer.parameters())
        if self.config.camera_optimizer.mode != 'off':
            assert len(camera_opt_params) > 0
            param_groups[self.config.camera_optimizer.param_group] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0

        return param_groups
