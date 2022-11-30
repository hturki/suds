import hashlib
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

from suds.stream_utils import (buffer_from_stream, image_from_stream, get_filesystem, table_from_stream)


class ImageMetadata:
    def __init__(self, image_path: str, c2w: torch.Tensor, W: int, H: int, intrinsics: torch.Tensor, image_index: int,
                 time: float, video_id: int, depth_path: str, mask_path: Optional[str], sky_mask_path: Optional[str],
                 feature_path: Optional[str], backward_flow_path: Optional[str], forward_flow_path: Optional[str],
                 backward_neighbor_index: Optional[int], forward_neighbor_index: Optional[int], is_val: bool,
                 pose_scale_factor: float, local_cache: Optional[Path]):
        self.image_path = image_path
        self.c2w = c2w
        self.W = W
        self.H = H
        self.intrinsics = intrinsics
        self.image_index = image_index
        self.time = time
        self.video_id = video_id
        self.depth_path = depth_path
        self.mask_path = mask_path
        self.sky_mask_path = sky_mask_path
        self.feature_path = feature_path
        self.backward_flow_path = backward_flow_path
        self.forward_flow_path = forward_flow_path
        self.backward_neighbor_index = backward_neighbor_index
        self.forward_neighbor_index = forward_neighbor_index
        self.is_val = is_val

        self._pose_scale_factor = pose_scale_factor
        self._local_cache = local_cache

    def load_image(self) -> torch.Tensor:
        if self._local_cache is not None and not self.image_path.startswith(str(self._local_cache)):
            self.image_path = self._load_from_cache(self.image_path)

        rgbs = image_from_stream(self.image_path).convert('RGB')
        size = rgbs.size

        if size[0] != self.W or size[1] != self.H:
            rgbs = rgbs.resize((self.W, self.H), Image.LANCZOS)

        return torch.ByteTensor(np.asarray(rgbs))

    def load_mask(self) -> torch.Tensor:
        if self.mask_path is None:
            return torch.ones(self.H, self.W, dtype=torch.bool)

        if self._local_cache is not None and not self.mask_path.startswith(str(self._local_cache)):
            self.mask_path = self._load_from_cache(self.mask_path)

        mask = image_from_stream(self.mask_path)
        size = mask.size

        if size[0] != self.W or size[1] != self.H:
            mask = mask.resize((self.W, self.H), Image.NEAREST)

        return torch.BoolTensor(np.asarray(mask))

    def load_sky_mask(self) -> torch.Tensor:
        if self.sky_mask_path is None:
            return torch.zeros(self.H, self.W, dtype=torch.bool)

        if self._local_cache is not None and not self.sky_mask_path.startswith(str(self._local_cache)):
            self.sky_mask_path = self._load_from_cache(self.sky_mask_path)

        sky_mask = image_from_stream(self.sky_mask_path)
        size = sky_mask.size

        if size[0] != self.W or size[1] != self.H:
            sky_mask = sky_mask.resize((self.W, self.H), Image.NEAREST)

        return torch.BoolTensor(np.asarray(sky_mask))

    def load_features(self, resize: bool = True) -> torch.Tensor:
        assert self.feature_path is not None

        if self._local_cache is not None and not self.feature_path.startswith(str(self._local_cache)):
            self.feature_path = self._load_from_cache(self.feature_path)

        table = table_from_stream(self.feature_path)
        features = torch.FloatTensor(table['pca'].to_numpy()).view(
            [int(x) for x in table.schema.metadata[b'shape'].split()])

        if (features.shape[0] != self.H or features.shape[1] != self.W) and resize:
            features = F.interpolate(features.permute(2, 0, 1).unsqueeze(0), size=(self.H, self.W)).squeeze() \
                .permute(1, 2, 0)

        return features

    def load_depth(self) -> torch.Tensor:
        if self._local_cache is not None and not self.depth_path.startswith(str(self._local_cache)):
            self.depth_path = self._load_from_cache(self.depth_path)

        if self.depth_path.endswith('.parquet'):
            table = table_from_stream(self.depth_path)

            # Get original depth dimensions
            size = image_from_stream(self.image_path).size

            depth = torch.FloatTensor(table['depth'].to_numpy()).view(size[1], size[0])
        else:
            # Assume it's vkitti2 format
            depth = np.array(image_from_stream(self.depth_path))
            depth[depth == 65535] = -1

            # depth is in cm - convert to meters
            depth = torch.FloatTensor(depth / 100)

        if depth.shape[0] != self.H or depth.shape[1] != self.W:
            depth = F.interpolate(depth.unsqueeze(0).unsqueeze(0), size=(self.H, self.W)).squeeze()

        return depth / self._pose_scale_factor

    def load_backward_flow(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._load_flow(self.backward_flow_path, False)

    def load_forward_flow(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._load_flow(self.forward_flow_path, True)

    def _load_flow(self, flow_path: Optional[str], is_forward: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if flow_path is None:
            return torch.zeros(self.H, self.W, 2), torch.zeros(self.H, self.W, dtype=torch.bool)

        if self._local_cache is not None and not flow_path.startswith(str(self._local_cache)):
            flow_path = self._load_from_cache(flow_path)
            if is_forward:
                self.forward_flow_path = flow_path
            else:
                self.backward_flow_path = flow_path

        if flow_path.endswith('.parquet'):
            table = pq.read_table(flow_path, filesystem=get_filesystem(flow_path))

            if 'flow' in table.column_names:
                flow = torch.FloatTensor(table['flow'].to_numpy()).view(
                    [int(x) for x in table.schema.metadata[b'shape'].split()])
                if len(flow.shape) == 4:
                    flow = flow.squeeze().permute(1, 2, 0)

                flow_valid = torch.ones_like(flow[:, :, 0], dtype=torch.bool)
            else:
                point1 = torch.LongTensor(table.to_pandas()[['point1_x', 'point1_y']].to_numpy())
                point2 = torch.LongTensor(table.to_pandas()[['point2_x', 'point2_y']].to_numpy())

                correspondences = (point2 - point1) if is_forward else (point1 - point2)
                to_index = point1 if is_forward else point2

                orig_H, orig_W = [int(x) for x in table.schema.metadata[b'shape'].split()]
                flow = torch.zeros(orig_H, orig_W, 2)
                flow_valid = torch.zeros(orig_H, orig_W, dtype=torch.bool)
                flow.view(-1, 2)[to_index[:, 0] + to_index[:, 1] * orig_W] = correspondences.float()
                flow_valid.view(-1)[to_index[:, 0] + to_index[:, 1] * orig_W] = True
        else:
            quantized_flow = cv2.imdecode(np.frombuffer(buffer_from_stream(flow_path).getbuffer(), dtype=np.uint8),
                                          cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            # From https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
            _, _, _c = quantized_flow.shape
            assert quantized_flow.dtype == np.uint16 and _c == 3
            # b == invalid flow flag == 0 for sky or other invalid flow
            invalid = quantized_flow[:, :, 0] == 0
            # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 – 1]
            flow = 2.0 / (2 ** 16 - 1.0) * quantized_flow[:, :, 2:0:-1].astype(np.float32) - 1
            flow[:, :, 0] *= flow.shape[1] - 1
            flow[:, :, 1] *= flow.shape[0] - 1
            flow[invalid] = 0  # or another value (e.g., np.nan)

            flow = torch.FloatTensor(flow)
            flow_valid = torch.BoolTensor(quantized_flow[:, :, 0] != 0)

        if flow.shape[0] != self.H or flow.shape[1] != self.W:
            flow[:, :, 0] *= (self.W / flow.shape[1])
            flow[:, :, 1] *= (self.H / flow.shape[0])
            flow = F.interpolate(flow.permute(2, 0, 1).unsqueeze(0), size=(self.H, self.W)).squeeze().permute(1, 2, 0)
            flow_valid = F.interpolate(flow_valid.unsqueeze(0).unsqueeze(0).float(),
                                       size=(self.H, self.W)).bool().squeeze()

        return flow, flow_valid

    def _load_from_cache(self, remote_path: str) -> str:
        sha_hash = hashlib.sha256()
        sha_hash.update(remote_path.encode('utf-8'))
        hashed = sha_hash.hexdigest()
        cache_path = self._local_cache / hashed[:2] / hashed[2:4] / '{}{}'.format(hashed, Path(remote_path).suffix)

        if cache_path.exists():
            return str(cache_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = '{}.{}'.format(cache_path, uuid.uuid4())
        remote_filesystem = get_filesystem(remote_path)

        if remote_filesystem is not None:
            remote_filesystem.get(remote_path, tmp_path)
        else:
            shutil.copy(remote_path, tmp_path)

        os.rename(tmp_path, cache_path)
        return str(cache_path)
