import json
from pathlib import Path
from typing import List, Tuple, Optional, Set

import numpy as np
import torch
from nerfstudio.cameras.cameras import CameraType, Cameras
from smart_open import open

from suds.data.image_metadata import ImageMetadata
from suds.stream_utils import get_filesystem

OPENCV_TO_OPENGL = torch.DoubleTensor([[1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, -1, 0],
                                       [0, 0, 0, 1]])


def write_metadata(output_path: str, metadata_items: List[ImageMetadata], static_masks: List[str], origin: torch.Tensor,
                   pose_scale_factor: float, scene_bounds: torch.Tensor) -> None:
    if len(static_masks) > 0:
        assert len(metadata_items) == len(static_masks), \
            'Number of metadata items and static masks not equal: {} {}'.format(len(metadata_items), len(static_masks))

    frames = []
    for i, item in enumerate(metadata_items):
        frame_metadata = {
            'image_index': item.image_index,
            'rgb_path': item.image_path,
            'depth_path': item.depth_path,
            'feature_path': item.feature_path,
            'backward_flow_path': item.backward_flow_path,
            'forward_flow_path': item.forward_flow_path,
            'backward_neighbor_index': item.backward_neighbor_index,
            'forward_neighbor_index': item.forward_neighbor_index,
            'c2w': item.c2w.tolist(),
            'W': item.W,
            'H': item.H,
            'intrinsics': item.intrinsics.tolist(),
            'time': item.time,
            'video_id': item.video_id,
            'is_val': item.is_val
        }

        if len(static_masks) > 0:
            frame_metadata['static_mask_path'] = static_masks[i]

        if item.mask_path is not None:
            frame_metadata['mask_path'] = item.mask_path

        if item.sky_mask_path is not None:
            frame_metadata['sky_mask_path'] = item.sky_mask_path

        frames.append(frame_metadata)

    if get_filesystem(output_path) is None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'origin': origin.tolist(),
            'scene_bounds': scene_bounds.tolist(),
            'pose_scale_factor': pose_scale_factor,
            'frames': frames
        }, f, indent=2)


def get_bounds_from_depth(item: ImageMetadata, cur_min_bounds: Optional[torch.Tensor],
                          cur_max_bounds: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    ray_bundle = Cameras(camera_to_worlds=item.c2w,
                         fx=item.intrinsics[0],
                         fy=item.intrinsics[1],
                         cx=item.intrinsics[2],
                         cy=item.intrinsics[3],
                         width=item.W,
                         height=item.H,
                         camera_type=CameraType.PERSPECTIVE).generate_rays(0)

    directions = ray_bundle.directions.view(-1, 3)
    depth = item.load_depth().view(-1)

    filtered_directions = directions[depth > 0]
    filtered_depth = depth[depth > 0].unsqueeze(-1)
    filtered_z_scale = ray_bundle.metadata['directions_norm'].view(-1, 1)[depth > 0]

    points = item.c2w[:, 3].unsqueeze(0) + filtered_directions * filtered_depth * filtered_z_scale
    bounds = [item.c2w[:, 3].unsqueeze(0), points]

    if cur_min_bounds is not None:
        bounds.append(cur_min_bounds.unsqueeze(0))
        bounds.append(cur_max_bounds.unsqueeze(0))

    bounds = torch.cat(bounds)
    return bounds.min(dim=0)[0], bounds.max(dim=0)[0]


def scale_bounds(
        all_items: List[ImageMetadata],
        min_bounds: torch.Tensor,
        max_bounds: torch.Tensor) -> Tuple[torch.Tensor, float, torch.Tensor]:
    positions = torch.cat([x.c2w[:, 3].unsqueeze(0) for x in all_items])

    print('Camera range in metric space: {} {}'.format(positions.min(dim=0)[0], positions.max(dim=0)[0]))

    origin = (max_bounds + min_bounds) * 0.5
    print('Calculated origin: {} {} {}'.format(origin, min_bounds, max_bounds))

    pose_scale_factor = torch.linalg.norm((max_bounds - min_bounds) * 0.5).item()
    print('Calculated pose scale factor: {}'.format(pose_scale_factor))

    for item in all_items:
        item.c2w[:, 3] = (item.c2w[:, 3] - origin) / pose_scale_factor
        assert torch.logical_and(item.c2w >= -1, item.c2w <= 1).all(), item.c2w

    scene_bounds = (torch.stack([min_bounds, max_bounds]) - origin) / pose_scale_factor

    return origin, pose_scale_factor, scene_bounds


def normalize_timestamp(item: ImageMetadata, min_frame: int, max_frame: int) -> None:
    divisor = 0.5 * (max_frame - min_frame)
    assert divisor > 0
    item.time = (item.time - min_frame) / divisor - 1
    assert -1 <= item.time <= 1


def get_frame_range(frame_ranges: List[Tuple[int]], frame: int) -> Optional[Tuple[int]]:
    for frame_range in frame_ranges:
        if frame_range[0] <= frame <= frame_range[1]:
            return frame_range

    return None


def get_val_frames(num_frames: int, test_every: int, train_every: int) -> Set[int]:
    assert train_every is None or test_every is None
    if train_every is None:
        val_frames = set(np.arange(test_every, num_frames, test_every))
    else:
        train_frames = set(np.arange(0, num_frames, train_every))
        val_frames = (set(np.arange(num_frames)) - train_frames) if train_every > 1 else train_frames

    return val_frames


def get_neighbor(image_index: int, val_frames: Set[int], dir: int) -> int:
    diff = dir
    while (image_index + diff) // 2 in val_frames:
        diff += dir

    return image_index + diff
