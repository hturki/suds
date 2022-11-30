""" Data parser for SUDS datasets. """

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, List, Optional, Set, Dict, Any

import torch
import tyro
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from rich.console import Console
from smart_open import open

from suds.data.image_metadata import ImageMetadata

CONSOLE = Console(width=120)
ALL_ITEMS = 'all_items'
ALL_CAMERAS = 'all_cameras'
POSE_SCALE_FACTOR = 'pose_scale_factor'
ORIGIN = 'origin'


@dataclass
class SUDSDataParserConfig(DataParserConfig):
    """SUDS dataset config"""

    _target: Type = field(default_factory=lambda: SUDSDataParser)
    """target class to instantiate"""
    metadata_path: str = 'metadata.json'
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_scale: float = 1.0
    """How much to scale the region of interest by."""
    train_downscale_factor: float = 1
    """How much to downscale images used for training."""
    eval_downscale_factor: float = 1
    """How much to downscale images used for evaluation."""
    train_with_val_images: bool = False
    """Whether to include the validation images when training."""
    static_only: bool = False
    """Whether to include static pixels when training."""
    local_cache_path: Optional[str] = None
    """Caches images and metadata in specific path if set."""

    metadata: tyro.conf.Suppress[Optional[Dict[str, Any]]] = None


@dataclass
class SUDSDataParser(DataParser):
    """SUDS DatasetParser"""

    config: SUDSDataParserConfig

    def get_dataparser_outputs(self, split='train', indices: Optional[Set[int]] = None) -> DataparserOutputs:
        # Cache json load - SUDSManager will clear it when it's no longer needed
        if self.config.metadata is None:
            with open(self.config.metadata_path) as f:
                self.config.metadata = json.load(f)

            if all([f['is_val'] for f in self.config.metadata['frames']]):
                self.config.train_with_val_images = True

        downscale_factor = self.config.train_downscale_factor if split == 'train' else self.config.eval_downscale_factor
        all_items = []
        split_items = []
        image_filenames = []
        mask_filenames = []

        local_cache_path = Path(self.config.local_cache_path) if self.config.local_cache_path is not None else None
        frames = self.config.metadata['frames']
        for frame_index in range(len(frames)):
            frame = frames[frame_index]
            c2w = torch.FloatTensor(frame['c2w'])
            c2w[:, 3] /= self.config.scale_factor

            item = ImageMetadata(frame['rgb_path'],
                                 c2w,
                                 int(frame['W'] // downscale_factor),
                                 int(frame['H'] // downscale_factor),
                                 torch.FloatTensor(frame['intrinsics']) / downscale_factor,
                                 frame['image_index'],
                                 frame['time'],
                                 frame['video_id'],
                                 frame['depth_path'],
                                 frame.get('static_mask_path' if self.config.static_only else 'mask_path', None),
                                 frame.get('sky_mask_path', None),
                                 frame.get('feature_path', None),
                                 frame.get('backward_flow_path', None),
                                 frame.get('forward_flow_path', None),
                                 frame.get('backward_neighbor_index', None),
                                 frame.get('forward_neighbor_index', None),
                                 frame['is_val'],
                                 self.config.metadata['pose_scale_factor'],
                                 local_cache_path)

            all_items.append(item)

            # Keep the image indices consistent between training and validation
            if split == 'train':
                if frame['is_val'] and not self.config.train_with_val_images:
                    continue
            elif not frame['is_val']:
                continue

            if indices is not None and frame_index not in indices:
                continue

            split_items.append(item)
            image_filenames.append(Path(item.image_path))
            if item.mask_path is not None:
                mask_filenames.append(Path(item.mask_path))

        assert (
                len(image_filenames) != 0
        ), """
        No image files found. 
        You should check the file_paths in the transforms.json file to make sure they are correct.
        """
        assert len(mask_filenames) == 0 or (
                len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
        """

        scene_box = SceneBox(
            aabb=torch.tensor(self.config.metadata['scene_bounds']) * self.config.scene_scale
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=self.create_cameras(split_items),
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            metadata={
                ALL_ITEMS: all_items,
                ALL_CAMERAS: self.create_cameras(all_items),
                POSE_SCALE_FACTOR: self.config.metadata['pose_scale_factor'],
                ORIGIN: self.config.metadata['origin']
            }
        )

        return dataparser_outputs

    @staticmethod
    def create_cameras(metadata_items: List[ImageMetadata]) -> Cameras:
        return Cameras(
            camera_to_worlds=torch.stack([x.c2w for x in metadata_items]),
            fx=torch.FloatTensor([x.intrinsics[0] for x in metadata_items]),
            fy=torch.FloatTensor([x.intrinsics[1] for x in metadata_items]),
            cx=torch.FloatTensor([x.intrinsics[2] for x in metadata_items]),
            cy=torch.FloatTensor([x.intrinsics[3] for x in metadata_items]),
            width=torch.IntTensor([x.W for x in metadata_items]),
            height=torch.IntTensor([x.H for x in metadata_items]),
            camera_type=CameraType.PERSPECTIVE,
            times=torch.FloatTensor([x.time for x in metadata_items]).unsqueeze(-1)
        )
