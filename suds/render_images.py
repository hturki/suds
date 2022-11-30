#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Set

import numpy as np
import torch
import torch.distributed as dist
import tyro
from PIL import Image
from nerfstudio.utils.comms import get_world_size
from nerfstudio.utils.eval_utils import eval_setup
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn

from suds.suds_constants import RGB, DEPTH, FEATURES, VIDEO_ID

CONSOLE = Console(width=120)


@dataclass
class RenderImages:
    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path

    generate_ring_view: bool
    video_ids: Optional[Set[int]] = None
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

    focal_mult: Optional[float] = None
    pos_shift: Optional[List[float]] = None

    feature_filter: Optional[List[int]] = None
    sigma_threshold: Optional[float] = None
    max_altitude: Optional[float] = None
    static_only: bool = False

    @torch.inference_mode()
    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path = eval_setup(self.load_config)
        pipeline.eval()

        dataloader = pipeline.datamanager.all_indices_eval_dataloader(self.generate_ring_view, self.video_ids,
                                                                      self.start_frame, self.end_frame, self.focal_mult,
                                                                      torch.FloatTensor(self.pos_shift)
                                                                      if self.pos_shift is not None else None)
        num_images = len(dataloader)

        render_options = {'static_only': self.static_only}
        if self.sigma_threshold is not None:
            render_options['sigma_threshold'] = self.sigma_threshold
        if self.max_altitude is not None:
            render_options['max_altitude'] = self.max_altitude
        if self.feature_filter is not None:
            render_options['feature_filter'] = self.feature_filter

        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)

            ring_buffer = []
            for camera_ray_bundle, batch in dataloader:
                images = {}

                to_check = [RGB, DEPTH]
                for key in pipeline.model.config.feature_clusters:
                    to_check.append(f'{FEATURES}_{key}')

                frame_id = int(camera_ray_bundle.camera_indices[0, 0, 0])
                video_id = str(camera_ray_bundle.metadata[VIDEO_ID][0, 0, 0].item())

                all_present = True
                for key in to_check:
                    candidiate = (self.output_path / video_id / '{0}-{1:06d}.jpg'.format(key, frame_id))
                    if candidiate.exists():
                        images[key] = np.asarray(Image.open(candidiate))
                    else:
                        all_present = False
                        break

                if not all_present:
                    outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle,
                                                                               render_options=render_options)

                    images[RGB] = (outputs[RGB] * 255).byte().cpu().numpy()
                    images[DEPTH] = (pipeline.model.apply_depth_colormap(outputs[DEPTH]) * 255).byte().cpu().numpy()
                    for key in pipeline.model.config.feature_clusters:
                        images[f'{FEATURES}_{key}'] = (outputs[f'{FEATURES}_{key}'] * 255).byte().cpu().numpy()

                (self.output_path / video_id).mkdir(parents=True, exist_ok=True)
                for key, val in images.items():
                    Image.fromarray(val).save(
                        self.output_path / video_id / '{0}-{1:06d}.jpg'.format(key, frame_id))

                if self.generate_ring_view:
                    ring_buffer.append(images)

                if len(ring_buffer) == 7:
                    merged_W = ring_buffer[1][RGB].shape[1] + \
                               ring_buffer[0][RGB].shape[1] + ring_buffer[2][RGB].shape[1]
                    merged_H = max(ring_buffer[0][RGB].shape[0], ring_buffer[1][RGB].shape[0] +
                                   ring_buffer[5][RGB].shape[0] + ring_buffer[3][RGB].shape[0])

                    offsets = [
                        (ring_buffer[1][RGB].shape[1], 0),
                        (0, 0),
                        (ring_buffer[1][RGB].shape[1] + ring_buffer[0][RGB].shape[1], 0),
                        (0, ring_buffer[1][RGB].shape[0] + ring_buffer[5][RGB].shape[0]),
                        (ring_buffer[1][RGB].shape[1] + ring_buffer[0][RGB].shape[1],
                         ring_buffer[1][RGB].shape[0] + ring_buffer[5][RGB].shape[0]),
                        (0, ring_buffer[1][RGB].shape[0]),
                        (ring_buffer[1][RGB].shape[1] + ring_buffer[0][RGB].shape[1], ring_buffer[1][RGB].shape[0])
                    ]

                    merged_images = []
                    for key, val in ring_buffer[0].items():
                        merged = np.zeros((merged_H, merged_W, 3), dtype=np.uint8)
                        for i, (offset_W, offset_H) in enumerate(offsets):
                            image = ring_buffer[i][key]
                            merged[offset_H:offset_H + image.shape[0], offset_W:offset_W + image.shape[1]] = image

                        merged_images.append(merged)
                        Image.fromarray(merged).save(
                            self.output_path / video_id / 'merged-{0}-{1:06d}.jpg'.format(key, frame_id // 7))

                    Image.fromarray(np.concatenate(merged_images, 1)).save(
                        self.output_path / video_id / 'merged-all-{0:06d}.jpg'.format(frame_id // 7))

                    ring_buffer = []

                progress.advance(task)

        if get_world_size() > 1:
            dist.barrier()


def entrypoint():
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(0, hours=24))
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderImages).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderImages)  # noqa
