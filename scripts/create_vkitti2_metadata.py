from argparse import Namespace
from pathlib import Path
from typing import List, Optional, Tuple

import configargparse
import torch
from smart_open import open
from tqdm import tqdm

from metadata_utils import get_frame_range, get_bounds_from_depth, get_neighbor, \
    write_metadata, get_val_frames, scale_bounds, OPENCV_TO_OPENGL, normalize_timestamp
from suds.data.image_metadata import ImageMetadata
from suds.stream_utils import image_from_stream, get_filesystem

GROUND_PLANE_Z = torch.DoubleTensor([[1, 0, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, 0, 1]])


def get_vkitti2_items(vkitti2_path: str,
                      frame_ranges: Optional[List[Tuple[int]]],
                      train_every: Optional[int],
                      test_every: Optional[int],
                      use_gt_flow: bool) -> \
        Tuple[List[ImageMetadata], List[str], torch.Tensor, float, torch.Tensor]:
    with open('{}/intrinsic.txt'.format(vkitti2_path), 'r') as in_f, \
            open('{}/extrinsic.txt'.format(vkitti2_path), 'r') as ex_f:
        # frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
        next(in_f)
        # frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1
        next(ex_f)

        num_frames = 0
        for in_line, ex_line in zip(in_f, ex_f):
            in_entry = in_line.strip().split()
            frame = int(in_entry[0])
            if frame_ranges is not None and get_frame_range(frame_ranges, frame) is None:
                continue
            num_frames += 1

    val_frames = get_val_frames(num_frames, test_every, train_every)
    metadata_items: List[ImageMetadata] = []
    item_frame_ranges: List[Tuple[int]] = []
    static_masks = []
    min_bounds = None
    max_bounds = None

    use_masks = True
    with open('{}/intrinsic.txt'.format(vkitti2_path), 'r') as in_f, \
            open('{}/extrinsic.txt'.format(vkitti2_path), 'r') as ex_f:
        # frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
        next(in_f)
        # frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1
        next(ex_f)

        min_frame = None
        max_frame = None
        for in_line, ex_line in tqdm(zip(in_f, ex_f)):
            in_entry = in_line.strip().split()
            frame = int(in_entry[0])
            frame_range = get_frame_range(frame_ranges, frame) if frame_ranges is not None else None
            if frame_ranges is not None and frame_range is None:
                continue

            min_frame = min(frame, min_frame) if min_frame is not None else frame
            max_frame = max(frame, max_frame) if max_frame is not None else frame
            cameraID = int(in_entry[1])

            w2c = torch.DoubleTensor([float(x) for x in ex_line.strip().split()[2:]]).view(4, 4)
            c2w = (GROUND_PLANE_Z @ (torch.inverse(w2c) @ OPENCV_TO_OPENGL))[:3]

            image_index = len(metadata_items)
            is_val = image_index // 2 in val_frames

            if is_val:
                backward_neighbor = image_index - 2
                forward_neighbor = image_index + 2
            else:
                backward_neighbor = get_neighbor(image_index, val_frames, -2)
                forward_neighbor = get_neighbor(image_index, val_frames, 2)

            backward_suffix = '' if (image_index - backward_neighbor) // 2 == 1 else '-{}'.format(
                (image_index - backward_neighbor) // 2)
            forward_suffix = '' if (forward_neighbor - image_index) // 2 == 1 else '-{}'.format(
                (forward_neighbor - image_index) // 2)

            if use_gt_flow:
                backward_flow_path = '{0}/frames/backwardFlow{1}/Camera_{2}/backwardFlow_{3:05d}.png'.format(
                    vkitti2_path, backward_suffix, cameraID, frame)
                forward_flow_path = '{0}/frames/forwardFlow{1}/Camera_{2}/flow_{3:05d}.png'.format(vkitti2_path,
                                                                                                   forward_suffix,
                                                                                                   cameraID, frame)
            else:
                backward_flow_path = '{0}/frames/dino_correspondences{1}/Camera_{2}/rgb_{3:05d}.parquet'.format(
                    vkitti2_path, forward_suffix, cameraID, frame - (image_index - backward_neighbor) // 2)
                forward_flow_path = '{0}/frames/dino_correspondences{1}/Camera_{2}/rgb_{3:05d}.parquet'.format(
                    vkitti2_path, forward_suffix, cameraID, frame)

            image_path = '{0}/frames/rgb/Camera_{1}/rgb_{2:05d}.jpg'.format(vkitti2_path, cameraID, frame)
            image = image_from_stream(image_path)

            sky_mask_path = '{0}/frames/sky_mask/Camera_{1}/sky_mask_{2:05d}.png'.format(vkitti2_path, cameraID, frame) \
                if use_masks else None
            if sky_mask_path is not None and use_masks:
                fs = get_filesystem(sky_mask_path)
                if (fs is None and (not Path(sky_mask_path).exists())) or \
                        (fs is not None and (not fs.exists(sky_mask_path))):
                    print('Did not find sky mask at {} - not including static or sky masks in metadata'.format(
                        sky_mask_path))
                    use_masks = False
                    sky_mask_path = None

            item = ImageMetadata(
                image_path,
                c2w,
                image.size[0],
                image.size[1],
                torch.FloatTensor([float(x) for x in in_line.strip().split()[2:]]),
                image_index,
                frame,
                0,
                '{0}/frames/depth/Camera_{1}/depth_{2:05d}.png'.format(vkitti2_path, cameraID, frame),
                None,
                sky_mask_path,
                '{0}/frames/dino/Camera_{1}/dino_{2:05d}.parquet'.format(vkitti2_path, cameraID, frame),
                backward_flow_path,
                forward_flow_path,
                backward_neighbor,
                forward_neighbor,
                is_val,
                1,
                None
            )

            metadata_items.append(item)
            item_frame_ranges.append(frame_range)

            if use_masks:
                static_mask_path = '{0}/frames/static_mask/Camera_{1}/static_mask_{2:05d}.png'.format(vkitti2_path,
                                                                                                      cameraID,
                                                                                                      frame)

                static_masks.append(static_mask_path)

            min_bounds, max_bounds = get_bounds_from_depth(item, min_bounds, max_bounds)

    for item in metadata_items:
        normalize_timestamp(item, min_frame, max_frame)

    for item in metadata_items:
        if item.backward_neighbor_index < 0 \
                or item_frame_ranges[item.image_index] != item_frame_ranges[item.backward_neighbor_index]:
            item.backward_flow_path = None
            item.backward_neighbor_index = None

        if item.forward_neighbor_index >= len(metadata_items) \
                or item_frame_ranges[item.image_index] != item_frame_ranges[item.forward_neighbor_index]:
            item.forward_flow_path = None
            item.forward_neighbor_index = None

    origin, pose_scale_factor, scene_bounds = scale_bounds(metadata_items, min_bounds, max_bounds)

    return metadata_items, static_masks, origin, pose_scale_factor, scene_bounds


def _get_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config_file', is_config_file=True)

    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--vkitti2_path', type=str, required=True)
    parser.add_argument('--frame_ranges', type=int, nargs='+', default=None)
    parser.add_argument('--train_every', type=int, default=None)
    parser.add_argument('--test_every', type=int, default=None)
    parser.add_argument('--use_gt_flow', default=False, action='store_true')

    return parser.parse_args()


def main(hparams: Namespace) -> None:
    assert hparams.train_every is not None or hparams.test_every is not None, \
        'Exactly one of train_every or test_every must be specified'

    assert hparams.train_every is None or hparams.test_every is None, \
        'Only one of train_every or test_every must be specified'

    if hparams.frame_ranges is not None:
        frame_ranges = []
        for i in range(0, len(hparams.frame_ranges), 2):
            frame_ranges.append([hparams.frame_ranges[i], hparams.frame_ranges[i + 1]])
    else:
        frame_ranges = None

    metadata_items, static_masks, origin, pose_scale_factor, scene_bounds = get_vkitti2_items(hparams.vkitti2_path,
                                                                                              frame_ranges,
                                                                                              hparams.train_every,
                                                                                              hparams.test_every,
                                                                                              hparams.use_gt_flow)

    write_metadata(hparams.output_path, metadata_items, static_masks, origin, pose_scale_factor, scene_bounds)


if __name__ == '__main__':
    main(_get_opts())
