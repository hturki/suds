from argparse import Namespace
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import configargparse
import numpy as np
import torch
from smart_open import open
from tqdm import tqdm

from metadata_utils import get_frame_range, get_bounds_from_depth, normalize_timestamp, scale_bounds, get_neighbor, \
    write_metadata, get_val_frames, OPENCV_TO_OPENGL
from suds.data.image_metadata import ImageMetadata
from suds.stream_utils import image_from_stream, get_filesystem


# From https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/devkits/convertOxtsPose/python/utils.py
def _lat_lon_to_mercator(lat: float, lon: float, scale: float) -> Tuple[float, float]:
    ''' converts lat/lon coordinates to mercator coordinates using mercator scale '''
    er = 6378137.  # average earth radius at the equator

    mx = scale * lon * np.pi * er / 180
    my = scale * er * np.log(np.tan((90 + lat) * np.pi / 360))

    return mx, my


# From https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/devkits/convertOxtsPose/python/utils.py
def _lat_to_scale(lat: float) -> float:
    ''' compute mercator scale from latitude '''
    scale = np.cos(lat * np.pi / 180.0)
    return scale


def get_kitti_items(kitti_root: str,
                    kitti_sequence: str,
                    frame_ranges: Optional[List[Tuple[int]]],
                    train_every: Optional[int],
                    test_every: Optional[int]) -> \
        Tuple[List[ImageMetadata], List[str], torch.Tensor, float, torch.Tensor]:
    calib: Dict[str, torch.Tensor] = {}
    with open('{}/calib/{}.txt'.format(kitti_root, kitti_sequence), 'r') as f:
        for line in f:
            tokens = line.strip().split()
            calib[tokens[0]] = torch.DoubleTensor([float(x) for x in tokens[1:]])

    imu2velo = torch.eye(4, dtype=torch.float64)
    imu2velo[:3] = calib['Tr_imu_velo'].view(3, 4)

    velo2cam_base = torch.eye(4, dtype=torch.float64)
    velo2cam_base[:3] = calib['Tr_velo_cam'].view(3, 4)

    cam_base2rect = torch.eye(4, dtype=torch.float64)
    cam_base2rect[:3, :3] = calib['R_rect'].view(3, 3)

    P2 = calib['P2:'].view(3, 4)
    K_inv = torch.inverse(P2[:, :3])
    R_t = P2[:, 3]
    rect2P2 = torch.eye(4, dtype=torch.float64)
    rect2P2[:3, 3] = torch.matmul(K_inv, R_t)
    P22imu = torch.inverse(rect2P2 @ cam_base2rect @ velo2cam_base @ imu2velo)

    P3 = calib['P3:'].view(3, 4)
    K_inv = torch.inverse(P3[:, :3])
    R_t = P3[:, 3]
    rect2P3 = torch.eye(4, dtype=torch.float64)
    rect2P3[:3, 3] = torch.matmul(K_inv, R_t)
    P32imu = torch.inverse(rect2P3 @ cam_base2rect @ velo2cam_base @ imu2velo)

    num_frames = 0
    with open('{}/oxts/{}.txt'.format(kitti_root, kitti_sequence), 'r') as f:
        for frame, line in enumerate(f):
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
    with open('{}/oxts/{}.txt'.format(kitti_root, kitti_sequence), 'r') as f:
        min_frame = None
        max_frame = None
        scale = None
        for frame, line in enumerate(tqdm(f)):
            frame_range = get_frame_range(frame_ranges, frame) if frame_ranges is not None else None
            if frame_ranges is not None and frame_range is None:
                continue

            min_frame = min(frame, min_frame) if min_frame is not None else frame
            max_frame = max(frame, max_frame) if max_frame is not None else frame
            oxts = [float(x) for x in line.strip().split()]
            if scale is None:
                scale = _lat_to_scale(oxts[0])

            imu_pose = torch.eye(4, dtype=torch.float64)
            imu_pose[0, 3], imu_pose[1, 3] = _lat_lon_to_mercator(oxts[0], oxts[1], scale)
            imu_pose[2, 3] = oxts[2]

            # From https://github.com/autonomousvision/kitti360Scripts/blob/master/kitti360scripts/devkits/convertOxtsPose/python/convertOxtsToPose.py
            rx = oxts[3]  # roll
            ry = oxts[4]  # pitch
            rz = oxts[5]  # heading
            Rx = torch.DoubleTensor([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)],
                                     [0, np.sin(rx), np.cos(rx)]])  # base => nav  (level oxts => rotated oxts)
            Ry = torch.DoubleTensor([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0],
                                     [-np.sin(ry), 0, np.cos(ry)]])  # base => nav  (level oxts => rotated oxts)
            Rz = torch.DoubleTensor([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0],
                                     [0, 0, 1]])  # base => nav  (level oxts => rotated oxts)
            imu_pose[:3, :3] = torch.matmul(torch.matmul(Rz, Ry), Rx)

            for camera, transformation, intrinsics in [('2', P22imu, P2), ('3', P32imu, P3)]:
                image_index = len(metadata_items)
                c2w = ((imu_pose @ transformation) @ OPENCV_TO_OPENGL)[:3]

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

                backward_flow_path = '{0}/dino_correspondences_0{1}{2}/{3}/{4:06d}.parquet'.format(kitti_root,
                                                                                                   camera,
                                                                                                   backward_suffix,
                                                                                                   kitti_sequence,
                                                                                                   frame - (
                                                                                                           image_index - backward_neighbor) // 2)
                forward_flow_path = '{0}/dino_correspondences_0{1}{2}/{3}/{4:06d}.parquet'.format(kitti_root,
                                                                                                  camera,
                                                                                                  forward_suffix,
                                                                                                  kitti_sequence,
                                                                                                  frame)

                image_path = '{0}/image_0{1}/{2}/{3:06d}.png'.format(kitti_root, camera, kitti_sequence, frame)
                image = image_from_stream(image_path)

                sky_mask_path = '{0}/sky_0{1}/{2}/{3:06d}.png'.format(kitti_root, camera, kitti_sequence, frame) \
                    if (camera == '2' and use_masks) else None
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
                    torch.DoubleTensor([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]),
                    image_index,
                    frame,
                    0,
                    '{0}/depth_0{1}/{2}/{3:06d}.parquet'.format(kitti_root, camera, kitti_sequence, frame),
                    None,
                    sky_mask_path,
                    '{0}/dino_0{1}/{2}/{3:06d}.parquet'.format(kitti_root, camera, kitti_sequence, frame),
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
                    static_mask_path = '{0}/static_02/{1}/{2:06d}.png'.format(kitti_root, kitti_sequence, frame) \
                        if camera == '2' else '{0}/all-false.png'.format(kitti_root)
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
    parser.add_argument('--kitti_root', type=str, default='data/kitti/training')
    parser.add_argument('--kitti_sequence', type=str, required=True)
    parser.add_argument('--frame_ranges', type=int, nargs='+', default=None)
    parser.add_argument('--train_every', type=int, default=None)
    parser.add_argument('--test_every', type=int, default=None)

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

    metadata_items, static_masks, origin, pose_scale_factor, scene_bounds = get_kitti_items(hparams.kitti_root,
                                                                                            hparams.kitti_sequence,
                                                                                            frame_ranges,
                                                                                            hparams.train_every,
                                                                                            hparams.test_every)

    write_metadata(hparams.output_path, metadata_items, static_masks, origin, pose_scale_factor, scene_bounds)


if __name__ == '__main__':
    main(_get_opts())
