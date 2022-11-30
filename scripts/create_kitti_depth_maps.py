from smart_open import open
from tqdm import tqdm

MOVER_CLASSES = [11, 12, 13, 14, 15, 16, 17, 18, 255]
SKY_CLASS = 10

from argparse import Namespace
from pathlib import Path

import configargparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch_scatter import scatter_min

from suds.stream_utils import image_from_stream, get_filesystem, buffer_from_stream


@torch.inference_mode()
def write_depth_maps(kitti_root: str, kitti_sequence: str) -> None:
    fs = get_filesystem(kitti_root)

    if fs is None:
        (Path(kitti_root) / 'depth_02' / kitti_sequence).mkdir(parents=True)
        (Path(kitti_root) / 'depth_03' / kitti_sequence).mkdir(parents=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open('{}/calib/{}.txt'.format(kitti_root, kitti_sequence), 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if tokens[0] == 'P2:':
                P2 = torch.eye(4, dtype=torch.float64, device=device)
                P2[:3] = torch.DoubleTensor([float(x) for x in tokens[1:]]).view(3, 4)
            if tokens[0] == 'P3:':
                P3 = torch.eye(4, dtype=torch.float64, device=device)
                P3[:3] = torch.DoubleTensor([float(x) for x in tokens[1:]]).view(3, 4)
            if tokens[0] == 'R_rect':
                R_rect = torch.eye(4, dtype=torch.float64, device=device)
                R_rect[:3, :3] = torch.DoubleTensor([float(x) for x in tokens[1:]]).view(3, 3)
            if tokens[0] == 'Tr_velo_cam':
                Tr_velo_cam = torch.eye(4, dtype=torch.float64, device=device)
                Tr_velo_cam[:3] = torch.DoubleTensor([float(x) for x in tokens[1:]]).view(3, 4)

    with open('{}/oxts/{}.txt'.format(kitti_root, kitti_sequence), 'r') as f:
        for frame, line in enumerate(tqdm(f)):
            lidar_points = np.frombuffer(buffer_from_stream(
                '{0}/velodyne/{1}/{2:06d}.bin'.format(kitti_root, kitti_sequence, frame)).getbuffer(),
                                         dtype=np.float32).reshape(-1, 4)
            lidar_points = torch.DoubleTensor(lidar_points).to(device)
            lidar_points[:, 3] = 1
            lidar_points_T = lidar_points.T
            for camera, transform in [('2', P2), ('3', P3)]:
                points_cam = (transform @ R_rect @ Tr_velo_cam @ lidar_points_T).T[:, :3]

                image_path = '{0}/image_0{1}/{2}/{3:06d}.png'.format(kitti_root, camera, kitti_sequence, frame)
                image = image_from_stream(image_path)
                depth_map = torch.ones(image.size[1], image.size[0], device=device, dtype=torch.float64) \
                            * torch.finfo(torch.float64).max
                points_cam[:, :2] = points_cam[:, :2] / points_cam[:, 2].view(-1, 1)
                is_valid_x = torch.logical_and(0 <= points_cam[:, 0], points_cam[:, 0] < image.size[0] - 1)
                is_valid_y = torch.logical_and(0 <= points_cam[:, 1], points_cam[:, 1] < image.size[1] - 1)
                is_valid_z = points_cam[:, 2] > 0
                is_valid_points = torch.logical_and(torch.logical_and(is_valid_x, is_valid_y), is_valid_z)

                assert is_valid_points.sum() > 0

                u = torch.round(points_cam[:, 0][is_valid_points]).long().cuda()
                v = torch.round(points_cam[:, 1][is_valid_points]).long().cuda()
                z = points_cam[:, 2][is_valid_points].cuda()
                scatter_min(z, v * image.size[0] + u, out=depth_map.view(-1))

                depth_map[depth_map >= torch.finfo(torch.float64).max - 1e-5] = 0
                pq.write_table(pa.table({'depth': depth_map.cpu().float().numpy().flatten()}),
                               '{0}/depth_0{1}/{2}/{3:06d}.parquet'.format(kitti_root, camera, kitti_sequence, frame),
                               filesystem=fs, compression='BROTLI')


def _get_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config_file', is_config_file=True)

    parser.add_argument('--kitti_root', type=str, default='data/kitti/training')
    parser.add_argument('--kitti_sequence', type=str, required=True)

    return parser.parse_known_args()[0]


def main(hparams: Namespace) -> None:
    write_depth_maps(hparams.kitti_root, hparams.kitti_sequence)


if __name__ == '__main__':
    main(_get_opts())
