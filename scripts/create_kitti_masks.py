from argparse import Namespace
from pathlib import Path

import configargparse
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from suds.stream_utils import image_from_stream, image_to_stream, get_filesystem

MOVER_CLASSES = [11, 12, 13, 14, 15, 16, 17, 18, 255]
SKY_CLASS = 10


def write_static_masks(kitti_root: str, kitti_step_path: str, kitti_sequence: str, dilation: int) -> None:
    if get_filesystem(kitti_root) is None:
        (Path(kitti_root) / 'static_02' / kitti_sequence).mkdir(parents=True)
        (Path(kitti_root) / 'sky_02' / kitti_sequence).mkdir(parents=True)

    sequence_path = '{}/train/{}'.format(kitti_step_path, kitti_sequence)
    fs = get_filesystem(sequence_path)
    if (fs is None and (not Path(sequence_path).exists())) or (fs is not None and (not fs.exists(sequence_path))):
        sequence_path = '{}/val/{}'.format(kitti_step_path, kitti_sequence)

    with open('{}/oxts/{}.txt'.format(kitti_root, kitti_sequence), 'r') as f:
        for frame, line in enumerate(tqdm(f)):
            category_path_2 = '{0}/{1:06d}.png'.format(sequence_path, frame)
            category = torch.LongTensor(np.asarray(image_from_stream(category_path_2)))[:, :, 0]

            mover = torch.zeros_like(category, dtype=torch.bool)
            for mover_class in MOVER_CLASSES:
                mover[category == mover_class] = True
            mover = mover.float().numpy()
            kernel = np.ones((dilation, dilation), dtype=np.float32)
            mover = cv2.dilate(mover, kernel)
            static_mask = Image.fromarray(mover <= 0)

            image_to_stream(static_mask, '{0}/static_02/{1}/{2:06d}.png'.format(kitti_root, kitti_sequence, frame))

            sky_mask = Image.fromarray((category == SKY_CLASS).numpy())
            image_to_stream(sky_mask, '{0}/sky_02/{1}/{2:06d}.png'.format(kitti_root, kitti_sequence, frame))

            if frame == 0:
                # create all-false for camera 3
                all_false = Image.fromarray(np.zeros_like(mover <= 0))
                image_to_stream(all_false, '{0}/all-false.png'.format(kitti_root))


def _get_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config_file', is_config_file=True)

    parser.add_argument('--kitti_root', type=str, default='data/kitti/training')
    parser.add_argument('--kitti_step_path', type=str, default='data/kitti/kitti-step/panoptic_maps')
    parser.add_argument('--kitti_sequence', type=str, required=True)
    parser.add_argument('--dilation', type=int, default=30)

    return parser.parse_known_args()[0]


def main(hparams: Namespace) -> None:
    write_static_masks(hparams.kitti_root, hparams.kitti_step_path, hparams.kitti_sequence, hparams.dilation)


if __name__ == '__main__':
    main(_get_opts())
