from argparse import Namespace
from pathlib import Path

import configargparse
import cv2
import numpy as np
import torch
from PIL import Image
from smart_open import open
from tqdm import tqdm

from suds.stream_utils import image_from_stream, image_to_stream, get_filesystem

MOVER_CLASSES = [280, 462, 278]
SKY_CLASS = 545

def write_masks(vkitti2_path: str, dilation: int) -> None:
    if get_filesystem(vkitti2_path) is None:
        for i in range(2):
            (Path(vkitti2_path) / 'frames' / 'static_mask' / 'Camera_{}'.format(i)).mkdir(parents=True)
            (Path(vkitti2_path) / 'frames' / 'sky_mask' / 'Camera_{}'.format(i)).mkdir(parents=True)

    with open('{}/intrinsic.txt'.format(vkitti2_path), 'r') as in_f, \
            open('{}/extrinsic.txt'.format(vkitti2_path), 'r') as ex_f:
        # frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
        next(in_f)
        # frame cameraID r1,1 r1,2 r1,3 t1 r2,1 r2,2 r2,3 t2 r3,1 r3,2 r3,3 t3 0 0 0 1
        next(ex_f)

        for in_line, ex_line in tqdm(zip(in_f, ex_f)):
            in_entry = in_line.strip().split()
            frame = int(in_entry[0])
            cameraID = int(in_entry[1])

            category_path = '{0}/frames/classSegmentation/Camera_{1}/classgt_{2:05d}.png'.format(
                vkitti2_path, cameraID, frame)
            category = torch.LongTensor(np.asarray(image_from_stream(category_path))).sum(dim=-1)
            mover = torch.zeros_like(category, dtype=torch.bool)
            for mover_class in MOVER_CLASSES:
                mover[category == mover_class] = True
            mover = mover.float().numpy()
            kernel = np.ones((dilation, dilation), dtype=np.float32)
            mover = cv2.dilate(mover, kernel)
            static_mask = Image.fromarray(mover <= 0)

            image_to_stream(static_mask, '{0}/frames/static_mask/Camera_{1}/static_mask_{2:05d}.png'.format(
                vkitti2_path, cameraID, frame))

            sky_mask = Image.fromarray((category == SKY_CLASS).numpy())
            image_to_stream(sky_mask, '{0}/frames/sky_mask/Camera_{1}/sky_mask_{2:05d}.png'.format(
                vkitti2_path, cameraID, frame))


def _get_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config_file', is_config_file=True)

    parser.add_argument('--vkitti2_path', type=str, required=True)
    parser.add_argument('--dilation', type=int, default=30)

    return parser.parse_known_args()[0]


def main(hparams: Namespace) -> None:
    write_masks(hparams.vkitti2_path, hparams.dilation)


if __name__ == '__main__':
    main(_get_opts())
