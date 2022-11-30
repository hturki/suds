"""Code adapted and modified from https://github.com/ShirAmir/dino-vit-features"""

import argparse
import datetime
import json
import os
import traceback
from io import BytesIO
from pathlib import Path

import configargparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from sklearn.decomposition import PCA
from smart_open import open
from tqdm import tqdm

from suds.stream_utils import get_filesystem, buffer_from_stream


def _get_opts() -> argparse.Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config_file', is_config_file=True)

    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--n_components', default=64, type=int, help="number of pca components to produce.")
    parser.add_argument('--no_tmp_cleanup', dest='tmp_cleanup', default=True, action='store_false')

    return parser.parse_known_args()[0]


@torch.inference_mode()
def main(hparams: argparse.Namespace) -> None:
    with open(hparams.metadata_path) as f:
        metadata = json.load(f)

    descriptors_list = []
    num_patches_list = []

    frames = metadata['frames']
    for frame in tqdm(frames):
        descriptor = torch.load(buffer_from_stream('{}.pt'.format(frame['feature_path'])), map_location='cpu')
        num_patches_list.append([descriptor.shape[0], descriptor.shape[1]])
        descriptor /= descriptor.norm(dim=-1, keepdim=True)
        descriptors_list.append(descriptor.view(-1, descriptor.shape[2]).numpy())

    descriptors = np.concatenate(descriptors_list, axis=0)
    print('Running PCA on descriptors of dim: {}'.format(descriptors.shape))
    pca_descriptors = PCA(n_components=hparams.n_components, random_state=42).fit_transform(descriptors)
    split_idxs = np.array([num_patches[0] * num_patches[1] for num_patches in num_patches_list])
    split_idxs = np.cumsum(split_idxs)
    pca_per_image = np.split(pca_descriptors, split_idxs[:-1], axis=0)

    results = [(frame, img_pca.reshape((num_patches[0], num_patches[1], hparams.n_components))) for
               (frame, img_pca, num_patches) in zip(frames, pca_per_image, num_patches_list)]

    for frame, img_pca in tqdm(results):
        fs = get_filesystem(frame['feature_path'])
        pq.write_table(
            pa.table({'pca': img_pca.flatten()}, metadata={'shape': ' '.join([str(x) for x in img_pca.shape])}),
            frame['feature_path'], filesystem=fs, compression='BROTLI')

        if hparams.tmp_cleanup:
            tmp_path = '{}.pt'.format(frame['feature_path'])
            if fs is None:
                Path(tmp_path).unlink()
            else:
                fs.rm(tmp_path)


if __name__ == '__main__':
    main(_get_opts())
