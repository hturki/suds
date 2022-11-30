import json
import multiprocessing
import random
from argparse import Namespace
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple, Any

import configargparse
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from smart_open import open
from tqdm import tqdm

from suds.stream_utils import get_filesystem, image_from_stream

CITYSCAPE_COLORS = torch.ByteTensor([[128, 64, 128],
                                     [244, 35, 232],
                                     [70, 70, 70],
                                     [102, 102, 156],
                                     [190, 153, 153],
                                     [153, 153, 153],
                                     [250, 170, 30],
                                     [220, 220, 0],
                                     [107, 142, 35],
                                     [152, 251, 152],
                                     [70, 130, 180],
                                     [220, 20, 60],
                                     [255, 0, 0],
                                     [0, 0, 142],
                                     [0, 0, 70],
                                     [0, 60, 100],
                                     [0, 80, 100],
                                     [0, 0, 230],
                                     [119, 11, 32]])


def load_class_features(sequence_path: str, item: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, torch.Tensor]]:
    feature_path = item['feature_path']

    table = pq.read_table(feature_path)
    features = torch.FloatTensor(table['pca'].to_numpy()).view(
        [int(x) for x in table.schema.metadata[b'shape'].split()])

    if (features.shape[0] != item['H'] or features.shape[1] != item['W']):
        features = F.interpolate(features.permute(2, 0, 1).unsqueeze(0), size=(item['H'], item['W'])).squeeze() \
            .permute(1, 2, 0)

    frame = Path(feature_path).stem
    categories = torch.LongTensor(np.asarray(image_from_stream('{}/{}.png'.format(sequence_path, frame))))[:, :,
                 0].view(-1)

    sorted_categories, ordering = categories.sort()
    unique_categories, counts = torch.unique_consecutive(sorted_categories, return_counts=True)

    category_features = {}
    category_counts = {}

    offset = 0
    for category, category_count in zip(unique_categories, counts):
        if category > 18:
            continue

        category_counts[category.item()] = category_count.item()
        category_features[category.item()] = features.view(-1, features.shape[-1])[
            ordering[offset:offset + category_count]].sum(dim=0)
        offset += category_count

    return category_features, category_counts


def _get_opts() -> Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config_file', is_config_file=True)

    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--kitti_step_path', type=str, default='data/kitti/kitti-step/panoptic_maps')
    parser.add_argument('--subset_ratio', type=float, default=0.1)

    return parser.parse_known_args()[0]


def main(hparams: Namespace) -> None:
    with open(hparams.metadata_path) as f:
        metadata = json.load(f)

    class_features = {}
    class_counts = defaultdict(int)

    frames = metadata['frames']
    frames_with_sem = list(filter(lambda x: '/dino_02/' in x['feature_path'], frames))
    indices = np.linspace(0, len(frames_with_sem), int(len(frames) * hparams.subset_ratio), endpoint=False,
                          dtype=np.int32)

    random.seed(42)
    random.shuffle(indices)

    kitti_sequence = Path(metadata['frames'][0]['rgb_path']).parent.name
    sequence_path = '{}/train/{}'.format(hparams.kitti_step_path, kitti_sequence)
    fs = get_filesystem(sequence_path)
    if (fs is None and (not Path(sequence_path).exists())) or (fs is not None and (not fs.exists(sequence_path))):
        sequence_path = '{}/val/{}'.format(hparams.kitti_step_path, kitti_sequence)

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {}
        for index in indices:
            futures[index] = executor.submit(load_class_features, sequence_path, frames[index])

        for index in tqdm(indices):
            frame_class_features, frame_class_counts = futures[index].result()
            for key, val in frame_class_features.items():
                if key not in class_features:
                    class_features[key] = val
                else:
                    class_features[key] = class_features[key] + val

                class_counts[key] += frame_class_counts[key]
            del futures[index]

    class_clusters = []
    category_colors = []
    for key in sorted(class_features.keys()):
        class_clusters.append((class_features[key] / class_counts[key]).unsqueeze(0))
        category_colors.append(CITYSCAPE_COLORS[key:key + 1])

    class_clusters = torch.cat(class_clusters)
    category_colors = torch.cat(category_colors)

    buffer = BytesIO()
    torch.save({'centroids': class_clusters, 'colors': category_colors}, buffer)

    if get_filesystem(hparams.output_path) is None:
        Path(hparams.output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(hparams.output_path, 'wb') as f:
        f.write(buffer.getbuffer())


if __name__ == '__main__':
    main(_get_opts())
