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


def load_class_features(item: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, torch.Tensor]]:
    feature_path = item['feature_path']

    table = pq.read_table(feature_path)
    features = torch.FloatTensor(table['pca'].to_numpy()).view(
        [int(x) for x in table.schema.metadata[b'shape'].split()])

    if (features.shape[0] != item['H'] or features.shape[1] != item['W']):
        features = F.interpolate(features.permute(2, 0, 1).unsqueeze(0), size=(item['H'], item['W'])).squeeze() \
            .permute(1, 2, 0)

    frame = Path(feature_path).stem
    categories = torch.LongTensor(np.asarray(image_from_stream(
        item['depth_path'].replace('/depth/', '/classSegmentation/').replace('depth_', 'classgt_')))).sum(dim=-1)

    sorted_categories, ordering = categories.sort()
    unique_categories, counts = torch.unique_consecutive(sorted_categories, return_counts=True)

    category_features = {}
    category_counts = {}

    offset = 0
    for category, category_count in zip(unique_categories, counts):
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
    parser.add_argument('--vkitti2_path', type=str, required=True)
    parser.add_argument('--subset_ratio', type=float, default=0.1)

    return parser.parse_known_args()[0]


def main(hparams: Namespace) -> None:
    with open(hparams.metadata_path) as f:
        metadata = json.load(f)

    class_features = {}
    class_counts = defaultdict(int)

    frames = metadata['frames']

    with open('{}/colors.txt'.format(hparams.vkitti2_path)) as f:
        # Category r g b
        next(f)  # skip header
        # Terrain 210 0 200
        color_mappings = []
        for line in f:
            color_mappings.append([int(x) for x in line.strip().split()[1:]])
        color_mappings = torch.LongTensor(color_mappings)
        colors = color_mappings.sum(dim=-1)
        assert torch.unique(colors).shape[0] == colors.shape[0]

        all_category_colors = torch.zeros(colors.max() + 2, 3, dtype=torch.uint8)
        for color, color_mapping in zip(colors, color_mappings):
            all_category_colors[color] = color_mapping

    indices = np.linspace(0, len(frames), int(len(frames) * hparams.subset_ratio), endpoint=False,
                          dtype=np.int32)

    random.seed(42)
    random.shuffle(indices)

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {}
        for index in indices:
            futures[index] = executor.submit(load_class_features, frames[index])

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
        category_colors.append(all_category_colors[key:key + 1])

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
