"""Code adapted and modified from https://github.com/ShirAmir/dino-vit-features"""

import argparse
import datetime
import json
import os
import traceback
from pathlib import Path
from typing import List, Tuple

import configargparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from PIL import Image
from sklearn.cluster import KMeans
from smart_open import open
from tqdm import tqdm

from extract_dino_features import ViTExtractor
from suds.stream_utils import get_filesystem


def find_correspondences(extractor: ViTExtractor, image_path1: str, image_path2: str, num_pairs: int = 10,
                         load_size: int = 224, layer: int = 9,
                         facet: str = 'key', bin: bool = True, thresh: float = 0.05) -> Tuple[
    List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:
    """
    finding point correspondences between two images.
    :param image_path1: path to the first image.
    :param image_path2: path to the second image.
    :param num_pairs: number of outputted corresponding pairs.
    :param load_size: size of the smaller edge of loaded images. If None, does not resize.
    :param layer: layer to extract descriptors from.
    :param facet: facet to extract descriptors from.
    :param bin: if True use a log-binning descriptor.
    :param thresh: threshold of saliency maps to distinguish fg and bg.
    :param model_type: type of model to extract descriptors from.
    :param stride: stride of the model.
    :return: list of points from image_path1, list of corresponding points from image_path2, the processed pil image of
    image_path1, and the processed pil image of image_path2.
    """
    # extracting descriptors for each image
    image1_batch, image1_pil = extractor.preprocess(image_path1, load_size)
    descriptors1 = extractor.extract_descriptors(image1_batch.to(extractor.device), layer, facet, bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size
    image2_batch, image2_pil = extractor.preprocess(image_path2, load_size)
    descriptors2 = extractor.extract_descriptors(image2_batch.to(extractor.device), layer, facet, bin)
    num_patches2, load_size2 = extractor.num_patches, extractor.load_size

    # extracting saliency maps for each image
    saliency_map1 = extractor.extract_saliency_maps(image1_batch.to(extractor.device))[0]
    saliency_map2 = extractor.extract_saliency_maps(image2_batch.to(extractor.device))[0]
    # threshold saliency maps to get fg / bg masks
    fg_mask1 = saliency_map1 > thresh
    fg_mask2 = saliency_map2 > thresh

    # calculate similarity between image1 and image2 descriptors
    similarities = chunk_cosine_sim(descriptors1, descriptors2)

    # calculate best buddies
    image_idxs = torch.arange(num_patches1[0] * num_patches1[1], device=extractor.device)
    sim_1, nn_1 = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_2, nn_2 = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
    sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
    sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]
    bbs_mask = nn_2[nn_1] == image_idxs

    # remove best buddies where at least one descriptor is marked bg by saliency mask.
    fg_mask2_new_coors = nn_2[fg_mask2]
    fg_mask2_mask_new_coors = torch.zeros(num_patches1[0] * num_patches1[1], dtype=torch.bool, device=extractor.device)
    fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

    # applying k-means to extract k high quality well distributed correspondence pairs
    bb_descs1 = descriptors1[0, 0, bbs_mask, :].cpu().numpy()
    bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].cpu().numpy()
    # apply k-means on a concatenation of a pairs descriptors.
    all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
    n_clusters = min(num_pairs, len(all_keys_together))  # if not enough pairs, show all found pairs.
    length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
    normalized = all_keys_together / length
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(normalized)
    bb_topk_sims = np.full((n_clusters), -np.inf)
    bb_indices_to_show = np.full((n_clusters), -np.inf)

    # rank pairs by their mean saliency value
    bb_cls_attn1 = saliency_map1[bbs_mask]
    bb_cls_attn2 = saliency_map2[nn_1[bbs_mask]]
    bb_cls_attn = (bb_cls_attn1 + bb_cls_attn2) / 2
    ranks = bb_cls_attn

    for k in range(n_clusters):
        for i, (label, rank) in enumerate(zip(kmeans.labels_, ranks)):
            if rank > bb_topk_sims[label]:
                bb_topk_sims[label] = rank
                bb_indices_to_show[label] = i

    # get coordinates to show
    indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
        bb_indices_to_show]  # close bbs
    img1_indices_to_show = torch.arange(num_patches1[0] * num_patches1[1], device=extractor.device)[indices_to_show]
    img2_indices_to_show = nn_1[indices_to_show]
    # coordinates in descriptor map's dimensions
    img1_y_to_show = (img1_indices_to_show / num_patches1[1]).cpu().numpy()
    img1_x_to_show = (img1_indices_to_show % num_patches1[1]).cpu().numpy()
    img2_y_to_show = (img2_indices_to_show / num_patches2[1]).cpu().numpy()
    img2_x_to_show = (img2_indices_to_show % num_patches2[1]).cpu().numpy()
    points1, points2 = [], []
    for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
        x1_show = (int(x1) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y1_show = (int(y1) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        x2_show = (int(x2) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y2_show = (int(y2) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        points1.append((y1_show, x1_show))
        points2.append((y2_show, x2_show))
    return points1, points2, image1_pil, image2_pil


def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: a tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def _get_opts() -> argparse.Namespace:
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add_argument('--config_file', is_config_file=True)

    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--load_size', default=375, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=4, type=int, help="""stride of first convolution layer. 
                                                              small stride -> higher resolution.""")
    parser.add_argument('--model_type', default='dino_vits8', type=str,
                        help="""type of model to extract. 
                        Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                        vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                    options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default=False, action='store_true', help='create a binned descriptor if True.')
    parser.add_argument('--thresh', default=0.05, type=float, help='saliency maps threshold to distinguish fg / bg.')
    parser.add_argument('--num_pairs', default=50000, type=int, help='Final number of correspondences.')

    return parser.parse_known_args()[0]


@torch.inference_mode()
def main(hparams: argparse.Namespace) -> None:
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(0, hours=24))
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = 0
        world_size = 1

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if rank == 0:
        # Create the extractor on only one device to avoid race conditions when downloading the weights across
        # multiple processes
        extractor = ViTExtractor(hparams.model_type, hparams.stride, device=device)
        if world_size > 1:
            dist.barrier()
    else:
        if world_size > 1:
            dist.barrier()
        extractor = ViTExtractor(hparams.model_type, hparams.stride, device=device)

    with open(hparams.metadata_path) as f:
        metadata = json.load(f)

    frames = metadata['frames']
    for i in tqdm(np.arange(rank, len(frames), world_size)):
        frame = frames[i]
        for flow_path, neighbor_index in [(frame['forward_flow_path'], frame['forward_neighbor_index']), (
                frame['backward_flow_path'], frame['backward_neighbor_index'])]:
            if flow_path is None:
                continue

            fs = get_filesystem(flow_path)

            if (fs is None and Path(flow_path).exists()) or (fs is not None and fs.exists(flow_path)):
                try:
                    pq.read_table(flow_path, filesystem=fs)['point1_x'].to_numpy().sum()
                    continue
                except:
                    traceback.print_exc()

            if fs is None:
                parent = Path(flow_path).parent
                if not parent.exists():
                    parent.mkdir(parents=True, exist_ok=True)

            first = frames[min(i, neighbor_index)]['rgb_path']
            second = frames[max(i, neighbor_index)]['rgb_path']
            points1, points2, image1_pil, image2_pil = find_correspondences(extractor, first, second,
                                                                            hparams.num_pairs, hparams.load_size,
                                                                            hparams.layer, hparams.facet, hparams.bin,
                                                                            hparams.thresh)

            pq.write_table(pa.table({
                'point1_x': np.array([x[1] for x in points1]),
                'point1_y': np.array([x[0] for x in points1]),
                'point2_x': np.array([x[1] for x in points2]),
                'point2_y': np.array([x[0] for x in points2])
            }, metadata={'shape': ' '.join([str(image1_pil.size[1]), str(image1_pil.size[0])])}),
                flow_path, filesystem=fs, compression='BROTLI')

    if world_size > 1:
        dist.barrier()


if __name__ == '__main__':
    main(_get_opts())
