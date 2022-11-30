from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto
from typing import Dict, Set, List

import torch
from rich.console import Console
from torch.utils.data import Dataset

from suds.data.dataset_utils import get_w2c_and_K
from suds.data.image_metadata import ImageMetadata
from suds.suds_constants import RGB, PIXEL_INDEX, IMAGE_INDEX, RAY_INDEX, TIME, VIDEO_ID, DEPTH, FEATURES, \
    BACKWARD_FLOW, BACKWARD_FLOW_VALID, FORWARD_FLOW, FORWARD_FLOW_VALID, BACKWARD_NEIGHBOR_TIME_DIFF, \
    FORWARD_NEIGHBOR_TIME_DIFF, BACKWARD_NEIGHBOR_W2C, BACKWARD_NEIGHBOR_K, FORWARD_NEIGHBOR_W2C, FORWARD_NEIGHBOR_K, \
    SKY, MASK

CONSOLE = Console(width=120)


class Split(Enum):
    TRAIN = auto()
    VAL = auto()
    ALL = auto()


class SUDSDataset(Dataset):

    def __init__(self,
                 all_items: List[ImageMetadata],
                 load_depth: bool,
                 load_features: bool,
                 load_flow: bool,
                 load_sky: bool,
                 load_on_demand: Set[str],
                 items_per_chunk: int,
                 load_random_subset: bool,
                 split: Split):
        super(SUDSDataset, self).__init__()

        self.all_items = all_items
        self.load_depth = load_depth
        self.load_features = load_features
        self.load_flow = load_flow
        self.load_sky = load_sky
        self.items_per_chunk = items_per_chunk
        self.load_on_demand = load_on_demand
        self.split = split

        if load_flow:
            K = []
            w2c = []
            for item in all_items:
                item_w2c, item_K = get_w2c_and_K(item)

                K.append(item_K.unsqueeze(0))
                w2c.append(item_w2c.unsqueeze(0))

            self.K = torch.cat(K)
            self.w2c = torch.cat(w2c)

        self.chunk_load_executor = ThreadPoolExecutor(max_workers=1)
        self.on_demand_executor = ThreadPoolExecutor(max_workers=16)

        self.load_random_subset = load_random_subset
        if load_random_subset:
            image_indices_to_sample = []
            self.pixel_indices_to_sample = torch.empty(max([x.image_index for x in all_items]) + 1, dtype=torch.long)
            for item in all_items:
                should_sample = split == Split.ALL or (item.is_val and split == Split.VAL) \
                                or ((not item.is_val) and split == Split.TRAIN)

                if should_sample:
                    image_indices_to_sample.append(item.image_index)
                    self.pixel_indices_to_sample[item.image_index] = item.W * item.H

            self.image_indices_to_sample = torch.LongTensor(image_indices_to_sample)
            assert len(self.image_indices_to_sample) > 0, self.split
        else:
            self.memory_fields = None

        self.loaded_fields = None
        self.loaded_field_offset = 0
        self.chunk_future = None
        self.loaded_chunk = None

    def load_chunk(self) -> None:
        if self.chunk_future is None:
            self.chunk_future = self.chunk_load_executor.submit(self._load_chunk_inner)

        self.loaded_chunk = self.chunk_future.result()
        self.chunk_future = self.chunk_load_executor.submit(self._load_chunk_inner)

    def __len__(self) -> int:
        return self.loaded_chunk[RGB].shape[0] if self.loaded_chunk is not None else 0

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = {}
        for key, value in self.loaded_chunk.items():
            if key != PIXEL_INDEX:
                item[key] = value[idx]

        metadata_item = self.all_items[item[IMAGE_INDEX]]
        width = metadata_item.W

        # image index, row, col
        item[RAY_INDEX] = torch.LongTensor([
            item[IMAGE_INDEX],
            self.loaded_chunk[PIXEL_INDEX][idx] // width,
            self.loaded_chunk[PIXEL_INDEX][idx] % width])

        item[TIME] = torch.FloatTensor([metadata_item.time])
        item[VIDEO_ID] = torch.IntTensor([metadata_item.video_id])

        if self.load_flow:
            if metadata_item.backward_neighbor_index is not None:
                assert 0 <= metadata_item.backward_neighbor_index < len(self.all_items)
                item[BACKWARD_NEIGHBOR_W2C] = self.w2c[metadata_item.backward_neighbor_index]
                item[BACKWARD_NEIGHBOR_K] = self.K[metadata_item.backward_neighbor_index]
                item[BACKWARD_NEIGHBOR_TIME_DIFF] = \
                    torch.FloatTensor([metadata_item.time - self.all_items[metadata_item.backward_neighbor_index].time])
            else:
                item[BACKWARD_NEIGHBOR_W2C] = torch.zeros(4, 4)
                item[BACKWARD_NEIGHBOR_K] = torch.zeros(3, 3)
                item[BACKWARD_NEIGHBOR_TIME_DIFF] = torch.zeros(1)

            if metadata_item.forward_neighbor_index is not None:
                assert 0 <= metadata_item.forward_neighbor_index < len(self.all_items)
                item[FORWARD_NEIGHBOR_W2C] = self.w2c[metadata_item.forward_neighbor_index]
                item[FORWARD_NEIGHBOR_K] = self.K[metadata_item.forward_neighbor_index]
                item[FORWARD_NEIGHBOR_TIME_DIFF] = \
                    torch.FloatTensor([self.all_items[metadata_item.forward_neighbor_index].time - metadata_item.time])
            else:
                item[FORWARD_NEIGHBOR_W2C] = torch.zeros(4, 4)
                item[FORWARD_NEIGHBOR_K] = torch.zeros(3, 3)
                item[FORWARD_NEIGHBOR_TIME_DIFF] = torch.zeros(1)

            assert item[BACKWARD_NEIGHBOR_TIME_DIFF].min() >= 0, item[BACKWARD_NEIGHBOR_TIME_DIFF].min()
            assert item[FORWARD_NEIGHBOR_TIME_DIFF].min() >= 0, item[FORWARD_NEIGHBOR_TIME_DIFF].min()

        return item

    def _load_chunk_inner(self) -> Dict[str, torch.Tensor]:
        loaded_chunk = defaultdict(list)
        loaded = 0

        while loaded < self.items_per_chunk:
            if self.loaded_fields is None or self.loaded_field_offset >= len(self.loaded_fields[IMAGE_INDEX]):
                self.loaded_fields = {}
                self.loaded_field_offset = 0

                if self.load_random_subset:
                    to_shuffle = self._load_random_subset()
                else:
                    if self.memory_fields is None:
                        self.memory_fields = self._load_metadata_into_memory()
                    to_shuffle = self.memory_fields

                shuffled_indices = torch.randperm(len(to_shuffle[IMAGE_INDEX]))
                for key, val in to_shuffle.items():
                    self.loaded_fields[key] = val[shuffled_indices]

            to_add = self.items_per_chunk - loaded
            for key, val in self.loaded_fields.items():
                loaded_chunk[key].append(val[self.loaded_field_offset:self.loaded_field_offset + to_add])

            added = len(self.loaded_fields[IMAGE_INDEX][self.loaded_field_offset:self.loaded_field_offset + to_add])
            loaded += added
            self.loaded_field_offset += added

        loaded_chunk = {k: torch.cat(v) for k, v in loaded_chunk.items()}

        if len(self.load_on_demand) > 0:
            loaded_fields = self._load_fields(loaded_chunk[IMAGE_INDEX], loaded_chunk[PIXEL_INDEX],
                                              self.load_on_demand, True)
            for key, val in loaded_fields.items():
                loaded_chunk[key] = val

        return loaded_chunk

    def _load_random_subset(self) -> Dict[str, torch.Tensor]:
        image_indices = self.image_indices_to_sample[
            torch.randint(0, len(self.image_indices_to_sample), (self.items_per_chunk,))]
        pixel_indices = (
                torch.rand((self.items_per_chunk,)) * self.pixel_indices_to_sample[image_indices]).floor().long()

        mask = self._load_fields(image_indices, pixel_indices, {MASK})[MASK]

        return {
            IMAGE_INDEX: image_indices[mask > 0],
            PIXEL_INDEX: pixel_indices[mask > 0]
        }

    def _load_metadata_into_memory(self) -> Dict[str, torch.Tensor]:
        image_indices = []
        pixel_indices = []

        if RGB not in self.load_on_demand:
            rgbs = []

        if self.load_depth and DEPTH not in self.load_on_demand:
            depths = []

        if self.load_features and FEATURES not in self.load_on_demand:
            features = []

        if self.load_flow and BACKWARD_FLOW not in self.load_on_demand:
            backward_flow = []
            backward_flow_valid = []

        if self.load_flow and FORWARD_FLOW not in self.load_on_demand:
            forward_flow = []
            forward_flow_valid = []

        if self.load_sky and SKY not in self.load_on_demand:
            sky = []

        CONSOLE.log('Loading fields into memory')

        for metadata_item in self.all_items:
            # This is needed for both for chunks and when loading directly from metadata
            if metadata_item.is_val and self.split == Split.TRAIN:
                continue
            if (not metadata_item.is_val) and self.split == Split.VAL:
                continue

            image_keep_mask = metadata_item.load_mask().view(-1)
            if not torch.any(image_keep_mask > 0):
                continue

            image_indices.append(
                torch.ones_like(image_keep_mask, dtype=torch.long)[image_keep_mask > 0] * metadata_item.image_index)
            pixel_indices.append(torch.arange(metadata_item.W * metadata_item.H, dtype=torch.long)[image_keep_mask > 0])

            if RGB not in self.load_on_demand:
                image_rgbs = metadata_item.load_image().view(-1, 3)[image_keep_mask > 0].float() / 255.
                rgbs.append(image_rgbs)

            if self.load_depth and DEPTH not in self.load_on_demand:
                image_depth = metadata_item.load_depth().view(-1)[image_keep_mask > 0]
                depths.append(image_depth)

            if self.load_features and FEATURES not in self.load_on_demand:
                image_features = metadata_item.load_features()
                image_features = image_features.view(-1, image_features.shape[-1])[image_keep_mask > 0]
                max_feature = image_features.abs().max()
                assert max_feature <= 1.05, '{} {}'.format(metadata_item.feature_path, max_feature)
                features.append(image_features)

            if self.load_flow and BACKWARD_FLOW not in self.load_on_demand:
                image_backward_flow, image_backward_flow_valid = metadata_item.load_backward_flow()
                backward_flow.append(image_backward_flow.view(-1, 2)[image_keep_mask > 0])
                backward_flow_valid.append(image_backward_flow_valid.view(-1, 1)[image_keep_mask > 0])

            if self.load_flow and FORWARD_FLOW not in self.load_on_demand:
                image_forward_flow, image_forward_flow_valid = metadata_item.load_forward_flow()
                forward_flow.append(image_forward_flow.view(-1, 2)[image_keep_mask > 0])
                forward_flow_valid.append(image_forward_flow_valid.view(-1, 1)[image_keep_mask > 0])

            if self.load_sky and SKY not in self.load_on_demand:
                image_sky = metadata_item.load_sky_mask().view(-1)[image_keep_mask > 0]
                sky.append(image_sky)

        CONSOLE.log('Finished loading fields')

        fields = {IMAGE_INDEX: torch.cat(image_indices), PIXEL_INDEX: torch.cat(pixel_indices)}
        if RGB not in self.load_on_demand:
            fields[RGB] = torch.cat(rgbs)

        if self.load_depth and DEPTH not in self.load_on_demand:
            fields[DEPTH] = torch.cat(depths)

        if self.load_features and FEATURES not in self.load_on_demand:
            fields[FEATURES] = torch.cat(features)

        if self.load_flow and BACKWARD_FLOW not in self.load_on_demand:
            fields[BACKWARD_FLOW] = torch.cat(backward_flow)
            fields[BACKWARD_FLOW_VALID] = torch.cat(backward_flow_valid)

        if self.load_flow and FORWARD_FLOW not in self.load_on_demand:
            fields[FORWARD_FLOW] = torch.cat(forward_flow)
            fields[FORWARD_FLOW_VALID] = torch.cat(forward_flow_valid)

        if self.load_sky and SKY not in self.load_on_demand:
            fields[SKY] = torch.cat(sky)

        return fields

    def _load_fields(self, image_indices: torch.Tensor, pixel_indices: torch.Tensor, fields_to_load: Set[str],
                     verbose: bool = False) -> Dict[str, torch.Tensor]:
        assert image_indices.shape == pixel_indices.shape

        sorted_image_indices, ordering = image_indices.sort()
        unique_image_indices, counts = torch.unique_consecutive(sorted_image_indices, return_counts=True)
        load_futures = {}

        offset = 0
        for image_index, image_count in zip(unique_image_indices, counts):
            load_futures[int(image_index)] = self.on_demand_executor.submit(
                self._load_image_fields, image_index, pixel_indices[ordering[offset:offset + image_count]],
                fields_to_load)
            offset += image_count

        loaded = {}
        offset = 0
        for i, (image_index, image_count) in enumerate(zip(unique_image_indices, counts)):
            if i % 1000 == 0 and verbose:
                CONSOLE.log('Loading image {} of {}'.format(i, len(unique_image_indices)))
            loaded_features = load_futures[int(image_index)].result()
            to_put = ordering[offset:offset + image_count]

            for key, value in loaded_features.items():
                if i == 0:
                    loaded[key] = torch.zeros(image_indices.shape[0:1] + value.shape[1:], dtype=value.dtype)
                loaded[key][to_put] = value

            offset += image_count
            del load_futures[int(image_index)]

        return loaded

    def _load_image_fields(self, image_index: int, pixel_indices: torch.Tensor, fields_to_load: Set[str]) -> \
            Dict[str, torch.Tensor]:
        fields = {}

        item = self.all_items[image_index]
        for field in fields_to_load:
            if field == RGB:
                fields[RGB] = item.load_image().view(-1, 3)[pixel_indices].float() / 255.
            elif field == DEPTH:
                fields[DEPTH] = item.load_depth().view(-1)[pixel_indices]
            elif field == FEATURES:
                # We special case features since they're usually at a smaller resolution and we want to skip
                # the expensive resizing
                u = pixel_indices % item.W
                v = pixel_indices // item.W
                features = item.load_features(False)
                max_feature = features.abs().max()
                assert max_feature <= 1.05, '{} {}'.format(item.feature_path, max_feature)

                sub_pixel_indices = torch.floor(u * features.shape[1] / item.W).long() + \
                                    torch.floor(v * features.shape[0] / item.H).long() * features.shape[1]
                fields[FEATURES] = features.view(-1, features.shape[-1])[sub_pixel_indices]
            elif field == BACKWARD_FLOW:
                backward_flow, backward_flow_valid = item.load_backward_flow()
                fields[BACKWARD_FLOW] = backward_flow.view(-1, 2)[pixel_indices]
                fields[BACKWARD_FLOW_VALID] = backward_flow_valid.view(-1, 1)[pixel_indices]
            elif field == FORWARD_FLOW:
                forward_flow, forward_flow_valid = item.load_forward_flow()
                fields[FORWARD_FLOW] = forward_flow.view(-1, 2)[pixel_indices]
                fields[FORWARD_FLOW_VALID] = forward_flow_valid.view(-1, 1)[pixel_indices]
            elif field == SKY:
                fields[SKY] = item.load_sky_mask().view(-1)[pixel_indices]
            elif field == MASK:
                fields[MASK] = item.load_mask().view(-1)[pixel_indices]
            else:
                raise Exception('Unrecognized field: {}'.format(field))

        return fields
