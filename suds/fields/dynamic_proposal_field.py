# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Proposal network field.
"""

from typing import Optional

import numpy as np
import tinycudann as tcnn
import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.fields.base_field import Field
from torchtyping import TensorType

from suds.suds_constants import VIDEO_ID


class DynamicProposalField(Field):
    """A lightweight density field module.

    Args:
        aabb: parameters of scene aabb bounds
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        spatial_distortion: spatial distortion module
        use_linear: whether to skip the MLP and use a single linear layer instead
    """

    def __init__(
            self,
            num_layers: int = 2,
            hidden_dim: int = 16,
            num_levels: int = 5,
            base_resolution: int = 16,
            max_resolution: int = 256,
            log2_hashmap_size: int = 18,
            features_per_level: int = 2,
            network_activation: str = 'ReLU'
    ) -> None:
        super().__init__()

        growth_factor = np.exp((np.log(max_resolution) - np.log(base_resolution)) / (num_levels - 1))

        self.encoding = tcnn.Encoding(
            n_input_dims=5,
            encoding_config={
                'otype': 'SequentialGrid',
                'n_levels': num_levels,
                'n_features_per_level': features_per_level,
                'log2_hashmap_size': log2_hashmap_size,
                'base_resolution': base_resolution,
                'per_level_scale': growth_factor,
                'include_static': False
            }
        )

        self.mlp_base = tcnn.Network(
            n_input_dims=num_levels * features_per_level,
            n_output_dims=1,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': network_activation,
                'output_activation': 'None',
                'n_neurons': hidden_dim,
                'n_hidden_layers': num_layers - 1,
            }
        )

    def density_fn(self, positions: TensorType["bs":..., 3], times: TensorType, video_ids: TensorType) -> \
            TensorType["bs":..., 1]:
        ray_samples = RaySamples(
            frustums=Frustums(
                origins=positions,
                directions=torch.ones_like(positions),
                starts=torch.zeros_like(positions[..., :1]),
                ends=torch.zeros_like(positions[..., :1]),
                pixel_area=torch.ones_like(positions[..., :1]),
            ),
            times=times.unsqueeze(-2).expand(*times.shape[:-1], positions.shape[-2], -1),
            metadata={VIDEO_ID: video_ids.unsqueeze(-2).expand(*video_ids.shape[:-1], positions.shape[-2], -1)}
        )
        density, _ = self.get_density(ray_samples)
        return density

    def get_density(self, ray_samples: RaySamples):
        if ray_samples.times is None:
            raise AttributeError('Times are not provided.')
        if VIDEO_ID not in ray_samples.metadata:
            raise AttributeError('Video ids are not provided.')

        positions = ray_samples.frustums.get_positions()
        times = ray_samples.times

        base_input = torch.cat(
            [positions.view(-1, 3), times.reshape(-1, 1), ray_samples.metadata[VIDEO_ID].reshape(-1, 1)], -1)
        density_before_activation = self.mlp_base(self.encoding(base_input)).view(*ray_samples.frustums.shape, -1)

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = F.softplus(density_before_activation.to(ray_samples.frustums.directions) - 1)

        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        return {}
