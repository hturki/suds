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

from typing import Optional, List

import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RaySamples, Frustums
from nerfstudio.fields.base_field import Field
from torch import nn
from torchtyping import TensorType

from suds.fields.dynamic_proposal_field import DynamicProposalField
from suds.suds_constants import VIDEO_ID


class ShardedDynamicProposalField(Field):
    def __init__(
            self,
            centroids: torch.Tensor,
            origin: torch.Tensor,
            centroid_origins: torch.Tensor,
            scale: float,
            centroid_scales: List[float],
            delegates: List[DynamicProposalField]
    ) -> None:
        super().__init__()

        self.register_buffer('centroids', centroids)
        self.register_buffer('origin', origin)
        self.register_buffer('centroid_origins', centroid_origins)

        self.scale = scale
        self.centroid_scales = centroid_scales
        self.delegates = nn.ModuleList(delegates)

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

        positions = ray_samples.frustums.get_positions().view(-1, 3)
        times = ray_samples.times.reshape(-1, 1)
        video_ids = ray_samples.metadata[VIDEO_ID].reshape(-1, 1)

        density = None
        cluster_assignments = torch.cdist(positions, self.centroids).argmin(dim=1)
        for i, delegate in enumerate(self.delegates):
            cluster_mask = cluster_assignments == i

            if torch.any(cluster_mask):
                shifted_positions = (positions[cluster_mask].double() * self.scale + self.origin -
                                     self.centroid_origins[i]) / self.centroid_scales[i]
                del_input = torch.cat([shifted_positions.float(), times[cluster_mask], video_ids[cluster_mask]], -1)

                del_density_before_activation = delegate.mlp_base(delegate.encoding(del_input))
                del_density = F.softplus(del_density_before_activation.to(ray_samples.frustums.directions) - 1)

                if density is None:
                    density = torch.empty(ray_samples.frustums.starts.shape, dtype=del_density.dtype,
                                          device=del_density.device)

                density.view(-1, 1)[cluster_mask] = del_density

        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        return {}
