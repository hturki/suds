from typing import List, Optional

import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.fields.base_field import Field
from torch import nn
from torchtyping import TensorType

from suds.fields.static_proposal_field import StaticProposalField


class ShardedStaticProposalField(Field):

    def __init__(
            self,
            centroids: torch.Tensor,
            origin: torch.Tensor,
            centroid_origins: torch.Tensor,
            scale: float,
            centroid_scales: List[float],
            delegates: List[StaticProposalField]
    ) -> None:
        super().__init__()

        self.register_buffer('centroids', centroids)
        self.register_buffer('origin', origin)
        self.register_buffer('centroid_origins', centroid_origins)

        self.scale = scale
        self.centroid_scales = centroid_scales
        self.delegates = nn.ModuleList(delegates)

    def get_density(self, ray_samples: RaySamples):
        positions = ray_samples.frustums.get_positions().view(-1, 3)

        density = None
        cluster_assignments = torch.cdist(positions, self.centroids).argmin(dim=1)
        for i, delegate in enumerate(self.delegates):
            cluster_mask = cluster_assignments == i

            if torch.any(cluster_mask):
                shifted_positions = ((positions[cluster_mask].double() * self.scale + self.origin -
                                      self.centroid_origins[i]) / self.centroid_scales[i]).float()

                del_density_before_activation = delegate.mlp_base(delegate.encoding(shifted_positions))
                del_density = F.softplus(del_density_before_activation.to(ray_samples.frustums.directions) - 1)

                if density is None:
                    density = torch.empty(ray_samples.frustums.starts.shape, dtype=del_density.dtype,
                                          device=del_density.device)

                density.view(-1, 1)[cluster_mask] = del_density

        return density, None

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None):
        return {}
