from typing import Literal, Optional

import nerfacc
import torch
from torch import nn
from torchtyping import TensorType


class SUDSDepthRenderer(nn.Module):
    def __init__(self, method: Literal["median", "expected"] = "expected") -> None:
        super().__init__()
        self.method = method

    def forward(
        self,
        weights: TensorType[..., "num_samples", 1],
        z_vals: torch.Tensor,
        ray_indices: Optional[TensorType["num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> TensorType[..., 1]:
        """Composite samples along ray and calculate depths.

        Args:
            weights: Weights for each sample.
            ray_samples: Set of ray samples.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs of depth values.
        """

        if self.method == "median":
            # steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

            if ray_indices is not None and num_rays is not None:
                raise NotImplementedError("Median depth calculation is not implemented for packed samples.")
            cumulative_weights = torch.cumsum(weights[..., 0], dim=-1)  # [..., num_samples]
            split = torch.ones((*weights.shape[:-2], 1), device=weights.device) * 0.5  # [..., 1]
            median_index = torch.searchsorted(cumulative_weights, split, side="left")  # [..., 1]
            median_index = torch.clamp(median_index, 0, z_vals.shape[-2] - 1)  # [..., 1]
            median_depth = torch.gather(z_vals[..., 0], dim=-1, index=median_index)  # [..., 1]
            return median_depth
        if self.method == "expected":
            eps = 1e-10
            # steps = (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2

            if ray_indices is not None and num_rays is not None:
                # Necessary for packed samples from volumetric ray sampler
                depth = nerfacc.accumulate_along_rays(weights, ray_indices, z_vals, num_rays)
                accumulation = nerfacc.accumulate_along_rays(weights, ray_indices, None, num_rays)
                depth = depth / (accumulation + eps)
            else:
                depth = torch.sum(weights * z_vals, dim=-2) / (torch.sum(weights, -2) + eps)

            depth = torch.clip(depth, z_vals.min(), z_vals.max())

            return depth

        raise NotImplementedError(f"Method {self.method} not implemented")
