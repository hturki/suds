import torch


# From RaySamplers.get_weights
@torch.jit.script
def _get_weights(deltas: torch.Tensor, density: torch.Tensor, filter_nan: bool) -> torch.Tensor:
    delta_density = deltas * density
    alphas = 1 - torch.exp(-delta_density)
    transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
    transmittance = torch.cat(
        [torch.zeros((transmittance.shape[0], 1, 1), device=density.device),
         transmittance], dim=-2
    )
    transmittance = torch.exp(-transmittance)  # [..., "num_samples"]
    weights = alphas * transmittance  # [..., "num_samples"]

    if filter_nan:
        weights = torch.nan_to_num(weights)

    return weights
