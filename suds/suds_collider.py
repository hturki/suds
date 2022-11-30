from typing import Optional, Tuple

import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.model_components.scene_colliders import NearFarCollider, _intersect_with_sphere

from suds.suds_constants import BG_INTERSECTION


class SUDSCollider(NearFarCollider):

    def __init__(self,
                 near: float,
                 far: float,
                 scene_bounds: Optional[torch.Tensor],
                 sphere_center: Optional[torch.Tensor],
                 sphere_radius: Optional[torch.Tensor]) -> None:
        super().__init__(near, far if sphere_center is None else 1e10)
        self.far = far  # we clamp to far after finding sphere intersections
        self.scene_bounds = scene_bounds
        self.sphere_center = sphere_center
        self.sphere_radius = sphere_radius

    def forward(self, ray_bundle: RayBundle) -> RayBundle:
        ray_bundle = super().forward(ray_bundle)

        if self.scene_bounds is not None:
            _truncate_with_plane_intersection(ray_bundle.origins, ray_bundle.directions, self.scene_bounds[1, 2],
                                              ray_bundle.nears)
            _truncate_with_plane_intersection(ray_bundle.origins, ray_bundle.directions, self.scene_bounds[0, 2],
                                              ray_bundle.fars)

        if self.sphere_center is not None:
            device = ray_bundle.origins.device
            rays_d, rays_o = _ellipse_to_sphere_coords(ray_bundle.origins.view(-1, 3),
                                                       ray_bundle.directions.view(-1, 3),
                                                       self.sphere_center,
                                                       self.sphere_radius,
                                                       device)

            _, sphere_fars = _intersect_with_sphere(rays_o, rays_d, torch.zeros(3, device=device))
            ray_bundle.metadata[BG_INTERSECTION] = torch.zeros_like(sphere_fars)
            rays_with_bg = ray_bundle.fars > sphere_fars
            ray_bundle.metadata[BG_INTERSECTION][rays_with_bg] = sphere_fars[rays_with_bg]
            ray_bundle.fars = torch.minimum(ray_bundle.fars, sphere_fars)

        assert ray_bundle.nears.min() >= 0, ray_bundle.nears.min()
        assert ray_bundle.fars.min() >= 0, ray_bundle.fars.min()

        ray_bundle.nears = ray_bundle.nears.clamp_min(self.near_plane)
        ray_bundle.fars = ray_bundle.fars.clamp_min(ray_bundle.nears + 1e-6).clamp_max(self.far)

        return ray_bundle


@torch.jit.script
def _ellipse_to_sphere_coords(rays_o: torch.Tensor, rays_d: torch.Tensor, sphere_center: torch.Tensor,
                              sphere_radius: torch.Tensor, device: torch.device) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    sphere_radius = sphere_radius.to(device)
    rays_o = (rays_o - sphere_center.to(device)) / sphere_radius
    rays_d = rays_d / sphere_radius
    return rays_d, rays_o


@torch.jit.script
def _truncate_with_plane_intersection(rays_o: torch.Tensor, rays_d: torch.Tensor, altitude: float,
                                      default_bounds: torch.Tensor) -> None:
    starts_before = rays_o[..., 2] > altitude
    goes_down = rays_d[..., 2] < 0

    boundable_rays = torch.minimum(starts_before, goes_down)

    ray_points = rays_o[boundable_rays]
    ray_dirs = rays_d[boundable_rays]
    if ray_points.shape[0] == 0:
        return

    default_bounds[boundable_rays] = ((altitude - ray_points[..., 2]) / ray_dirs[..., 2]).unsqueeze(-1)

    assert torch.all(default_bounds[boundable_rays] > 0)
