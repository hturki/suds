from typing import Dict, Union, Tuple, List

import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn
from torch import nn
from torchtyping import TensorType

from suds.fields.dynamic_field import DynamicField
from suds.fields.suds_field_head_names import SUDSFieldHeadNames
from suds.suds_constants import BACKWARD_NEIGHBOR_TIME_DIFF, FORWARD_NEIGHBOR_TIME_DIFF, VIDEO_ID, OUTPUT_TYPE, RGB, \
    FEATURES, STATIC_RGB, NO_ENV_MAP_RGB, BACKWARD_FLOW, FORWARD_FLOW


class ShardedDynamicField(Field):

    def __init__(
            self,
            centroids: torch.Tensor,
            origin: torch.Tensor,
            centroid_origins: torch.Tensor,
            scale: float,
            centroid_scales: List[float],
            delegates: List[DynamicField]
    ) -> None:
        super().__init__()

        self.register_buffer('centroids', centroids)
        self.register_buffer('origin', origin)
        self.register_buffer('centroid_origins', centroid_origins)

        self.scale = scale
        self.centroid_scales = centroid_scales
        self.delegates = nn.ModuleList(delegates)

    def get_density(self, ray_samples: RaySamples):
        if ray_samples.times is None:
            raise AttributeError('Times are not provided.')
        if VIDEO_ID not in ray_samples.metadata:
            raise AttributeError('Video ids are not provided.')

        positions = ray_samples.frustums.get_positions().view(-1, 3)
        times = ray_samples.times.reshape(-1, 1)
        video_ids = ray_samples.metadata[VIDEO_ID].reshape(-1, 1)
        base_input = torch.empty(positions.shape[0], 5, dtype=positions.dtype, device=positions.device)

        density = None
        base_mlp_out = None
        cluster_assignments = torch.cdist(positions, self.centroids).argmin(dim=1)
        for i, delegate in enumerate(self.delegates):
            cluster_mask = cluster_assignments == i

            if torch.any(cluster_mask):
                shifted_positions = (positions[cluster_mask].double() * self.scale + self.origin -
                                     self.centroid_origins[i]) / self.centroid_scales[i]
                del_input = torch.cat([shifted_positions.float(), times[cluster_mask], video_ids[cluster_mask]], -1)
                base_input[cluster_mask] = del_input

                h = delegate.mlp_base(delegate.encoding(del_input))
                del_density_before_activation, del_base_mlp_out = torch.split(h, [1, delegate.geo_feat_dim], dim=-1)
                del_density = F.softplus(del_density_before_activation.to(ray_samples.frustums.directions) - 1)

                if density is None:
                    density = torch.empty(ray_samples.frustums.starts.shape, dtype=del_density.dtype,
                                          device=del_density.device)
                    base_mlp_out = torch.empty(*ray_samples.frustums.starts.shape[:-1], del_base_mlp_out.shape[-1],
                                               dtype=del_base_mlp_out.dtype, device=del_base_mlp_out.device)

                density.view(-1, 1)[cluster_mask] = del_density
                base_mlp_out.view(-1, base_mlp_out.shape[-1])[cluster_mask] = del_base_mlp_out

        return density, (base_mlp_out, base_input, cluster_assignments)

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Tuple[TensorType, TensorType, TensorType]) \
            -> Dict[Union[FieldHeadNames, SUDSFieldHeadNames], TensorType]:
        density_embedding, base_input, cluster_assignments = density_embedding
        density_embedding = density_embedding.view(-1, self.delegates[0].geo_feat_dim)
        directions = ray_samples.frustums.directions

        outputs = {}

        if ray_samples.metadata[OUTPUT_TYPE] is None or RGB in ray_samples.metadata[OUTPUT_TYPE]:
            rgb_inputs = []
            if self.delegates[0].num_directions > 0:
                # Using spherical harmonics - need to map directions to [0, 1]
                rgb_inputs.append(shift_directions_for_tcnn(directions).view(-1, 3))

            rgb_inputs.append(density_embedding)
            rgb_inputs = torch.cat(rgb_inputs, -1)

            rgb = torch.empty_like(directions)
            for i, delegate in enumerate(self.delegates):
                cluster_mask = cluster_assignments == i

                if torch.any(cluster_mask):
                    rgb.view(-1, 3)[cluster_mask] = delegate.mlp_head(rgb_inputs[cluster_mask]).to(directions)

            outputs[FieldHeadNames.RGB] = rgb * (1 + 2e-3) - 1e-3

        if self.delegates[0].feature_dim > 0 \
                and (ray_samples.metadata[OUTPUT_TYPE] is None or FEATURES in ray_samples.metadata[OUTPUT_TYPE]):
            features = torch.empty(*directions.shape[:-1], self.delegates[0].feature_dim, dtype=directions.dtype,
                                   device=directions.device)
            for i, delegate in enumerate(self.delegates):
                cluster_mask = cluster_assignments == i

                if torch.any(cluster_mask):
                    features.view(-1, delegate.feature_dim)[cluster_mask] = delegate.mlp_feature(
                        delegate.encoding_feature(base_input[cluster_mask])).to(directions)

            if self.delegates[0].feature_output_activation.casefold() == 'tanh':
                features = features * 1.1

            outputs[SUDSFieldHeadNames.FEATURES] = features

        if self.delegates[0].predict_shadow and (ray_samples.metadata[OUTPUT_TYPE] is None
                                                 or ray_samples.metadata[OUTPUT_TYPE] in {RGB, STATIC_RGB,
                                                                                          NO_ENV_MAP_RGB}):
            shadows = torch.empty(*directions.shape[:-1], 1, dtype=density_embedding.dtype, device=directions.device)
            for i, delegate in enumerate(self.delegates):
                cluster_mask = cluster_assignments == i

                if torch.any(cluster_mask):
                    shadows.view(-1, 1)[cluster_mask] = delegate.mlp_shadow(density_embedding[cluster_mask]).to(
                        directions)

            outputs[SUDSFieldHeadNames.SHADOWS] = shadows

        if self.delegates[0].predict_flow and (ray_samples.metadata[OUTPUT_TYPE] is None
                                               or ray_samples.metadata[OUTPUT_TYPE] in {BACKWARD_FLOW, FORWARD_FLOW}):
            flow = torch.empty(*directions.shape[:-1], 6, dtype=density_embedding.dtype, device=directions.device)
            for i, delegate in enumerate(self.delegates):
                cluster_mask = cluster_assignments == i

                if torch.any(cluster_mask):
                    flow.view(-1, 6)[cluster_mask] = delegate.mlp_flow(
                        delegate.encoding_flow(base_input[cluster_mask])).to(directions)

            flow = torch.tanh(flow)

            backward_flow = flow[..., :3]
            forward_flow = flow[..., 3:]

            if BACKWARD_NEIGHBOR_TIME_DIFF in ray_samples.metadata:
                backward_time_diff = ray_samples.metadata[BACKWARD_NEIGHBOR_TIME_DIFF]
                backward_flow = backward_flow * backward_time_diff / self.delegates[0].flow_unit

            if FORWARD_NEIGHBOR_TIME_DIFF in ray_samples.metadata:
                forward_time_diff = ray_samples.metadata[FORWARD_NEIGHBOR_TIME_DIFF]
                forward_flow = forward_flow * forward_time_diff / self.delegates[0].flow_unit

            outputs[SUDSFieldHeadNames.BACKWARD_FLOW] = backward_flow
            outputs[SUDSFieldHeadNames.FORWARD_FLOW] = forward_flow

        return outputs
