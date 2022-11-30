from typing import Optional, Dict, Union, Tuple

import tinycudann as tcnn
import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn
from torchtyping import TensorType

from suds.fields.suds_field_head_names import SUDSFieldHeadNames
from suds.suds_constants import BACKWARD_NEIGHBOR_TIME_DIFF, FORWARD_NEIGHBOR_TIME_DIFF, VIDEO_ID, OUTPUT_TYPE, RGB, \
    FEATURES, STATIC_RGB, NO_ENV_MAP_RGB, BACKWARD_FLOW, FORWARD_FLOW, FILTER_FEATURES


class DynamicField(Field):

    def __init__(
            self,
            num_layers: int = 2,
            hidden_dim: int = 64,
            geo_feat_dim: int = 15,
            num_layers_color: int = 3,
            hidden_dim_color: int = 64,
            num_levels: int = 16,
            features_per_level: int = 2,
            log2_hashmap_size: int = 16,
            base_resolution: int = 16,
            num_directions: int = 4,
            feature_dim: int = 64,
            num_layers_feature: int = 3,
            hidden_dim_feature: int = 64,
            predict_shadow: bool = True,
            num_layers_shadow: int = 3,
            hidden_dim_shadow: int = 64,
            predict_flow: bool = True,
            num_layers_flow: int = 3,
            hidden_dim_flow: int = 64,
            flow_unit: float = 1,
            network_activation: str = 'ReLU',
            feature_output_activation: str = 'Tanh'
    ) -> None:
        super().__init__()

        self.geo_feat_dim = geo_feat_dim
        self.feature_output_activation = feature_output_activation

        self.encoding = tcnn.Encoding(
            n_input_dims=5,
            encoding_config={
                'otype': 'SequentialGrid',
                'n_levels': num_levels,
                'n_features_per_level': features_per_level,
                'log2_hashmap_size': log2_hashmap_size,
                'base_resolution': base_resolution,
                'include_static': False
            }
        )

        self.mlp_base = tcnn.Network(
            n_input_dims=features_per_level * num_levels,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': network_activation,
                'output_activation': 'None',
                'n_neurons': hidden_dim,
                'n_hidden_layers': num_layers - 1,
            },
        )

        color_network_config = {
            'otype': 'FullyFusedMLP',
            'activation': network_activation,
            'output_activation': 'Sigmoid',
            'n_neurons': hidden_dim_color,
            'n_hidden_layers': num_layers_color - 1,
        }

        self.num_directions = num_directions
        if num_directions > 0:
            dir_encoding = {
                'n_dims_to_encode': 3,
                'otype': 'SphericalHarmonics',
                'degree': num_directions
            }

            self.mlp_head = tcnn.NetworkWithInputEncoding(
                n_input_dims=3 + geo_feat_dim,
                n_output_dims=3,
                encoding_config={
                    'otype': 'Composite',
                    'nested': [
                        dir_encoding,
                        {
                            'n_dims_to_encode': geo_feat_dim,
                            'otype': 'Identity'
                        }
                    ]
                },
                network_config=color_network_config,
            )
        else:
            self.mlp_head = tcnn.Network(
                n_input_dims=geo_feat_dim,
                n_output_dims=3,
                network_config=color_network_config,
            )

        self.feature_dim = feature_dim
        if feature_dim > 0:
            self.encoding_feature = tcnn.Encoding(
                n_input_dims=5,
                encoding_config={
                    'otype': 'SequentialGrid',
                    'n_levels': num_levels,
                    'n_features_per_level': features_per_level,
                    'log2_hashmap_size': log2_hashmap_size,
                    'base_resolution': base_resolution,
                    'include_static': False
                },
            )

            self.mlp_feature = tcnn.Network(
                n_input_dims=(features_per_level * num_levels),
                n_output_dims=feature_dim,
                network_config={
                    'otype': 'FullyFusedMLP',
                    'activation': network_activation,
                    'output_activation': self.feature_output_activation,
                    'n_neurons': hidden_dim_feature,
                    'n_hidden_layers': num_layers_feature - 1,
                },
            )

        self.predict_shadow = predict_shadow
        if predict_shadow:
            self.mlp_shadow = tcnn.Network(
                n_input_dims=geo_feat_dim,
                n_output_dims=1,
                network_config={
                    'otype': 'FullyFusedMLP',
                    'activation': network_activation,
                    'output_activation': 'Sigmoid',
                    'n_neurons': hidden_dim_shadow,
                    'n_hidden_layers': num_layers_shadow - 1,
                },
            )

        self.predict_flow = predict_flow
        if predict_flow:
            self.encoding_flow = tcnn.Encoding(
                n_input_dims=5,
                encoding_config={
                    'otype': 'SequentialGrid',
                    'n_levels': num_levels,
                    'n_features_per_level': features_per_level,
                    'log2_hashmap_size': log2_hashmap_size,
                    'base_resolution': base_resolution,
                    'include_static': False
                },
            )

            self.mlp_flow = tcnn.Network(
                n_input_dims=(features_per_level * num_levels),
                n_output_dims=6,
                network_config={
                    'otype': 'FullyFusedMLP',
                    'activation': network_activation,
                    'output_activation': 'None',
                    'n_neurons': hidden_dim_flow,
                    'n_hidden_layers': num_layers_flow - 1,
                },
            )

        self.flow_unit = flow_unit

    def get_density(self, ray_samples: RaySamples, position_offsets: Optional[torch.Tensor] = None,
                    time_offsets: Optional[torch.Tensor] = None):
        if ray_samples.times is None:
            raise AttributeError('Times are not provided.')
        if VIDEO_ID not in ray_samples.metadata:
            raise AttributeError('Video ids are not provided.')

        positions = ray_samples.frustums.get_positions()
        times = ray_samples.times
        if position_offsets is not None:
            positions = positions + position_offsets
        if time_offsets is not None:
            times = times + time_offsets

        base_input = torch.cat(
            [positions.view(-1, 3), times.reshape(-1, 1), ray_samples.metadata[VIDEO_ID].reshape(-1, 1)], -1)
        h = self.mlp_base(self.encoding(base_input)).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        density = F.softplus(density_before_activation.to(ray_samples.frustums.directions) - 1)

        return density, (base_mlp_out, base_input)

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Tuple[TensorType, TensorType],
                    is_warped: bool = False) -> Dict[Union[FieldHeadNames, SUDSFieldHeadNames], TensorType]:
        density_embedding, base_input = density_embedding
        density_embedding = density_embedding.view(-1, self.geo_feat_dim)
        directions = ray_samples.frustums.directions

        outputs = {}

        if ray_samples.metadata[OUTPUT_TYPE] is None or RGB in ray_samples.metadata[OUTPUT_TYPE]:
            rgb_inputs = []
            if self.num_directions > 0:
                # Using spherical harmonics - need to map directions to [0, 1]
                rgb_inputs.append(shift_directions_for_tcnn(directions).view(-1, 3))

            rgb_inputs.append(density_embedding)

            rgb = self.mlp_head(torch.cat(rgb_inputs, -1)).view(*directions.shape[:-1], -1).to(directions)
            outputs[FieldHeadNames.RGB] = rgb * (1 + 2e-3) - 1e-3

        if self.feature_dim > 0 \
                and (ray_samples.metadata[FILTER_FEATURES] or
                     (ray_samples.metadata[OUTPUT_TYPE] is None or FEATURES in ray_samples.metadata[OUTPUT_TYPE])):
            features = self.mlp_feature(self.encoding_feature(base_input)).view(*directions.shape[:-1], -1).to(
                directions)

            if self.feature_output_activation.casefold() == 'tanh':
                features = features * 1.1

            outputs[SUDSFieldHeadNames.FEATURES] = features  # * 1.1

        if self.predict_shadow and (not is_warped) \
                and (ray_samples.metadata[OUTPUT_TYPE] is None or ray_samples.metadata[OUTPUT_TYPE] in {RGB, STATIC_RGB,
                                                                                                        NO_ENV_MAP_RGB}):
            shadows = self.mlp_shadow(density_embedding).view(*directions.shape[:-1], -1).to(directions)
            outputs[SUDSFieldHeadNames.SHADOWS] = shadows

        if self.predict_flow and (ray_samples.metadata[OUTPUT_TYPE] is None
                                  or ray_samples.metadata[OUTPUT_TYPE] in {BACKWARD_FLOW, FORWARD_FLOW}):
            flow = torch.tanh(
                self.mlp_flow(self.encoding_flow(base_input)).view(*directions.shape[:-1], -1).to(directions))

            backward_flow = flow[..., :3]
            forward_flow = flow[..., 3:]

            if not is_warped:
                if BACKWARD_NEIGHBOR_TIME_DIFF in ray_samples.metadata:
                    backward_time_diff = ray_samples.metadata[BACKWARD_NEIGHBOR_TIME_DIFF]
                    backward_flow = backward_flow * backward_time_diff / self.flow_unit

                if FORWARD_NEIGHBOR_TIME_DIFF in ray_samples.metadata:
                    forward_time_diff = ray_samples.metadata[FORWARD_NEIGHBOR_TIME_DIFF]
                    forward_flow = forward_flow * forward_time_diff / self.flow_unit

                if self.training:
                    backward_jitter = torch.rand_like(backward_time_diff)
                    backward_density, backward_outputs = self._get_training_flow_outputs(
                        ray_samples, backward_flow * backward_jitter, -backward_time_diff * backward_jitter)
                    outputs[SUDSFieldHeadNames.BACKWARD_DENSITY] = backward_density
                    outputs[SUDSFieldHeadNames.BACKWARD_RGB] = backward_outputs[FieldHeadNames.RGB]
                    if self.feature_dim > 0:
                        outputs[SUDSFieldHeadNames.BACKWARD_FEATURES] = backward_outputs[SUDSFieldHeadNames.FEATURES]

                    outputs[SUDSFieldHeadNames.BACKWARD_FLOW_CYCLE_DIFF] = \
                        (backward_flow + backward_outputs[SUDSFieldHeadNames.FORWARD_FLOW]
                         * backward_time_diff / self.flow_unit).square().mean(dim=-1, keepdim=True)

                    forward_jitter = torch.rand_like(forward_time_diff)
                    forward_density, forward_outputs = self._get_training_flow_outputs(
                        ray_samples, forward_flow * forward_jitter, forward_time_diff * forward_jitter)
                    outputs[SUDSFieldHeadNames.FORWARD_DENSITY] = forward_density
                    outputs[SUDSFieldHeadNames.FORWARD_RGB] = forward_outputs[FieldHeadNames.RGB]
                    if self.feature_dim > 0:
                        outputs[SUDSFieldHeadNames.FORWARD_FEATURES] = forward_outputs[SUDSFieldHeadNames.FEATURES]

                    outputs[SUDSFieldHeadNames.FORWARD_FLOW_CYCLE_DIFF] = \
                        (forward_flow + forward_outputs[SUDSFieldHeadNames.BACKWARD_FLOW]
                         * forward_time_diff / self.flow_unit).square().mean(dim=-1, keepdim=True)

            outputs[SUDSFieldHeadNames.BACKWARD_FLOW] = backward_flow
            outputs[SUDSFieldHeadNames.FORWARD_FLOW] = forward_flow

        return outputs

    def _get_training_flow_outputs(self, ray_samples: RaySamples, flow: torch.Tensor, time_diff: torch.Tensor):
        warped_density, warped_base_mlp_out = self.get_density(ray_samples, flow, time_diff)
        return warped_density, self.get_outputs(ray_samples, warped_base_mlp_out, True)
