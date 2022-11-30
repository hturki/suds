from typing import Dict, Union, Tuple

import tinycudann as tcnn
import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn
from torchtyping import TensorType

from suds.fields.suds_field_head_names import SUDSFieldHeadNames
from suds.suds_constants import APPEARANCE_EMBEDDING, OUTPUT_TYPE, RGB, FEATURES, FILTER_FEATURES


class StaticField(Field):

    def __init__(
            self,
            num_layers: int = 2,
            hidden_dim: int = 64,
            geo_feat_dim: int = 15,
            num_layers_color: int = 3,
            hidden_dim_color: int = 64,
            appearance_embedding_dim: int = 32,
            num_levels: int = 16,
            features_per_level: int = 2,
            log2_hashmap_size: int = 19,
            base_resolution: int = 16,
            num_directions: int = 4,
            feature_dim: int = 64,
            num_layers_feature: int = 3,
            hidden_dim_feature: int = 64,
            network_activation: str = 'ReLU',
            feature_output_activation: str = 'Tanh',
    ) -> None:
        super().__init__()

        self.geo_feat_dim = geo_feat_dim
        self.appearance_embedding_dim = appearance_embedding_dim
        self.feature_output_activation = feature_output_activation

        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                'otype': 'HashGrid',
                'n_levels': num_levels,
                'n_features_per_level': features_per_level,
                'log2_hashmap_size': log2_hashmap_size,
                'base_resolution': base_resolution,
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
                n_input_dims=3 + geo_feat_dim + appearance_embedding_dim,
                n_output_dims=3,
                encoding_config={
                    'otype': 'Composite',
                    'nested': [
                        dir_encoding,
                        {
                            'n_dims_to_encode': geo_feat_dim + appearance_embedding_dim,
                            'otype': 'Identity'
                        }
                    ]
                },
                network_config=color_network_config,
            )
        else:
            self.mlp_head = tcnn.Network(
                n_input_dims=geo_feat_dim + appearance_embedding_dim,
                n_output_dims=3,
                network_config=color_network_config,
            )

        self.feature_dim = feature_dim
        if feature_dim > 0:
            self.encoding_feature = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    'otype': 'HashGrid',
                    'n_levels': num_levels,
                    'n_features_per_level': features_per_level,
                    'log2_hashmap_size': log2_hashmap_size,
                    'base_resolution': base_resolution,
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

    def get_density(self, ray_samples: RaySamples):
        positions = ray_samples.frustums.get_positions()
        positions_flat = positions.view(-1, 3)

        h = self.mlp_base(self.encoding(positions_flat)).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)

        density = F.softplus(density_before_activation.to(positions) - 1)
        return density, (base_mlp_out, positions_flat)

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Tuple[TensorType, TensorType]) \
            -> Dict[Union[FieldHeadNames, SUDSFieldHeadNames], TensorType]:
        density_embedding, base_input = density_embedding
        density_embedding = density_embedding.view(-1, self.geo_feat_dim)
        directions = ray_samples.frustums.directions

        outputs = {}

        if ray_samples.metadata[OUTPUT_TYPE] is None or RGB in ray_samples.metadata[OUTPUT_TYPE]:
            rgb_inputs = []
            if self.num_directions > 0:
                rgb_inputs.append(shift_directions_for_tcnn(directions).view(-1, 3))

            rgb_inputs.append(density_embedding)

            if self.appearance_embedding_dim > 0:
                rgb_inputs.append(ray_samples.metadata[APPEARANCE_EMBEDDING].reshape(-1, self.appearance_embedding_dim))

            rgb = self.mlp_head(torch.cat(rgb_inputs, -1)).view(*directions.shape[:-1], -1).to(directions)
            outputs[FieldHeadNames.RGB] = rgb * (1 + 2e-3) - 1e-3

        if self.feature_dim > 0 \
                and (ray_samples.metadata[FILTER_FEATURES]
                     or (ray_samples.metadata[OUTPUT_TYPE] is None or FEATURES in ray_samples.metadata[OUTPUT_TYPE])):
            features = self.mlp_feature(self.encoding_feature(base_input)).view(*directions.shape[:-1], -1).to(
                directions)

            if self.feature_output_activation.casefold() == 'Tanh':
                features = features * 1.1

            outputs[SUDSFieldHeadNames.FEATURES] = features

        return outputs
