from typing import Dict, Union, Optional

import tinycudann as tcnn
import torch
from nerfstudio.field_components.field_heads import FieldHeadNames
from torch import nn

from suds.fields.suds_field_head_names import SUDSFieldHeadNames
from suds.suds_constants import RGB, FEATURES, FILTER_FEATURES


class EnvMapField(nn.Module):

    def __init__(
            self,
            num_layers: int = 2,
            hidden_dim: int = 64,
            appearance_embedding_dim: int = 32,
            num_levels: int = 16,
            features_per_level: int = 2,
            log2_hashmap_size: int = 16,
            base_resolution: int = 16,
            feature_dim: int = 64,
            network_activation: str = 'ReLU',
            feature_output_activation: str = 'Tanh'
    ) -> None:
        super().__init__()
        self.appearance_embedding_dim = appearance_embedding_dim
        self.feature_dim = feature_dim
        self.feature_output_activation = feature_output_activation

        self.encoding = tcnn.Encoding(
            n_input_dims=4,
            encoding_config={
                'otype': 'SequentialGrid',
                'n_levels': num_levels,
                'n_features_per_level': features_per_level,
                'log2_hashmap_size': log2_hashmap_size,
                'base_resolution': base_resolution,
                'include_static': False
            }
        )

        self.mlp_head = tcnn.Network(
            n_input_dims=features_per_level * num_levels + appearance_embedding_dim,
            n_output_dims=3,
            network_config={
                'otype': 'FullyFusedMLP',
                'activation': network_activation,
                'output_activation': 'Sigmoid',
                'n_neurons': hidden_dim,
                'n_hidden_layers': num_layers - 1,
            },
        )

        if feature_dim > 0:
            self.mlp_feature = tcnn.Network(
                n_input_dims=features_per_level * num_levels,
                n_output_dims=feature_dim,
                network_config={
                    'otype': 'FullyFusedMLP',
                    'activation': network_activation,
                    'output_activation': self.feature_output_activation,
                    'n_neurons': hidden_dim,
                    'n_hidden_layers': num_layers - 1,
                },
            )

    def forward(self, directions: torch.Tensor, video_ids: torch.Tensor, appearance_embedding: Optional[torch.Tensor],
                output_type: Optional[str], filter_features: bool) \
            -> Dict[Union[FieldHeadNames, SUDSFieldHeadNames], torch.Tensor]:
        embedding = self.encoding(torch.cat([directions, video_ids], -1))

        outputs = {}

        if output_type is None or RGB in output_type:
            outputs[FieldHeadNames.RGB] = self.mlp_head(torch.cat([embedding, appearance_embedding], -1) \
                                                            if self.appearance_embedding_dim > 0 else embedding) \
                                          * (1 + 2e-3) - 1e-3

        if self.feature_dim > 0 and (filter_features or (output_type is None or FEATURES in output_type)):
            features = self.mlp_feature(embedding)
            if self.feature_output_activation.casefold() == 'tanh':
                features = features * 1.1

            outputs[SUDSFieldHeadNames.FEATURES] = features

        return outputs
