import copy
import glob
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Type, Optional, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
import tyro
import yaml
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallback, TrainingCallbackLocation
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, SemanticRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, misc
from rich.console import Console
from torch.nn import Parameter, MSELoss, L1Loss
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchtyping import TensorType

from suds.composite_proposal_network_sampler import CompositeProposalNetworkSampler
from suds.data.suds_dataparser import ALL_ITEMS, POSE_SCALE_FACTOR, ORIGIN
from suds.data.suds_dataset import RGB, DEPTH
from suds.draw_utils import scene_flow_to_rgb, cat_imgflo
from suds.fields.dynamic_field import DynamicField
from suds.fields.dynamic_proposal_field import DynamicProposalField
from suds.fields.env_map_field import EnvMapField
from suds.fields.sharded.sharded_dynamic_field import ShardedDynamicField
from suds.fields.sharded.sharded_dynamic_proposal_field import ShardedDynamicProposalField
from suds.fields.sharded.sharded_static_field import ShardedStaticField
from suds.fields.sharded.sharded_static_proposal_field import ShardedStaticProposalField
from suds.fields.static_field import StaticField
from suds.fields.static_proposal_field import StaticProposalField
from suds.fields.suds_field_head_names import SUDSFieldHeadNames
from suds.fields.video_embedding import VideoEmbedding
from suds.kmeans import kmeans_predict
from suds.metrics import ssim
from suds.sample_utils import _get_weights
from suds.suds_collider import SUDSCollider, BG_INTERSECTION
from suds.suds_constants import FEATURES, MASK, VIDEO_ID, APPEARANCE_EMBEDDING, RAY_INDEX, FORWARD_FLOW_VALID, \
    BACKWARD_FLOW_VALID, BACKWARD_NEIGHBOR_TIME_DIFF, FORWARD_NEIGHBOR_TIME_DIFF, BACKWARD_NEIGHBOR_W2C, \
    BACKWARD_NEIGHBOR_K, FORWARD_NEIGHBOR_W2C, FORWARD_NEIGHBOR_K, FORWARD_FLOW, BACKWARD_FLOW, SKY, NO_SHADOW_RGB, \
    STATIC_RGB, DYNAMIC_RGB, NO_ENV_MAP_RGB, OUTPUT_TYPE, FILTER_FEATURES
from suds.suds_depth_renderer import SUDSDepthRenderer

CONSOLE = Console(width=120)

ENTROPY = 'entropy'
DYNAMIC = 'dynamic'
SHADOW = 'shadow'

DYNAMIC_WEIGHT = 'dynamic_weight'
BACKWARD_DYNAMIC_WEIGHT = 'backward_dynamic_weight'
FORWARD_DYNAMIC_WEIGHT = 'forward_dynamic_weight'

DISTORTION = 'distortion'

INTERLEVEL = 'interlevel'
STATIC_INTERLEVEL = 'static_interlevel'
DYNAMIC_INTERLEVEL = 'dynamic_interlevel'

FLOW_CYCLE = 'flow_cycle'
FLOW_SLOW = 'flow_slow'
FLOW_SMOOTH_SPATIAL = 'flow_smooth_spatial'
FLOW_SMOOTH_TEMPORAL = 'flow_smooth_temporal'

OPTICAL_FLOW = 'optical_flow'
BACKWARD_RGB = 'backward_rgb'
BACKWARD_FEATURES = 'backward_features'
FORWARD_RGB = 'forward_rgb'
FORWARD_FEATURES = 'forward_features'

BACKWARD_UV = 'backward_uv'
FORWARD_UV = 'forward_uv'
BACKWARD_DISOCC = 'backward_disocc'
FORWARD_DISOCC = 'forward_disocc'
BACKWARD_FLOW_CYCLE_DIFF = 'backward_flow_cycle_diff'
FORWARD_FLOW_CYCLE_DIFF = 'forward_flow_cycle_diff'

WEIGHTS_SQ = 'weights_sq'

WEIGHTS_LIST = 'weights_list'
RAY_SAMPLES_LIST = 'ray_samples_list'
PROP_DEPTH = 'prop_depth'

STATIC_WEIGHTS_LIST = 'static_weights_list'
DYNAMIC_WEIGHTS_LIST = 'dynamic_weights_list'


@dataclass
class SUDSModelConfig(ModelConfig):
    """SUDS Model Config"""

    _target: Type = field(default_factory=lambda: SUDSModel)

    num_levels: int = 16
    """Number of levels in static and dynamic hash tables"""

    features_per_level: int = 2
    """Features per hash table level in static and dynamic hash tables"""

    base_resolution: int = 16
    """Resolution at first level of static and dynamic hash tables"""

    log2_hashmap_size_static: int = 19
    """Size of static hash table"""

    log2_hashmap_size_dynamic: int = 19
    """Size of dynamic hash table"""

    dynamic_only: bool = False
    """Whether to only use a dynamic branch"""

    static_only: bool = False
    """Whether to only use a static branch"""

    num_directions: int = 4
    """Spherical harmonics degree used when encoding directions"""

    geo_feat_dim: int = 15
    """Input feature dimension of downstream MLP heads"""

    network_activation: str = 'ReLU'
    """Network activation used in MLPs"""

    num_layers_density: int = 2
    """Number of layers in density MLP"""

    hidden_dim_density: int = 64
    """Dimension of hidden layers in density MLP"""

    num_layers_color: int = 3
    """Number of layers in color MLP"""

    hidden_dim_color: int = 64
    """Dimension of hidden layers in color MLP"""

    num_layers_feature: int = 3
    """Number of layers in feature MLP"""

    hidden_dim_feature: int = 64
    """Dimension of hidden layers in feature MLP"""

    feature_output_activation: str = 'Tanh'
    """Output activation for feature MLP"""

    predict_shadow: bool = True
    """Enable shadow prediction"""

    num_layers_shadow: int = 3
    """Number of layers in shadow MLP"""

    hidden_dim_shadow: int = 64
    """Dimension of hidden layers in shadow MLP"""

    predict_flow: bool = True
    """Enable flow prediction"""

    num_layers_flow: int = 3
    """Number of layers in flow MLP"""

    hidden_dim_flow: int = 64
    """Dimension of hidden layers in flow MLP"""

    appearance_embedding_dim: int = 48
    """Dimension of appearance embedding used to compute static color. Set to 0 to disable"""

    video_frequencies: int = 6
    """Dimension of fourier encoding applied to time when computing appearance embeddings"""

    num_layers_env_map: int = 2
    """Number of layers in environment map MLPs. Set to 0 to disable the environment map."""

    hidden_dim_env_map: int = 64
    """Dimension of hidden layers in environment map MLPs"""

    num_levels_env_map: int = 6
    """Number of levels in environment map hash table"""

    features_per_level_env_map: int = 2
    """Features per hash table level in environment map hash table"""

    base_resolution_env_map: int = 16
    """Resolution at first level of environment map hash tables"""

    log2_hashmap_size_env_map: int = 19
    """Size of environment map hash table"""

    num_render_samples: int = 192
    """Number of samples used to render a ray"""

    use_composite_prop: bool = True
    """Model the proposal network with separate static and dynamic branches"""

    log2_hashmap_size_proposal_static: int = 19
    """Size of static proposal network hash table"""

    log2_hashmap_size_proposal_dynamic: int = 19
    """Size of dynamic proposal network hash table"""

    num_levels_proposal: int = 9
    """Number of levels in proposal hash tables"""

    hidden_dim_proposal: int = 16
    """Dimension of hidden layers in proposal MLPs"""

    proposal_max_resolutions: Tuple[int, ...] = (4096, 8192)
    """Max resolution for each proposal network"""

    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 512)
    """Number of samples per ray for each proposal network"""

    proposal_warmup: int = 5000
    """Scales n from 1 to proposal_update_every over this many steps"""

    num_proposal_iterations: int = 2
    """Number of proposal network iterations"""

    use_proposal_weight_anneal: bool = True
    """Whether to use proposal weight annealing"""

    proposal_weights_anneal_slope: float = 10.0
    """Slope of the annealing function for the proposal weights"""

    proposal_weights_anneal_max_num_iters: int = 1000
    """Max num iterations for the annealing function"""

    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {RGB: 1.0, DEPTH: 0.1, FEATURES: 1.0, DYNAMIC: 0.001, SHADOW: 0.05, FLOW_CYCLE: 1, FLOW_SLOW: 0.1,
         FLOW_SMOOTH_SPATIAL: 0.1, FLOW_SMOOTH_TEMPORAL: 0.1, SKY: 0.1, DISTORTION: 0.002, INTERLEVEL: 1.0})

    optical_flow_decrease_max_iters: float = 250000
    """Period over which optical flow loss coefficient decreases"""

    optical_flow_loss_start: float = 0.02
    """Starting optical flow loss coefficient"""

    optical_flow_loss_end: float = 0.0000002
    """Ending optical flow loss coefficient"""

    entropy_increase_max_iters: float = 250000
    """Period over which entropy loss coefficient increases"""

    entropy_loss_start: float = 0.0001
    """Starting entropy loss coefficient"""

    entropy_loss_end: float = 0.01
    """Ending entropy loss coefficient"""

    entropy_skewness: float = 1.75
    """Entropy skewness factor"""

    use_occ_weights: bool = True
    """Downsample warping-related losses with occlusion weights"""

    collider_params: Optional[Dict[str, float]] = to_immutable_dict({'near_plane': 0.01, 'far_plane': -1})
    """Near and far values used by collider. Set far to -1 to automatically determine it based on ellipse intersection"""

    use_altitude_collider: bool = True
    """Set to true to terminate ray sampling at ground level as defined by scene bounds"""

    centroids_path: Optional[Path] = None
    shards_pattern: Optional[str] = None

    feature_clusters: tyro.conf.Suppress[Dict[str, torch.Tensor]] = to_immutable_dict({})
    feature_colors: tyro.conf.Suppress[Dict[str, torch.Tensor]] = to_immutable_dict({})


class SUDSModel(Model):
    config: SUDSModelConfig

    def __init__(
            self,
            config: SUDSModelConfig,
            scene_box: SceneBox,
            metadata: Dict[str, Any],
            **kwargs,
    ) -> None:
        super().__init__(config=config, scene_box=scene_box, **kwargs)
        # self.config.num_proposal_samples_per_ray = (2048, 2048)
        # self.config.num_render_samples = 2048

        feature_dim = metadata[ALL_ITEMS][0].load_features(False).shape[-1] \
            if self.config.loss_coefficients[FEATURES] > 0 else 0
        self.predict_feature = feature_dim > 0
        self.max_video_id = max(x.video_id for x in metadata[ALL_ITEMS])  # for viewer
        num_videos = self.max_video_id + 1

        if self.config.predict_flow:
            time_diffs = []
            for item in metadata[ALL_ITEMS]:
                if not item.is_val:
                    if item.backward_neighbor_index is not None:
                        time_diffs.append(item.time - metadata[ALL_ITEMS][item.backward_neighbor_index].time)
                    if item.forward_neighbor_index is not None:
                        time_diffs.append(metadata[ALL_ITEMS][item.forward_neighbor_index].time - item.time)

            if len(time_diffs) == 0:
                for item in metadata[ALL_ITEMS]:
                    if item.backward_neighbor_index is not None:
                        time_diffs.append(item.time - metadata[ALL_ITEMS][item.backward_neighbor_index].time)
                    if item.forward_neighbor_index is not None:
                        time_diffs.append(metadata[ALL_ITEMS][item.forward_neighbor_index].time - item.time)

            diff_quantiles = torch.quantile(torch.FloatTensor(time_diffs), torch.linspace(0, 1, 11))
            CONSOLE.print('Time difference quantiles: {}'.format(diff_quantiles))
            flow_unit = diff_quantiles[5].item()
        else:
            flow_unit = 0

        self.altitude_base = scene_box.aabb[0, 2]
        self.pose_scale_factor = metadata[POSE_SCALE_FACTOR]
        self.collider = self._create_collider(scene_box, self.pose_scale_factor)

        if self.config.centroids_path is not None:
            self.config.static_only = True
            self.config.use_composite_prop = False

        if not self.config.dynamic_only:
            self.static = StaticField(
                num_layers=self.config.num_layers_density,
                hidden_dim=self.config.hidden_dim_density,
                geo_feat_dim=self.config.geo_feat_dim,
                num_layers_color=self.config.num_layers_color,
                hidden_dim_color=self.config.hidden_dim_color,
                appearance_embedding_dim=self.config.appearance_embedding_dim,
                num_levels=self.config.num_levels,
                features_per_level=self.config.features_per_level,
                log2_hashmap_size=self.config.log2_hashmap_size_static,
                base_resolution=self.config.base_resolution,
                num_directions=self.config.num_directions,
                feature_dim=feature_dim,
                num_layers_feature=self.config.num_layers_feature,
                hidden_dim_feature=self.config.hidden_dim_feature,
                network_activation=self.config.network_activation,
                feature_output_activation=self.config.feature_output_activation,
            )

        if not self.config.static_only:
            self.dynamic = DynamicField(
                num_layers=self.config.num_layers_density,
                hidden_dim=self.config.hidden_dim_density,
                geo_feat_dim=self.config.geo_feat_dim,
                num_layers_color=self.config.num_layers_color,
                hidden_dim_color=self.config.hidden_dim_color,
                num_levels=self.config.num_levels,
                features_per_level=self.config.features_per_level,
                log2_hashmap_size=self.config.log2_hashmap_size_dynamic,
                base_resolution=self.config.base_resolution,
                num_directions=self.config.num_directions,
                feature_dim=feature_dim,
                num_layers_feature=self.config.num_layers_feature,
                hidden_dim_feature=self.config.hidden_dim_feature,
                predict_shadow=self.config.predict_shadow,
                num_layers_shadow=self.config.num_layers_shadow,
                hidden_dim_shadow=self.config.hidden_dim_shadow,
                predict_flow=self.config.predict_flow,
                num_layers_flow=self.config.num_layers_flow,
                hidden_dim_flow=self.config.hidden_dim_flow,
                flow_unit=flow_unit,
                network_activation=self.config.network_activation,
                feature_output_activation=self.config.feature_output_activation,
            )

        if self.config.num_layers_env_map > 0:
            self.env_map = EnvMapField(
                num_layers=self.config.num_layers_env_map,
                hidden_dim=self.config.hidden_dim_env_map,
                appearance_embedding_dim=self.config.appearance_embedding_dim,
                num_levels=self.config.num_levels_env_map,
                features_per_level=self.config.features_per_level_env_map,
                log2_hashmap_size=self.config.log2_hashmap_size_env_map,
                base_resolution=self.config.base_resolution_env_map,
                feature_dim=feature_dim,
                network_activation=self.config.network_activation,
                feature_output_activation=self.config.feature_output_activation,
            )

        if self.config.centroids_path is not None:
            centroids = torch.load(self.config.centroids_path, map_location='cpu')
            centroid_origins = []
            centroid_scales = []
            static_dicts = []
            dynamic_dicts = []
            static_prop_dicts = []
            dynamic_prop_dicts = []

            for i in range(centroids.shape[0]):
                CONSOLE.log('Loading shard {} of {}'.format(i, centroids.shape[0]))
                checkpoint_paths = glob.glob(self.config.shards_pattern.format(i))
                assert len(checkpoint_paths) == 1, i
                checkpoint_path = Path(checkpoint_paths[0])

                subdirs = list((checkpoint_path / 'suds').iterdir())
                assert len(subdirs) == 1, '{} {}'.format(checkpoint_path, len(subdirs))

                config = yaml.load((subdirs[0] / 'config.yml').read_text(), Loader=yaml.Loader)
                assert isinstance(config, TrainerConfig)
                with Path(config.pipeline.datamanager.dataparser.metadata_path.replace('/project_data/ramanan',
                                                                                       '/data3')).open() as f:
                    shard_metadata = json.load(f)
                    centroid_origins.append(shard_metadata['origin'])
                    centroid_scales.append(shard_metadata['pose_scale_factor'])

                model_paths = list((subdirs[0] / 'nerfstudio_models').iterdir())
                assert len(model_paths) == 1, '{} {}'.format(subdirs[0], len(model_paths))
                state = torch.load(model_paths[0], map_location='cpu')['pipeline']
                state = {key.replace('module.', '').replace('_model.', ''): value for key, value in state.items()}
                static_dicts.append(
                    {k.replace('static.', ''): v for k, v in
                     filter(lambda x: x[0].startswith('static.'), state.items())})
                dynamic_dicts.append(
                    {k.replace('dynamic.', ''): v for k, v in
                     filter(lambda x: x[0].startswith('dynamic.'), state.items())})

                ckpt_static_prop_dicts = []
                ckpt_dynamic_prop_dicts = []
                for i in range(self.config.num_proposal_iterations):
                    skey = f'static_proposal_networks.{i}.'
                    ckpt_static_prop_dicts.append(
                        {k.replace(skey, ''): v for k, v in filter(lambda x: x[0].startswith(skey), state.items())})
                    dkey = f'proposal_networks.{i}.'
                    ckpt_dynamic_prop_dicts.append(
                        {k.replace(dkey, ''): v for k, v in filter(lambda x: x[0].startswith(dkey), state.items())})

                static_prop_dicts.append(ckpt_static_prop_dicts)
                dynamic_prop_dicts.append(ckpt_dynamic_prop_dicts)

            centroid_origins = torch.DoubleTensor(centroid_origins)

            if self.config.static_only:
                del dynamic_dicts, dynamic_prop_dicts

            if not self.config.dynamic_only:
                static_shards = []
                for static_dict in static_dicts:
                    static = copy.deepcopy(self.static)
                    static.load_state_dict(static_dict)
                    static.encoding_feature = None
                    static.mlp_feature = None
                    static.feature_dim = 0
                    static_shards.append(static)

                self.static = ShardedStaticField(
                    centroids=centroids,
                    origin=torch.DoubleTensor(metadata[ORIGIN]),
                    centroid_origins=centroid_origins,
                    scale=metadata[POSE_SCALE_FACTOR],
                    centroid_scales=centroid_scales,
                    delegates=static_shards
                )

            if not self.config.static_only:
                dynamic_shards = []
                for dynamic_dict in dynamic_dicts:
                    dynamic = copy.deepcopy(self.dynamic)
                    dynamic.load_state_dict(dynamic_dict)
                    dynamic.encoding_feature = None
                    dynamic.mlp_feature = None
                    dynamic.feature_dim = 0
                    dynamic_shards.append(dynamic)

                self.dynamic = ShardedDynamicField(
                    centroids=centroids,
                    origin=torch.DoubleTensor(metadata[ORIGIN]),
                    centroid_origins=centroid_origins,
                    scale=metadata[POSE_SCALE_FACTOR],
                    centroid_scales=centroid_scales,
                    delegates=dynamic_shards
                )

            self.config.num_layers_env_map = 0
            self.predict_feature = False

        if self.config.appearance_embedding_dim > 0:
            self.appearance_embedding = VideoEmbedding(num_videos, self.config.video_frequencies,
                                                       self.config.appearance_embedding_dim)

        update_schedule = lambda step: np.clip(np.interp(step, [0, self.config.proposal_warmup], [0, 1]), 1, 1)

        if self.config.use_composite_prop:
            assert not self.config.dynamic_only
            assert not self.config.static_only

        if self.config.static_only or self.config.use_composite_prop:
            self.static_proposal_networks = torch.nn.ModuleList()

        if not self.config.static_only:
            self.proposal_networks = torch.nn.ModuleList()

        for i in range(self.config.num_proposal_iterations):
            if self.config.static_only or self.config.use_composite_prop:
                static_prop_field = StaticProposalField(hidden_dim=self.config.hidden_dim_proposal,
                                                        num_levels=self.config.num_levels_proposal,
                                                        log2_hashmap_size=self.config.log2_hashmap_size_proposal_static,
                                                        max_resolution=self.config.proposal_max_resolutions[i],
                                                        network_activation=self.config.network_activation)

                if self.config.centroids_path is not None:
                    static_prop_shards = []
                    for static_prop_dict in static_prop_dicts:
                        static_prop = copy.deepcopy(static_prop_field)
                        static_prop.load_state_dict(static_prop_dict[i])
                        static_prop_shards.append(static_prop)

                    static_prop_field = ShardedStaticProposalField(
                        centroids=centroids,
                        origin=torch.DoubleTensor(metadata[ORIGIN]),
                        centroid_origins=centroid_origins,
                        scale=metadata[POSE_SCALE_FACTOR],
                        centroid_scales=centroid_scales,
                        delegates=static_prop_shards
                    )

                self.static_proposal_networks.append(static_prop_field)

            if not self.config.static_only:
                dynamic_prop_field = DynamicProposalField(hidden_dim=self.config.hidden_dim_proposal,
                                                          num_levels=self.config.num_levels_proposal,
                                                          log2_hashmap_size=self.config.log2_hashmap_size_proposal_dynamic,
                                                          max_resolution=self.config.proposal_max_resolutions[i],
                                                          network_activation=self.config.network_activation)

                if self.config.centroids_path is not None:
                    dynamic_prop_shards = []
                    for dynamic_prop_dict in dynamic_prop_dicts:
                        dynamic_prop = copy.deepcopy(dynamic_prop_field)
                        dynamic_prop.load_state_dict(dynamic_prop_dict[i])
                        dynamic_prop_shards.append(dynamic_prop)

                    dynamic_prop_field = ShardedDynamicProposalField(
                        centroids=centroids,
                        origin=torch.DoubleTensor(metadata[ORIGIN]),
                        centroid_origins=centroid_origins,
                        scale=metadata[POSE_SCALE_FACTOR],
                        centroid_scales=centroid_scales,
                        delegates=dynamic_prop_shards
                    )

                self.proposal_networks.append(dynamic_prop_field)

        if not self.config.dynamic_only:
            self.static_density_fns = [network.density_fn for network in self.static_proposal_networks]

        if self.config.use_composite_prop:
            self.proposal_sampler = CompositeProposalNetworkSampler(
                num_nerf_samples_per_ray=self.config.num_render_samples,
                num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
                num_proposal_network_iterations=self.config.num_proposal_iterations,
                update_sched=update_schedule,
                single_jitter=True
            )
        else:
            self.proposal_sampler = ProposalNetworkSampler(
                num_nerf_samples_per_ray=self.config.num_render_samples,
                num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
                num_proposal_network_iterations=self.config.num_proposal_iterations,
                update_sched=update_schedule,
                single_jitter=True
            )

        self.renderer_weight = SemanticRenderer()
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = SUDSDepthRenderer()

        self.mse_loss = MSELoss()
        self.l1_loss = L1Loss()

        self.mse_loss_none = MSELoss(reduction='none')
        self.l1_loss_none = L1Loss(reduction='none')

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = ssim
        self.lpips = LearnedPerceptualImagePatchSimilarity()

        for key, val in self.config.feature_clusters.items():
            self.register_buffer(self._feature_cluster_key(key), val, False)
            self.register_buffer(self._feature_color_key(key), self.config.feature_colors[key], False)
        self.has_feature_clusters = self.predict_feature and len(self.config.feature_clusters) > 0  # for viewer

        if self.has_feature_clusters:
            self.filter_key = min(self.config.feature_clusters.keys())  # TODO: support multiple feature filters
            self.register_buffer('filtered_classes',
                                 torch.ones(len(self.config.feature_clusters[self.filter_key]), dtype=torch.bool),
                                 False)

        # Make a mutable copy as entropy and optical flow loss evolve over time
        self.loss_coefficients = {k: v for k, v in self.config.loss_coefficients.items()}

        if not self.config.static_only:
            self.register_buffer('temporal_distortion', torch.ones(1))  # To trigger time in UI

        self.output_keys = None

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        fields = []
        if not self.config.static_only:
            fields.append(self.dynamic.named_children())
            fields.append(self.proposal_networks.named_children())

        if not self.config.dynamic_only:
            fields.append(self.static.named_children())
            fields.append(self.static_proposal_networks.named_children())

        if self.config.num_layers_env_map > 0:
            fields.append(self.env_map.named_children())

        flow = []
        flow_mlp = []
        mlps = []
        other_fields = []
        for children in fields:
            for name, child in children:
                if 'flow' in name and 'mlp' in name:
                    flow_mlp += child.parameters()
                elif 'flow' in name:
                    flow += child.parameters()
                elif 'mlp' in name:
                    mlps += child.parameters()
                else:
                    other_fields += child.parameters()

        if self.config.appearance_embedding_dim > 0:
            other_fields += self.appearance_embedding.parameters()

        param_groups = {
            'mlps': mlps,
            'fields': other_fields,
        }

        if self.config.predict_flow:
            param_groups['flow'] = flow
            param_groups['flow_mlp'] = flow_mlp

        return param_groups

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[
        TrainingCallback]:
        callbacks = [TrainingCallback(
            where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
            update_every_num_iters=1,
            func=self._update_loss_weights,
        )]

        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )

        return callbacks

    def _update_loss_weights(self, step: int):
        self.step = step

        if self.config.optical_flow_loss_start == 0:
            self.loss_coefficients[OPTICAL_FLOW] = 0
        else:
            self.loss_coefficients[OPTICAL_FLOW] = self.config.optical_flow_loss_start * \
                                                   ((
                                                            self.config.optical_flow_loss_end / self.config.optical_flow_loss_start) ** (
                                                            1 / self.config.optical_flow_decrease_max_iters)) \
                                                   ** step

        self.loss_coefficients[ENTROPY] = self.config.entropy_loss_start * \
                                          ((self.config.entropy_loss_end / self.config.entropy_loss_start) ** (
                                                  1 / self.config.entropy_increase_max_iters)) \
                                          ** step

    def get_outputs(self, ray_bundle: RayBundle, render_options: Optional[Dict[str, Any]] = None) -> Dict[
        str, torch.Tensor]:
        with torch.inference_mode(not self.training):
            filter_features = False

            if render_options is None:
                render_options = {}

            if self.has_feature_clusters and (not self.training):
                if len(render_options.get('feature_filter', [])) > 0:
                    filter_features = True
                    feature_filter = set(render_options['feature_filter'])
                    for i in range(len(self.filtered_classes)):
                        self.filtered_classes[i] = i in feature_filter

            sigma_threshold = render_options.get('sigma_threshold', None) if (not self.training) else None
            max_altitude = render_options.get('max_altitude', None) if (not self.training) else None
            min_altitude = render_options.get('min_altitude', None) if (not self.training) else None
            static_only = (render_options.get('static_only', False) and (not self.training)) or self.config.static_only
            custom_embedding = render_options.get('custom_embedding', None) if (not self.training) else None

            if self.output_keys is None:
                results = self.get_outputs_inner(ray_bundle, None, sigma_threshold, min_altitude, max_altitude,
                                                 filter_features, static_only, custom_embedding)
                self.output_keys = set(results.keys())
                return results

            output_type = render_options.get('output_type', None)
            if output_type not in self.output_keys:
                output_type = None

            results = self.get_outputs_inner(ray_bundle, output_type, sigma_threshold, min_altitude, max_altitude,
                                             filter_features, static_only, custom_embedding)

            if output_type is not None:
                # Do this so that keys appear in viewer
                for key in self.output_keys:
                    if key != output_type:
                        assert key not in results, key
                        results[key] = results[output_type]

            if not self.training:
                results = {k: torch.nan_to_num(v) for k, v in results.items()}

            return results

    def get_outputs_inner(self, ray_bundle: RayBundle, output_type: Optional[str], sigma_threshold: Optional[float],
                          min_altitude: Optional[float], max_altitude: Optional[float], filter_features: bool,
                          static_only: bool, custom_embedding: Optional[torch.Tensor]) \
            -> Dict[str, torch.Tensor]:
        if ray_bundle.times is None:
            raise AttributeError('Times are not provided.')
        if VIDEO_ID not in ray_bundle.metadata:
            raise AttributeError('Video ids are not provided.')

        need_embedding = output_type is None or output_type in {RGB, STATIC_RGB, NO_SHADOW_RGB, NO_ENV_MAP_RGB}

        if self.config.appearance_embedding_dim > 0 and need_embedding:
            if custom_embedding is not None:
                to_expand = 1
                for dim in ray_bundle.times.shape[:-1]:
                    to_expand *= dim
                appearance_embedding = custom_embedding.expand(to_expand, -1).view(*ray_bundle.times.shape[:-1], -1)
            elif self.config.centroids_path is not None:
                appearance_embedding = torch.zeros(*ray_bundle.times.shape[:-1], self.config.appearance_embedding_dim,
                                                   device=ray_bundle.times.device)
            else:
                times = ray_bundle.times.view(-1)
                video_ids = ray_bundle.metadata[VIDEO_ID].view(-1)
                appearance_embedding = self.appearance_embedding(times, video_ids).view(
                    (*ray_bundle.times.shape[:-1], -1))

        results = {}

        density_fns = self.static_density_fns if self.config.static_only else \
            [lambda p: x.density_fn(p, ray_bundle.times, ray_bundle.metadata[VIDEO_ID]) for x in self.proposal_networks]

        if filter_features:
            filter_fn = lambda x: self._filter_ray_samples(x, static_only or (output_type == STATIC_RGB),
                                                           output_type == DYNAMIC_RGB)
        else:
            filter_fn = None

        if self.config.use_composite_prop:
            ray_samples, weights_list, ray_samples_list, static_weights_list, dynamic_weights_list \
                = self.proposal_sampler(ray_bundle, static_density_fns=self.static_density_fns,
                                        dynamic_density_fns=density_fns,
                                        static_only=static_only or (output_type == STATIC_RGB),
                                        dynamic_only=output_type == DYNAMIC_RGB,
                                        filter_fn=filter_fn)
        else:
            ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle,
                                                                                density_fns=density_fns)
        if self.config.appearance_embedding_dim > 0 and need_embedding:
            ray_samples.metadata[APPEARANCE_EMBEDDING] = appearance_embedding.unsqueeze(-2).expand(
                *appearance_embedding.shape[:-1], ray_samples.frustums.starts.shape[-2], -1)

        ray_samples.metadata[FILTER_FEATURES] = filter_features
        ray_samples.metadata[OUTPUT_TYPE] = output_type

        if output_type == STATIC_RGB or static_only:
            outputs = self.static.forward(ray_samples)
        elif output_type == DYNAMIC_RGB:
            outputs = self.dynamic.forward(ray_samples)
        else:
            if self.config.dynamic_only:
                outputs = self.dynamic.forward(ray_samples)
                self._add_dynamic_weight(outputs)
            else:
                static_outputs = self.static.forward(ray_samples)
                dynamic_outputs = self.dynamic.forward(ray_samples)
                outputs = self._composite_outputs(static_outputs, dynamic_outputs, output_type, filter_features)

        if filter_features:
            self._filter_by_feature(outputs)

        if sigma_threshold is not None:
            outputs[FieldHeadNames.DENSITY][outputs[FieldHeadNames.DENSITY] < sigma_threshold] = 0

        if min_altitude is not None:
            outputs[FieldHeadNames.DENSITY].view(-1)[
                ray_samples.frustums.get_positions().view(-1, 3)[:, 2] <
                self.altitude_base + max_altitude / self.pose_scale_factor] = 0

        if max_altitude is not None:
            outputs[FieldHeadNames.DENSITY].view(-1)[
                ray_samples.frustums.get_positions().view(-1, 3)[:, 2] >
                self.altitude_base + max_altitude / self.pose_scale_factor] = 0

        weights = _get_weights(ray_samples.deltas, outputs[FieldHeadNames.DENSITY], not self.training)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        if self.training:
            results[WEIGHTS_LIST] = weights_list
            results[RAY_SAMPLES_LIST] = ray_samples_list
            if self.config.use_composite_prop:
                static_weights_list.append(
                    _get_weights(ray_samples.deltas, static_outputs[FieldHeadNames.DENSITY], not self.training))
                dynamic_weights_list.append(
                    _get_weights(ray_samples.deltas, dynamic_outputs[FieldHeadNames.DENSITY], not self.training))
                results[STATIC_WEIGHTS_LIST] = static_weights_list
                results[DYNAMIC_WEIGHTS_LIST] = dynamic_weights_list

        for i in range(self.config.num_proposal_iterations):
            prop_depth_key = f'{PROP_DEPTH}_{i}'
            if output_type is None or output_type == prop_depth_key:
                prop_z_vals = (ray_samples_list[i].frustums.starts + ray_samples_list[i].frustums.ends) / \
                              (2 * ray_bundle.metadata['directions_norm'].unsqueeze(-1))
                results[prop_depth_key] = self.renderer_depth(weights=weights_list[i], z_vals=prop_z_vals)

        z_vals = (ray_samples.frustums.starts + ray_samples.frustums.ends) / \
                 (2 * ray_bundle.metadata['directions_norm'].unsqueeze(-1))
        deltas = ray_samples.deltas

        if output_type is None or output_type == RGB:
            results[RGB] = self.renderer_weight(semantics=outputs[FieldHeadNames.RGB], weights=weights)

        if output_type is None or output_type == DEPTH:
            results[DEPTH] = self.renderer_depth(weights=weights, z_vals=z_vals)

        if self.predict_feature and (output_type is None or FEATURES in output_type):
            results[FEATURES] = self.renderer_weight(semantics=outputs[SUDSFieldHeadNames.FEATURES],
                                                     weights=weights.detach())

        if self.config.predict_shadow and (output_type is None or output_type == SHADOW) and (not static_only):
            if not self.training:
                results[SHADOW] = self.renderer_weight(
                    semantics=outputs[SUDSFieldHeadNames.SHADOWS] * (1 - outputs[SUDSFieldHeadNames.DYNAMIC_WEIGHT]),
                    weights=weights.detach())
            else:
                results[SHADOW] = outputs[SUDSFieldHeadNames.SHADOWS].squeeze().square().mean(dim=-1).unsqueeze(-1)

        if (not self.training) and (not static_only) and (output_type is None or output_type == ENTROPY):
            results[ENTROPY] = self.renderer_weight(semantics=outputs[SUDSFieldHeadNames.DYNAMIC_WEIGHT],
                                                    weights=weights)

        if SUDSFieldHeadNames.BACKWARD_FLOW in outputs:
            backward_flow = outputs[SUDSFieldHeadNames.BACKWARD_FLOW]

        if SUDSFieldHeadNames.FORWARD_FLOW in outputs:
            forward_flow = outputs[SUDSFieldHeadNames.FORWARD_FLOW]

        if self.training:
            if not static_only:
                results[DYNAMIC_WEIGHT] = outputs[SUDSFieldHeadNames.DYNAMIC_WEIGHT]

            if self.config.predict_flow:
                results[BACKWARD_DYNAMIC_WEIGHT] = outputs[SUDSFieldHeadNames.BACKWARD_DYNAMIC_WEIGHT]
                results[FORWARD_DYNAMIC_WEIGHT] = outputs[SUDSFieldHeadNames.FORWARD_DYNAMIC_WEIGHT]

                backward_weights = _get_weights(deltas, outputs[SUDSFieldHeadNames.BACKWARD_DENSITY], not self.training)
                results[BACKWARD_RGB] = self.renderer_weight(semantics=outputs[SUDSFieldHeadNames.BACKWARD_RGB],
                                                             weights=backward_weights)
                forward_weights = _get_weights(deltas, outputs[SUDSFieldHeadNames.FORWARD_DENSITY], not self.training)
                results[FORWARD_RGB] = self.renderer_weight(semantics=outputs[SUDSFieldHeadNames.FORWARD_RGB],
                                                            weights=forward_weights)

                if self.predict_feature:
                    results[BACKWARD_FEATURES] = self.renderer_weight(
                        semantics=outputs[SUDSFieldHeadNames.BACKWARD_FEATURES], weights=backward_weights.detach())
                    results[FORWARD_FEATURES] = self.renderer_weight(
                        semantics=outputs[SUDSFieldHeadNames.FORWARD_FEATURES], weights=forward_weights.detach())

                results[FLOW_SLOW] = (torch.linalg.norm(backward_flow, dim=-1, ord=1, keepdim=True) \
                                      + torch.linalg.norm(forward_flow, dim=-1, ord=1, keepdim=True)).mean(dim=1)

                results[FLOW_SMOOTH_TEMPORAL] = (backward_flow + forward_flow).square().mean(dim=1)

                spatial_weights = torch.exp(-2 * deltas[:, :-1]).detach()
                results[FLOW_SMOOTH_SPATIAL] = \
                    (spatial_weights * torch.linalg.norm(backward_flow[:, 1:] - backward_flow[:, :-1], dim=-1, ord=1,
                                                         keepdim=True)).mean(dim=1) \
                    + (spatial_weights * torch.linalg.norm(forward_flow[:, 1:] - forward_flow[:, :-1], dim=-1, ord=1,
                                                           keepdim=True)).mean(dim=1)

                backward_occs = (outputs[SUDSFieldHeadNames.BACKWARD_DYNAMIC_WEIGHT] - outputs[
                    SUDSFieldHeadNames.DYNAMIC_WEIGHT]).detach()
                forward_occs = (outputs[SUDSFieldHeadNames.FORWARD_DYNAMIC_WEIGHT] - outputs[
                    SUDSFieldHeadNames.DYNAMIC_WEIGHT]).detach()
                results[BACKWARD_FLOW_CYCLE_DIFF] = (1 - backward_occs.abs()) * outputs[
                    SUDSFieldHeadNames.BACKWARD_FLOW_CYCLE_DIFF]
                results[FORWARD_FLOW_CYCLE_DIFF] = (1 - forward_occs.abs()) * outputs[
                    SUDSFieldHeadNames.FORWARD_FLOW_CYCLE_DIFF]

                results[BACKWARD_DISOCC] = 1 - self.renderer_weight(semantics=backward_occs,
                                                                    weights=backward_weights.detach()).abs()
                results[FORWARD_DISOCC] = 1 - self.renderer_weight(semantics=forward_occs,
                                                                   weights=forward_weights.detach()).abs()
        else:
            if output_type == STATIC_RGB:
                results[STATIC_RGB] = self.renderer_weight(semantics=outputs[FieldHeadNames.RGB], weights=weights)
            if output_type == DYNAMIC_RGB:
                results[DYNAMIC_RGB] = self.renderer_weight(semantics=outputs[FieldHeadNames.RGB], weights=weights)

            if self.config.predict_shadow and (not self.config.dynamic_only) and (not static_only) \
                    and (output_type is None or output_type == NO_SHADOW_RGB):
                results[NO_SHADOW_RGB] = self.renderer_weight(
                    semantics=outputs[SUDSFieldHeadNames.NO_SHADOW_RGB], weights=weights)

            if self.config.predict_flow and not static_only:
                if output_type is None or output_type == BACKWARD_FLOW:
                    backward_flow_comp = self.renderer_weight(semantics=backward_flow, weights=weights)
                    results[BACKWARD_FLOW] = scene_flow_to_rgb(backward_flow_comp)

                if output_type is None or output_type == FORWARD_FLOW:
                    forward_flow_comp = self.renderer_weight(semantics=forward_flow, weights=weights)
                    results[FORWARD_FLOW] = scene_flow_to_rgb(forward_flow_comp)

        if (not self.training and (output_type is None or output_type == WEIGHTS_SQ)) \
                or (self.training and self.loss_coefficients[SKY] > 0):
            results[WEIGHTS_SQ] = self.renderer_accumulation(weights.square())

        if self.config.num_layers_env_map > 0:
            bg_intersection = ray_bundle.metadata[BG_INTERSECTION]
            bg_to_eval = bg_intersection > 0
            has_bg = torch.any(bg_to_eval > 0)

            if not self.training:
                if output_type is None:
                    results[NO_ENV_MAP_RGB] = results[RGB].clone() if has_bg else results[RGB]
                elif output_type == NO_ENV_MAP_RGB:
                    results[NO_ENV_MAP_RGB] = self.renderer_weight(semantics=outputs[FieldHeadNames.RGB],
                                                                   weights=weights)

            if has_bg and output_type not in {DYNAMIC_RGB, NO_ENV_MAP_RGB}:
                env_appearance_embedding = appearance_embedding[bg_to_eval.view(-1) > 0] \
                    .reshape(-1, self.config.appearance_embedding_dim) \
                    if (self.config.appearance_embedding_dim > 0 and need_embedding) \
                    else None

                env_outputs = self.env_map.forward(ray_bundle.directions.view(-1, 3)[bg_to_eval.view(-1) > 0],
                                                   ray_bundle.metadata[VIDEO_ID][bg_to_eval > 0].reshape(-1, 1),
                                                   env_appearance_embedding, output_type, filter_features)

                env_weight = 1 - torch.sum(weights, dim=-2)[bg_to_eval > 0]
                rgb_keys = {}
                if output_type is None or output_type == RGB:
                    rgb_keys[RGB] = env_weight

                if self.config.predict_shadow and (not self.config.dynamic_only) and (not static_only) \
                        and ((output_type is None and not self.training) or output_type == NO_SHADOW_RGB):
                    rgb_keys[NO_SHADOW_RGB] = env_weight

                if output_type == STATIC_RGB:
                    rgb_keys[STATIC_RGB] = env_weight

                if self.training and self.config.predict_flow:
                    rgb_keys[BACKWARD_RGB] = 1 - torch.sum(backward_weights, dim=-2)[bg_to_eval > 0]
                    rgb_keys[FORWARD_RGB] = 1 - torch.sum(forward_weights, dim=-2)[bg_to_eval > 0]

                for rgb, weight in rgb_keys.items():
                    results[rgb][bg_to_eval.squeeze() > 0] += weight.unsqueeze(-1) * env_outputs[FieldHeadNames.RGB]

                bg_depth = bg_intersection[bg_to_eval > 0] / ray_bundle.metadata['directions_norm'][bg_to_eval > 0]
                depth_keys = {}
                if output_type is None or output_type == DEPTH:
                    depth_keys[DEPTH] = env_weight

                if not self.training:
                    for i in range(self.config.num_proposal_iterations):
                        prop_depth_key = f'{PROP_DEPTH}_{i}'
                        if output_type is None or output_type == prop_depth_key:
                            depth_keys[prop_depth_key] = 1 - torch.sum(weights_list[i], dim=-2)[bg_to_eval > 0]

                for depth, weight in depth_keys.items():
                    results[depth][bg_to_eval > 0] += weight * bg_depth

                if self.predict_feature:
                    feature_keys = {}
                    if output_type is None or FEATURES in output_type:
                        feature_keys[FEATURES] = env_weight

                    if self.training and self.config.predict_flow:
                        feature_keys[BACKWARD_FEATURES] = rgb_keys[BACKWARD_RGB]
                        feature_keys[FORWARD_FEATURES] = rgb_keys[FORWARD_RGB]

                    for feature, weight in feature_keys.items():
                        results[feature][bg_to_eval.squeeze() > 0] += weight.unsqueeze(-1).detach() \
                                                                      * env_outputs[SUDSFieldHeadNames.FEATURES]

        # Doesn't happen in viewer
        has_backward = BACKWARD_NEIGHBOR_TIME_DIFF in ray_bundle.metadata
        has_forward = FORWARD_NEIGHBOR_TIME_DIFF in ray_bundle.metadata
        if self.config.optical_flow_loss_start > 0 and (has_backward or has_forward) and output_type is None:
            with torch.cuda.amp.autocast(enabled=False):
                positions = ray_samples.frustums.get_positions().detach()
                position_weights = weights.detach()

                if has_backward:
                    backward_pos_to_eval = (ray_bundle.metadata[BACKWARD_FLOW_VALID] > 0) \
                        if self.training else torch.ones_like(ray_bundle.metadata[BACKWARD_NEIGHBOR_TIME_DIFF],
                                                              dtype=torch.bool)
                    backward_pos_to_eval = backward_pos_to_eval.view(backward_pos_to_eval.shape[:-1])
                    backward_flow_to_use = backward_flow[backward_pos_to_eval > 0]

                if has_forward:
                    forward_pos_to_eval = (ray_bundle.metadata[FORWARD_FLOW_VALID] > 0) \
                        if self.training else torch.ones_like(ray_bundle.metadata[FORWARD_NEIGHBOR_TIME_DIFF],
                                                              dtype=torch.bool)
                    forward_pos_to_eval = forward_pos_to_eval.view(forward_pos_to_eval.shape[:-1])
                    forward_flow_to_use = forward_flow[forward_pos_to_eval > 0]

                if self.config.num_layers_env_map > 0 and has_bg:
                    # Make the sky position very far so optical flow is small
                    positions = torch.cat([
                        positions, (ray_bundle.origins + ray_bundle.directions * bg_intersection).unsqueeze(1)], 1)
                    position_weights = torch.cat([position_weights, torch.zeros_like(weights[:, -1:])], -2)
                    position_weights[bg_intersection.squeeze() > 0, -1] = env_weight.unsqueeze(-1).detach()

                    if has_backward and len(backward_pos_to_eval) > 0:
                        backward_flow_to_use = torch.cat(
                            [backward_flow_to_use, torch.zeros_like(backward_flow_to_use[:, -1:])], 1)

                    if has_forward and len(forward_pos_to_eval) > 0:
                        forward_flow_to_use = torch.cat(
                            [forward_flow_to_use, torch.zeros_like(forward_flow_to_use[:, -1:])], 1)

                if has_backward and len(backward_pos_to_eval) > 0:
                    backward_neighbor_w2cs = ray_bundle.metadata[BACKWARD_NEIGHBOR_W2C].view(
                        ray_bundle.metadata[BACKWARD_NEIGHBOR_W2C].shape[:-1] + (4, 4))[backward_pos_to_eval > 0]
                    backward_K = ray_bundle.metadata[BACKWARD_NEIGHBOR_K].view(
                        ray_bundle.metadata[BACKWARD_NEIGHBOR_K].shape[:-1] + (3, 3))[backward_pos_to_eval > 0]
                    backward_positions_world = positions[backward_pos_to_eval > 0] + backward_flow_to_use
                    results[BACKWARD_UV] = _get_flow_uv(backward_neighbor_w2cs, backward_K, backward_positions_world,
                                                        position_weights[backward_pos_to_eval > 0])

                if has_forward and len(forward_pos_to_eval) > 0:
                    forward_neighbor_w2cs = ray_bundle.metadata[FORWARD_NEIGHBOR_W2C].view(
                        ray_bundle.metadata[FORWARD_NEIGHBOR_W2C].shape[:-1] + (4, 4))[forward_pos_to_eval > 0]
                    forward_K = ray_bundle.metadata[FORWARD_NEIGHBOR_K].view(
                        ray_bundle.metadata[FORWARD_NEIGHBOR_K].shape[:-1] + (3, 3))[forward_pos_to_eval > 0]
                    forward_positions_world = positions[forward_pos_to_eval > 0] + forward_flow_to_use
                    results[FORWARD_UV] = _get_flow_uv(forward_neighbor_w2cs, forward_K, forward_positions_world,
                                                       position_weights[forward_pos_to_eval > 0])

        if not self.training and self.predict_feature and (output_type is None or FEATURES in output_type):
            device = results[FEATURES].device

            for key in self.config.feature_clusters:
                feature_key = f'{FEATURES}_{key}'
                if output_type is None or output_type == feature_key:
                    results[feature_key] = self.get_buffer(self._feature_color_key(key))[
                        kmeans_predict(results[FEATURES], self.get_buffer(self._feature_cluster_key(key)),
                                       device=device, tqdm_flag=False)]
            del results[FEATURES]

        if not self.training and output_type is None and (not static_only) and (not self.config.dynamic_only):
            for rgb in [STATIC_RGB, DYNAMIC_RGB]:
                for key, val in self.get_outputs_inner(ray_bundle, rgb, sigma_threshold, min_altitude,
                                                       max_altitude, filter_features, static_only,
                                                       custom_embedding).items():
                    assert key not in results, key
                    results[key] = val

        return results

    def get_metrics_dict(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) \
            -> Dict[str, torch.Tensor]:
        self.outputs = outputs
        self.batch = batch
        device = outputs[RGB].device
        rgb_gt = batch[RGB].to(device)

        metrics_dict = {
            'psnr': self.psnr(batch[RGB].to(device), outputs[RGB]),
            RGB: self.mse_loss(outputs[RGB], rgb_gt)
        }

        if self.loss_coefficients[DEPTH] > 0:
            depth = batch[DEPTH].to(device)
            true_depth = depth > 0
            depth_to_eval = depth[true_depth > 0]

            depth_keys = [DEPTH]
            for i in range(self.config.num_proposal_iterations):
                depth_keys.append(f'{PROP_DEPTH}_{i}')

            for depth_key in depth_keys:
                if depth_to_eval.shape[0] > 0:
                    depth_loss = self.mse_loss(outputs[depth_key][true_depth > 0].view(depth_to_eval.shape),
                                               depth_to_eval)
                else:
                    depth_loss = torch.tensor(0.0)

                metrics_dict[depth_key] = depth_loss

        if not self.training:
            return metrics_dict

        for loss in [OPTICAL_FLOW, ENTROPY]:
            metrics_dict[f'{loss}_loss_coefficient'] = self.loss_coefficients[loss]

        if self.predict_feature:
            feature_gt = batch[FEATURES].to(device)
            metrics_dict[FEATURES] = self.l1_loss(outputs[FEATURES], feature_gt)

        if (not self.config.dynamic_only) and (not self.config.static_only):
            blendw = outputs[DYNAMIC_WEIGHT].pow(self.config.entropy_skewness)
            rev_blendw = 1 - blendw
            metrics_dict[ENTROPY] = -(
                    blendw * torch.log(blendw + 1e-8) + rev_blendw * torch.log(rev_blendw + 1e-8)).mean()
            metrics_dict[DYNAMIC] = outputs[DYNAMIC_WEIGHT].squeeze().max(dim=-1)[0].mean()

        if self.config.predict_flow:
            for rgb, disocc in [(BACKWARD_RGB, BACKWARD_DISOCC), (FORWARD_RGB, FORWARD_DISOCC)]:
                metrics_dict[disocc] = outputs[disocc].mean()
                if self.config.use_occ_weights:
                    metrics_dict[rgb] = (outputs[disocc] * self.mse_loss_none(outputs[rgb], rgb_gt)).mean()
                else:
                    metrics_dict[rgb] = self.mse_loss(outputs[rgb], rgb_gt)

            if self.predict_feature:
                for feature, disocc in [(BACKWARD_FEATURES, BACKWARD_DISOCC), (FORWARD_FEATURES, FORWARD_DISOCC)]:
                    if self.config.use_occ_weights:
                        metrics_dict[feature] = (
                                outputs[disocc] * self.l1_loss_none(outputs[feature], feature_gt)).mean()
                    else:
                        metrics_dict[feature] = self.l1_loss(outputs[feature], feature_gt)

            metrics_dict[BACKWARD_FLOW_CYCLE_DIFF] = outputs[BACKWARD_FLOW_CYCLE_DIFF].mean()
            metrics_dict[FORWARD_FLOW_CYCLE_DIFF] = outputs[FORWARD_FLOW_CYCLE_DIFF].mean()
            metrics_dict[FLOW_SLOW] = outputs[FLOW_SLOW].mean()
            metrics_dict[FLOW_SMOOTH_SPATIAL] = outputs[FLOW_SMOOTH_SPATIAL].mean()
            metrics_dict[FLOW_SMOOTH_TEMPORAL] = outputs[FLOW_SMOOTH_TEMPORAL].mean()

            if self.config.optical_flow_loss_start > 0:
                for uv_key, flow_key, flow_valid_key in [(BACKWARD_UV, BACKWARD_FLOW, BACKWARD_FLOW_VALID),
                                                         (FORWARD_UV, FORWARD_FLOW, FORWARD_FLOW_VALID)]:
                    flow_valid = batch[flow_valid_key].squeeze() > 0
                    flow_gt = batch[flow_key][flow_valid > 0].to(device)
                    if flow_gt.shape[0] > 0:
                        optical_flow = outputs[uv_key] - batch[RAY_INDEX][flow_valid > 0][..., [2, 1]].to(device)
                        metrics_dict[flow_key] = self.l1_loss(optical_flow, flow_gt)

        if self.config.predict_shadow:
            metrics_dict[SHADOW] = outputs[SHADOW].mean()

        if self.loss_coefficients[SKY] > 0:
            valid_sky = batch[SKY] > 0
            sky_pixels = outputs[WEIGHTS_SQ][valid_sky]
            if sky_pixels.shape[0] > 0:
                metrics_dict[SKY] = sky_pixels.mean()
            else:
                metrics_dict[SKY] = torch.tensor(0.0)

        if self.loss_coefficients[DISTORTION] > 0:
            metrics_dict[DISTORTION] = distortion_loss(outputs[WEIGHTS_LIST], outputs[RAY_SAMPLES_LIST])

        if self.loss_coefficients[INTERLEVEL] > 0:
            if self.config.use_composite_prop:
                metrics_dict[STATIC_INTERLEVEL] = interlevel_loss(outputs[STATIC_WEIGHTS_LIST],
                                                                  outputs[RAY_SAMPLES_LIST])
                metrics_dict[DYNAMIC_INTERLEVEL] = interlevel_loss(outputs[DYNAMIC_WEIGHTS_LIST],
                                                                   outputs[RAY_SAMPLES_LIST])
            else:
                metrics_dict[INTERLEVEL] = interlevel_loss(outputs[WEIGHTS_LIST], outputs[RAY_SAMPLES_LIST])

        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        loss_dict = {RGB: metrics_dict[RGB].clone()}

        if self.loss_coefficients[DEPTH] > 0:
            loss_dict[DEPTH] = metrics_dict[DEPTH].clone()

        if not self.training:
            return loss_dict

        if self.predict_feature:
            loss_dict[FEATURES] = metrics_dict[FEATURES].clone()

        if (not self.config.dynamic_only) and (not self.config.static_only):
            loss_dict[ENTROPY] = metrics_dict[ENTROPY].clone()
            loss_dict[DYNAMIC] = metrics_dict[DYNAMIC].clone()

        if self.config.predict_flow:
            for rgb in [BACKWARD_RGB, FORWARD_RGB]:
                loss_dict[RGB] += metrics_dict[rgb]

            if self.predict_feature:
                for feature in [BACKWARD_FEATURES, FORWARD_FEATURES]:
                    loss_dict[FEATURES] += metrics_dict[feature]

            loss_dict[FLOW_CYCLE] = metrics_dict[BACKWARD_FLOW_CYCLE_DIFF] + metrics_dict[FORWARD_FLOW_CYCLE_DIFF]
            for key in [FLOW_SLOW, FLOW_SMOOTH_TEMPORAL, FLOW_SMOOTH_SPATIAL]:
                loss_dict[key] = metrics_dict[key].clone()

            if self.config.optical_flow_loss_start > 0:
                loss_dict[OPTICAL_FLOW] = metrics_dict.get(BACKWARD_FLOW, 0) + metrics_dict.get(FORWARD_FLOW, 0)

        if self.config.predict_shadow:
            loss_dict[SHADOW] = metrics_dict[SHADOW].clone()

        if self.loss_coefficients[SKY] > 0:
            loss_dict[SKY] = metrics_dict[SKY].clone()

        if self.loss_coefficients[DISTORTION] > 0:
            loss_dict[DISTORTION] = metrics_dict[DISTORTION].clone()

        if self.loss_coefficients[INTERLEVEL] > 0:
            if self.config.use_composite_prop:
                loss_dict[INTERLEVEL] = metrics_dict[STATIC_INTERLEVEL] + metrics_dict[DYNAMIC_INTERLEVEL]
            else:
                loss_dict[INTERLEVEL] = metrics_dict[INTERLEVEL].clone()

        loss_dict = misc.scale_dict(loss_dict, self.loss_coefficients)

        for key, val in loss_dict.items():
            if not math.isfinite(val):
                raise Exception('Loss is not finite: {}'.format(loss_dict))

        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        device = outputs[RGB].device
        rgb_gt_base = batch[RGB].to(device)
        rgb_base = outputs[RGB]
        acc = colormaps.apply_colormap(outputs[WEIGHTS_SQ], cmap='inferno')
        depth = self.apply_depth_colormap(outputs[DEPTH])

        if self.config.dynamic_only or self.config.static_only:
            combined_rgb = torch.cat([rgb_gt_base, rgb_base], dim=1)
        else:
            combined_rgb = torch.cat([torch.cat([rgb_gt_base, rgb_base], dim=1),
                                      torch.cat([outputs[STATIC_RGB], outputs[DYNAMIC_RGB]], dim=1)], dim=0)

        mask = batch[MASK].to(device)
        rgb_gt = rgb_gt_base.clone()
        rgb = rgb_base.clone()
        rgb_gt[mask <= 0] = 0
        rgb[mask <= 0] = 0

        images_dict = {
            RGB: combined_rgb,
            WEIGHTS_SQ: acc,
            DEPTH: depth,
        }

        if not self.config.static_only:
            images_dict[ENTROPY] = colormaps.apply_colormap(outputs[ENTROPY], cmap='inferno')

        if self.predict_feature:
            for key in self.config.feature_clusters:
                images_dict[f'{FEATURES}_{key}'] = torch.cat([batch[f'{FEATURES}_{key}'].to(device),
                                                              outputs[f'{FEATURES}_{key}']], dim=1)

        if self.config.predict_shadow:
            shadow_intensity = colormaps.apply_colormap(outputs[SHADOW], cmap='inferno')
            if self.config.dynamic_only:
                images_dict[SHADOW] = shadow_intensity
            else:
                images_dict[NO_SHADOW_RGB] = torch.cat([outputs[NO_SHADOW_RGB], shadow_intensity], dim=1)

        if self.config.optical_flow_loss_start > 0:
            for uv_key, flow_key in [(BACKWARD_UV, BACKWARD_FLOW), (FORWARD_UV, FORWARD_FLOW)]:
                if flow_key in batch:
                    flow_gt = batch[flow_key].to(device)
                    flow = outputs[uv_key] - batch[RAY_INDEX][..., [2, 1]].to(device)

                    # Make flow more visible when using sparse correspondences
                    gt_skip = 1 if flow_gt.view(-1, 2)[flow_gt.view(-1, 2).abs().sum(dim=-1) > 0].shape[0] < 0.1 * \
                                   flow_gt.view(-1, 2).shape[0] else None

                    images_dict[flow_key] = torch.cat([
                        cat_imgflo(rgb_gt_base, flow_gt, skip=gt_skip),
                        cat_imgflo(rgb_base, flow)
                    ], dim=1)

        if self.config.num_layers_env_map > 0:
            if self.loss_coefficients[SKY] > 0:
                rgb_gt_sky = rgb_gt_base.clone()
                rgb_gt_sky[batch[SKY] <= 0] = 0
                images_dict[NO_ENV_MAP_RGB] = torch.cat([outputs[NO_ENV_MAP_RGB], rgb_gt_sky], dim=1)
            else:
                images_dict[NO_ENV_MAP_RGB] = outputs[NO_ENV_MAP_RGB]

        for i in range(self.config.num_proposal_iterations):
            images_dict[f'{PROP_DEPTH}_{i}'] = self.apply_depth_colormap(outputs[f'{PROP_DEPTH}_{i}'])

        # SSIM needs to be [H, W, C]
        ssim = self.ssim(rgb_gt, rgb)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        rgb_gt = torch.moveaxis(rgb_gt, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(rgb_gt, rgb)
        lpips = self.lpips(rgb_gt, rgb)

        metrics_dict = {
            'psnr': float(psnr),
            'ssim': float(ssim),
            'lpips': float(lpips),
        }

        return metrics_dict, images_dict

    def apply_depth_colormap(self, depth: torch.Tensor) -> torch.Tensor:
        log_depth = torch.log(depth + 1e-8)  # Increase visualization contrast
        to_use = log_depth.view(-1)
        while to_use.shape[0] > 2 ** 24:
            to_use = to_use[::2]

        mi, ma = torch.quantile(to_use.clamp_max(np.log(self.collider.far)),
                                torch.FloatTensor([0.05, 0.95]).to(to_use.device))
        log_depth = (log_depth - mi) / (ma - mi + 1e-10)
        log_depth = torch.clip(log_depth, 0, 1)
        return colormaps.apply_colormap(1 - log_depth, cmap='inferno')

    def _composite_outputs(self, static_outputs: Dict[Union[FieldHeadNames, SUDSFieldHeadNames], TensorType],
                           dynamic_outputs: Dict[Union[FieldHeadNames, SUDSFieldHeadNames], TensorType],
                           output_type: Optional[str], filter_features: bool) -> \
            Dict[Union[FieldHeadNames, SUDSFieldHeadNames], TensorType]:
        composite_weights = F.normalize(
            torch.cat([static_outputs[FieldHeadNames.DENSITY], dynamic_outputs[FieldHeadNames.DENSITY]], -1), p=1,
            dim=-1)

        outputs = {
            FieldHeadNames.DENSITY: static_outputs[FieldHeadNames.DENSITY] + dynamic_outputs[FieldHeadNames.DENSITY],
            SUDSFieldHeadNames.DYNAMIC_WEIGHT: composite_weights[..., 1:]
        }

        if output_type is None or RGB in output_type:
            static_rgb_component = static_outputs[FieldHeadNames.RGB]

            if self.config.predict_shadow and (output_type is None or output_type != NO_SHADOW_RGB):
                static_rgb_component = (1 - dynamic_outputs[SUDSFieldHeadNames.SHADOWS]) * static_rgb_component

            outputs[FieldHeadNames.RGB] = composite_weights[..., 0:1] * static_rgb_component \
                                          + composite_weights[..., 1:] * dynamic_outputs[FieldHeadNames.RGB]

        if self.predict_feature and (filter_features or (output_type is None or FEATURES in output_type)):
            outputs[SUDSFieldHeadNames.FEATURES] = composite_weights[..., 0:1] * static_outputs[
                SUDSFieldHeadNames.FEATURES] + composite_weights[..., 1:] * dynamic_outputs[SUDSFieldHeadNames.FEATURES]

        if self.config.predict_shadow and (output_type is None or output_type == SHADOW):
            outputs[SUDSFieldHeadNames.SHADOWS] = dynamic_outputs[SUDSFieldHeadNames.SHADOWS]

        if self.config.predict_shadow and not self.training and (output_type is None or output_type == NO_SHADOW_RGB):
            outputs[SUDSFieldHeadNames.NO_SHADOW_RGB] = \
                composite_weights[..., 0:1] * static_outputs[FieldHeadNames.RGB] + \
                composite_weights[..., 1:] * dynamic_outputs[FieldHeadNames.RGB]

        if self.config.predict_flow and (output_type is None or output_type in {BACKWARD_FLOW, FORWARD_FLOW}):
            flow_keys = [SUDSFieldHeadNames.BACKWARD_FLOW, SUDSFieldHeadNames.FORWARD_FLOW]
            if self.training:
                flow_keys += [SUDSFieldHeadNames.BACKWARD_FLOW_CYCLE_DIFF, SUDSFieldHeadNames.FORWARD_FLOW_CYCLE_DIFF]
                # SUDSFieldHeadNames.FLOW_SLOW, SUDSFieldHeadNames.FLOW_SMOOTH_TEMPORAL]

            if not self.training:
                for key in flow_keys:
                    outputs[key] = composite_weights[..., 1:].detach() * dynamic_outputs[key]
            else:
                for key in flow_keys:
                    outputs[key] = dynamic_outputs[key]

            if self.training:
                backward_composite_weights = F.normalize(
                    torch.cat([static_outputs[FieldHeadNames.DENSITY],
                               dynamic_outputs[SUDSFieldHeadNames.BACKWARD_DENSITY]], -1), p=1, dim=-1)
                outputs[SUDSFieldHeadNames.BACKWARD_RGB] = \
                    backward_composite_weights[..., 0:1] * static_rgb_component \
                    + backward_composite_weights[..., 1:] * dynamic_outputs[SUDSFieldHeadNames.BACKWARD_RGB]

                outputs[SUDSFieldHeadNames.BACKWARD_DENSITY] = \
                    static_outputs[FieldHeadNames.DENSITY] + dynamic_outputs[SUDSFieldHeadNames.BACKWARD_DENSITY]

                if self.predict_feature:
                    outputs[SUDSFieldHeadNames.BACKWARD_FEATURES] = \
                        backward_composite_weights[..., 0:1] * static_outputs[SUDSFieldHeadNames.FEATURES] + \
                        backward_composite_weights[..., 1:] * dynamic_outputs[SUDSFieldHeadNames.BACKWARD_FEATURES]

                outputs[SUDSFieldHeadNames.BACKWARD_DYNAMIC_WEIGHT] = backward_composite_weights[..., 1:]

                forward_composite_weights = F.normalize(
                    torch.cat([static_outputs[FieldHeadNames.DENSITY],
                               dynamic_outputs[SUDSFieldHeadNames.FORWARD_DENSITY]], -1), p=1, dim=-1)
                outputs[SUDSFieldHeadNames.FORWARD_RGB] = \
                    forward_composite_weights[..., 0:1] * static_rgb_component \
                    + forward_composite_weights[..., 1:] * dynamic_outputs[SUDSFieldHeadNames.FORWARD_RGB]

                outputs[SUDSFieldHeadNames.FORWARD_DENSITY] = \
                    static_outputs[FieldHeadNames.DENSITY] + dynamic_outputs[SUDSFieldHeadNames.FORWARD_DENSITY]

                if self.predict_feature:
                    outputs[SUDSFieldHeadNames.FORWARD_FEATURES] = \
                        forward_composite_weights[..., 0:1] * static_outputs[SUDSFieldHeadNames.FEATURES] + \
                        forward_composite_weights[..., 1:] * dynamic_outputs[SUDSFieldHeadNames.FORWARD_FEATURES]

                outputs[SUDSFieldHeadNames.FORWARD_DYNAMIC_WEIGHT] = forward_composite_weights[..., 1:]

        return outputs

    def _add_dynamic_weight(self, outputs: Dict[Union[FieldHeadNames, SUDSFieldHeadNames], TensorType]) -> None:
        outputs[SUDSFieldHeadNames.DYNAMIC_WEIGHT] = torch.ones_like(outputs[FieldHeadNames.DENSITY])
        if self.config.predict_flow:
            outputs[SUDSFieldHeadNames.BACKWARD_DYNAMIC_WEIGHT] = torch.ones_like(outputs[FieldHeadNames.DENSITY])
            outputs[SUDSFieldHeadNames.FORWARD_DYNAMIC_WEIGHT] = torch.ones_like(outputs[FieldHeadNames.DENSITY])

    def _filter_by_feature(self, outputs: Dict[Union[FieldHeadNames, SUDSFieldHeadNames], TensorType]) -> None:
        if (not self.has_feature_clusters) or self.training:
            return

        feature_classes = []
        to_process = outputs[SUDSFieldHeadNames.FEATURES].view(-1, outputs[SUDSFieldHeadNames.FEATURES].shape[-1])

        for i in range(0, to_process.shape[0], self.config.eval_num_rays_per_chunk):
            item_chunk = to_process[i:i + self.config.eval_num_rays_per_chunk]
            feature_classes.append(
                kmeans_predict(item_chunk, self.get_buffer(self._feature_cluster_key(self.filter_key)),
                               device=item_chunk.device, tqdm_flag=False))

        feature_classes = torch.cat(feature_classes)
        outputs[FieldHeadNames.DENSITY].view(-1)[self.filtered_classes[feature_classes] <= 0] = 0

    def _filter_ray_samples(self, ray_samples: RaySamples, static_only: bool, dynamic_only: bool) -> torch.Tensor:
        ray_samples.metadata[OUTPUT_TYPE] = FEATURES
        ray_samples.metadata[FILTER_FEATURES] = True

        if static_only:
            outputs = self.static.forward(ray_samples)
        elif dynamic_only:
            outputs = self.dynamic.forward(ray_samples)
        else:
            static_outputs = self.static.forward(ray_samples)
            dynamic_outputs = self.dynamic.forward(ray_samples)
            outputs = self._composite_outputs(static_outputs, dynamic_outputs, FEATURES, True)

        self._filter_by_feature(outputs)
        return outputs[FieldHeadNames.DENSITY] > 0

    def _create_collider(self, scene_box: SceneBox, pose_scale_factor: float) -> SUDSCollider:
        if self.config.collider_params['far_plane'] > 0:
            far = self.config.collider_params['far_plane'] / pose_scale_factor
        elif self.config.num_layers_env_map > 0:
            far = 1e10
        else:
            far = 2
        if self.config.num_layers_env_map > 0:
            sphere_center = scene_box.get_center()
            sphere_radius = (scene_box.aabb[1] - scene_box.aabb[0]) * math.sqrt(3) / 2
        else:
            sphere_center = None
            sphere_radius = None

        return SUDSCollider(near=self.config.collider_params['near_plane'] / pose_scale_factor,
                            far=far,
                            scene_bounds=scene_box.aabb if self.config.use_altitude_collider else None,
                            sphere_center=sphere_center,
                            sphere_radius=sphere_radius)

    @staticmethod
    def _merge_coarse_fine(key: Union[FieldHeadNames, SUDSFieldHeadNames],
                           outputs_coarse: Dict[Union[FieldHeadNames, SUDSFieldHeadNames], torch.Tensor],
                           outputs_fine: Dict[Union[FieldHeadNames, SUDSFieldHeadNames], torch.Tensor],
                           ordering: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [torch.gather(torch.cat([outputs_coarse[key][..., i], outputs_fine[key][..., i]], -1), 1,
                          ordering).unsqueeze(-1) for i
             in range(outputs_coarse[key].shape[-1])], -1)

    @staticmethod
    def _feature_cluster_key(key):
        return f'__cluster_{key}'

    @staticmethod
    def _feature_color_key(key):
        return f'__color_{key}'


@torch.jit.script
def _get_flow_uv(neighbor_w2cs: torch.Tensor, K: torch.Tensor, positions_world: torch.Tensor,
                 position_weights: torch.Tensor) -> torch.Tensor:
    positions_cam = neighbor_w2cs @ torch.cat([positions_world,
                                               torch.ones_like(
                                                   positions_world[..., :1])],
                                              -1).transpose(1, 2)  # (n, 4, samples)
    positions_cam = positions_cam[:, :3].transpose(1, 2)  # (n, samples, 3)
    positions_cam[..., 1:] *= -1  # RUB to RDF
    z_pos = positions_cam[..., 2]
    is_valid = z_pos > 1e-5
    valid_weights = position_weights.clone()

    valid_weights[is_valid <= 0] = 0
    positions_cam[is_valid <= 0] = 0
    valid_weights = F.normalize(valid_weights, p=1.0, dim=-2)
    positions_cam = torch.sum(valid_weights * positions_cam, dim=-2)
    uv = (K @ positions_cam.reshape(-1, 3, 1)).view(-1, 3)
    ret = uv[..., :2] / uv[..., 2:]

    return ret
