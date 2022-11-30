import faulthandler
import signal

import nerfstudio.configs.method_configs
import nerfstudio.data.datamanagers.base_datamanager
import tyro
from nerfstudio.configs.config_utils import convert_markup_to_ansi
from nerfstudio.configs.method_configs import descriptions, method_configs
from nerfstudio.data.dataparsers.arkitscenes_dataparser import ARKitScenesDataParserConfig
from nerfstudio.data.dataparsers.blender_dataparser import BlenderDataParserConfig
from nerfstudio.data.dataparsers.dnerf_dataparser import DNeRFDataParserConfig
from nerfstudio.data.dataparsers.dycheck_dataparser import DycheckDataParserConfig
from nerfstudio.data.dataparsers.instant_ngp_dataparser import InstantNGPDataParserConfig
from nerfstudio.data.dataparsers.minimal_dataparser import MinimalDataParserConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.data.dataparsers.nuscenes_dataparser import NuScenesDataParserConfig
from nerfstudio.data.dataparsers.phototourism_dataparser import PhototourismDataParserConfig
from nerfstudio.data.dataparsers.scannet_dataparser import ScanNetDataParserConfig
from nerfstudio.data.dataparsers.sdfstudio_dataparser import SDFStudioDataParserConfig
from nerfstudio.data.dataparsers.sitcoms3d_dataparser import Sitcoms3DDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from scripts.train import main

from suds.data.suds_datamanager import SUDSDataManagerConfig
from suds.data.suds_dataparser import SUDSDataParserConfig
from suds.data.suds_pipeline import SUDSPipelineConfig
from suds.suds_model import SUDSModelConfig


def suds_entrypoint():
    faulthandler.register(signal.SIGUSR1)

    descriptions['suds'] = 'Scalable Urban Dynamic Scenes'

    method_configs['suds'] = TrainerConfig(
        method_name='suds',
        steps_per_eval_batch=500,
        steps_per_save=10000,
        max_num_iterations=250001,
        mixed_precision=True,
        steps_per_eval_all_images=250000,
        steps_per_eval_image=1000,
        log_gradients=True,
        pipeline=SUDSPipelineConfig(
            datamanager=SUDSDataManagerConfig(
                dataparser=SUDSDataParserConfig(),
            ),
            model=SUDSModelConfig()
        ),
        optimizers={
            'fields': {
                'optimizer': AdamOptimizerConfig(lr=5e-3, eps=1e-8),
                'scheduler': ExponentialDecaySchedulerConfig(lr_final=5e-4, max_steps=250000),
            },
            'mlps': {
                'optimizer': AdamOptimizerConfig(lr=5e-3, eps=1e-8, weight_decay=1e-8),
                'scheduler': ExponentialDecaySchedulerConfig(lr_final=5e-4, max_steps=250000),
            },
            'flow': {
                'optimizer': AdamOptimizerConfig(lr=1e-4, eps=1e-8),
                'scheduler': ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=250000),
            },
            'flow_mlp': {
                'optimizer': AdamOptimizerConfig(lr=1e-4, eps=1e-8, weight_decay=1e-8),
                'scheduler': ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=250000),
            }
        }
    )

    AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[
        # Don't show unparseable (fixed) arguments in helptext.
        tyro.conf.FlagConversionOff[
            tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
        ]
    ]

    nerfstudio.data.datamanagers.base_datamanager.AnnotatedDataParserUnion = tyro.conf.OmitSubcommandPrefixes[
        # Omit prefixes of flags in subcommands.
        tyro.extras.subcommand_type_from_defaults(
            {
                "nerfstudio-data": NerfstudioDataParserConfig(),
                "minimal-parser": MinimalDataParserConfig(),
                "arkit-data": ARKitScenesDataParserConfig(),
                "blender-data": BlenderDataParserConfig(),
                "instant-ngp-data": InstantNGPDataParserConfig(),
                "nuscenes-data": NuScenesDataParserConfig(),
                "dnerf-data": DNeRFDataParserConfig(),
                "phototourism-data": PhototourismDataParserConfig(),
                "dycheck-data": DycheckDataParserConfig(),
                "scannet-data": ScanNetDataParserConfig(),
                "sdfstudio-data": SDFStudioDataParserConfig(),
                "sitcoms3d-data": Sitcoms3DDataParserConfig(),
                "suds-data": SUDSDataParserConfig(),
            },
            prefix_names=False,  # Omit prefixes in subcommands themselves.
        )
    ]

    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    suds_entrypoint()
