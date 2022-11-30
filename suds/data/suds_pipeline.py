from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Type, List, Optional

import torch
from PIL import Image
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.utils import profiler
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from typing_extensions import Literal

from suds.data.suds_datamanager import SUDSDataManagerConfig
from suds.draw_utils import label_colormap
from suds.stream_utils import buffer_from_stream
from suds.suds_constants import DEPTH, FEATURES, SKY
from suds.suds_model import SUDSModelConfig


@dataclass
class SUDSPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: SUDSPipeline)
    """target class to instantiate"""
    datamanager: SUDSDataManagerConfig = SUDSDataManagerConfig()
    """specifies the datamanager config"""
    model: SUDSModelConfig = SUDSModelConfig()
    """specifies the model config"""

    feature_clusters: List[str] = field(default_factory=lambda: [])
    """clusters to use for feature visualization"""


class SUDSPipeline(VanillaPipeline):
    config: SUDSPipelineConfig

    def __init__(
            self,
            config: SUDSPipelineConfig,
            device: str,
            test_mode: Literal['test', 'val', 'inference'] = 'val',
            world_size: int = 1,
            local_rank: int = 0):

        feature_clusters = {}
        feature_colors = {}
        for feature_cluster_path in config.feature_clusters:
            feature_cluster = torch.load(buffer_from_stream(feature_cluster_path), map_location='cpu')
            feature_clusters[Path(feature_cluster_path).stem] = feature_cluster['centroids']
            cluster_colors = feature_cluster['colors'] / 255. if 'colors' in feature_cluster \
                else label_colormap(feature_cluster['centroids'].shape[0])
            feature_colors[Path(feature_cluster_path).stem] = cluster_colors

        config.model.feature_clusters = feature_clusters
        config.model.feature_colors = feature_colors

        config.datamanager.feature_clusters = feature_clusters
        config.datamanager.feature_colors = feature_colors
        config.datamanager.load_depth = config.model.loss_coefficients[DEPTH] > 0
        config.datamanager.load_features = config.model.loss_coefficients[FEATURES] > 0
        config.datamanager.load_flow = config.model.predict_flow
        config.datamanager.load_sky = config.model.loss_coefficients[SKY] > 0

        super().__init__(config, device, test_mode, world_size, local_rank)

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None, image_save_dir: Optional[Path] = None):
        """Iterate over all the images in the eval dataset and get the average.

        Returns:
            metrics_dict: dictionary of metrics
        """
        self.eval()
        metrics_dict_list = []
        num_images = len(self.datamanager.fixed_indices_eval_dataloader)
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all eval images...", total=num_images)
            for camera_ray_bundle, batch in self.datamanager.fixed_indices_eval_dataloader:
                # time this the following line
                inner_start = time()
                height, width = camera_ray_bundle.shape
                num_rays = height * width
                outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                metrics_dict, images = self.model.get_image_metrics_and_images(outputs, batch)

                if image_save_dir is not None:
                    for key, val in images.items():
                        Image.fromarray((val * 255).byte().cpu().numpy()).save(
                            image_save_dir / '{0:06d}-{1}.jpg'.format(int(camera_ray_bundle.camera_indices[0, 0, 0]),
                                                                      key))

                assert "num_rays_per_sec" not in metrics_dict
                metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                fps_str = "fps"
                assert fps_str not in metrics_dict
                metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)
        # average the metrics list
        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            metrics_dict[key] = float(
                torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
            )
        self.train()
        return metrics_dict
