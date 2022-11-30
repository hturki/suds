#!/usr/bin/env python
"""
eval.py
"""
from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import tyro
from nerfstudio.utils.comms import get_world_size, get_rank, is_main_process
from nerfstudio.utils.eval_utils import eval_setup
from rich.console import Console

CONSOLE = Console(width=120)


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path = eval_setup(self.load_config)

        self.output_path.mkdir(parents=True, exist_ok=True)
        metrics_dict = pipeline.get_average_eval_image_metrics(image_save_dir=self.output_path)

        output_json_path = self.output_path / 'metrics.json'

        if is_main_process():
            if get_world_size() > 1:
                dist.barrier()
            num_images = len(pipeline.datamanager.fixed_indices_eval_dataloader)
            for key in metrics_dict:
                metrics_dict[key] = metrics_dict[key] * num_images

            for i in range(1, get_world_size()):
                shard_path = Path(str(output_json_path) + f'.{i}')
                with shard_path.open() as f:
                    shard_results = json.load(f)

                for key in metrics_dict:
                    metrics_dict[key] += (shard_results['results'][key] * shard_results['num_images'])

                num_images += shard_results['num_images']

                shard_path.unlink()

            for key in metrics_dict:
                metrics_dict[key] = metrics_dict[key] / num_images

            benchmark_info = {
                "experiment_name": config.experiment_name,
                "method_name": config.method_name,
                "checkpoint": str(checkpoint_path),
                "results": metrics_dict,
                "num_images": num_images
            }

            output_json_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")

            CONSOLE.print(f"Saved results to: {self.output_path}")
        else:
            shard_output_path = Path(str(output_json_path) + f'.{get_rank()}.tmp')
            # Get the output and define the names to save to
            benchmark_info = {
                "results": metrics_dict,
                "num_images": len(pipeline.datamanager.fixed_indices_eval_dataloader)
            }

            # Save output to output file

            shard_output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
            shard_output_path = shard_output_path.rename(Path(str(output_json_path) + f'.{get_rank()}'))
            CONSOLE.print(f"Saved shard results to: {shard_output_path}")
            if get_world_size() > 1:
                dist.barrier()


def entrypoint():
    if 'RANK' in os.environ:
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(0, hours=24))
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
