# SUDS: Scalable Urban Dynamic Scenes

[Haithem Turki](https://haithemturki.com), [Jason Y. Zhang](https://jasonyzhang.com/), [Francesco Ferroni](https://www.francescoferroni.com/), [Deva Ramanan](http://www.cs.cmu.edu/~deva)

[Project Page](https://haithemturki.com/suds) / [Paper](https://haithemturki.com/suds/paper.pdf)


This repository contains the code needed to train [SUDS](https://haithemturki.com/suds/) models.

## Citation

```
@misc{turki2023suds,
   title={SUDS: Scalable Urban Dynamic Scenes},
   author={Haithem Turki and Jason Y. Zhang and Francesco Ferroni and Deva Ramanan},
   year={2023},
   eprint={2303.14536},
   archivePrefix={arXiv},
   primaryClass={cs.CV}
}
```

## Setup

```
conda env create -f environment.yml
conda activate suds
python setup.py install
```

The codebase has been mainly tested against CUDA >= 11.3 and A100/A6000 GPUs. GPUs with compute capability greater or equal to 7.5 should generally work, although you may need to adjust batch sizes to fit within GPU memory constraints.

## Data Preparation

### KITTI

1. Download the following from the [KITTI MOT dataset](http://www.cvlibs.net/datasets/kitti/eval_tracking.php):
   1. [Left color images](http://www.cvlibs.net/download.php?file=data_tracking_image_2.zip)
   2. [Right color images](http://www.cvlibs.net/download.php?file=data_tracking_image_3.zip)
   3. [GPS/IMU data](http://www.cvlibs.net/download.php?file=data_tracking_oxts.zip)
   4. [Camera calibration files](http://www.cvlibs.net/download.php?file=data_tracking_calib.zip)
   5. [Velodyne point clouds](http://www.cvlibs.net/download.php?file=data_tracking_velodyne.zip)
   6. (Optional) [Semantic labels](https://storage.googleapis.com/gresearch/tf-deeplab/data/kitti-step.tar.gz)

2. Extract everything to ```./data/kitti``` and keep the data structure
3. Generate depth maps from the Velodyne point clouds: ```python scripts/create_kitti_depth_maps.py --kitti_sequence $SEQUENCE```
4. (Optional) Generate sky and static masks from semantic labels: ```python scripts/create_kitti_masks.py --kitti_sequence $SEQUENCE```
5. Create metadata file: ```python scripts/create_kitti_metadata.py --config_file scripts/configs/$CONFIG_FILE```
6. Extract DINO features:
   1. ```python scripts/extract_dino_features.py --metadata_path $METADATA_PATH``` or ```python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node $NUM_GPUS scripts/extract_dino_features.py --metadata_path $METADATA_PATH``` for multi-GPU extraction
   2. ```python scripts/run_pca.py --metadata_path $METADATA_PATH```
7. Extract DINO correspondences: ```python scripts/extract_dino_correspondences.py --metadata_path $METADATA_PATH``` or ```python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node $NUM_GPUS scripts/extract_dino_correspondences.py --metadata_path $METADATA_PATH``` for multi-GPU extraction
8. (Optional) Generate feature clusters for visualization: ```python scripts/create_kitti_feature_clusters.py --metadata_path $METADATA_PATH --output_path $OUTPUT_PATH```

### VKITTI2

1. Download the following from the [VKITTI2 dataset](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/):
   1. [RGB images](http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar)
   2. [Depth images](http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_depth.tar)
   3. [Camera intrinsics/extrinsics](http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_textgt.tar.gz)
   4. (Optional) [Ground truth forward flow](http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_forwardFlow.tar)
   5. (Optional) [Ground truth backward flow](http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_backwardFlow.tar)
   6. (Optional) [Semantic labels](http://download.europe.naverlabs.com//virtual_kitti_2.0.3/vkitti_2.0.3_classSegmentation.tar)

2. Extract everything to ```./data/vkitti2``` and keep the data structure
3. (Optional) Generate sky and static masks from semantic labels: ```python scripts/create_vkitti2_masks.py --vkitti2_path $SCENE_PATH```
4. Create metadata file: ```python scripts/create_vkitti2_metadata.py --config_file scripts/configs/$CONFIG_FILE```
5. Extract DINO features:
   1. ```python scripts/extract_dino_features.py --metadata_path $METADATA_PATH``` or ```python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node $NUM_GPUS scripts/extract_dino_features.py --metadata_path $METADATA_PATH``` for multi-GPU extraction
   2. ```python scripts/run_pca.py --metadata_path $METADATA_PATH```
6. If not using the ground truth flow provided by VKITTI2, extract DINO correspondences: ```python scripts/extract_dino_correspondences.py --metadata_path $METADATA_PATH``` or ```python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node $NUM_GPUS scripts/extract_dino_correspondences.py --metadata_path $METADATA_PATH``` for multi-GPU extraction
7. (Optional) Generate feature clusters for visualization: ```python scripts/create_vkitti2_feature_clusters.py --metadata_path $METADATA_PATH --vkitti2_path $SCENE_PATH --output_path $OUTPUT_PATH```

## Training

```python suds/train.py suds --experiment-name $EXPERIMENT_NAME --pipeline.datamanager.dataparser.metadata_path $METADATA_PATH [--pipeline.feature_clusters $FEATURE_CLUSTERS]```

## Evaluation

```python suds/eval.py --load_config $SAVED_MODEL_PATH``` or ```python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node $NUM_GPUS suds/eval.py --load_config $SAVED_MODEL_PATH``` for multi-GPU evaluation

## Acknowledgements

This project is built on [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn). The DINO feature extraction scripts are based on [ShirAmir's implementation](https://github.com/ShirAmir/dino-vit-features) and parts of the KITTI processing code from [Neural Scene Graphs](https://github.com/princeton-computational-imaging/neural-scene-graphs).