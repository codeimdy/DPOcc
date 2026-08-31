# DPOcc: Dual-Path Depth Modeling and Pillar-Based Encoding for Vision-Based 3D Semantic Occupancy Prediction

## Introduction

Vision-based 3D semantic occupancy prediction is essential for autonomous
driving because it provides dense geometric and semantic understanding of
the surrounding environment. However, dense 3D occupancy modeling usually
involves high-resolution voxel features, 2D-to-3D view transformation, and expen￾sive 3D feature encoding, which brings considerable computational burden for
GPU-accelerated perception systems. To address these challenges, this paper pro￾poses DPOcc, a vision-based 3D semantic occupancy prediction framework that
combines dual-path adaptive depth fusion with a pillar-based occupancy encoder.
The dual-path adaptive depth fusion module integrates implicit context-based
depth modeling and explicit depth-distribution-based modeling, and uses nearest￾neighbor matching between non-empty 3D voxels to improve 2D-to-3D feature
transformation. To reduce the cost of dense 3D occupancy encoding, we introduce
a pillar-based occupancy representation that aggregates vertical voxel informa￾tion and shifts part of the feature encoding to more efficient 2D operations while
retaining geometric expressiveness through sparse vertical aggregation and fea￾ture enhancement. Experiments on SemanticKITTI and nuScenes demonstrate
that DPOcc achieves competitive occupancy prediction and LiDAR segmentation
performance. In addition, single-GPU runtime evaluation on the SemanticKITTI
validation set shows that DPOcc reaches 244.45 ms per frame, corresponding to 4.09 FPS on an NVIDIA RTX 3090, which is faster than representative single frame methods such as MonoScene, TPVFormer, OccFormer, and PanoSSC under
the same evaluation setting. These results indicate that DPOcc provides a favor able accuracy–efficiency trade-off for GPU-accelerated vision-based occupancy
inference. 


## Installation

Python 3.8 and newer may not be compatible with this legacy OpenMMLab stack.
The complete pinned environment is also available in `docs/DPOcc.yaml`.

### 1. Create and activate the Conda environment

```bash
conda create -n DPOcc python=3.7 -y
conda activate DPOcc
```

Alternatively, create the fully pinned environment directly:

```bash
conda env create -f docs/DPOcc.yaml
conda activate DPOcc
```

### 2. Install PyTorch and torchvision

Follow the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/),
or install the versions used by this project:

```bash
conda install pytorch==1.10.1 torchvision==0.11.2 \
  torchaudio==0.10.1 cudatoolkit=11.3 \
  -c pytorch -c conda-forge
```

These versions are selected because MMDetection3D 0.17.1 does not support
PyTorch 1.11 or newer, and the reference CUDA environment is 11.3.

### 3. Install MMCV, MMDetection and MMSegmentation

```bash
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

### 4. Install MMDetection3D 0.17.1

Compared with the official release, the `mmdetection3d` directory in this
repository includes additional operations such as BEV pooling.

```bash
cd mmdetection3d
pip install -r requirements/runtime.txt
pip install -v -e .
cd ..
```

### 5. Configure the project path

Run this command from the DPOcc repository root:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

### 6. Install the remaining dependencies

Install packages such as timm, einops and torchmetrics. Select an spconv build
that matches the local CUDA version.

```bash
pip install -r docs/requirements.txt
```

The CUDA toolkit used to compile MMCV and MMDetection3D extensions must be
compatible with the CUDA version used by PyTorch.

## Data preparation

Download nuScenes v1.0, CAN bus expansion data and lidarseg labels, then use
the following layout:

```text
data/
  nuscenes/
    maps/
    samples/
    sweeps/
    v1.0-trainval/
    can_bus/
    lidarseg/
  nuscenes_infos_temporal_train.pkl
  nuscenes_infos_temporal_val.pkl
```

Generate the temporal information files from the repository root:

```bash
python tools/create_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data \
  --extra-tag nuscenes \
  --version v1.0 \
  --canbus ./data
```

Before training, change `data_root` in the selected file under
`projects/configs/DPOcc_nusc/` to the local nuScenes path. Also ensure that the
ResNet `pretrained` path points to an existing checkpoint, or set it to
`None`.

## Training

Single GPU:

```bash
python tools/train.py \
  projects/configs/DPOcc_nusc/DPOcc_nusc_r50_256x704.py
```

Distributed training (replace `8` with the GPU count):

```bash
bash tools/dist_train.sh \
  projects/configs/DPOcc_nusc/DPOcc_nusc_r50_256x704.py 8
```

## Evaluation and prediction

```bash
bash tools/dist_test.sh \
  projects/configs/DPOcc_nusc/DPOcc_nusc_r50_256x704.py \
  work_dirs/DPOcc_nusc_r50_256x704/latest.pth 8
```

Save predictions for visualization:

```bash
bash tools/dist_test.sh \
  projects/configs/DPOcc_nusc/DPOcc_nusc_r50_256x704.py \
  work_dirs/DPOcc_nusc_r50_256x704/latest.pth 8 \
  --pred-save work_dirs/predictions
```

See `docs/prepare_dataset.md`, `docs/train_and_eval.md` and
`docs/predict_and_visualize.md` for task-specific details.


## Related Projects

- [TPVFormer](https://github.com/wzzheng/TPVFormer): tri-perspective-view
  representation for 3D semantic occupancy.
- [OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy): a large-scale
  benchmark extending nuScenes for surrounding semantic occupancy perception.
- [OccFormer](https://github.com/zhangyp15/OccFormer): a dual-path Transformer
  for vision-based 3D semantic occupancy prediction.

## Acknowledgement

This project is developed with reference to the following open-source
projects: [MonoScene](https://github.com/astra-vision/MonoScene),
[BEVDet](https://github.com/HuangJunJie2017/BEVDet),
[BEVFormer](https://github.com/fundamentalvision/BEVFormer),
[Mask2Former](https://github.com/facebookresearch/Mask2Former), and
[OccFormer](https://github.com/zhangyp15/OccFormer). We thank their authors for
their excellent work.

## License

This repository is distributed under the terms in `LICENSE`.
