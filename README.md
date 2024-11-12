# OccFormer: Dual-path Transformer for Vision-based 3D Semantic Occupancy Prediction

## News

Code will be released soon.

## Introduction
3D occupancy prediction represents a crucial task within the domain of autonomous driving, with recent approaches demonstrating substantial advancements in vision-only 3D occupancy prediction. Nevertheless, two primary challenges persist in current models for vision-only monocular occupancy prediction:accurate deep modeling of the input RGB image and efficient encoding of occupancy features with reduced computational costs. To address these issues, this paper proposes a novel vision-only 3D occupancy prediction framework, called DPOcc,which primarily consists of a dual-path adaptive depth fusion module and a pillar-based occupancy encoder-decoder architecture. Firstly, in the dual-path adaptive depth fusion module, we use a nearest-neighbor matching algorithm to find distance-prioritized relevant voxels between explicit and implicit encoded 3D features to achieve accurate 2D to 3D view transformation. Then we propose for the first time to perform a pillar-based occupancy representation for 3D voxel features, which reduces the computational overhead while ensuring effective feature encoding. Finally, by including a dense feature extractor, spatial attention and channel attention encoder-decoder architecture for 3D semantic occupancy prediction. We conducted an extensive ablation study to validate the effectiveness and efficiency of the proposed method. The experimental results show that DPOcc significantly outperforms existing methods in semantic scene complementation on the SemanticKITTI dataset and LiDAR semantic segmentation on the nuScenes dataset, achieving an effective balance between performance and efficiency.

## Installation

**1. Create a conda virtual environment and activate**

python 3.8 may not be supported.
```shell
conda create -n sparseocc python=3.7 -y
conda activate sparseocc
```

**2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```
We select this pytorch version because mmdet3d 0.17.1 do not supports pytorch >= 1.11 and our cuda version is 11.3 .

**3. Install mmcv, mmdet, and mmseg**
```shell
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**4. Install mmdet3d 0.17.1**

Compared with the offical version, the mmdetection3d folder in this repo further includes operations like bev-pooling. 

```shell
cd mmdetection3d
pip install -r requirements/runtime.txt
pip install -v -e .
cd ..
```

**5. Build dependencies**
```shell
cd SparseOcc
export PYTHONPATH=“.”
python setup.py develop
```

**d. Install other dependencies, like timm, einops, torchmetrics, etc.**

Please change the spconv version according to your cuda version.
```shell
pip install -r docs/requirements.txt
```

## Related Projects

[TPVFormer](https://github.com/wzzheng/TPVFormer): Tri-perspective view (TPV) representation for 3D semantic occupancy.

[OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy): A large scale benchmark extending nuScenes for surrounding semantic occupancy perception.

[OccFormer](https://github.com/zhangyp15/OccFormer): Dual-path Transformer for Vision-based 3D Semantic Occupancy Prediction.

## Acknowledgement

This project is developed based on the following open-sourced projects: [MonoScene](https://github.com/astra-vision/MonoScene), [BEVDet](https://github.com/HuangJunJie2017/BEVDet), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [Mask2Former](https://github.com/facebookresearch/Mask2Former),[OccFormer](https://github.com/zhangyp15/OccFormer). Thanks for their excellent work.


