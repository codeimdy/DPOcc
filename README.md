# OccFormer: Dual-path Transformer for Vision-based 3D Semantic Occupancy Prediction

## News

Code will be released soon.

## Introduction
3D occupancy prediction represents a crucial task within the domain of autonomous driving, with recent approaches demonstrating substantial advancements in vision-only 3D occupancy prediction. Nevertheless, two primary challenges persist in current models for vision-only monocular occupancy prediction:accurate deep modeling of the input RGB image and efficient encoding of occupancy features with reduced computational costs. To address these issues, this paper proposes a novel vision-only 3D occupancy prediction framework, called DPOcc,which primarily consists of a dual-path adaptive depth fusion module and a pillar-based occupancy encoder-decoder architecture. Firstly, in the dual-path adaptive depth fusion module, we use a nearest-neighbor matching algorithm to find distance-prioritized relevant voxels between explicit and implicit encoded 3D features to achieve accurate 2D to 3D view transformation. Then we propose for the first time to perform a pillar-based occupancy representation for 3D voxel features, which reduces the computational overhead while ensuring effective feature encoding. Finally, by including a dense feature extractor, spatial attention and channel attention encoder-decoder architecture for 3D semantic occupancy prediction. We conducted an extensive ablation study to validate the effectiveness and efficiency of the proposed method. The experimental results show that DPOcc significantly outperforms existing methods in semantic scene complementation on the SemanticKITTI dataset and LiDAR semantic segmentation on the nuScenes dataset, achieving an effective balance between performance and efficiency.


## Related Projects

[TPVFormer](https://github.com/wzzheng/TPVFormer): Tri-perspective view (TPV) representation for 3D semantic occupancy.

[OpenOccupancy](https://github.com/JeffWang987/OpenOccupancy): A large scale benchmark extending nuScenes for surrounding semantic occupancy perception.

[OccFormer](https://github.com/zhangyp15/OccFormer): Dual-path Transformer for Vision-based 3D Semantic Occupancy Prediction.

## Acknowledgement

This project is developed based on the following open-sourced projects: [MonoScene](https://github.com/astra-vision/MonoScene), [BEVDet](https://github.com/HuangJunJie2017/BEVDet), [BEVFormer](https://github.com/fundamentalvision/BEVFormer), [Mask2Former](https://github.com/facebookresearch/Mask2Former),[OccFormer](https://github.com/zhangyp15/OccFormer). Thanks for their excellent work.


