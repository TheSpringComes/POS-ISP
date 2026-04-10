# [CVPR 2026 Findings] POS-ISP: Pipeline Optimization at the Sequence Level for Task-aware ISP

<p align="center">
<a href="https://w1jyun.github.io/">Jiyun Won</a><sup>1</sup> &nbsp;&nbsp;
<a href="https://hmyang0727.github.io/">Heemin Yang</a><sup>1</sup> &nbsp;&nbsp;
<a href="https://woo525.github.io/">Woohyeok Kim</a><sup>2</sup> &nbsp;&nbsp;
<a href="https://sites.google.com/view/jungseulok">Jungseul Ok</a><sup>1,2</sup> &nbsp;&nbsp;
<a href="http://www.scho.pe.kr">Sunghyun Cho</a><sup>1,2</sup>
</p>

<p align="center">
POSTECH CSE<sup>1</sup> & GSAI<sup>2</sup>
</p>

![teaser](overview.jpg)

Official repository for **POS-ISP**, a sequence-level reinforcement learning framework for task-aware ISP optimization.

## Overview

Task-aware ISP optimization aims to adapt image signal processing pipelines for downstream vision tasks. Prior approaches typically optimize ISP modules in a stage-wise manner, which often leads to unstable training and high computational overhead.

POS-ISP reformulates modular ISP optimization as a sequence-level prediction problem. The model predicts the entire ISP pipeline and its parameters in a single forward pass and optimizes it using a terminal task reward. This design improves optimization stability while keeping the predictor lightweight and efficient.

This repository contains the current training and validation code used in this project.

## Project Page

More details, visualizations, and results are available on the project page:

[https://w1jyun.github.io/POS-ISP/](https://w1jyun.github.io/POS-ISP/)

## Tasks

- Detection: `lod`
- Segmentation: `lis`
- Depth estimation: `kitti`

## Repository Structure

- [agent.py](C:\Users\yun\POS-ISP\agent.py): ISP policy network
- [config.py](C:\Users\yun\POS-ISP\config.py): filter configuration and dataset paths
- [train_detection.py](C:\Users\yun\POS-ISP\train_detection.py): detection training
- [train_segmentation.py](C:\Users\yun\POS-ISP\train_segmentation.py): segmentation training
- [train_depth.py](C:\Users\yun\POS-ISP\train_depth.py): depth training
- [validate_detection.py](C:\Users\yun\POS-ISP\validate_detection.py): detection validation
- [validate_segmentation.py](C:\Users\yun\POS-ISP\validate_segmentation.py): segmentation validation
- [validate_depth.py](C:\Users\yun\POS-ISP\validate_depth.py): depth validation
- [depth](C:\Users\yun\POS-ISP\depth): depth estimator code
- [isp](C:\Users\yun\POS-ISP\isp): ISP filter implementations

## Dataset Setup

### Detection

Detection uses:

```text
yolov3/data/lod.yaml
```

The LOD dataset can be downloaded through the AdaptiveISP repository:

[https://github.com/OpenImagingLab/AdaptiveISP](https://github.com/OpenImagingLab/AdaptiveISP)

Direct download link:

[https://onedrive.live.com/?id=FD75B81D284B4FAD%21115&resid=FD75B81D284B4FAD%21115&e=KURDwo&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBcTFQU3lnZHVIWDljekhCOVdrVU5VVFV4OG8%5FZT1LVVJEd28&cid=fd75b81d284b4fad&v=validatepermission](https://onedrive.live.com/?id=FD75B81D284B4FAD%21115&resid=FD75B81D284B4FAD%21115&e=KURDwo&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBcTFQU3lnZHVIWDljekhCOVdrVU5VVFV4OG8%5FZT1LVVJEd28&cid=fd75b81d284b4fad&v=validatepermission)

### Segmentation

Segmentation uses:

```text
yolov3/data/lis_raw_all.yaml
```

The LIS dataset can be downloaded from:

[https://github.com/Linwei-Chen/LIS](https://github.com/Linwei-Chen/LIS)

### Depth

Depth dataset paths are configured in [config.py](C:\Users\yun\POS-ISP\config.py):

```python
cfg.depth_train_dir = 'Dataset/kitti/KITTI_depth/KITTI_sc'
cfg.depth_test_dir = 'Dataset/kitti/KITTI_depth/kitti_depth_test'
```

The `KITTI_depth` dataset is the same one used in DRL-ISP. Download it from:

[https://kaistackr-my.sharepoint.com/personal/shinwc159_kaist_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fshinwc159%5Fkaist%5Fac%5Fkr%2FDocuments%2FDRL%5FISP%2FDRL%5FISP%5FRAW&ga=1](https://kaistackr-my.sharepoint.com/personal/shinwc159_kaist_ac_kr/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fshinwc159%5Fkaist%5Fac%5Fkr%2FDocuments%2FDRL%5FISP%2FDRL%5FISP%5FRAW&ga=1)

## Depth Estimator

Depth training and validation use `DispResNet(18, False)` from [depth](C:\Users\yun\POS-ISP\depth), which is based on:

[SC-SfMLearner-Release](https://github.com/JiawangBian/SC-SfMLearner-Release)

Download the pretrained depth weight from that repository and place it at:

```text
depth/weight/dispnet_model_best.pth.tar
```

## Training

Detection:

```bash
python train_detection.py \
    --data_name=lod \
    --data_cfg=yolov3/data/lod.yaml \
    --save_dir_name=experiments \
    --save_path=posisp_det
```

Segmentation:

```bash
python train_segmentation.py \
    --data_name=lis \
    --data_cfg=yolov3/data/lis_raw_all.yaml \
    --save_dir_name=experiments \
    --save_path=posisp_seg
```

Depth:

```bash
python train_depth.py \
    --data_name=kitti \
    --depth_weights=depth/weight/dispnet_model_best.pth.tar \
    --save_dir_name=experiments \
    --save_path=posisp_depth
```

## Validation

Detection:

```bash
python validate_detection.py \
    --project=results \
    --name=lod_eval \
    --isp_weights=experiments/lod-posisp/ckpt/PosISP_iter_15000.pth \
    --batch-size=1 \
    --save_image \
    --save_param
```

Segmentation:

```bash
python validate_segmentation.py \
    --project=results \
    --isp_weights=/home1/w1jyun/CVPR26/ModISP/experiments_seed/lis-raw_all_yolov13_seg_no_cond_seed_3/ckpt/DynamicISP_iter_15000.pth \
    --data_name=lis \
    --data=/home1/w1jyun/CVPR26/ModISP/yolov3/data/lis_raw_all.yaml \
    --batch-size=1 \
    --steps=5 \
    --name=lis_raw_all_re_seed3 \
    --save_image \
    --save_param
```

Depth:

```bash
python validate_depth.py \
    --project=results \
    --isp_weights=/home1/w1jyun/CVPR26/ModISP/experiments_depth/lod-kitti_seed2/ckpt/DynamicISP_iter_15000.pth \
    --batch-size=1 \
    --name=kitti_seed1 \
    --save_image \
    --save_param
```

## Notes

- `validate_depth.py` loads the depth backbone from `depth/weight/dispnet_model_best.pth.tar` by default.
- Validation outputs are saved under the directory specified by `--project` and `--name`.
- ISP checkpoints are loaded from the `agent_model` key in the saved checkpoint.

## Citation

```bibtex
@inproceedings{won2026posisp,
  title={POS-ISP: Pipeline Optimization at the Sequence Level for Task-aware ISP},
  author={Won, Jiyun and Yang, Heemin and Kim, Woohyeok and Ok, Jungseul and Cho, Sunghyun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  year={2026}
}
```
