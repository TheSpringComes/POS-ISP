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

Task-aware ISP optimization aims to adapt image signal processing (ISP) pipelines for downstream vision tasks. Prior approaches typically optimize ISP modules in a stage-wise manner, which often leads to unstable training and high computational overhead.

POS-ISP reformulates modular ISP optimization as a **sequence-level prediction problem**. The model predicts the entire ISP pipeline and its parameters in a single forward pass and optimizes it using a terminal task reward. This design improves optimization stability while keeping the predictor lightweight and efficient.

## Project Page

More details, visualizations, and results are available on the project page:

https://w1jyun.github.io/POS-ISP/

## Code Release

The training and inference code will be released soon.

## Citation

```bibtex
@inproceedings{won2026posisp,
  title={POS-ISP: Pipeline Optimization at the Sequence Level for Task-aware ISP},
  author={Won, Jiyun and Yang, Heemin and Kim, Woohyeok and Ok, Jungseul and Cho, Sunghyun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  year={2026}
}
```
