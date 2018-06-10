## Introduction

The code is originally from Shiyu's GitHub repository [(gathierry/FashionAI-KeyPointsDetectionOfApparel/)](https://github.com/gathierry/FashionAI-KeyPointsDetectionOfApparel/), which provided his solution that achieved [LB 3.82% in Tianchi FashionAI Global Challenge, 17th place out 2322 teams](https://tianchi.aliyun.com/competition/rankingList.htm?spm=5176.11165320.0.0.29762af1InLUXu&raceId=231648&season=1&_lang=en_US). For educational purpose, I've refactored most of his code and added plenty of documentations to help me understand what's happening behind the scene.

I could have made some mistakes along the way. For the most accurate implementation, please check the original repository.

## Model overview

The code uses [Cascaded Pyramid Network (CPN)](https://arxiv.org/abs/1711.07319), which wins the 2017 COCO Keypoints Challenge. In Shiyu's implementation, he made a variety of modification and found two models perform the best: 1) CPN with pretrained ResNet-152 backbone and 2) CPN with pretrained [SENet-154](https://arxiv.org/abs/1709.01507).

Here are some additional literatures that I've found useful to understand the models:

* [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
* [Training Region-based Object Detectors with Online Hard Example Mining](https://arxiv.org/abs/1604.03540)

## Requirements

Here are the key libraries that I've used to run the models:

* Python 3.6.5
* CUDA Toolkit 9.0+
* `torch` 0.4.0
* `cv2`
* `pandas`
* `numpy`
* `fire`
* `nvidia-ml-py3`
* `py3nvml`
* `visdom`

The easiest way to run the code is to use Docker image provided by [FloydHub](https://hub.docker.com/r/floydhub/pytorch/tags/): `docker pull floydhub/pytorch:0.4.0-gpu.cuda9cudnn7-py3.30`. You'll need [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) if you're running Docker on GPUs.

## Data Preparation

I follow the same directory structure as [gathierry/FashionAI-KeyPointsDetectionOfApparel/](https://github.com/gathierry/FashionAI-KeyPointsDetectionOfApparel/#data-preparation). You can download FashionAI dataset from [here](https://tianchi.aliyun.com/competition/information.htm?spm=5176.11165261.5678.2.34b72ec5iFguTn&raceId=231648&_lang=en_US) (Login required).

Make sure to change the `proj_path` and `db_path` in `utils/config.py`.

## Training / Evaluation / Prediction

Below are sample commands for running the models. Feel free to change/add keyword arguments by looking at `utils/config.py`.

To train:

```bash
python3 trainval.py main --category=skirt --model=cpn-senet --lr=1e-5
python3 trainval.py main --category=outwear --model=cpn-resnet
python3 trainval.py main --category=blouse --model=ensemble --batch_size=8
```

To evaluate:

```bash
python3 evaluate/predict_one.py main --category=outwear --model=cpn-resnet
python3 evaluate/predict_ensemble.py main --category=dress --model=ensemble
```

To predict (for submission):

```bash
python3 submission/predict_one.py main --category=outwear --model=cpn-senet
python3 submission/predict_ensemble.py main --category=dress --model=ensemble
```
