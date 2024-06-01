# Learning-based Inference of Longitudinal imAge Changes (LILAC)

[//]: # ([[Website]&#40;&#41;] [[Paper]&#40;&#41;])

## Overview

LILAC is a method that learns to compare images, designed to automatically filter out 
nuisance variations and focus on clinically significant changes in longitudinal imaging data.

This repository includes models and train/evaluation code for the LILAC paper. 

For more information, please read the LILAC [paper](). 
[README.md.html](..%2F..%2FDownloads%2FREADME.md.html)
For questions and feedback, please [open issues](https://github.com/heejong-kim/lilac/issues) or email the [corresponding author](https://heejongkim.com)

[//]: # (## Updates)
[//]: # (- [May 2024] The preprint for LILAC is available on [arXiv]&#40;https://arxiv.org/abs/2405.14019&#41;!)

## Installation
```bash
git clone https://github.com/heejong-kim/lilac.git
cd lilac
pip install -e .
```

### Requirements
The LILAC package depends on the following requirements:

- numpy>=1.19.1
- ogb>=1.2.6
- outdated>=0.2.0
- pandas>=1.1.0
- pillow>=7.2.0
- pytz>=2020.4
- torch>=1.7.0
- torch-scatter>=2.0.5
- torch-geometric>=2.0.1
- torchvision>=0.8.2
- tqdm>=4.53.0
- scikit-learn>=0.20.0
- scipy>=1.5.4

Running `pip install -e .` will automatically check for and install all of these requirements.


All baseline experiments in the paper were run on Python 3.8.5 and CUDA 10.1.


## Examples
### Learning to temporally order embryo images (LILAC-o)
```bash
python run.py \
    --groupwise \
    --num_keypoints 256 \
    --variant S \
    --weights_dir ./weights/ \
    --moving ./example_data/ \
    --fixed ./example_data/ \
    --moving_seg ./example_data/ \
    --fixed_seg ./example_data/ \
    --list_of_aligns rigid affine tps_1 \
    --list_of_metrics mse harddice \
    --save_eval_to_disk \
    --save_dir ./register_output/ \
    --visualize \
    --download
```
### Learning to temporally order wound-healing images (LILAC-o)
```bash
python scripts/register.py \
    --groupwise \
    --num_keypoints 256 \
    --variant S \
    --weights_dir ./weights/ \
    --moving ./example_data/ \
    --fixed ./example_data/ \
    --moving_seg ./example_data/ \
    --fixed_seg ./example_data/ \
    --list_of_aligns rigid affine tps_1 \
    --list_of_metrics mse harddice \
    --save_eval_to_disk \
    --save_dir ./register_output/ \
    --visualize \
    --download
```
### Learning to predict temporal changes (LILAC-t)
```bash
python scripts/register.py \
    --groupwise \
    --num_keypoints 256 \
    --variant S \
    --weights_dir ./weights/ \
    --moving ./example_data/ \
    --fixed ./example_data/ \
    --moving_seg ./example_data/ \
    --fixed_seg ./example_data/ \
    --list_of_aligns rigid affine tps_1 \
    --list_of_metrics mse harddice \
    --save_eval_to_disk \
    --save_dir ./register_output/ \
    --visualize \
    --download
```
### Learning to predict specific changes (LILAC-s)
```bash
python scripts/register.py \
    --groupwise \
    --num_keypoints 256 \
    --variant S \
    --weights_dir ./weights/ \
    --moving ./example_data/ \
    --fixed ./example_data/ \
    --moving_seg ./example_data/ \
    --fixed_seg ./example_data/ \
    --list_of_aligns rigid affine tps_1 \
    --list_of_metrics mse harddice \
    --save_eval_to_disk \
    --save_dir ./register_output/ \
    --visualize \
    --download
```

