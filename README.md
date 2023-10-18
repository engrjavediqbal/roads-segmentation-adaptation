
# Leveraging Topology for Domain Adaptive Road Segmentation in Satellite and Aerial Imagery

By Javed Iqbal, Aliza Masood, Waqas Sultani and Mohsen Ali

### Contents
0. [Introduction](#introduction)
0. [Requirements](#requirements)
0. [Setup](#models)
0. [Usage](#usage)
0. [Results](#results)
0. [Note](#note)
0. [Citation](#citation)

### Introduction
This repository contains the source code for Leveraging Topology for Road Segmentation and Adaptation in Satellite and Aerial Imagery based on the work presented in our paper "[Leveraging Topology for Domain Adaptive Road Segmentation in Satellite and Aerial Imagery]". 
(https://arxiv.org/pdf/2309.15625.pdf).

### Requirements:
The code is tested in Ubuntu 16.04. with the following libraries.

- Python: 3.6
- numpy: 1.18.5
- opencv-python: 4.4.0.42
- torch: 1.5.0
- torchvision: 0.6.0

For GPU usage, the maximum GPU memory consumption is about 10.8 GB in a single GTX 1080ti.


### Setup
We assume you are working in roadAdapt.

0. Datasets:
- Download [SpaceNet-V3](link) dataset. 
- Download [DeepGlobe](link).
- Put downloaded data in the "data" folder.
1. Source pre-trained models:
- Download [source model](link) trained on SpaceNet dataset.
- Put source-trained model in the "models/" folder

### Usage

cd roadAdapt


1. Evaluate source only or final model:

~~~~
python evaluate.py
~~~~


2. Adapt model SpaceNet to DeepGlobe

~~~~
python road_adapt.py
~~~~



### Note
- This code is partially based on [DLinkNet](https://github.com/ShenweiXie/D-LinkNet).
- Due to the randomness, the self-training-based domain adaptation results may slightly vary in each run.


### Results
A leaderboard for state-of-the-art methods is available

### Citation:
If you found this useful, please cite our [paper](https://arxiv.org/pdf/2309.15625.pdf). 

@inproceedings{iqbal2020roadDa,  
&nbsp; &nbsp; &nbsp;    title={Leveraging Topology for Domain Adaptive Road Segmentation in Satellite and Aerial Imagery},  
&nbsp; &nbsp; &nbsp;     author={Javed Iqbal, Aliza Masood, Waqas Sultani and Mohsen Ali},  
&nbsp; &nbsp; &nbsp;     booktitle={---}, 
&nbsp; &nbsp; &nbsp;     pages={--}, 
&nbsp; &nbsp; &nbsp;     year={2023} 
}


Contact: javed.iqbal@itu.edu.pk
