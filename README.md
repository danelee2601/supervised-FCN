# Description
Fully Convolutional Network (FCN) was proposed in a paper [1]. FCN's architecture is shown in the figure below. \
![Architecture of FCN](.imgs/fcn.png)

This repository offers a supervised-training code of FCN on all the subset datasets of the UCR archive. 
The trained FCN will be used to compute the FID (Fréchet Inception Distance) score for evaluation of generated time series.

# Training History
The training histories of FCN on all the UCR subset datasets are available [here](https://wandb.ai/daesoolee/supervised-FCN?workspace=user-daesoolee).

A training and test dataset split is the same as provided in the UCR archive, and a test dataset is used as a validation set during training for better training progress tracking.

# Reference
[1] Wang, Zhiguang, Weizhong Yan, and Tim Oates. "Time series classification from scratch with deep neural networks: A strong baseline." 2017 International joint conference on neural networks (IJCNN). IEEE, 2017. \
[2]