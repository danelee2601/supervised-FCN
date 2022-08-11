# Description
Fully Convolutional Network (FCN) was proposed in a paper [1]. FCN's architecture is shown in the figure below. \
![Architecture of FCN](.imgs/fcn.png)

This repository offers a supervised-training code of FCN on all the subset datasets of the UCR archive. 
The trained FCN will be used to compute the FID (Fréchet Inception Distance) score for evaluation of generated time series.

# Run
- `train_fcn.py`: runs training of FCN on a subset dataset from the UCR archive. 
- `configs/config.yaml`: is where you can set parameters and hyper-parameters for dataset loading and training. 

# Prerequisite
You need to download the UCR archive dataset and put it in `datasets/`. You can download it from [here](https://studntnu-my.sharepoint.com/:u:/g/personal/daesool_ntnu_no/EUVHWAlJRrZBnCZMAOdTR-cB3m_LP7Q10Y78meuzUAuIBQ?e=h9aGhi).
Then, your `datasets` directory should have `datasets/UCRArchive_2018`.

# Training History
The training histories of FCN on all the UCR subset datasets are available [here](https://wandb.ai/daesoolee/supervised-FCN?workspace=user-daesoolee).

A training and test dataset split is the same as provided in the UCR archive, and a test dataset is used as a validation set during training for better training progress tracking.

# Optimizer
- initial learning rate: 1e-3
- learning rate scheduler: cosine learning rate scheduler
- optimizer: AdamW
- weight decay: 1e-5
- max epochs: 1000
- batch size: 256

# Example Templates
- `example_data_loading.py`: a template for loading a subset dataset from the UCR archive.
- `example_pretrained_model_loading.py`: a template for loading a pretrained FCN.

# Reference
[1] Wang, Zhiguang, Weizhong Yan, and Tim Oates. "Time series classification from scratch with deep neural networks: A strong baseline." 2017 International joint conference on neural networks (IJCNN). IEEE, 2017. \
