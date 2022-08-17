# Description
Fully Convolutional Network (FCN) was proposed in a paper [1]. FCN's architecture is shown in the figure below. \
![Architecture of FCN](.imgs/fcn.png)

This repository offers a supervised-training code of FCN on all the subset datasets of the UCR archive. 
The trained FCN will be used to compute the FID (Fréchet Inception Distance) score for evaluation of generated time series.

# `pip` installation
```angular2html
pip install supervised-fcn
```

# Load a Pretrained FCN 
```angular2html
from supervised_FCN.example_pretrained_model_loading import load_pretrained_FCN

subset_dataset_name = ...  # 'Adiac'
fcn = load_pretrained_FCN(subset_dataset_name)
```
You can do the _forward propagation_ as follows:
```angular2html
x = torch.rand((1, 1, 176))  # (batch_size, in_channels, length)
out = fcn(x)  # (batch_size, n_classes)
```
You can obtain the representation (feature) vector (_i.e.,_ a vector right after the last pooling layer) as follows:
```angular2html
x = torch.rand((1, 1, 176))  # (batch_size, in_channels, length)
out = fcn(x, return_feature_vector=True)  # (batch_size, feature_dim)
```

# Training

## Prerequisite for Training
You need to download the UCR archive dataset and put it in `supervised_FCN/datasets/`. You can download it from [here](https://studntnu-my.sharepoint.com/:u:/g/personal/daesool_ntnu_no/EUVHWAlJRrZBnCZMAOdTR-cB3m_LP7Q10Y78meuzUAuIBQ?e=h9aGhi).
Then, your `supervised_FCN/datasets` directory should have `supervised_FCN/datasets/UCRArchive_2018`.

## Training
`supervised_FCN/train_fcn.py`: runs training of FCN on a subset dataset from the UCR archive.

`supervised_FCN/configs/config.yaml`: is where you can set parameters and hyper-parameters for dataset loading and training. 

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
`supervised_FCN/example_data_loading.py`: a template for loading a subset dataset from the UCR archive.

`supervised_FCN/example_pretrained_model_loading.py`: a template for loading a pretrained FCN.

`supervised_FCN/example_compute_FID.py`: an example of computing the FID score.

`supervised_FCN/example_compute_IS.py`: an example of computing the IS (Inception Score).

# Reference
[1] Wang, Zhiguang, Weizhong Yan, and Tim Oates. "Time series classification from scratch with deep neural networks: A strong baseline." 2017 International joint conference on neural networks (IJCNN). IEEE, 2017.
