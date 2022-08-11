import os
import torch
from models.fcn import FCNBaseline


if __name__ == '__main__':
    # config
    subset_dataset_name = 'Adiac'
    in_channels = 1  # univariate model
    n_classes = 37  # can be found here [https://www.cs.ucr.edu/~eamonn/time_series_data_2018/]

    # build
    fcn = FCNBaseline(in_channels, n_classes)

    # load
    ckpt_fname = os.path.join('saved_models', f'{subset_dataset_name}.ckpt')
    fcn.load_state_dict(torch.load(ckpt_fname))

    print(fcn)
