import os
import shutil

import pandas as pd
import torch
import gdown

from supervised_FCN.models.fcn import FCNBaseline
from supervised_FCN.utils import get_root_dir


def load_pretrained_FCN(subset_dataset_name: str, in_channels: int = 1):
    """
    load a pretrained FCN (Fully Convolutional Network)
    :param subset_dataset_name:
    :param in_channels: 1; univariate for the UCR subset datasets.
    :return:
    """
    pretrained_zip_fnames = [fname for fname in os.listdir(get_root_dir().joinpath('saved_models')) if '.zip' in fname]
    if len(pretrained_zip_fnames) == 0:
        url = "https://drive.google.com/u/0/uc?id=14F-x1Ef5UTNrAVrzohKUe6trKB2IhUDm&export=download"
        zipped_pretrained_model_fname = str(get_root_dir().joinpath('saved_models', 'supervised-FCN-saved_models.zip'))
        gdown.download(url, zipped_pretrained_model_fname)
        shutil.unpack_archive(zipped_pretrained_model_fname, extract_dir=get_root_dir().joinpath('saved_models'))

    ucr_summary = pd.read_csv(get_root_dir().joinpath('datasets', 'DataSummary_UCR.csv'))
    q = ucr_summary.query(f"Name == '{subset_dataset_name}'")
    n_classes = q['Class'].iloc[0]

    # build
    fcn = FCNBaseline(in_channels, n_classes)

    # load
    ckpt_fname = get_root_dir().joinpath('saved_models', f'{subset_dataset_name}.ckpt')
    fcn.load_state_dict(torch.load(ckpt_fname))
    return fcn


if __name__ == '__main__':
    fcn = load_pretrained_FCN('Adiac')
    print(fcn)
