import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import torch
# import gdown
import wget

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
    # if len(pretrained_zip_fnames) == 0:
    #     temp_dir = Path(tempfile.gettempdir())
    #     pretrained_zip_fnames = [fname for fname in os.listdir(temp_dir) if '.zip' in fname]
    #     zipped_pretrained_dirname = temp_dir
    # else:
    zipped_pretrained_dirname = get_root_dir().joinpath('saved_models')

    # is_temp_dir = False
    if len(pretrained_zip_fnames) == 0:
        url = "https://figshare.com/ndownloader/files/38378411"
        # try:
        zipped_pretrained_model_fname = str(zipped_pretrained_dirname.joinpath('supervised-FCN-saved_models.zip'))
        # gdown.download(url, zipped_pretrained_model_fname)
        wget.download(url, zipped_pretrained_model_fname)
        shutil.unpack_archive(zipped_pretrained_model_fname, extract_dir=zipped_pretrained_dirname)
        # except PermissionError:
        #     # is_temp_dir = True
        #     # temp_dir = tempfile.TemporaryDirectory()
        #     # zipped_pretrained_dirname = Path(temp_dir.name)
        #     temp_dir = tempfile.gettempdir()
        #     zipped_pretrained_dirname = Path(temp_dir)
        #     zipped_pretrained_model_fname = str(zipped_pretrained_dirname.joinpath('supervised-FCN-saved_models.zip'))
        #     # gdown.download(url, zipped_pretrained_model_fname)
        #     wget.download(url, zipped_pretrained_model_fname)
        #     shutil.unpack_archive(zipped_pretrained_model_fname, extract_dir=zipped_pretrained_dirname)

    ucr_summary = pd.read_csv(get_root_dir().joinpath('datasets', 'DataSummary_UCR.csv'))
    q = ucr_summary.query(f"Name == '{subset_dataset_name}'")
    n_classes = q['Class'].item()

    # build
    fcn = FCNBaseline(in_channels, n_classes)

    # load
    ckpt_fname = zipped_pretrained_dirname.joinpath(f'{subset_dataset_name}.ckpt')
    fcn.load_state_dict(torch.load(ckpt_fname))

    # if is_temp_dir:
    #     temp_dir.cleanup()
    return fcn


if __name__ == '__main__':
    fcn = load_pretrained_FCN('Adiac')
    print(fcn)
