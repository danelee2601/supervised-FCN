"""
`Dataset` (pytorch) class is defined.
"""
import math

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from scipy.io.arff import loadarff
from sktime.datasets import load_from_tsfile, load_from_arff_to_dataframe

from sklearn.preprocessing import LabelEncoder

from supervised_FCN.utils import get_root_dir


def extract_data(data):
    res_data = []
    res_labels = []
    for t_data, t_label in data:
        t_data = np.array([d.tolist() for d in t_data])
        t_label = t_label.decode("utf-8")
        res_data.append(t_data)
        res_labels.append(t_label)
    return np.array(res_data).swapaxes(1, 2), np.array(res_labels)


class DatasetImporterUEA(object):
    """
    This uses train and test sets as given.
    To compare with the results from ["Unsupervised scalable representation learning for multivariate time series"]
    """

    def __init__(self, subset_name: str, data_scaling: bool, **kwargs):
        """
        :param subset_name: e.g., "Cricket"

        following https://github.com/yuezhihan/ts2vec/blob/631bd533aab3547d1310f4e02a20f3eb53de26be/datautils.py#L79
        """
        # download_ucr_datasets()
        self.data_root = get_root_dir().joinpath("datasets", "UEAArchive_2018", subset_name)

        # fetch an entire dataset
        train_data = loadarff(str(self.data_root.joinpath(f'{subset_name}_TRAIN.arff')))[0]
        test_data = loadarff(str(self.data_root.joinpath(f'{subset_name}_TEST.arff')))[0]

        self.X_train, self.Y_train = extract_data(train_data)
        self.X_test, self.Y_test = extract_data(test_data)

        if data_scaling:
            # from [https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/dcc674541a94ca8a54fbb5503bb75a297a5231cb/uea.py#L31]
            # Normalizing dimensions independently
            nb_dims = self.X_train.shape[1]
            for j in range(nb_dims):
                mean = np.mean(self.X_train[:, j])
                var = np.var(self.X_train[:, j])
                self.X_train[:, j] = (self.X_train[:, j] - mean) / math.sqrt(var)
                self.X_test[:, j] = (self.X_test[:, j] - mean) / math.sqrt(var)

        le = LabelEncoder()
        self.Y_train = le.fit_transform(self.Y_train)
        self.Y_test = le.transform(self.Y_test)

        print('self.X_train.shape:', self.X_train.shape)
        print('self.X_test.shape:', self.X_test.shape)

        print("# unique labels (train):", np.unique(self.Y_train.reshape(-1)))
        print("# unique labels (test):", np.unique(self.Y_test.reshape(-1)))


class UEADataset(Dataset):
    def __init__(self,
                 kind: str,
                 dataset_importer,
                 **kwargs):
        """
        :param kind: "train" / "test"
        :param dataset_importer: instance of the `DatasetImporter` class.
        :param augs: instance of the `Augmentations` class.
        :param used_augmentations: e.g., ["RC", "AmpR", "Vshift"]
        """
        super().__init__()
        self.kind = kind

        if kind == "train":
            self.X, self.Y = dataset_importer.X_train, dataset_importer.Y_train
        elif kind == "test":
            self.X, self.Y = dataset_importer.X_test, dataset_importer.Y_test
        else:
            raise ValueError

        self._len = self.X.shape[0]

    @staticmethod
    def _assign_float32(*xs):
        """
        assigns `dtype` of `float32`
        so that we wouldn't have to change `dtype` later before propagating data through a model.
        """
        new_xs = []
        for x in xs:
            new_xs.append(x.astype(np.float32))
        return new_xs[0] if (len(xs) == 1) else new_xs

    def getitem_default(self, idx):
        x, y = self.X[idx, :, :], self.Y[idx]

        if len(x.shape) == 1:
            x = x[np.newaxis, :]  # to make a channel dim of 1 for a univariate time series

        return x, y

    def __getitem__(self, idx):
        return self.getitem_default(idx)

    def __len__(self):
        return self._len


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    os.chdir("../")

    # data pipeline
    dataset_importer = DatasetImporterUEA("Cricket", data_scaling=True)
    dataset = UEADataset("train", dataset_importer)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=False)

    # get a mini-batch of samples
    for batch in data_loader:
        x, y = batch
        break
    print('x.shape:', x.shape)

    # test if all datasets can be loaded
    subset_names = sorted(os.listdir(os.path.join('datasets', 'UEAArchive_2018')))
    for subset_name in subset_names:
        print('subset_name:', subset_name)
        try:
            dataset_importer = DatasetImporterUEA(subset_name, data_scaling=True)
            dataset = UEADataset("train", dataset_importer)
            data_loader = DataLoader(dataset, batch_size=32, num_workers=0, shuffle=True)
            for batch in data_loader:
                x, y = batch
                break
            print('x.shape:', x.shape)
            print()
        except:
            pass
