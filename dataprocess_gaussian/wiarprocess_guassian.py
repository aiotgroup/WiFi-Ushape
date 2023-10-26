import numpy as np
import scipy.io as scio
import torch.utils.data as data
import pickle
import torch


class DataSource:
    def __init__(self, filename):
        self.filename = filename
        self.data_amps = None
        self.labels = None
        self._load_data()

    def _load_data(self):
        data = scio.loadmat(self.filename)
        self.data_amps = data['amp'].astype(np.float32)  # shape = 1116 * 52 * 192
        self.labels = data['smooth_label'].astype(np.float32)
        n = len(self.data_amps)


class DataSourceTest:
    def __init__(self, filename):
        self.filename = filename
        self.data_amps = None
        self.labels = None
        self._load_data()

    def _load_data(self):
        data = scio.loadmat(self.filename)
        self.data_amps = data['amp'].astype(np.float32)  # shape = 1116 * 52 * 192
        self.labels = data['smooth_label'].astype(np.float32)
        n = len(self.data_amps)



class WiARDataGaussian(data.Dataset):
    def __init__(self, filename, trainortest):
        super().__init__()
        self.filename = filename
        if trainortest == 'train':
            self.datasource = DataSource(self.filename)
        elif trainortest == 'test':
            self.datasource = DataSourceTest(self.filename)

        self.data_amps = self.datasource.data_amps
        self.labels = self.datasource.labels

    def __getitem__(self, index):
        self.__data_amp = self.data_amps[index]
        self.__label = self.labels[index]
        return torch.Tensor(
            self.__data_amp), self.__label

    def __len__(self):
        return len(self.data_amps)