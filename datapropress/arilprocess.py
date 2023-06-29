import numpy as np
import scipy.io as scio
import torch.utils.data as data
import pickle
import torch


class DataSource:
    def __init__(self, filename):
        self.filename = filename
        self.data_amps = None
        self.data_phas = None
        self.data_label_bags = None
        self.train_instance_masks = None
        self.instance_label_time = None
        self._load_data()

    def _load_data(self):
        data = scio.loadmat(self.filename)
        self.data_amps = data['train_data_amp'].astype(np.float32)  # shape = 1116 * 52 * 192
        self.data_phas = data['train_data_pha'].astype(np.float32)
        self.data_label_instance = data['train_label_instance'].astype(np.int32)  # shape = 1116 * 192
        self.instance_label_masks = data['train_label_mask'].astype(np.float32)
        self.instance_label_time = data['train_label_time'].astype(np.int32)
        n = len(self.data_amps)

        self.data_label_bags = np.zeros((n, 1))

        for i in range(n):
            self.data_label_bags[i, 0] = np.max(self.data_label_instance[i])


class DataSourceTest:
    def __init__(self, filename):
        self.filename = filename
        self.data_amps = None
        self.data_phas = None
        self.data_label_bags = None
        self.instance_masks = None
        self.instance_label_time = None
        self._load_data()

    def _load_data(self):
        data = scio.loadmat(self.filename)
        self.data_amps = data['test_data_amp'].astype(np.float32)  # shape = 1116 * 52 * 192
        self.data_phas = data['test_data_pha'].astype(np.float32)
        self.data_label_instance = data['test_label_instance'].astype(np.int32)  # shape = 1116 * 192
        self.instance_label_masks = data['test_label_mask'].astype(np.float32)
        self.instance_label_time = data['test_label_time'].astype(np.int32)
        n = len(self.data_amps)

        self.data_label_bags = np.zeros((n, 1))

        for i in range(n):
            self.data_label_bags[i, 0] = np.max(self.data_label_instance[i])


class ARILData(data.Dataset):
    def __init__(self, filename, trainortest):
        super().__init__()
        self.filename = filename
        if trainortest == 'train':
            self.datasource = DataSource(self.filename)
        elif trainortest == 'test':
            self.datasource = DataSourceTest(self.filename)

        self.data_amps = self.datasource.data_amps
        self.data_phas = self.datasource.data_phas
        self.data_label_bags = self.datasource.data_label_bags
        self.instance_label_masks = self.datasource.instance_label_masks
        self.instance_label_time = self.datasource.instance_label_time
        self.frame_label = self.datasource.data_label_instance

    def __getitem__(self, index):
        self.__data_amp = self.data_amps[index]
        self.__data_pha = self.data_phas[index]

        self.__data_label_bag = self.data_label_bags[index]
        self.__instance_label_mask = np.expand_dims(self.instance_label_masks[index], 0)
        self.__instance_label_time = self.instance_label_time[index]
        self.__frame_label = self.frame_label[index]
        return torch.Tensor(
            self.__data_amp), self.__data_pha, self.__data_label_bag, self.__frame_label, self.__instance_label_mask, self.__instance_label_time

    def __len__(self):
        return len(self.data_amps)