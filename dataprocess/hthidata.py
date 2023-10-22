import numpy as np
import scipy.io as scio
import torch.utils.data as data
import pickle




class DataSource:
    def __init__(self, filename):
        self.filename = filename
        self.data_amps = None
        self.data_labels = None
        self.data_classes = None
        self._load_data()


    def _load_data(self):
        data = scio.loadmat(self.filename)
        self.data_amps = data['trainAmp'].astype(np.float32)           # shape = 1116 * 52 * 192
        self.data_labels = data['trainLabel'].astype(np.float32)         # shape = 1116 * 192
        self.data_classes = data['trainClass']
        n = len(self.data_amps)


class DataSourceTest:
    def __init__(self, filename):
        self.filename = filename
        self.data_amps = None
        self.data_labels = None
        self.data_classes = None
        self._load_data()
    
    def _load_data(self):
        data = scio.loadmat(self.filename)
        self.data_amps = data['testAmp'].astype(np.float32)
        self.data_labels = data['testLabel'].astype(np.float32)
        self.data_classes = data['testClass']
        n = len(self.data_amps)

class HTHIData(data.Dataset):
    def __init__(self, filename, trainortest):
        super().__init__()
        self.filename = filename
        if trainortest == 'train':
            self.datasource = DataSource(self.filename)
        elif trainortest == 'test':
            self.datasource = DataSourceTest(self.filename)

        self.data_amps = self.datasource.data_amps
        self.data_labels = self.datasource.data_labels
        self.data_classes = self.datasource.data_classes

    
    def __getitem__(self, index):
        self.__data_amp = self.data_amps[index]
        self.__data_label = self.data_labels[index]
        self.__data_class = self.data_classes[index]
        return self.__data_amp, self.__data_label, self.__data_class

    def __len__(self):
        return len(self.data_amps)
