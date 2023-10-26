import numpy as np
import scipy.io as scio
import torch.utils.data as data
import pickle


class DataSource:
    def __init__(self, filename):
        self.filename = filename
        self.amps = None
        self.labels = None
        self.salient_labels = None
        self.segment_labels = None
        self._load_data()


    def _load_data(self):
        data = scio.loadmat(self.filename)
        self.amps = data['trainAmp'].astype(np.float32)
        self.labels = data['trainLabel'].astype(np.float32)
        self.salient_labels = data['trainSalientLabel'].astype(np.float32)
        self.segment_labels = data['trainSegmentLabel'].astype(np.float32)
        n = len(self.amps)


class DataSourceTest:
    def __init__(self, filename):
        self.filename = filename
        self.amps = None
        self.labels = None
        self.salient_labels = None
        self.segment_labels = None
        self._load_data()
    
    def _load_data(self):
        data = scio.loadmat(self.filename)
        self.amps = data['testAmp'].astype(np.float32)
        self.labels = data['testLabel'].astype(np.float32)
        self.salient_labels = data['testSalientLabel'].astype(np.float32)
        self.segment_labels = data['testSegmentLabel'].astype(np.float32)
        n = len(self.amps)

class WiARData(data.Dataset):
    def __init__(self, filename, trainortest):
        super().__init__()
        self.filename = filename
        if trainortest == 'train':
            self.datasource = DataSource(self.filename)
        elif trainortest == 'test':
            self.datasource = DataSourceTest(self.filename)

        self.amps = self.datasource.amps
        self.labels = self.datasource.labels
        self.salient_labels = self.datasource.salient_labels
        self.segment_labels = self.datasource.segment_labels

    
    def __getitem__(self, index):
        self.__amp = self.amps[index]
        self.__label = self.labels[index]
        self.__salient_label = self.salient_labels[index]
        self.__segment_label = self.segment_labels[index]
        return self.__amp, self.__label, self.__salient_label, self.__segment_label

    def __len__(self):
        return len(self.amps)


