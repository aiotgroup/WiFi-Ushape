import numpy as np
import scipy.io as scio
import torch.utils.data as data
import pickle
import torch

from dataprocess.arilprocess import ARILData
from dataprocess.wiarprocess import WiARData
from dataprocess.hthidata import HTHIData

from dataprocess_gaussian.arilprocess_gaussian import ARILDataGaussian
from dataprocess_gaussian.hthiprocess_gaussian import HTHIDataGaussian
from dataprocess_gaussian.wiarprocess_guassian import WiARDataGaussian


def getdataloader(dataset_name, filepath, batch_size, trainortest, shuffle=True, detection_gaussian=False):

    dataset = None
    if dataset_name not in ['ARIL', 'WiAR', 'HTHI']:
        raise ValueError("Dataset name error, expected to enter WiAR, ARIL, or HTHI")
    if dataset_name == 'ARIL':
        if detection_gaussian == "Yes":
            dataset = ARILDataGaussian(filepath, trainortest)
        else:
            dataset = ARILData(filepath, trainortest)

    elif dataset_name == 'WiAR':
        if detection_gaussian == "Yes":
            dataset = WiARDataGaussian(filepath, trainortest)
        else:
            dataset = WiARData(filepath, trainortest)
    else:
        if detection_gaussian == "Yes":
            dataset = HTHIDataGaussian(filepath, trainortest)
        else:
            dataset = HTHIData(filepath, trainortest)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return data_loader
