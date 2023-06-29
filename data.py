import numpy as np
import scipy.io as scio
import torch.utils.data as data
import pickle
import torch

from datapropress.arilprocess import ARILData
from datapropress.wiarprocess import WiARData
from datapropress.hthidata import HTHIData


def getdataloader(dataset_name, filepath, batch_size, trainortest, shuffle=True):

    dataset = None
    if dataset_name not in ['ARIL', 'WiAR', 'HTHI']:
        raise ValueError("Dataset name error, expected to enter WiAR, ARIL, or HTHI")

    if dataset_name == 'ARIL':
        dataset = ARILData(filepath, trainortest)
    elif dataset_name == 'WiAR':
        dataset = WiARData(filepath, trainortest)
    else:
        dataset = HTHIData(filepath, trainortest)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return data_loader
