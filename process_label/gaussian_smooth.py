import scipy.io as scio
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d


def smooth_label(data, sigma=None):
    """

    data: the ground-truth annotations for action recognition -> (N, L)
    """

    if sigma is None:
        data_len = data.shape[1]
        sigma = data_len / 45

    result = []
    for i, label in enumerate(data):

        start_end = np.where(label > 0)[0]

        start = start_end[0]
        end = start_end[-1]

        smooth_start_label = np.zeros_like(label, dtype=np.float32)
        smooth_end_label = np.zeros_like(label, dtype=np.float32)

        smooth_start_label[start] = 1.0
        smooth_end_label[end] = 1.0

        smooth_start_label = gaussian_filter1d(smooth_start_label, sigma=sigma)
        smooth_end_label = gaussian_filter1d(smooth_end_label, sigma=sigma)

        smooth_start_label /= np.max(smooth_start_label)
        smooth_end_label /= np.max(smooth_end_label)

        smooth_start_label = np.expand_dims(smooth_start_label, axis=0)
        smooth_end_label = np.expand_dims(smooth_end_label, axis=0)

        smooth_label = np.concatenate([smooth_start_label, smooth_end_label], axis=0)
        smooth_label = np.expand_dims(smooth_label, axis=0)
        result.append(smooth_label)

    result = np.concatenate(result, axis=0)

    return result

if __name__ == '__main__':
    train_data = scio.loadmat("trainDataset.mat")
    test_data = scio.loadmat('testDataset.mat')
    # ---------------------------------------------------------------------------------------------
    smooth_test_label = smooth_label(train_data['trainLabel'])
    label_data = {
        'amp': train_data['trainAmp'],
        'label': train_data['trainLabel'],
        'smooth_label': smooth_test_label
    }
    scio.savemat(f'out/TrainDataset.mat', label_data)
    # ---------------------------------------------------------------------------------------------
    smooth_test_label = smooth_label(test_data['testLabel'])
    label_data = {
        'amp': test_data['testAmp'],
        'label': test_data['testLabel'],
        'smooth_label': smooth_test_label
    }
    scio.savemat(f'out/TestDataset.mat', label_data)
    # ---------------------------------------------------------------------------------------------
