import numpy as np
import scipy.io as scio
from scipy.ndimage.filters import gaussian_filter1d

def label_encoding_gaussian(detection_label: np.ndarray, label_len: int, sigma: float=None):
    """
    Parameters
    ----------
    detection_label : np.ndarray
        The input action detection label is in the format:  [[[s1, e1],[s2,e2],...,[si, ei]]],
        where s denotes the subscript of action start and e denotes the subscript of action end.
        The action may occur multiple times, and i denotes the number of actions

    Returns
    -------
    gaussian_label : np.ndarray
        shape: (2, L)
        [gaussian_start_label, gaussian_end_label]
    """
    if sigma is None:
        sigma = label_len / 45

    gaussian_start_label_list = []
    gaussian_end_label_list = []
    for [start, end] in detection_label:
        smooth_start_label = np.zeros((label_len), dtype=np.float32)
        smooth_end_label = np.zeros((label_len), dtype=np.float32)

        smooth_start_label[start] = 1.0
        smooth_end_label[end] = 1.0

        smooth_start_label = gaussian_filter1d(smooth_start_label, sigma=sigma)
        smooth_end_label = gaussian_filter1d(smooth_end_label, sigma=sigma)

        smooth_start_label /= np.max(smooth_start_label)
        smooth_end_label /= np.max(smooth_end_label)

        gaussian_start_label_list.append(np.expand_dims(smooth_start_label, axis=0))
        gaussian_end_label_list.append(np.expand_dims(smooth_end_label, axis=0))

    gaussian_start_label_list = np.concatenate(gaussian_start_label_list, axis=0)
    gaussian_end_label_list = np.concatenate(gaussian_end_label_list, axis=0)

    gaussian_start_label = np.max(gaussian_start_label_list, axis=0, keepdims=True)
    gaussian_end_label = np.max(gaussian_end_label_list, axis=0, keepdims=True)

    gaussian_label = np.concatenate([gaussian_start_label, gaussian_end_label], axis=0)

    return gaussian_label

def smooth_label(labels):
    result = []
    for i, label in enumerate(labels):
        detection_label = get_detection_label(label)
        smooth_label = label_encoding_gaussian(detection_label, len(label))
        smooth_label = np.expand_dims(smooth_label, axis=0)
        result.append(smooth_label)
    result = np.concatenate(result, axis=0)
    return result
def draw_label(test_label, label_len, gaussian_label, sigma=None):
    """
    draw gaussian label
    """
    if sigma is None:
        sigma = label_len / 45

    gaussian_start_label_list = []
    gaussian_end_label_list = []

    for [start, end] in test_label:
        smooth_start_label = np.zeros((label_len), dtype=np.float32)
        smooth_end_label = np.zeros((label_len), dtype=np.float32)

        smooth_start_label[start] = 1.0
        smooth_end_label[end] = 1.0

        smooth_start_label = gaussian_filter1d(smooth_start_label, sigma=sigma)
        smooth_end_label = gaussian_filter1d(smooth_end_label, sigma=sigma)

        smooth_start_label /= np.max(smooth_start_label)
        smooth_end_label /= np.max(smooth_end_label)

        gaussian_start_label_list.append(smooth_start_label)
        gaussian_end_label_list.append(smooth_end_label)


    # plt.figure(figsize=(8,2))
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    for i, start_label in enumerate(gaussian_start_label_list):
        plt.plot(start_label, label=f'Gaussian {i+1}')
    plt.plot(gaussian_label[0], linestyle='--', color='black', label='MAX')
    plt.legend()
    plt.ylim(0, 1.5)

    plt.subplot(3, 1, 2)
    for i, start_label in enumerate(gaussian_end_label_list):
        plt.plot(start_label, label=f'Gaussian {i+1}')
    plt.plot(gaussian_label[1], linestyle='--', color='black', label='MAX')
    plt.legend()
    plt.ylim(0,1.5)

    plt.subplot(3, 1, 3)
    plt.plot(gaussian_label[0], label='start')
    plt.plot(gaussian_label[1], label='end')
    plt.legend()
    plt.ylim(0,1.5)
    plt.show()


def get_detection_label(label):
    detection_label = []
    start_end = np.where(label > 0)[0]
    start = start_end[0]
    end = start_end[-1]
    detection_label.append([start, end])

    return np.array(detection_label)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    """
        WiAR
    """
    data = scio.loadmat(r'<data path>')
    smooth_test_label = smooth_label(data['testSalientLabel'])
    label_data = {
        'amp': data['testAmp'],
        'label': data['testSalientLabel'],
        'smooth_label': smooth_test_label
    }
    scio.savemat(f'TestDataset_wiar_gauss.mat', label_data)

    data = scio.loadmat(r'<data path>')
    smooth_test_label = smooth_label(data['trainSalientLabel'])
    label_data = {
        'amp': data['trainAmp'],
        'label': data['trainSalientLabel'],
        'smooth_label': smooth_test_label
    }
    scio.savemat(f'TraintDataset_wiar_gauss.mat', label_data)

    """
        ARIL
    """
    train_data = scio.loadmat(r"<data path>")
    test_data = scio.loadmat(r'<data path>')

    smooth_test_label = smooth_label(train_data['train_label_mask'])
    label_data = {
        'amp': train_data['train_data_amp'],
        'label': train_data['train_label_mask'],
        'smooth_label': smooth_test_label
    }
    scio.savemat(f'TrainDataset_aril_gauss.mat', label_data)
    #
    smooth_test_label = smooth_label(test_data['test_label_mask'])
    label_data = {
        'amp': test_data['test_data_amp'],
        'label': test_data['test_label_mask'],
        'smooth_label': smooth_test_label
    }
    scio.savemat(f'TestDataset_aril_gauss.mat', label_data)

    """
        HTHI
    """
    train_data = scio.loadmat(r"<data path>")
    test_data = scio.loadmat(r'<data path>')

    smooth_test_label = smooth_label(train_data['trainLabel'])
    label_data = {
        'amp': train_data['trainAmp'],
        'label': train_data['trainLabel'],
        'smooth_label': smooth_test_label
    }
    scio.savemat(f'TrainDataset_hthi_gauss.mat', label_data)

    smooth_test_label = smooth_label(test_data['testLabel'])
    label_data = {
        'amp': test_data['testAmp'],
        'label': test_data['testLabel'],
        'smooth_label': smooth_test_label
    }
    scio.savemat(f'TestDataset_hthi_gauss.mat', label_data)