import csv
import os

import numpy as np
import torch
import torch.utils.data as Data
import pandas as pd

import scipy
import scipy.io as scio
import scipy.signal as signal
import matplotlib.pyplot as plt

from tqdm import tqdm

from  pathlib import Path


def plt_spec(data):

    Pxx, freqs, bins, im = plt.specgram(data, NFFT=150, Fs=1000, noverlap=10, cmap="jet", vmin=-100, vmax=20)

    a = 0
    for i in range(len(freqs)):
        if freqs[i] > 120:
            a = i
            break

    return bins, freqs[0:a], Pxx[0:a, :]

def scipy_spec(data):
    f, t, Sxx = scipy.signal.spectrogram(data, fs=1000,
                         window=('tukey', 0.25),
                         nperseg=None,
                         noverlap=None,
                         nfft=None,
                         detrend='constant',
                         return_onesided=True,
                         scaling='spectrum',
                         axis=- 1,
                         mode='psd')

    a = 0
    for i in range(len(f)):
        if f[i] > 120:
            a = i
            break

    return t, f[0:a], Sxx[0:a, :]

def filt_low(cutoff, data):
    cutoff = cutoff / (1000 / 2)
    b, a = signal.butter(6, cutoff, 'low', analog=False)
    data_f = signal.filtfilt(b, a, data)

    # plt.figure(figsize=(20, 5))
    # plt.plot(data, color='yellow')
    # plt.plot(data_f, color='red')
    # plt.show()

    return data_f


def filt_high(cutoff, data):
    cutoff = cutoff / (1000 / 2)
    b, a = signal.butter(3, cutoff, 'high', analog=False)
    data_f = signal.filtfilt(b, a, data)

    # plt.figure(figsize=(20, 5))
    # plt.plot(data, color='yellow')
    # plt.plot(data_f, color='red')
    # plt.show()

    return data_f

def sampling(data, time_stamp):
    intervel = (time_stamp[-1] - time_stamp[0]) / 15805
    cur_time = time_stamp[0] + intervel
    start_index = 0

    data_list = []
    for i in range(len(time_stamp)):
        if time_stamp[i] > cur_time:
            temp_data = data[start_index:i+1,:]

            temp_mean = np.mean(temp_data,axis=0)

            data_list.append(temp_mean.reshape(-1,90))

            start_index = i
            cur_time = cur_time + intervel

    if start_index < len(time_stamp) -1:
        temp_data = data[start_index:-1, :]
        temp_mean = np.mean(temp_data, axis=0)
        data_list.append(temp_mean.reshape(-1, 90))


    if len(data_list) < 15800:

        temp_data = data[-1, :]
        print("!-- < --!",len(data_list) ,temp_data.shape)
        data_list.append(temp_data.reshape(-1, 90))

    csi_amp = np.concatenate(data_list,axis=0)
    # print(csi_amp.shape)
    return csi_amp[:15800,:]

def load_data(root):
    # 1 stft
    flag = 1

    file_list_all = os.listdir(root)
    csi_path = Path('csi_data/')
    stft_list = []
    label_list = []
    aclist = ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']

    for index, file in enumerate(tqdm(file_list_all)):

        data = pd.read_csv(root/file, header=None).values
        csi_amp_temp = data[:, 1:91]
        time_stamp = data[:, 0]

        csi_amp_new = sampling(csi_amp_temp,time_stamp)

        temp_list = []
        csi_list = []
        # filtering
        for i in range(csi_amp_new.shape[1]):
            csi_amp_f = filt_low(120, csi_amp_new[:, i])
            csi_amp_f = filt_high(2, csi_amp_f)

            if flag:
                # two kinds of stft
                _,_,Pxx = plt_spec(csi_amp_f)
                # _, _, Pxx = scipy_spec(csi_amp_f)

                temp_list.append(Pxx.transpose())
                # print(Pxx.transpose().shape) -> (112, 19)
            else:
                csi_list.append(csi_amp_f.reshape(-1,1))

        if flag:
            stft_map = np.concatenate(temp_list, axis=1)
                # stft_map.shape -> (112, 1710)
            t,f = stft_map.shape
            stft_list.append(stft_map.reshape((1,t,f)))
        else:
            csi_amp = np.concatenate(csi_list, axis=1)

        for j in range(len(aclist)):
            if file.find(aclist[j]) != -1:
                label_list.append(j)
                if not flag:

                    scio.savemat(csi_path/'csi_amp-{}-{}.mat'.format(index, j), mdict={'csi_amp': csi_amp})


    if flag:
        stft_train = np.concatenate(stft_list, axis=0)
        print(stft_train.shape)
        scio.savemat('SF0_all.mat', mdict={'stft_train': stft_train, 'label_list': label_list})


def clean(root):
    file_list_all = os.listdir(root)
    for file in tqdm(file_list_all):
        if file.startswith('.') or file.startswith('annotation'):
            os.remove(root/file)


if __name__ == '__main__':
    path = Path('WiAR/Data/')
    # path = Path('data_s/')

    load_data(path)

    # clean(path)

    # file = 'input_161219_siamak_fall_7.csv'
    # data = pd.read_csv(path / file, header=None).values
    # csi_amp_temp = data[:, 1:91]
    # time_stamp = data[:, 0]
    #
    # csi_amp_new = sampling(csi_amp_temp, time_stamp)
    # print(data.shape)
