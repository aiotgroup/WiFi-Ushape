
import glob
import os.path
import re

import pandas as pd
import numpy as np
import scipy.io as scio

import tqdm

action_dict = dict(zip(["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"], [i for i in range(7)]))
action_dict['NoActivity'] = -1


def annotation2num(annotation):

    segment_label_list = []
    salient_label_list = []

    for label in annotation:
        l_num = action_dict[label[0]]
        segment_label_list.append(l_num)
        if l_num == -1:
            salient_label_list.append(0)
        else:
            salient_label_list.append(1)

    return salient_label_list, segment_label_list

def sampling(csi_data, time_stamp, salient_label_list, segment_label_list):

    intervel = (time_stamp[-1] - time_stamp[0]) / 15805
    cur_time = time_stamp[0] + intervel
    start_index = 0

    data_list = []
    salient_label = []
    segment_label = []

    for i in range(len(time_stamp)):

        if time_stamp[i] > cur_time:

            temp_data = csi_data[start_index:i+1,:]

            if i < len(salient_label_list):
                salient_label.append(salient_label_list[i])
                segment_label.append(segment_label_list[i])
            else:
                salient_label.append(salient_label_list[-1])
                segment_label.append(segment_label_list[-1])

            temp_mean = np.mean(temp_data,axis=0)

            data_list.append(temp_mean.reshape(-1,90))

            start_index = i
            cur_time = cur_time + intervel

    if start_index < len(time_stamp) -1:

        temp_data = csi_data[start_index:-1, :]
        temp_mean = np.mean(temp_data, axis=0)
        data_list.append(temp_mean.reshape(-1, 90))

        if start_index < len(salient_label_list):
            salient_label.append(salient_label_list[start_index])
            segment_label.append(segment_label_list[start_index])
        else:
            salient_label.append(salient_label_list[-1])
            segment_label.append(segment_label_list[-1])


    if len(data_list) < 15800:

        temp_data = csi_data[-1, :]
        print("!-- < --!",len(data_list) ,temp_data.shape)
        data_list.append(temp_data.reshape(-1, 90))

        salient_label.append(salient_label_list[-1])
        segment_label.append(segment_label_list[-1])

    csi_amp = np.concatenate(data_list,axis=0)
    # print(csi_amp.shape)
    return csi_amp[:15800,:], np.array(salient_label[:15800]), np.array(segment_label[:15800])

def main(root='Data'):
    count = 0

    csi_data_list = []
    label_list = []
    salient_label_list_list = []
    segment_label_list_list = []

    for i, label in enumerate (["bed", "fall", "pickup", "run", "sitdown", "standup", "walk"]):
        annotation_list = sorted(glob.glob(os.path.join(root, 'annotation_*' + label + '*.csv')))

        for j in range(len(annotation_list)):

            count += 1

            name = re.findall('annotation_(.*).csv', annotation_list[j])
            input_f = glob.glob(os.path.join(root, 'input_' + name[0] + '.csv'))
            if input_f == []:
                input_f = glob.glob(os.path.join(root, 'input_161219_' + name[0] + '.csv'))

            annotation = pd.read_csv(annotation_list[j], header=None).values
            data = pd.read_csv(input_f[0], header=None).values

            csi_data = data[:, 1:91]
            time_stamp = data[:, 0]


            if annotation.shape[0] != csi_data.shape[0]:
                print('!!', annotation_list[j], input_f[0], annotation.shape, csi_data.shape)

            salient_label_list, segment_label_list = annotation2num(annotation)

            csi_amp, salient_label, segment_label = sampling(csi_data, time_stamp, salient_label_list, segment_label_list)

            print(count,label, action_dict[label], csi_amp.shape, salient_label.shape, segment_label.shape)

            csi_data_list.append(csi_amp.reshape(1,csi_amp.shape[0],csi_amp.shape[1]))

            label_list.append(action_dict[label])

            salient_label_list_list.append(salient_label.reshape(1, 15800))

            segment_label_list_list.append(segment_label.reshape(1, 15800))

            # scio.savemat('csi_data\\csi_amp-{}-{}.mat'.format(count,action_dict[label]), mdict={'csi_amp': csi_amp,
            #                                                                                     'label': label,
            #                                                                                     'salient_label': salient_label,
            #                                                                                    'segment_label': segment_label})
    label_list_np = np.array(label_list)
    csi_data = np.concatenate(csi_data_list, axis=0).astype(np.float32)
    # csi_data = np.concatenate(csi_data_list, axis=0)

    salient_label_all = np.concatenate(salient_label_list_list, axis=0)
    segment_label_all = np.concatenate(segment_label_list_list, axis=0)

    scio.savemat('csi_amp_all.mat', mdict={'csi_amp': csi_data,
                                           'label': label_list_np,
                                           'salient_label': salient_label_all,
                                           'segment_label': segment_label_all})

    print(csi_data.shape, label_list_np.shape, salient_label_all.shape, segment_label_all.shape)



# !! Data\annotation_sankalp_walk_7.csv Data/input_161219_sankalp_walk_7.csv (19987, 1) (19989, 90)
# !! Data\annotation_sankalp_walk_8.csv Data/input_161219_sankalp_walk_8.csv (19989, 1) (19991, 90)


if __name__ == '__main__':

    """
        root_path
    """
    WiAR_root_path = 'Data'
    main(r'<data path>')
