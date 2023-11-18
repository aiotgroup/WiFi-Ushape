import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io as scio
import sys


dataset = scio.loadmat('csi_amp_all.mat')
datas = dataset['csi_amp']
labels = dataset['label'][0]
salient_labels = dataset['salient_label']
segment_labels = dataset['segment_label']

index = int(sys.argv[1])


data0, data1, data2, data3, data4, data5, data6 = [], [], [], [], [], [], []
label0, label1, label2, label3, label4, label5, label6 = [], [], [], [], [], [], []
salient_label0, salient_label1, salient_label2, salient_label3, salient_label4, \
    salient_label5, salient_label6 = [], [], [], [], [], [], []
segment_label0, segment_label1, segment_label2, segment_label3, segment_label4, \
    segment_label5, segment_label6 = [], [], [], [], [], [], []



for i in range(len(datas)):
    if labels[i] == 0:
        data0.append(datas[i])
        label0.append(labels[i])
        salient_label0.append(salient_labels[i])
        segment_label0.append(segment_labels[i])

    elif labels[i] == 1:
        data1.append(datas[i])
        label1.append(labels[i])
        salient_label1.append(salient_labels[i])
        segment_label1.append(segment_labels[i])

    elif labels[i] == 2:
        data2.append(datas[i])
        label2.append(labels[i])
        salient_label2.append(salient_labels[i])
        segment_label2.append(segment_labels[i])

    elif labels[i] == 3:
        data3.append(datas[i])
        label3.append(labels[i])
        salient_label3.append(salient_labels[i])
        segment_label3.append(segment_labels[i])

    elif labels[i] == 4:
        data4.append(datas[i])
        label4.append(labels[i])
        salient_label4.append(salient_labels[i])
        segment_label4.append(segment_labels[i])


    elif labels[i] == 5:
        data5.append(datas[i])
        label5.append(labels[i])
        salient_label5.append(salient_labels[i])
        segment_label5.append(segment_labels[i])

    elif labels[i] == 6:
        data6.append(datas[i])
        label6.append(labels[i])
        salient_label6.append(salient_labels[i])
        segment_label6.append(segment_labels[i])



train_data0, test_data0, train_label0, test_label0, train_salient_label0, test_salient_label0, train_segment_label0, test_segment_label0 \
      = train_test_split(data0, label0, salient_label0, segment_label0, test_size=0.1, shuffle=True)
train_data1, test_data1, train_label1, test_label1, train_salient_label1, test_salient_label1, train_segment_label1, test_segment_label1 \
      = train_test_split(data1, label1, salient_label1, segment_label1, test_size=0.1, shuffle=True)
train_data2, test_data2, train_label2, test_label2, train_salient_label2, test_salient_label2, train_segment_label2, test_segment_label2 \
      = train_test_split(data2, label2, salient_label2, segment_label2, test_size=0.1, shuffle=True)
train_data3, test_data3, train_label3, test_label3, train_salient_label3, test_salient_label3, train_segment_label3, test_segment_label3 \
      = train_test_split(data3, label3, salient_label3, segment_label3, test_size=0.1, shuffle=True)
train_data4, test_data4, train_label4, test_label4, train_salient_label4, test_salient_label4, train_segment_label4, test_segment_label4 \
      = train_test_split(data4, label4, salient_label4, segment_label4, test_size=0.1, shuffle=True)
train_data5, test_data5, train_label5, test_label5, train_salient_label5, test_salient_label5, train_segment_label5, test_segment_label5 \
      = train_test_split(data5, label5, salient_label5, segment_label5, test_size=0.1, shuffle=True)
train_data6, test_data6, train_label6, test_label6, train_salient_label6, test_salient_label6, train_segment_label6, test_segment_label6 \
      = train_test_split(data6, label6, salient_label6, segment_label6, test_size=0.1, shuffle=True)


traindata = train_data0 + train_data1 + train_data2 + train_data3 + train_data4 + train_data5 + train_data6
trainlabel = train_label0 + train_label1 + train_label2 + train_label3 + train_label4 + train_label5 + train_label6
testdata = test_data0 + test_data1 + test_data2 + test_data3 + test_data4 + test_data5 + test_data6
testlabel = test_label0 + test_label1 + test_label2 + test_label3 + test_label4 + test_label5 + test_label6
trainsalientlabel = train_salient_label0 + train_salient_label1 + train_salient_label2 + train_salient_label3 + train_salient_label4 + \
     train_salient_label5 + train_salient_label6
testsalientlabel = test_salient_label0 + test_salient_label1 + test_salient_label2 + test_salient_label3 + test_salient_label4 + \
     test_salient_label5 + test_salient_label6
trainsegmentlabel = train_segment_label0 + train_segment_label1 + train_segment_label2 + train_segment_label3 + train_segment_label4 + \
     train_segment_label5 + train_segment_label6
testsegmentlabel = test_segment_label0 + test_segment_label1 + test_segment_label2 + test_segment_label3 + test_segment_label4 + \
     test_segment_label5 + test_segment_label6





trainamp = np.array(traindata)
trainlabel = np.array(trainlabel)
trainsalientlabel = np.array(trainsalientlabel)
trainsegmentlabel = np.array(trainsegmentlabel)

testamp = np.array(testdata)
testlabel = np.array(testlabel)
testsalientlabel = np.array(testsalientlabel)
testsegmentlabel = np.array(testsegmentlabel)

scio.savemat('TrainDataset%d.mat'%index, mdict={'trainAmp':trainamp, 'trainLabel':trainlabel, \
                                                'trainSalientLabel':trainsalientlabel, 'trainSegmentLabel':trainsegmentlabel})
scio.savemat('TestDataset%d.mat'%index, mdict={'testAmp':testamp, 'testLabel':testlabel, 'testSalientLabel':testsalientlabel,\
                                               'testSegmentLabel':testsegmentlabel})

