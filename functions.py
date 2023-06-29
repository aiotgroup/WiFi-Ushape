import torch
import torch.nn as nn
import numpy as np

def onehot(batch_size, num_class, index):
    labels = torch.zeros((batch_size, num_class))
    for i in range(len(index)):
        labels[i, int(index[i, 0]-1)] = 1.0
    return labels

def onehot_first0(batch_size, num_class, index):
    labels = torch.zeros((batch_size, num_class))
    for i in range(len(index)):
        labels[i, int(index[i, 0])] = 1.0
    return labels

def segment_onehot(segment_label, num_class):
    batchsize, seqlen = segment_label.shape
    labels = torch.zeros((batchsize, seqlen, num_class))
    for i in range(batchsize):
        for j in range(seqlen):
            labels[i, j, int(segment_label[i, j])] = 1.0
    return labels

def segmentonehot_negone(frame_label, num_class):
    batchsize, seqlen = frame_label.shape
    labels = torch.zeros((batchsize, seqlen, num_class))
    for i in range(batchsize):
        for j in range(seqlen):
            labels[i, j, int(frame_label[i, j])+1] = 1.0
    return labels

def cal_segment_acc(data, label):
    predict = data.argmax(dim=-1)
    target = label.argmax(dim=-1)
    result = predict - target
    correct_sum = 0
    for i in range(len(result)):
        temp = result[i]
        correct_sum += len(temp[temp==0]) / len(temp)
    return correct_sum




def cal_F(predict, gt):
    predict = predict.squeeze()
    gt = gt.squeeze()

    predict[predict >= 0.5] = 1
    predict[predict < 0.5] = 0

    result = predict - gt
    FP = result[result == 1].sum()
    FN = abs(result[result == -1].sum())

    temp = predict[gt == 1]
    TP = temp[temp == 1].sum()

    P = TP / (TP + FP)
    R = TP / (TP + FN)

    F = 2 * P * R / (P + R)

    if torch.isnan(F):
        F = 0
    return F


def cal_MAE(predict, gt):
    predict = predict.squeeze()
    gt = gt.squeeze()
    predict[predict >= 0.5] = 1
    predict[predict < 0.5] = 0

    result = (abs(predict - gt)).sum()
    mae = result / len(predict)
    return mae


def cal_acc(out, label):
    correct = 0
    predict = out.argmax(dim=1)
    target = label.argmax(dim=1)
    result = predict - target
    for i in result:
        if i == 0:
            correct += 1
    return correct



def detection_start_end(out, detection_label, sample_rate):
    label = detection_label.squeeze()
    out = out.squeeze()
    label = label.data.cpu().numpy()
    out = out.data.cpu().numpy()
    out[out > 0.5] = 1
    out[out <= 0.5] = 0
    if len(np.where(out == 1)[0]) == 0:
        start_error = 1
        end_error = 1
    else:
        label_start, prediction_start = np.where(label == 1)[0][0], np.where(out == 1)[0][0]
        label_end, prediction_end = np.where(label == 1)[0][-1], np.where(out == 1)[0][-1]
        start_error = np.abs(label_start - prediction_start) / sample_rate
        end_error = np.abs(label_end - prediction_end) / sample_rate

    return start_error, end_error