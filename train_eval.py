import os
import sys

import pickle
import argparse
from tqdm import tqdm

import numpy as np
from model import WholeNet
from data import getdataloader
import torch
import torch.optim as optim
import torch.nn as nn
import random
from sklearn.metrics import confusion_matrix

from functions import onehot, onehot_first0, segment_onehot, segmentonehot_negone, \
    cal_segment_acc, cal_F, cal_MAE, cal_acc, detection_start_end
from Config import Config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='fcn',
                        help='{unet} / {unetpp} / {fcn}')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--epoches', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate')
    parser.add_argument('--decay_epoch', type=list, default=[],
                        help='every n epochs decay learning rate')
    parser.add_argument('--task', type=str, default='segment',
                        help='choose target of this task: {classify} / {detection} / {segment}')

    parser.add_argument('--dataset_name', type=str, default='WiAR',
                        help='{HTHI} / {WiAR} / {ARIL}')
    parser.add_argument('--train_dataset_path', type=str, default='create_wiar_dataset/TrainDataset1.mat',
                        help='train dataset path')
    parser.add_argument('--test_dataset_path', type=str, default='create_wiar_dataset/TestDataset1.mat',
                        help='test dataset path')

    parser.add_argument('--detection_gaussian', type=bool, default=False)
    args = parser.parse_args()

    return args


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


device = 'cuda'
seed = 42
seed_everything(seed)

args = get_args()

device_id = args.device_id
torch.cuda.set_device(device_id)
model_name = args.model_name
epoches = args.epoches
batch_size = args.batch_size
decay_epoch = args.decay_epoch
lr = args.lr
task = args.task
filename = args.train_dataset_path
testfilename = args.test_dataset_path
detection_gaussian = args.detection_gaussian

config = Config(dataset_name=args.dataset_name)



def model_opt_lossfn(model_name, lr, in_channel, num_class, segment_class, unet_depth, unetpp_depth, task, detection_gaussian):
    model = WholeNet(model_name=model_name, in_channel=in_channel, num_class=num_class, segment_class=segment_class, unet_depth=unet_depth,
                     unetpp_depth=unetpp_depth, task=task, detection_gaussian=detection_gaussian).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = None

    if task == 'classify':
        loss_fn = nn.CrossEntropyLoss(reduction='sum')

    elif task == 'detection':
        loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    elif task == 'segment':
        loss_fn = nn.CrossEntropyLoss(reduction='sum')

    return model, optimizer, loss_fn


# loading data
dataset = getdataloader(dataset_name=args.dataset_name, filepath=filename, batch_size=batch_size, trainortest='train', detection_gaussian=detection_gaussian)
testdataset = getdataloader(dataset_name=args.dataset_name, filepath=testfilename, batch_size=1, trainortest='test', shuffle=False, detection_gaussian=detection_gaussian)

# model, optimizer, loss_function
model, optimizer, loss_fn = model_opt_lossfn(model_name, lr, config.in_channel, config.num_class, config.segment_class, config.unet_depth, config.unetpp_depth, task, detection_gaussian=detection_gaussian)


# training, testing/evaluating

classify_max_result = 0
classify_matrix = None
detection_min_mae = 1
detection_min_error = 100
segment_max_result = 0
amp, label, detection_label, segment_label = None, None, None, None

if not os.path.exists("outputs/{}".format(task)):
    os.makedirs("outputs/{}".format(task))

if args.dataset_name == "HTHI" and (args.task == 'classify' or args.task == "segment"):
    raise ValueError("This dataset HTHI does not have a classify task and segment task")


for _ in range(epoches):
    loss_sum = 0
    correct_sum = 0
    whole_epoch_sum = 0
    if _ in decay_epoch:
        lr = lr * 0.1
        optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    '''
    training
    '''
    for data in tqdm(dataset, desc='Train'):
        optimizer.zero_grad()
        if args.dataset_name == 'ARIL':
            amp, pha, label, segment_label, detection_label, time = data
            amp = amp.to(device)
            pha = pha.to(device)
            label = onehot(batch_size, config.num_class, label).to(device)
            segment_label = segment_onehot(segment_label, config.segment_class).to(device)
            detection_label = detection_label.to(device)

        elif args.dataset_name == 'WiAR':
            amp, label, detection_label, segment_label = data
            amp = amp.to(device)
            label = onehot_first0(batch_size, config.num_class, label).to(device)
            detection_label = detection_label.to(device)
            segment_label = segmentonehot_negone(segment_label, config.segment_class).to(device)

        else:
            amp, detection_label, cla = data
            amp = amp.to(device)
            detection_label = detection_label.to(device)

        out = model(amp)

        if task == 'classify':
            correct_sum += cal_acc(out, label)
            loss = loss_fn(out, label)
        elif task == 'detection':
            loss = loss_fn(out, detection_label.squeeze())
        elif task == 'segment':
            loss = loss_fn(out, segment_label)
            correct_sum += cal_segment_acc(out, segment_label)

        loss.backward()

        optimizer.step()
        loss_sum += loss

    print("%d epoch's loss %.3f" % (_, loss_sum))

    ''' 
    evaluation 
    '''
    if task == 'classify':
        test_count = 0
        test_correct = 0
        pred = []
        gt = []
        model.eval()
        for testdata in tqdm(testdataset, desc='Test'):
            if args.dataset_name == 'ARIL':
                amp, pha, label, segment_label, detection_label, time = testdata
                amp = amp.to(device)
                label = onehot(1, config.num_class, label).to(device)
            elif args.dataset_name == 'WiAR':
                amp, label, detection_label, segment_label = testdata
                amp = amp.to(device)
                label = onehot_first0(1, config.num_class, label).to(device)
            else:
                raise ValueError("There are no classify tasks in the dataset")

            outputs = model(amp)
            p = int(outputs[0].argmax(dim=-1).data.cpu())
            g = int(label.argmax(dim=-1).data.cpu())
            pred.append(p)
            gt.append(g)
            if p == g:
                test_correct += 1
            test_count += 1
        print("Classify --> Test:  %d epoch's acc is %.3f" % (_ + 1, 100 * (test_correct / test_count)), "%")
        if 100 * (test_correct / test_count) > classify_max_result:
            classify_max_result = 100 * (test_correct / test_count)
            classify_matrix = confusion_matrix(gt, pred, labels=[1, 2, 3, 4, 5, 6])
            np.save('outputs/{}/{}_{}_matrix.npy'.format(task, args.dataset_name, args.model_name), classify_matrix)

            torch.save(model.state_dict(), "outputs/{}/{}_{}.pth".format(task, args.dataset_name, model_name))

    if task == 'detection':
        F_sum = 0
        Mae_sum = 0
        test_count = 0
        start_errors = []
        end_errors = []
        model.eval()

        for testdata in tqdm(testdataset, desc='Test'):
            if args.dataset_name == 'ARIL':
                amp, pha, label, segment_label, detection_label, time = testdata
                amp = amp.to(device)
                detection_label = detection_label.to(device)
            elif args.dataset_name == 'WiAR':
                amp, label, detection_label, segment_label = testdata
                amp = amp.to(device)
                detection_label = detection_label.to(device)
            else:
                amp, detection_label, cla = data
                amp = amp.to(device)
                detection_label = detection_label.to(device)

            out = model(amp)
            start_error, end_error = detection_start_end(out, detection_label, config.sample_rate)
            start_errors.append(start_error)
            end_errors.append(end_error)
            # F_sum += cal_F(out, detection_label)
            # Mae_sum += cal_MAE(out, detection_label)
            test_count += 1

        mean_start_error = sum(start_errors) / test_count
        mean_end_error = sum(end_errors) / test_count

        print(mean_start_error + mean_end_error)
        if mean_start_error + mean_end_error < detection_min_error:
            detection_min_error = mean_start_error + mean_end_error
            torch.save(model.state_dict(),
                       "outputs/{}/{}_{}.pth".format(task, args.dataset_name, model_name))
            with open('outputs/{}/{}_{}_starterror.data'.format(task, args.dataset_name, model_name),
                      'wb') as f:
                pickle.dump(start_errors, f)
            with open('outputs/{}/{}_{}_enderror.data'.format(task, args.dataset_name, model_name),
                      'wb') as f:
                pickle.dump(end_errors, f)



        # print("Test: F is %.3f" % (F_sum / test_count))
        # print("Test: MAE is %.3f" % (Mae_sum / test_count))
        #
        # if round(float((Mae_sum / test_count)), 3) < detection_min_mae:
        #     detection_min_mae = round(float((Mae_sum / test_count)), 3)
        #     torch.save(model.state_dict(), "{}_{}_{}.pth".format(task, args.dataset_name, model_name))
        #     with open('{}_{}_starterror.data'.format(args.dataset_name, model_name), 'wb') as f:
        #         pickle.dump(start_errors, f)
        #     with open('{}_{}_enderror.data'.format(args.dataset_name, model_name), 'wb') as f:
        #         pickle.dump(end_errors, f)

    if task == 'segment':
        print("Train: %d epoch's acc is %.3f" % (_ + 1, 100 * (correct_sum / len(dataset.dataset))), "%")
        test_count = 0
        test_correct = 0
        accs = []
        model.eval()
        for testdata in tqdm(testdataset, desc='Test'):
            if args.dataset_name == 'ARIL':
                amp, pha, label, segment_label, detection_label, time = testdata
                amp = amp.to(device)
                segment_label = segment_onehot(segment_label, config.num_class).to(device)
            elif args.dataset_name == 'WiAR':
                amp, label, detection_label, segment_label = testdata
                amp = amp.to(device)
                segment_label = segmentonehot_negone(segment_label, 8).to(device)
            else:
                raise ValueError("There are no classify tasks in the dataset")

            out = model(amp)
            p = out.argmax(dim=-1)
            g = segment_label.argmax(dim=-1)
            p = p.transpose(1, 0)
            g = g.transpose(1, 0)
            p = p.data.cpu().numpy()
            g = g.data.cpu().numpy()

            result = p - g
            frame_correct = len(result[result==0]) / amp.shape[2]
            #frame_correct = frame_correct.data.cpu().numpy()
            accs.append(frame_correct)
            # result = p - g
            # frame_correct = len(result[result == 0]) / amp.shape[2]
            # test_correct += frame_correct

            test_count += 1

        print("Segment --> Test: %d epoche's acc is %.3f" % (_ + 1, 100 * (sum(accs) / test_count)), "%")
        if 100 * sum(accs) / test_count > segment_max_result:
            segment_max_result = 100 * sum(accs) / test_count
            torch.save(model.state_dict(), "outputs/{}/{}_{}.pth".format(task, args.dataset_name, model_name))
            with open('outputs/{}/{}_{}_segment_accs.data'.format(task, args.dataset_name, model_name), 'wb') as f:
                pickle.dump(accs, f)

    print("----------------------------------------------------------------")
    print(" ")


