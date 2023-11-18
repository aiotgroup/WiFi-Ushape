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


from functions import start_end_gaussian
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
    parser.add_argument('--task', type=str, default='detection',
                        help='choose target of this task: {classify} / {detection} / {segment}')

    parser.add_argument('--dataset_name', type=str, default='ARIL',
                        help='{HTHI} / {WiAR} / {ARIL}')
    parser.add_argument('--train_dataset_path', type=str, default='process_label/TrainDataset_aril_gauss.mat',
                        help='train dataset path')
    parser.add_argument('--test_dataset_path', type=str, default='process_label/TestDataset_aril_gauss.mat',
                        help='test dataset path')

    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--detection_gaussian', type=str, default="Yes")
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
detection_gaussian = args.detection_gaussian
filename = args.train_dataset_path
testfilename = args.test_dataset_path


config = Config(dataset_name=args.dataset_name)
# sample_rate = None
#
# if args.dataset_name == 'ARIL':
#     sample_rate = 60
# elif args.dataset_name == 'WiAR':
#     sample_rate = 100
# else:
#     sample_rate = 160


def model_opt_lossfn(model_name, lr, in_channel, num_class, segment_class, unet_depth, unetpp_depth, task, detection_gaussian):
    model = WholeNet(model_name=model_name, in_channel=in_channel, num_class=num_class, segment_class=segment_class, unet_depth=unet_depth,
                     unetpp_depth=unetpp_depth, task=task, detection_gaussian=detection_gaussian).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='mean')

    return model, optimizer, loss_fn


# loading data
dataset = getdataloader(dataset_name=args.dataset_name, filepath=filename, batch_size=batch_size, trainortest='train', detection_gaussian=detection_gaussian)
testdataset = getdataloader(dataset_name=args.dataset_name, filepath=testfilename, batch_size=1, trainortest='test', detection_gaussian=detection_gaussian,
                            shuffle=False)

# model, optimizer, loss_function
model, optimizer, loss_fn = model_opt_lossfn(model_name, lr, config.in_channel, config.num_class, config.segment_class, config.unet_depth, config.unetpp_depth, task, detection_gaussian=detection_gaussian)

# training, testing/evaluating

classify_max_result = 0
classify_matrix = None
detection_min_error = 100
segment_max_result = 0
amp, label, detection_label, segment_label = None, None, None, None

if not os.path.exists("Detection_Gaussian/{}".format(args.model_name)):
    os.makedirs("Detection_Gaussian/{}".format(args.model_name))

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
            amp, label = data
            detection_label = label
            amp = amp.to(device)
            label = label.to(device)
            detection_label = detection_label.to(device)

        elif args.dataset_name == 'WiAR':
            amp, label = data
            detection_label = label
            amp = amp.to(device)
            label = label.to(device)
            detection_label = detection_label.to(device)

        else:
            amp, label = data
            detection_label = label
            amp = amp.to(device)
            label = label.to(device)
            detection_label = detection_label.to(device)

        out = model(amp)
        loss = loss_fn(out, detection_label.squeeze())

        loss.backward()

        optimizer.step()
        loss_sum += loss

    print("%d epoch's loss %.3f" % (_, loss_sum))

    ''' 
    evaluation 
    '''

    F_sum = 0
    Mae_sum = 0
    test_count = 0
    start_errors = []
    end_errors = []
    model.eval()

    for testdata in tqdm(testdataset, desc='Test'):
        if args.dataset_name == 'ARIL':
            amp, detection_label = testdata
            amp = amp.to(device)

        elif args.dataset_name == 'WiAR':
            amp, detection_label = testdata
            amp = amp.to(device)
        else:
            amp, detection_label = testdata
            amp = amp.to(device)

        out = model(amp)
        out = out.data.cpu().numpy()
        gt = detection_label.data.cpu().numpy()

        error_start, error_end = start_end_gaussian(out, gt, config.sample_rate)
        start_errors.append(error_start)
        end_errors.append(error_end)

        test_count += 1

    mean_start_error = sum(start_errors) / test_count
    mean_end_error = sum(end_errors) / test_count

    print(mean_start_error + mean_end_error)
    if mean_start_error + mean_end_error < detection_min_error:
        detection_min_error = mean_start_error + mean_end_error
        torch.save(model.state_dict(),
                   "Detection_Gaussian/{}/{}_{}_{}.pth".format(model_name, args.dataset_name, args.index, model_name))
        with open('Detection_Gaussian/{}/{}_{}_{}_starterror.data'.format(model_name, args.dataset_name, args.index, model_name),
                  'wb') as f:
            pickle.dump(start_errors, f)
        with open('Detection_Gaussian/{}/{}_{}_{}_enderror.data'.format(model_name, args.dataset_name, args.index, model_name),
                  'wb') as f:
            pickle.dump(end_errors, f)

