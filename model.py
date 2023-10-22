import torch
import torch.nn as nn

from models.unetpp import UNetPP
from models.fcn import FCN
from models.unet import UNet

from heads import Classify_Head, Detection_Head, Segment_Head, Detection_Head_Gaussian



class WholeNet(nn.Module):
    def __init__(self, model_name, in_channel, num_class, unet_depth, unetpp_depth, task, detection_gaussian):
        super(WholeNet, self).__init__()
        self.task = task

        self.model = None
        self.model_name = model_name
        if model_name == 'unetpp':
            self.model = UNetPP(in_channels=in_channel)
        elif model_name == "fcn":
            self.model = FCN(in_channel=in_channel, dim=64)
        elif model_name == 'unet':
            self.model = UNet(in_channel=in_channel, out_channel=64, depth=unet_depth)


        self.classifyhead = Classify_Head(in_channel=64, num_classes=num_class)
        if detection_gaussian:
            self.locationhead = Detection_Head_Gaussian()
        else:
            self.locationhead = Detection_Head()
        self.frameclassifyhead = Segment_Head(in_channel=64, num_classes=num_class)



        self.L = unetpp_depth

    def forward(self, data):
        if self.model_name == 'unetpp':
            output = self.model(data, self.L)
        else:
            output = self.model(data)
        if self.task == 'classify':
            output = self.classifyhead(output)
        elif self.task == 'detection':
            output = self.locationhead(output)
        elif self.task == 'segment':
            output = self.frameclassifyhead(output)
        return output
