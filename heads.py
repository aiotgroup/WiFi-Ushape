import torch
import torch.nn as nn


class Classify_Head(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(Classify_Head, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=in_channel)
        )
        self.out = nn.Linear(in_channel, num_classes)

    def forward(self, datas):
        outputs = 0
        if isinstance(datas, tuple):
            for data in datas:
                out = self.conv(data)
                outputs += out
            outputs = outputs / len(datas)
        else:
            outputs = self.conv(datas)
        outputs = outputs.permute(0, 2, 1)
        outputs = self.out(outputs)
        outputs = outputs.permute(0, 2, 1)
        classify_output = outputs.mean(dim=-1)

        return classify_output


class Detection_Head(nn.Module):
    def __init__(self):
        super(Detection_Head, self).__init__()
    def forward(self, datas):
        outputs = 0
        if isinstance(datas, tuple):
            for data in datas:
                outputs += data
            outputs = outputs / len(datas)
        else:
            outputs = datas
        location_output = outputs.mean(dim=1)
        return location_output


class Detection_Head_Gaussian(nn.Module):
    def __init__(self, in_channel):
        super(Detection_Head_Gaussian, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=2)
        )

    def forward(self, datas):
        outputs = 0
        if isinstance(datas, tuple):
            for data in datas:
                outputs += data
            outputs = outputs / len(datas)
        else:
            outputs = datas

        outputs = self.conv(outputs)
        return outputs

class Segment_Head(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(Segment_Head, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=num_classes, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=num_classes)
        )

    def forward(self, datas):
        outputs = 0
        if isinstance(datas, tuple):
            for data in datas:
                outputs += self.conv(data)
            outputs = outputs / len(datas)
        else:
            outputs = self.conv(datas)
        frameclassfiy_output = outputs.permute(0, 2, 1).contiguous()
        return frameclassfiy_output
