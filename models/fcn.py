import torch
import torch.nn as nn
import torch.nn.functional as F

def Unit_len(source, target):
    source = F.interpolate(source, target.shape[2], mode='linear', align_corners=True)
    return source

class FCN(nn.Module):
    def __init__(self, in_channel, dim):
        super(FCN,self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=96, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=96),
            nn.MaxPool1d(kernel_size=2, padding=0)
        )

        self.stage2 = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),
            nn.MaxPool1d(kernel_size=2, padding=0)
        )

        self.stage3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=384),

            nn.Conv1d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=384),

            nn.Conv1d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),

            nn.MaxPool1d(kernel_size=2, padding=0)
        )

        self.stage4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),

            nn.MaxPool1d(kernel_size=2, padding=0)
        )

        self.stage5 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=dim),

            nn.MaxPool1d(kernel_size=2, padding=0)
        )

        self.upsample_2 = nn.ConvTranspose1d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2)
        self.upsample_4 = nn.ConvTranspose1d(in_channels=dim, out_channels=dim, kernel_size=4, padding=0, stride=4)
        self.upsample_81 = nn.ConvTranspose1d(in_channels=512+dim+256, out_channels=512+dim+256, kernel_size=4, padding=0, stride=4)
        self.upsample_82 = nn.ConvTranspose1d(in_channels=512+dim+256, out_channels=512+dim+256, kernel_size=4, padding=1, stride=2)

        self.final = nn.Sequential(
            nn.Conv1d(512+dim+256, dim, kernel_size=7, padding=3),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        pool3 = x

        x = self.stage4(x)
        pool4 = self.upsample_2(x)
        if pool4.shape[2] != pool3.shape[2]:
            pool4 = Unit_len(pool4, pool3)
        
        x = self.stage5(x)
        conv7 = self.upsample_4(x)
        if conv7.shape[2] != pool4.shape[2]:
            conv7 = Unit_len(conv7, pool4)
            

        x = torch.cat([pool3, pool4, conv7], dim=1)
        output = self.upsample_81(x)
        output = self.upsample_82(output)
        output = self.final(output)
        return output
