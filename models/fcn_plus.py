import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN(nn.Module):
    def __init__(self, in_channel, dim):
        super(FCN, self).__init__()
        self.stage1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channel, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=64),
            nn.MaxPool1d(kernel_size=2, padding=0)
        )

        self.stage2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.MaxPool1d(kernel_size=2, padding=0)
        )

        self.stage3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=256),

            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
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

            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=512),

            nn.MaxPool1d(kernel_size=2, padding=0)
        )

        self.stage5 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=1024),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=1024),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=1024),
            nn.MaxPool1d(kernel_size=2, padding=0)

        )
        self.stage6 = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=dim),
            nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=dim),
            nn.MaxPool1d(kernel_size=2, padding=0)
        )

        self.upsample_2 = nn.ConvTranspose1d(in_channels=512, out_channels=512, kernel_size=4, padding=1, stride=2)
        self.upsample_4 = nn.ConvTranspose1d(in_channels=1024, out_channels=1024, kernel_size=4, padding=0, stride=4)
        self.upsample_8_1 = nn.ConvTranspose1d(in_channels=dim, out_channels=dim, kernel_size=4, padding=0, stride=4)
        self.upsample_8_2 = nn.ConvTranspose1d(in_channels=dim, out_channels=dim, kernel_size=4, padding=1, stride=2)

        self.upsample_81 = nn.ConvTranspose1d(in_channels=1024 + 512 + dim + 256, out_channels=1024 + 512 + dim + 256,
                                              kernel_size=4, padding=0, stride=4)
        self.upsample_82 = nn.ConvTranspose1d(in_channels=1024 + 512 + dim + 256, out_channels=1024 + 512 + dim + 256,
                                              kernel_size=4, padding=1, stride=2)

        self.final = nn.Sequential(
            nn.Conv1d(1024 + 512 + dim + 256, dim, kernel_size=7, padding=3),
        )

    def forward(self, x):
        seq_len = x.shape[2]
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        pool3 = x
        x = self.stage4(x)
        pool4 = self.upsample_2(x)
        if pool4.shape[2] != pool3.shape[2]:
            pool4 = F.interpolate(pool4, pool3.shape[2], mode='linear', align_corners=True)
        x = self.stage5(x)
        pool5 = self.upsample_4(x)
        if pool5.shape[2] != pool3.shape[2]:
            pool5 = F.interpolate(pool5, pool3.shape[2], mode='linear', align_corners=True)
        x = self.stage6(x)
        pool6 = self.upsample_8_1(x)
        pool6 = self.upsample_8_2(pool6)

        if pool6.shape[2] != pool3.shape[2]:
            pool6 = F.interpolate(pool6, pool3.shape[2], mode='linear', align_corners=True)

        x = torch.cat([pool3, pool4, pool5, pool6], dim=1)
        output = self.upsample_81(x)
        output = self.upsample_82(output)
        if output.shape[2] != seq_len:
            output = F.interpolate(output, seq_len, mode='linear', align_corners=True)
        output = self.final(output)
        return output
