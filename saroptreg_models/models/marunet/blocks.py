from torch import nn

from functools import partial

import torch
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )
        self.branch2 = nn.Conv2d(in_channels, out_channels, 1, padding="same")
        self.final = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ELU(), nn.Dropout(0.1))

    def forward(self, x):
        return self.final(self.branch1(x) + self.branch2(x))


class AttentionGates(nn.Module):
    def __init__(self, in_channels, gin_channels, common_channels):
        super(AttentionGates, self).__init__()
        self.xConv = nn.Conv2d(in_channels, common_channels, kernel_size=2, stride=2, bias=False)
        self.gConv = nn.Conv2d(gin_channels, common_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.psi = nn.Conv2d(common_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample = partial(F.interpolate, mode="bilinear")
        self.final = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels)
        )

    def forward(self, x, g):
        return self.final(
            self.upsample(
                self.sigmoid(self.psi(self.relu(self.xConv(x) + self.gConv(g)))), size=x.shape[-2:]
            )
            * x
        )


class ResidualUpsampling(nn.Module):
    def __init__(self, in_channels, gin_channels, out_channels):
        super(ResidualUpsampling, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(gin_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )
        self.attention = AttentionGates(in_channels, out_channels, out_channels)
        self.up = partial(F.interpolate, mode="bilinear")
        self.cat = partial(torch.cat, dim=1)
        self.final = ResidualBlock(in_channels + gin_channels, out_channels)

    def forward(self, x, g):
        return self.final(
            self.cat([self.up(g, size=x.shape[-2:]), self.attention(x, self.gate(g))])
        )
