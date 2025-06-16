from typing import Any

import torch
from torch import nn


from .blocks import ResidualBlock, ResidualUpsampling

from torch_crosscorr import FastNormalizedCrossCorrelation


class AttentionUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUnet, self).__init__()
        self.downsample = nn.MaxPool2d(2)
        self.encoders = nn.ModuleList(
            [
                ResidualBlock(in_channels, 32),
                ResidualBlock(32, 64),
                ResidualBlock(64, 128),
                ResidualBlock(128, 256),
            ]
        )
        self.latent = ResidualBlock(256, 512)

        self.decoders = nn.ModuleList(
            [
                ResidualUpsampling(256, 512, 256),
                ResidualUpsampling(128, 256, 128),
                ResidualUpsampling(64, 128, 64),
                ResidualUpsampling(32, 64, out_channels),
            ]
        )

    def forward(self, x):
        xres_1 = self.encoders[0](x)
        xres_2 = self.encoders[1](self.downsample(xres_1))
        xres_3 = self.encoders[2](self.downsample(xres_2))
        xres_4 = self.encoders[3](self.downsample(xres_3))

        x = self.downsample(xres_4)
        g = self.latent(x)
        g = self.decoders[0](xres_4, g)
        g = self.decoders[1](xres_3, g)
        g = self.decoders[2](xres_2, g)
        g = self.decoders[3](xres_1, g)

        return g


class MARUNet(nn.Module):
    def __init__(self, in_channels, siamese=True, temp_factor=None):
        super(MARUNet, self).__init__()
        self.in_channels = in_channels
        self.siamese = siamese
        self.network_opt = AttentionUnet(in_channels, 4)
        self.network_sar = self.network_opt if self.siamese else AttentionUnet(1, 4)

        self.cross_metric = FastNormalizedCrossCorrelation(
            "ncorr", "fft", tempFactor=temp_factor or 1 / 96
        )
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2, mode="bilinear")

    def expandTemplate(self, x):
        if self.siamese:
            return x.expand(-1, self.in_channels, -1, -1)
        else:
            return x

    def backbone(self, im, template):
        template = self.expandTemplate(template)
        opt = torch.cat(
            [
                self.network_opt(im),
                self.upscale(self.network_opt(self.downscale(im))),
            ],
            dim=1,
        )

        sar = torch.cat(
            [
                self.network_sar(template),
                self.upscale(self.network_sar(self.downscale(template))),
            ],
            dim=1,
        )
        return (
            opt,
            sar,
        )

    def cross_op(self, dopt, dsar):
        return self.cross_metric(dopt, dsar).mean(dim=-3, keepdim=True)

    def forward(self, opt, sar):
        return self.cross_op(
            *self.backbone(opt, sar),
        )
