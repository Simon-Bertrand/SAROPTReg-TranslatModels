from functools import partial
from typing import Any

import torch
import torchvision
from torch import nn

from torch_crosscorr import FastNormalizedCrossCorrelation


class FFTUnet(nn.Module):
    def __init__(self, siamese=True, tempFactor=None):
        # tempFactor sharpens the softmax output
        # It is only useful if ncc is used instead of cc
        # Smaller tempFactor will make the algorithm more decisive/discriminative
        # Model must return logits as we compute softmax later
        super(FFTUnet, self).__init__()
        self.siamese = siamese
        self.network_opt = FFTUnetSharedNet(3, 4)
        self.network_sar = self.network_opt if siamese else FFTUnetSharedNet(1, 4)
        self.cross_metric = FastNormalizedCrossCorrelation(
            "ncorr",
            "fft",
            dtype=torch.double,
            padding="valid",
            tempFactor=tempFactor or 1 / 96,
        )

    def expandSar(self, x):
        if self.siamese:
            return x.expand(-1, 3, -1, -1)
        else:
            return x

    def backbone(self, opt, sar):
        return self.network_opt(opt), self.network_sar(self.expandSar(sar))

    def cross_op(self, dopt, dsar):
        return self.cross_metric(dopt, dsar).mean(dim=-3, keepdim=True)

    def forward(self, opt, sar):
        return self.cross_op(
            *self.backbone(opt, sar),
        )


class FFTUnetSharedNet(nn.Module):
    def _upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def __init__(self, in_channels, out_channels):
        super(FFTUnetSharedNet, self).__init__()
        self.downsample = nn.MaxPool2d(2)
        self.merge_upsample = partial(torch.cat, dim=1)
        self.crop = torchvision.transforms.CenterCrop
        self.encoders = nn.ModuleList(
            [
                self._block(in_channels, 64),
                self._block(64, 128),
                self._block(128, 256),
            ]
        )

        self.decoders = nn.ModuleList(
            [
                self._block(256, 128),
                self._block(128, 64),
            ]
        )

        self.upsamplers = nn.ModuleList(
            [
                self._upsample(256, 128),
                self._upsample(128, 64),
            ]
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        xres_1 = self.encoders[0](x)
        x = self.downsample(xres_1)
        xres_2 = self.encoders[1](x)
        x = self.downsample(xres_2)
        x = self.encoders[2](x)
        up_1 = self.upsamplers[0](x)
        x = self.merge_upsample([self.crop(up_1.shape[-2:])(xres_2), up_1])
        x = self.decoders[0](x)
        up_2 = self.upsamplers[1](x)
        x = self.merge_upsample([self.crop(up_2.shape[-2:])(xres_1), up_2])
        x = self.decoders[1](x)
        res = self.final(x)
        return res


class FFTUnetTrainableTempFactor(FFTUnet):
    def __init__(self, siamese=True):
        tempFactor = nn.Parameter(torch.tensor(1 / 96), requires_grad=True)
        super().__init__(siamese=siamese, tempFactor=tempFactor)
