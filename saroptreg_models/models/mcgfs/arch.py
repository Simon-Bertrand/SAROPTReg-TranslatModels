import logging
import math
from typing import Any

import torch
from torch import nn

from torch_crosscorr import FastNormalizedCrossCorrelation


class CrossCorrMean(FastNormalizedCrossCorrelation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, im, template):
        return super().forward(im, template) / (template.size(-2) * template.size(-1))


class MCGF(nn.Module):
    def __init__(
        self,
        siamese=False,
    ):
        super().__init__()
        self.nAngles = 9
        self.siamese = siamese
        self.cross_metric = CrossCorrMean("corr", "fft", center=True)

        def ConvBranch(relu=True):
            modls = [
                nn.Sequential(
                    nn.Conv2d(in_chans, out_chans, 3, 1, 1),
                    nn.BatchNorm2d(out_chans),
                    nn.ReLU(),
                )
                for in_chans, out_chans in [
                    (self.nAngles, 32),
                    (32, 32),
                    (32, self.nAngles),
                ]
            ]
            if isinstance(modls[-1][-1], nn.ReLU) and not relu:
                modls[-1][-1] = nn.Identity()
            return nn.Sequential(*modls)

        relu_end = not (
            isinstance(self.cross_metric, FastNormalizedCrossCorrelation)
            and self.cross_metric.normalize
        )
        self.imConv = nn.ModuleDict(
            dict(up=ConvBranch(relu_end), down=ConvBranch(relu_end))
        )
        self.templateConv = (
            nn.ModuleDict(dict(up=ConvBranch(relu_end), down=ConvBranch(relu_end)))
            if not self.siamese
            else self.imConv
        )

        self.downsampling = torch.nn.AvgPool2d(2)
        self.upsampling = torch.nn.Upsample(scale_factor=2, mode="bilinear")

    def og(self, im: torch.Tensor):
        # Two methods are available.
        # We use the second schema using a cosine and sine decomposition
        # of the gradients
        grads = torch.zeros_like(im, dtype=torch.complex64, device=im.device)
        grads[..., 1:-1] = im[..., 2:] - im[..., :-2]
        grads[..., 1:-1, :] += 1j * (im[..., 2:, :] - im[..., :-2, :])
        angles = torch.linspace(-math.pi, math.pi, self.nAngles, device=im.device)
        return (
            (
                torch.cos(angles) * grads.real.unsqueeze(-1)
                + torch.sin(angles) * grads.imag.unsqueeze(-1)
            )
            .abs()
            .amax(-4)
            .moveaxis(-1, -3)
        )

    def backbone(self, opt, sar):
        sarOg = self.templateConv["up"](self.og(sar))
        sarOgDown = self.templateConv["down"](self.downsampling(self.og(sar)))
        optOg = self.imConv["up"](self.og(opt))
        optOgDown = self.imConv["down"](self.downsampling(self.og(opt)))
        return (
            (self.upsampling(optOgDown) + optOg) / 2,
            (self.upsampling(sarOgDown) + sarOg) / 2,
        )

    def cross_op(self, dopt, dsar):
        return self.cross_metric(dopt, dsar).mean(dim=-3, keepdim=True)

    def forward(self, opt, sar):
        return self.cross_op(
            *self.backbone(opt, sar),
        )
