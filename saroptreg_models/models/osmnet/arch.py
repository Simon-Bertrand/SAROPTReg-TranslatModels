from functools import lru_cache
import logging
from typing import Any, Literal

import torch
from torch import nn


from .blocks import BasicBlock


class SSD(torch.nn.Module):
    def __init__(self, padding: Literal["same", "valid"] = "same"):
        super().__init__()
        self.padding = padding

    @lru_cache(maxsize=1)
    def get_padding_slice(self, x_shape, y_shape):
        match self.padding:
            case "same":
                hH = y_shape[-2] // 2
                hW = y_shape[-1] // 2
                validSupportI = slice(hH, -(hH) + x_shape[-2] % 2)
                validSupportJ = slice(hW, -(hW) + x_shape[-1] % 2)
            case "valid":
                validSupportI = slice(
                    y_shape[-2] - 1,
                    -(y_shape[-2]) + y_shape[-2] % 2 + x_shape[-2] % 2,
                )
                validSupportJ = slice(
                    y_shape[-1] - 1,
                    -(y_shape[-1]) + y_shape[-1] % 2 + x_shape[-1] % 2,
                )
            case _:
                raise ValueError(f"padding={self.padding} not supported")
        return validSupportI, validSupportJ

    def forward(self, x, y):
        # Optical and SAR Image Matching Using Pixelwise Deep Dense Features - Zhan and al.
        # DOI : 10.1109/LGRS.2020.3039473
        # logging.warning(x.shape, y.shape)
        # assert x.size(-3) == y.size(
        #     -3
        # ), "SSD expects the same number of channels for both x,y"
        validSupportI, validSupportJ = self.get_padding_slice(x.shape[1:], y.shape[2:])
        return (
            torch.fft.irfft2(
                (
                    torch.fft.rfft2(
                        x.pow(2),
                        s=(
                            padded_shape := (
                                (
                                    x.size(-2)
                                    + y.size(-2)
                                    - x.size(-2) % 2
                                    - y.size(-2) % 2
                                ),
                                (
                                    x.size(-1)
                                    + y.size(-1)
                                    - x.size(-1) % 2
                                    - y.size(-1) % 2
                                ),
                            )
                        ),
                    )
                    * torch.fft.rfft2(
                        torch.ones(y.shape[-2:], device=x.device), s=padded_shape
                    )
                ).sum(1, keepdim=True)
            )[..., validSupportI, validSupportJ]
            - 2
            * torch.fft.irfft2(
                (
                    (
                        torch.fft.rfft2(x, s=padded_shape)
                        * torch.fft.rfft2(torch.flip(y, dims=(-1, -2)), s=padded_shape)
                    ).sum(1, keepdim=True)
                )
            )[..., validSupportI, validSupportJ]
        )


class L2NormDense(nn.Module):
    def __init__(self):
        super(L2NormDense, self).__init__()
        self.eps = 1e-10

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x * x, dim=1) + self.eps)
        x = x / norm.unsqueeze(1).expand_as(x)
        return x


class SSLCNet(nn.Module):
    def __init__(self, block, num_block):
        super().__init__()

        self.in_channels = 32

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.conv2_x = self._make_layer(block, 32, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 64, num_block[1], 1)
        self.conv4_x = self._make_layer(block, 128, num_block[2], 1)

        self.conv2 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(160, 9, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(9),
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (
            x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, x):
        output1 = self.conv1(self.input_norm(x))
        output2 = self.conv2_x(output1)
        output = self.conv3_x(output2)
        output = self.conv_cat(torch.cat([output, output1], 1))
        output = self.conv4_x(output)
        output = self.conv2(torch.cat([output, output2], 1))

        return output


class SSLCNetPseudo(nn.Module):
    """SSLCNetPseudo model definition"""

    def __init__(self, siamese=False):
        super(SSLCNetPseudo, self).__init__()
        self.ResNet_Opt = SSLCNet(BasicBlock, [1, 1, 1, 1])
        self.ResNet_Sar = (
            SSLCNet(BasicBlock, [1, 1, 1, 1]) if not siamese else self.ResNet_Opt
        )

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return (
            x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        ) / sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x)

    def forward(self, input_opt, input_sar):
        input_opt = self.input_norm(input_opt)
        input_sar = self.input_norm(input_sar)
        features_opt = self.ResNet_Opt(input_opt)
        features_sar = self.ResNet_Sar(input_sar)

        return L2NormDense()(features_opt), L2NormDense()(features_sar)


class OSMNetSSD(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.ssd = SSD(**kwargs)

    def forward(self, input_opt, input_sar):
        return 1 - self.ssd(L2NormDense()(input_opt), L2NormDense()(input_sar)) / (
            input_sar.size(-2) * input_sar.size(-1)
        )


class OSMNet(nn.Module):
    def __init__(
        self,
        siamese=False,
        temp_factor=None,
        margin=0.25,
        gamma=32,
    ):
        super(OSMNet, self).__init__()
        self.net = SSLCNetPseudo(siamese=siamese)
        self.cross_metric = OSMNetSSD(padding="same")
        self.siamese = siamese
        if temp_factor is not None:
            logging.warning(
                "Temp_factor is not None for a model which isn't using it. It will be ignored."
            )
        self.margin = margin
        self.gamma = gamma

    def backbone(self, im, template):
        return self.net(im, template.repeat((1, 3, 1, 1)))

    def cross_op(self, dopt, dsar):
        return self.cross_metric(dopt, dsar).mean(dim=-3, keepdim=True)

    def forward(self, opt, sar):
        return self.cross_op(
            *self.backbone(opt, sar),
        )


class OSMNetTrainableGamma(OSMNet):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **(
                kwargs
                | dict(gamma=nn.Parameter(torch.tensor(32.0), requires_grad=True))
            ),
        )
