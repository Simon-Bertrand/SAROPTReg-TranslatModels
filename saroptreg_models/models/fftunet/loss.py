import torch
from torch import nn


class FFTUnetLoss(nn.Module):
    def __init__(self, templateSize, padding="valid"):
        super(FFTUnetLoss, self).__init__()
        self.tH, self.tW = templateSize
        self.padWl = (self.tW - 1) // 2
        self.padHt = (self.tH - 1) // 2
        self.padding = padding

    def unpad_if_needed(self, cond, truthMask):
        if cond:
            return truthMask[
                ...,
                self.padHt : self.padHt - self.tH + truthMask.size(-2) + 1,
                self.padWl : self.padWl - self.tW + truthMask.size(-1) + 1,
            ]
        else:
            return truthMask

    def forward(self, ypred, truthMask):
        # truthMask : Boolean (B, 1, H, W)
        # ypred : (B, 1, H-h+1, W-w+1)
        # returns (B,1)

        return (
            -(
                torch.nn.functional.log_softmax(
                    self.unpad_if_needed(
                        ypred.shape[-2:] == truthMask.shape[-2:],
                        ypred,
                    ).flatten(-2, -1),
                    dim=-1,
                ).where(
                    self.unpad_if_needed(True, truthMask).flatten(-2, -1),
                    0,
                )
            )
        ).sum(dim=-1)
