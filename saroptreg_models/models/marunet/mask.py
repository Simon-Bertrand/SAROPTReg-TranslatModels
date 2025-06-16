from typing import List

import torch

from models.saroptreg_models.masks import MaskGenerator


class MARUNetMask(MaskGenerator):
    def __init__(self, rGt: float = 2.0, stdSoft: float = 1.0):
        self.rGt = rGt
        self.stdSoft = stdSoft

    def __call__(self, meshCentered: torch.Tensor) -> List[torch.Tensor]:
        maskHardSample = (meshCentered[..., 0] == 0) & (meshCentered[..., 1] == 0)

        euclidianNorm = torch.linalg.norm(meshCentered, dim=-1)

        maskSoftSample = torch.where(
            euclidianNorm < self.rGt,
            (1 / (2 * torch.pi * self.stdSoft))
            * (-euclidianNorm.pow(2) / (2 * self.stdSoft)).exp(),
            0,
        )
        return [maskHardSample.unsqueeze(-3), maskSoftSample.unsqueeze(-3)]
