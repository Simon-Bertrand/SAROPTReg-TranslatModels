from abc import ABC, abstractmethod
from typing import List
import torch


class MaskGenerator(ABC):
    @abstractmethod
    def __call__(self, meshCentered: torch.Tensor) -> torch.Tensor | List[torch.Tensor]:
        pass


class HardMask(MaskGenerator):
    def __call__(self, meshCentered: torch.Tensor) -> torch.Tensor:
        return ((meshCentered[..., 0] == 0) & (meshCentered[..., 1] == 0)).unsqueeze(-3)


class CircleMask(MaskGenerator):
    def __init__(self, rGt: float = 1.0):
        self.rGt = rGt

    def __call__(self, meshCentered: torch.Tensor) -> torch.Tensor:
        return (meshCentered.norm(p=2, dim=-1) <= self.rGt).unsqueeze(-3)
