import torch

from ..utils.training_scheme import TrainingSchemeOutput


class MeanDist(torch.nn.Module):
    def __init__(self, ord=2):
        super().__init__()
        self.ord = ord

    def forward(self, training: TrainingSchemeOutput):
        return torch.linalg.norm(
            (training.ijPred - training.ijGTruth).float(),
            ord=self.ord,
            dim=-1,
        ).mean()
