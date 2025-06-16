from typing import List

import torch

from ..utils.training_scheme import TrainingSchemeOutput


class CorrectMatchingRate(torch.nn.Module):
    def __init__(self, matchingThreshold: List[int] = [1, 2, 3], ord=2):
        super().__init__()
        self.matchingThreshold = torch.tensor(
            matchingThreshold, dtype=torch.float32
        ).unsqueeze(1)
        self.ord = ord

    def forward(self, training: TrainingSchemeOutput):
        self.matchingThreshold = self.matchingThreshold.to(training.im.device)
        return (
            (
                torch.linalg.norm(
                    (training.ijPred - training.ijGTruth).float(),
                    ord=self.ord,
                    dim=-1,
                ).unsqueeze(0)
                <= self.matchingThreshold
            )
            .float()
            .mean(-1)
        )


# class CorrectMatchingRate(torch.nn.Module):
#     def __init__(self, maxThreshhold=3):
#         super().__init__()
#         self.ijPred = []
#         self.ijTrue = []
#         self.maxThreshhold = maxThreshhold

#     def append(self, ijPred, ijTrue):
#         assert (
#             len(ijPred) == len(ijTrue) == 2
#         ), f"CMR : Wrong input size during appending, got {len(ijPred)=} and {len(ijTrue)=}"
#         assert all(pred[0] > 0 and pred[1] > 0 for pred in ijPred) and all(
#             true[0] > 0 and true[1] > 0 for true in ijTrue
#         ), 3
#         f"CMR : Wrong input values during appending, got {ijPred=} and {ijTrue=}"
#         self.ijPred.append(ijPred)
#         self.ijTrue.append(ijTrue)

#     def distrib(self):
#         ans = self.forward()
#         return dict(
#             mean=ans.mean(dim=0),
#             std=ans.std(dim=0),
#             min=ans.min(dim=0),
#             max=ans.max(dim=0),
#             median=ans.median(dim=0),
#         )

#     def forward(self):
#         return torch.linalg.norm(
#             torch.Tensor(self.ijPred) - torch.Tensor(self.ijTrue), dim=1
#         ).unsqueeze(1) <= torch.arange(1, self.maxThreshhold + 1).unsqueeze(0)
