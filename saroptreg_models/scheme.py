from dataclasses import dataclass
from functools import lru_cache, partial
import random
from typing import List, Literal, Tuple

import torch
from torch.utils.data import Dataset
from torch import nn
from models.saroptreg_models.masks import MaskGenerator


@dataclass
class TrainingSchemeOutput:
    im: torch.Tensor
    templates: torch.Tensor
    loss: torch.Tensor
    output: torch.Tensor
    outputInterest: torch.Tensor
    ijPred: torch.Tensor
    mask: torch.Tensor | List[torch.Tensor]
    ijGTruth: torch.Tensor
    meshCentered: torch.Tensor
    mode_im: List[str]
    mode_templates: List[str]


class TrainingScheme(torch.nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model


class MeshGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ij, H, W):
        coords = self._coords(H, W)
        meshCentered = coords.unsqueeze(0) - ij.unsqueeze(-2).unsqueeze(-2)
        return meshCentered

    @lru_cache(maxsize=1)
    def _coords(self, H, W):
        return torch.stack(
            torch.meshgrid(
                torch.arange(0, H, dtype=torch.float32),
                torch.arange(0, W, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        )


class SAROptRegDatasetWrapper(Dataset):
    def __init__(
        self,
        dataset,
        resize=None,
        crop_size=(128, 128),
        sar_speckle_L=None,
        normalize=True,
    ):
        self.dataset = dataset
        self.crop_size = crop_size

        self.speckle_gamma = (
            torch.distributions.Gamma(
                torch.tensor(sar_speckle_L, dtype=torch.float32),
                torch.tensor(sar_speckle_L, dtype=torch.float32),
            )
            if isinstance(sar_speckle_L, int)
            else None
        )

        self.opt_stats: dict = dict(
            mean=torch.tensor(
                [89.49467695321677, 94.27883753546004, 66.8573059945736],
                dtype=torch.float32,
            ),
            std=torch.tensor(
                [45.20039252682808, 33.16180912278351, 29.78417331436597],
                dtype=torch.float32,
            ),
        )
        self.sar_stats: dict = dict(
            mean=torch.scalar_tensor(-17.47252257294352),
            std=torch.scalar_tensor(2.3839467795959663),
        )

        self.normalize = normalize
        self.resize = resize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.resize is not None:
            for k in ["opt", "sar"]:
                sample[k] = nn.functional.interpolate(
                    sample[k].unsqueeze(0),
                    size=self.resize,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

        if self.speckle_gamma is not None:
            gamma = self.speckle_gamma.sample(sample["sar"].shape)
            sample["sar"] = (20 * gamma.log10() + sample["sar"]).clamp(-25, 5)
        ij, sample["sar"] = self._randomValidCrop(sample["sar"], *self.crop_size)
        if self.normalize:
            sample["opt"] = (
                sample["opt"] - self.opt_stats["mean"].reshape(3, 1, 1)
            ) / self.opt_stats["std"].reshape(3, 1, 1)
            sample["sar"] = (
                sample["sar"] - self.sar_stats["mean"].reshape(1, 1, 1)
            ) / self.sar_stats["std"].reshape(1, 1, 1)
        return sample | {"ij": ij}

    def _randomValidCrop(self, x, tH, tW, gen=None):
        gen = gen or random
        H, W = x.shape[-2:]
        pH = (tH - 1) // 2
        pW = (tW - 1) // 2
        i, j = (
            gen.randint(pH, H - pH - 2 + tH % 2),
            gen.randint(pW, W - pW - 2 + tW % 2),
        )
        sI, sJ = i - pH, j - pW
        crop = x[..., sI : sI + tH, sJ : sJ + tW]
        return torch.Tensor((i, j)), crop


class GtGeneratorTrainingSchemeTemplate(TrainingScheme):
    def __init__(
        self,
        model,
        loss: torch.nn.Module,
        maskGenerator: MaskGenerator,
        dodgeBorder: bool = False,
        padding="same",
    ) -> None:
        super().__init__(model=model)
        self.loss = loss
        self.maskGenerator = maskGenerator
        self.dodgeBorder = dodgeBorder
        self.dodgeBordersValue: float | bool = False
        self.predArg = partial(torch.argmax, dim=-1)
        self.dodgeBordersValue = -torch.inf if self.dodgeBorder else False
        self.padding = padding
        self.mesh_generator = MeshGenerator()

    def forward(self, batch: dict) -> TrainingSchemeOutput:
        assert isinstance(batch, dict), "Sample must be a MEOW Sample"
        output = self.model(batch["opt"], batch["sar"])
        tH, tW = (
            batch["sar"].size(-2),
            batch["sar"].size(-1),
        )

        padWl = (tW - 1) // 2
        padHt = (tH - 1) // 2
        padHb = padHt + 1 - tH % 2
        padWr = padWl + 1 - tW % 2
        out = output
        if self.padding == "valid":
            output_padded = torch.nn.functional.pad(
                output,
                (padWl, padWr, padHt, padHb),
                mode="constant",
                value=self.dodgeBordersValue,
            )
            out = output_padded
        with torch.no_grad():
            if self.dodgeBorder:
                borderMask = torch.zeros_like(
                    out, dtype=torch.bool, device=batch["opt"].device
                )
                borderMask[..., padHt:-padHb, padWl:-padWr] = True
            else:
                borderMask = torch.ones_like(
                    out, dtype=torch.bool, device=batch["opt"].device
                )

            predictedPos = self.predArg(
                torch.where(
                    borderMask,
                    out,
                    self.dodgeBordersValue,
                )
                .mean(dim=-3)
                .flatten(-2)
            )
            ijPred = torch.stack(
                [predictedPos // out.size(-1), predictedPos % out.size(-1)],
                dim=-1,
            )
        meshCentered = self.mesh_generator(batch["ij"], *batch["opt"].shape[-2:])
        truthMask = self.maskGenerator(meshCentered)
        if isinstance(truthMask, list):
            assert all(el.ndim == 4 for el in truthMask)
        elif isinstance(truthMask, torch.Tensor):
            assert truthMask.ndim == 4, "truthMask must be (B, 1, H, W)"
        return TrainingSchemeOutput(
            im=batch["opt"],
            templates=batch["sar"],
            loss=self.loss(out, truthMask).mean(),
            mask=truthMask,
            outputInterest=output,
            output=out,
            ijPred=ijPred,
            ijGTruth=batch["ij"],
            meshCentered=meshCentered,
            mode_im=batch["mode_opt"],
            mode_templates=batch["mode_sar"],
        )
