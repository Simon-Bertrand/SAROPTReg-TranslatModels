from typing import Tuple

from models.saroptreg_models.masks import HardMask
from models.saroptreg_models.scheme import GtGeneratorTrainingSchemeTemplate


from .loss import FFTUnetLoss


class FFTUnetTrainingScheme(GtGeneratorTrainingSchemeTemplate):
    def __init__(
        self,
        model,
        templateSize: Tuple[int, int],
    ) -> None:
        super().__init__(
            model=model,
            loss=FFTUnetLoss(templateSize),
            maskGenerator=HardMask(),
            dodgeBorder=True,
            padding="valid",
        )
