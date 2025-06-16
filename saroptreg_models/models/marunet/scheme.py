from typing import Tuple

from models.saroptreg_models.scheme import GtGeneratorTrainingSchemeTemplate


from .loss import MARUNetLoss
from .mask import MARUNetMask


class MARUnetTrainingScheme(GtGeneratorTrainingSchemeTemplate):
    def __init__(
        self,
        model,
        rGt: float = 2.0,
        nNhs: int = 16,
        stdSoft: float = 1.0,
    ) -> None:
        super().__init__(
            model=model,
            loss=MARUNetLoss(nNhs),
            dodgeBorder=True,
            maskGenerator=MARUNetMask(rGt, stdSoft),
        )
