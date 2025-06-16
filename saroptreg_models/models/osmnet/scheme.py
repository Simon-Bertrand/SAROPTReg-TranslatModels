from typing import Tuple

from models.saroptreg_models.masks import CircleMask
from models.saroptreg_models.models.osmnet.loss import OSMNetLoss
from models.saroptreg_models.scheme import GtGeneratorTrainingSchemeTemplate


class OSMNetTrainingScheme(GtGeneratorTrainingSchemeTemplate):
    def __init__(
        self,
        model,
        templateSize: Tuple[int, int],
        rGt: float = 1.0,
    ) -> None:
        super().__init__(
            model=model,
            loss=OSMNetLoss(model, templateSize=templateSize),
            dodgeBorder=True,
            maskGenerator=CircleMask(rGt),
        )
