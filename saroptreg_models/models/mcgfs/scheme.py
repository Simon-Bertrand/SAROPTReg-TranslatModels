from models.saroptreg_models.masks import CircleMask
from models.saroptreg_models.models.mcgfs.loss import HardSampleLoss
from models.saroptreg_models.scheme import GtGeneratorTrainingSchemeTemplate


class MCGFTrainingScheme(GtGeneratorTrainingSchemeTemplate):
    def __init__(
        self,
        model,
        rGt: float = 1.0,
        nNhs: int = 40,
    ) -> None:
        super().__init__(
            model=model,
            loss=HardSampleLoss(nNhs),
            maskGenerator=CircleMask(rGt),
            dodgeBorder=True,
        )
