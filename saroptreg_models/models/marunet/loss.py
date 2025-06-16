import torch


class HardSampleBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ypred, ypredSoftmax, truthMask):
        return -torch.where(
            truthMask.flatten(-2, -1),
            torch.nn.functional.log_softmax(ypred.flatten(-2, -1), dim=(-1)),
            (1 - ypredSoftmax.flatten(-2, -1) + 1e-8).log(),
        ).sum(dim=(-1))


class SoftSampleLoss(torch.nn.Module):
    def __init__(self, nNhs: int = 16):
        super().__init__()
        self.nNhs = nNhs

    def forward(self, ypred, ypredSoftmax, truthMask):
        mr = (truthMask * ypredSoftmax).sum((-2, -1)) / truthMask.sum((-2, -1))
        nmr = (
            -(torch.where(truthMask > 0, -torch.inf, ypredSoftmax)
            .flatten(-2, -1)
            .topk(self.nNhs, dim=-1, largest=True)
            .values
            .mean(dim=-1))+1
        )
        return (-(mr - nmr)).clamp(min=0)


class MARUNetLoss(torch.nn.Module):
    def __init__(self, nNhs: int = 16):
        super().__init__()
        self.hardSampleBCELoss = HardSampleBCELoss()
        self.softSampleLoss = SoftSampleLoss(nNhs)

    def forward(self, ypred, truthMask):
        ypredSoftmax = torch.nn.functional.softmax(
            ypred.flatten(-2, -1), dim=(-1)
        ).unflatten(-1, ypred.shape[-2:])
        hardBce = self.hardSampleBCELoss(
            ypred,
            ypredSoftmax,
            truthMask[0],
        )
        soft = self.softSampleLoss(ypred, ypredSoftmax, truthMask[1])
        return hardBce + soft
