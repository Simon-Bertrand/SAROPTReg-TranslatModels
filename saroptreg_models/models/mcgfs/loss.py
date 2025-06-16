import torch


class HardSampleLoss(torch.nn.Module):
    def __init__(self, nNhs: int = 40):
        super().__init__()
        self.nNhs = nNhs

    def forward(self, ypred, truthMask):
        # truthMask : (B, 1, H, W)
        # ypred : (B, 1, H, W)
        # returns (B, )
        ns = (
            (
                torch.where(truthMask, -torch.inf, ypred)
                .flatten(-2, -1)
                .topk(self.nNhs, dim=-1)
                .values.exp()
                + 1
            )
            .log()
            .mean((-2, -1))
        )

        ps = torch.where(truthMask, ((-ypred).exp() + 1).log(), 0).sum(
            (-2, -1)
        ) / truthMask.sum((-2, -1))
        return ns + ps
