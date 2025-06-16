import torch
from torch import nn


class OSMNetLoss(torch.nn.Module):
    def __init__(self, model, templateSize):
        super().__init__()
        self.model = model
        self.tH, self.tW = templateSize

        self.softplus = torch.nn.Softplus()
        self.padWl = (self.tW - 1) // 2
        self.padHt = (self.tH - 1) // 2

    def forward(self, ypred, truthMask):
        # truthMask : Boolean (B, 1, H, W)
        # ypred : (B, 1, H-h+1, W-w+1) corresponding to the SSD output valid padded
        # returns (B,1)
        paddingValidMask = torch.zeros_like(ypred).type(dtype=torch.bool)
        paddingValidMask[
            ...,
            self.padHt
            - 1 : self.padHt
            - self.tH
            + truthMask.size(-2)
            + 2,  # On rajoute 1 à droite et à gauche pour le circle mask à 5 éls
            self.padWl
            - 1 : self.padWl
            - self.tW
            + truthMask.size(-1)
            + 2,  # On rajoute 1 à droite et à gauche pour le circle mask à 5 éls
        ] = True
        ypred = torch.where(ypred.isinf(), torch.nan, ypred)

        gt_mapx = truthMask.type(dtype=torch.bool)
        sp = ypred[gt_mapx]
        gt_map_negx = (paddingValidMask & ~truthMask).type(
            dtype=torch.bool
        )  # On filtre sur la zone de padding avec la marge Circle Mask et on prend les NMR
        sn = ypred[gt_map_negx]

        sp = sp.view(ypred.size()[0], -1)
        sn = sn.view(ypred.size()[0], -1)

        ap = torch.clamp_min(-sp.detach() + 1 + self.model.margin, min=0.0)
        an = torch.clamp_min(sn.detach() + self.model.margin, min=0.0)

        delta_p = 1 - self.model.margin
        delta_n = self.model.margin

        logit_p = -ap * (sp - delta_p) * self.model.gamma
        logit_n = an * (sn - delta_n) * self.model.gamma

        soft_plus = nn.Softplus()
        loss_circle = soft_plus(
            torch.logsumexp(logit_n.where(~logit_n.isnan(), -torch.inf), dim=1)
            + torch.logsumexp(
                logit_p.where(
                    ~logit_p.isnan(),
                    -torch.inf,
                ),
                dim=1,
            )
        )
        return loss_circle
