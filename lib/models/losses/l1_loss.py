import torch
from mmgen.models.builder import MODULES
from mmgen.models.losses import L1Loss
from mmgen.models.losses.utils import weighted_loss


@weighted_loss
def l1_loss_mod(pred, target):
    if isinstance(target, int):
        if target == 0:
            loss = torch.abs(pred)
        elif target == -1:
            loss = pred
        else:
            raise ValueError
    else:
        loss = torch.abs(pred - target)
    return loss


@MODULES.register_module()
class L1LossMod(L1Loss):

    def forward(self, pred, target, weight=None,
                avg_factor=None, reduction_override=None,):
        reduction = reduction_override if reduction_override else self.reduction
        return l1_loss_mod(
            pred, target, weight=weight, reduction=reduction, avg_factor=avg_factor
        ) * self.loss_weight
