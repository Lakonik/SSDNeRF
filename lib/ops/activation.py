import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd


class _trunc_exp(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # cast to float32
    def forward(ctx, x):
        exp_x = torch.exp(x)
        ctx.save_for_backward(exp_x)
        return exp_x

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        exp_x = ctx.saved_tensors[0]
        return g * exp_x.clamp(min=1e-6, max=1e6)


trunc_exp = _trunc_exp.apply


class TruncExp(nn.Module):
    # def __init__(self):
    #     super(TruncExp, self).__init__(eps=1e-6)
    #     self.eps = eps
    #     self.rec_eps = 1 / eps
    #     self.log_eps = math.log(eps)

    @staticmethod
    def forward(x):
        # return torch.where(
        #     x < self.log_eps,
        #     self.eps / (self.log_eps + 1 - x),
        #     torch.where(
        #         x < -self.log_eps,
        #         torch.exp(x),
        #         self.rec_eps * (self.log_eps + 1 + x)
        #     )
        # )
        return _trunc_exp.apply(x)
