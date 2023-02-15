# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from ..builder import NECKS


def gem(x: Tensor, p: Parameter, eps: float = 1e-6, clamp=True) -> Tensor:
    if clamp:
        x = x.clamp(min=eps)  # clamp（）函数的功能将输入input张量每个元素的值压缩到区间 [min,max]，并返回结果到一个新张量
    return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
    # pow 计算两个张量或者一个张量与一个标量的指数计算结果,
    # 张量：作对应位置指数，标量：作所有位置的指数
    # avg_pool2d https://blog.csdn.net/qq_38417994/article/details/114290934

@NECKS.register_module()
class GeneralizedMeanPooling(nn.Module):  # 继承自nn.Module
    """Generalized Mean Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        p (float): Parameter value.
            Default: 3.
        eps (float): epsilon.
            Default: 1e-6
        clamp (bool): Use clamp before pooling.
            Default: True
    """

    def __init__(self, p=3., eps=1e-6, clamp=True):
        assert p >= 1, "'p' must be a value greater then 1"
        super(GeneralizedMeanPooling, self).__init__()  # super: 调用父类中函数
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.clamp = clamp

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([
                gem(x, p=self.p, eps=self.eps, clamp=self.clamp)
                for x in inputs
            ])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = gem(inputs, p=self.p, eps=self.eps, clamp=self.clamp)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs
