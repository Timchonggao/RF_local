from dataclasses import dataclass

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from rfstudio.loss import BaseLoss


@dataclass
class L2Loss(BaseLoss):

    def __call__(self, outputs: Tensor, gt_outputs: Tensor, reduction: str = "mean") -> Float[Tensor, "1"]:
        return F.mse_loss(outputs, gt_outputs, reduction=reduction)
        # reduction 参数控制输出的形式，默认值为 'mean'。
        ## 'mean': 对所有元素求平均，返回一个标量。
        ## 'sum': 对所有元素求和，返回一个标量。
        ## 'none': 不