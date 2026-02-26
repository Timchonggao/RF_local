from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from rfstudio.data.dataset import MultiView2DDataset
from rfstudio.graphics._2d import Cameras2D, CircleShape2D, RGBA2DImages, Viser2D
from rfstudio.model.density_field.tiny_nerf2d import TinyNeRF2D
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


@dataclass
class TinyNeRF2DTrainer(BaseTrainer):

    lr: float = 5e-4

    eps: float = 1e-8

    max_norm: Optional[float] = None

    lr_decay: Optional[float] = None

    def setup(
        self,
        model: TinyNeRF2D,
        dataset: MultiView2DDataset,
    ) -> ModuleOptimizers:
        self._gt_shape: CircleShape2D = dataset.get_meta(split='train')
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict={
                'params': Optimizer(
                    torch.optim.RAdam,
                    modules=model,
                    lr=self.lr,
                    eps=self.eps,
                    max_norm=self.max_norm,
                    lr_decay=self.lr_decay
                )
            }
        )

    def step(
        self,
        model: TinyNeRF2D,
        inputs: Cameras2D,
        gt_outputs: RGBA2DImages,
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:

        outputs = model.render_rgba(inputs)
        rgba = outputs._tensors
        gt_rgba = gt_outputs._tensors
        bg = torch.rand(3).to(rgba)
        rgb = rgba[..., :3] + (1 - rgba[..., 3:]) * bg
        gt_rgb = gt_rgba[..., :3] + (1 - gt_rgba[..., 3:]) * bg
        loss = torch.nn.functional.mse_loss(rgb, gt_rgb)
        metrics = { 'psnr': -10 * loss.detach().log10() }
        image = None
        if visual:
            with torch.no_grad():
                rays = inputs.generate_rays(downsample_to=8)
                row1 = torch.cat((
                    outputs.visualize(width=800, height=800).item().cpu(),
                    gt_outputs.visualize(width=800, height=800).item().cpu(),
                ), dim=1)
                row2 = torch.cat((
                    Viser2D().show(model).show(inputs).show(rays).get().blend((1, 1, 1)).item(),
                    Viser2D().show(self._gt_shape).show(inputs).show(rays).get().blend((1, 1, 1)).item(),
                ), dim=1)
                image = torch.cat((row1[..., :3] + (1 - row1[..., 3:]), row2), dim=0)
        return loss, metrics, image
