from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, SfMDataset
from rfstudio.graphics import Cameras, RGBAImages, RGBImages
from rfstudio.loss import BasePhotometricLoss, ImageL1Loss, PSNRLoss
from rfstudio.model import VanillaNeuS
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


@dataclass
class VanillaNeuSTrainer(BaseTrainer):

    num_rays_per_batch: int = 1024
    lr: float = 5e-4
    eps: float = 1e-8
    loss: BasePhotometricLoss = ImageL1Loss()
    eikonal_weight: float = 0.1
    num_anneal_decay: Optional[int] = 50000

    def setup(
        self,
        model: VanillaNeuS,
        dataset: Union[MultiViewDataset, MeshViewSynthesisDataset, SfMDataset],
    ) -> ModuleOptimizers:
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict={
                'sdf_field': Optimizer(
                    torch.optim.Adam,
                    modules=model.sdf_field,
                    lr=self.lr,
                    eps=self.eps,
                    warm_up=5000,
                    lr_decay=295000,
                    lr_decay_mode='cos',
                ),
                'bg_field': Optimizer(
                    torch.optim.Adam,
                    modules=model.bg_field,
                    lr=self.lr,
                    eps=self.eps,
                    warm_up=5000,
                    lr_decay=295000,
                    lr_decay_mode='cos',
                ),
            }
        )

    def step(
        self,
        model: VanillaNeuS,
        inputs: Cameras,
        gt_outputs: Union[RGBAImages, RGBImages],
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        assert not (training and visual)

        if isinstance(gt_outputs, RGBAImages):
            gt_outputs = gt_outputs.blend((0, 0, 0))
        gt_outputs = gt_outputs.clamp(0, 1)

        if training:
            rays = inputs.generate_rays(self.num_rays_per_batch)
            outputs, eikonal_loss = model.render_rgb_along_rays(rays)
            eikonal_loss = eikonal_loss.mean()
            gt_pixels = gt_outputs.query_pixels(rays)
            l1_loss = self.loss._impl(gt_pixels, outputs)
            metrics = {
                'psnr': PSNRLoss()._impl(gt_pixels, outputs.detach()),
                'variance': model.sdf_field.get_variance().detach(),
                'eikonal_loss': eikonal_loss.detach(),
            }
            loss = l1_loss + eikonal_loss * self.eikonal_weight
        else:
            outputs = model.render_rgb(inputs)
            loss = self.loss(gt_outputs, outputs)
            psnr = PSNRLoss()(gt_outputs, outputs.detach())
            metrics = {
                'psnr': psnr,
                'variance': model.sdf_field.get_variance().detach(),
            }

        image = None
        if visual:
            image = torch.cat((outputs[0].item(), gt_outputs[0].item()), dim=1)      # [H 2W 3]
        return loss, metrics, image

    def before_update(self, model: VanillaNeuS, optimizers: Optimizer, *, curr_step: int) -> None:
        if self.num_anneal_decay is not None:
            model.cos_anneal_ratio = min(1.0, curr_step / self.num_anneal_decay)
