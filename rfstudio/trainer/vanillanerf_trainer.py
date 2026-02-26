from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, SfMDataset
from rfstudio.graphics import Cameras, RGBAImages, RGBImages
from rfstudio.loss import BasePhotometricLoss, ImageL2Loss
from rfstudio.model import VanillaNeRF
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


@dataclass
class VanillaNeRFTrainer(BaseTrainer):

    num_rays_per_batch: int = 4096

    lr: float = 5e-4

    eps: float = 1e-8

    max_norm: Optional[float] = None

    lr_decay: Optional[float] = None

    loss: BasePhotometricLoss = ImageL2Loss()

    def setup(
        self,
        model: VanillaNeRF,
        dataset: Union[MultiViewDataset, MeshViewSynthesisDataset, SfMDataset],
    ) -> ModuleOptimizers:
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
        model: VanillaNeRF,
        inputs: Cameras,
        gt_outputs: Union[RGBAImages, RGBImages],
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        assert not (training and visual)

        if isinstance(gt_outputs, RGBAImages):
            gt_outputs = gt_outputs.blend(model.get_background_color())
        gt_outputs = gt_outputs.clamp(0, 1)

        if training:
            rays = inputs.generate_rays(self.num_rays_per_batch)
            coarse_outputs, fine_outputs = model.render_rgb_along_rays(rays)
            gt_pixels = gt_outputs.query_pixels(rays)
            coarse_loss = self.loss._impl(gt_pixels, coarse_outputs)
            fine_loss = self.loss._impl(gt_pixels, fine_outputs)
            metrics = {
                'coarse psnr': -10 * coarse_loss.detach().log10(),
                'fine psnr': -10 * fine_loss.detach().log10(),
            }
            loss = coarse_loss + fine_loss
        else:
            outputs = model.render_rgb(inputs)
            loss = self.loss(gt_outputs, outputs)
            metrics = { 'psnr': -10 * loss.detach().log10() }

        image = None
        if visual:
            image = torch.cat((outputs[0].item(), gt_outputs[0].item()), dim=1)      # [H 2W 3]
        return loss, metrics, image
