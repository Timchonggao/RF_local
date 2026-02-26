from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset
from rfstudio.graphics import Cameras, RGBAImages
from rfstudio.graphics.shaders import NormalShader
from rfstudio.loss import MaskedPhotometricLoss, PSNRLoss, SSIML1Loss
from rfstudio.model.mesh_based.neufam import NeuFAM
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


@dataclass
class NeuFAMTrainer(BaseTrainer):

    grid_lr: float = 1e-3
    geometry_lr: float = 1e-3
    appearance_lr: float = 3e-4

    base_eps: float = 1e-8

    sdf_reg_begin: float = 0.2
    sdf_reg_end: float = 0.01
    sdf_reg_decay: Optional[int] = None

    def setup(
        self,
        model: NeuFAM,
        dataset: MeshViewSynthesisDataset,
    ) -> ModuleOptimizers:
        assert isinstance(dataset, MeshViewSynthesisDataset)

        optim_dict = {
            'grid': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='grid'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                lr_decay=1500
            ),
            'geometry': Optimizer(
                category=torch.optim.Adam,
                modules=model.geom_decoder,
                lr=self.geometry_lr,
                eps=self.base_eps,
                lr_decay=1500
            ),
            'appearance': Optimizer(
                category=torch.optim.Adam,
                modules=model.app_decoder,
                lr=self.appearance_lr,
                eps=self.base_eps,
                lr_decay=1500
            ),
        }
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def step(
        self,
        model: NeuFAM,
        inputs: Cameras,
        gt_outputs: RGBAImages,
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        bg_color = model.get_background_color().to(inputs.device)
        outputs, mesh, reg_loss = model.render_report(inputs)
        gt_rgb = gt_outputs.blend(bg_color)
        pred_rgb = outputs.detach().blend(bg_color)
        loss = MaskedPhotometricLoss(
            photometric_term=SSIML1Loss(),
        )(gt_outputs=gt_outputs, outputs=outputs)
        metrics = {
            'ssim-l1': loss.detach(),
            'psnr': PSNRLoss()(gt_rgb.clamp(0, 1), pred_rgb.clamp(0, 1)),
            'reg-loss': reg_loss.detach(),
            '#vertex': mesh.num_vertices,
            '#triangle': mesh.num_faces,
        }
        image = None
        if visual:
            with torch.no_grad():
                normal_map = mesh.render(inputs[0], shader=NormalShader(antialias=model.antialias))
                image = torch.cat((
                    gt_rgb.item(),
                    pred_rgb.detach().item(),
                    normal_map.visualize(bg_color).item(),
                ), dim=1).clamp(0, 1)
        return loss + reg_loss, metrics, image

    def before_update(self, model: NeuFAM, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        if self.sdf_reg_decay is not None:
            model.sdf_weight = (
                self.sdf_reg_begin -
                (self.sdf_reg_begin - self.sdf_reg_end) * min(1.0, curr_step / self.sdf_reg_decay)
            )
