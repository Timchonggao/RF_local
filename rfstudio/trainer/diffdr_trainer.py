from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.data import DepthSynthesisDataset
from rfstudio.graphics import Cameras, DepthImages
from rfstudio.graphics.shaders import NormalShader, PrettyShader
from rfstudio.model import DiffDR
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


def get_tv(x: Float32[Tensor, "H W 1"]) -> Float32[Tensor, "H W 1"]:
    return (x[:-1, :-1, :] - x[1:, :-1, :]).square() + (x[:-1, :-1, :] - x[:-1, 1:, :]).square()

@dataclass
class DiffDRTrainer(BaseTrainer):

    geometry_lr: float = 0.01

    base_eps: float = 1e-8

    sdf_reg_begin: float = 0.2
    sdf_reg_end: float = 0.01
    sdf_reg_decay: Optional[int] = 1250

    use_mask: bool = True

    def setup(
        self,
        model: DiffDR,
        dataset: DepthSynthesisDataset,
    ) -> ModuleOptimizers:
        assert isinstance(dataset, DepthSynthesisDataset)

        optim_dict = {
            'deforms': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='deforms'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                lr_decay=1500
            ),
            'sdfs': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='sdfs'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                lr_decay=1500
            ),
            'weights': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='weights'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                lr_decay=1500
            ),
        }
        self.real_geometry_lr = self.geometry_lr
        self.gt_geometry = dataset.get_meta(split='train')
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def step(
        self,
        model: DiffDR,
        inputs: Cameras,
        gt_depths: DepthImages,
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        depths, mesh, reg_loss = model.render_report(inputs)
        results = []
        for depth, gt_depth in zip(depths, gt_depths, strict=True):
            item = 10 * torch.nn.functional.l1_loss(
                depth[..., :1] * gt_depth[..., 1:],
                gt_depth[..., :1] * gt_depth[..., 1:],
            )
            if self.use_mask:
                item = item + torch.nn.functional.l1_loss(gt_depth[..., 1:], depth[..., 1:])
            results.append(item)

        loss = torch.stack(results).mean()
        metrics = {
            'geom-loss': loss.detach(),
            'reg-loss': reg_loss.detach(),
            'learning-rate': self.real_geometry_lr,
            '#vertex': mesh.num_vertices,
            '#triangle': mesh.num_faces,
        }
        image = None
        if visual:
            with torch.no_grad():
                row0 = torch.cat((
                    mesh.render(
                        inputs[0],
                        shader=PrettyShader(culling=False, z_up=model.z_up, occlusion_type='none'),
                    ).blend((1, 1, 1)).item(),
                    self.gt_geometry.render(
                        inputs[0],
                        shader=PrettyShader(culling=False, z_up=model.z_up, occlusion_type='none'),
                    ).blend((1, 1, 1)).item(),
                ), dim=1).clamp(0, 1)
                row1 = torch.cat((
                    mesh.render(
                        inputs[0],
                        shader=NormalShader(culling=False, normal_type='flat'),
                    ).visualize((1, 1, 1)).item(),
                    self.gt_geometry.render(
                        inputs[0],
                        shader=NormalShader(culling=False, normal_type='flat'),
                    ).visualize((1, 1, 1)).item(),
                ), dim=1).clamp(0, 1)
                image = torch.cat((row0, row1))
        return loss + reg_loss, metrics, image

    def before_update(self, model: DiffDR, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        if self.sdf_reg_decay is not None:
            model.sdf_weight = (
                self.sdf_reg_begin -
                (self.sdf_reg_begin - self.sdf_reg_end) * min(1.0, curr_step / self.sdf_reg_decay)
            )

    def after_update(self, model: DiffDR, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        self.real_geometry_lr = optimizers.optimizers['deforms'].param_groups[0]['lr']
