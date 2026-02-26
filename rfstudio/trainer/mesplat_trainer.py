from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset
from rfstudio.graphics import Cameras, RGBAImages
from rfstudio.graphics.shaders import NormalShader
from rfstudio.loss import PSNRLoss, SSIML1Loss
from rfstudio.model import MeshSplatter
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


@dataclass
class MeSplatTrainer(BaseTrainer):

    base_lr: float = 5e-3

    base_eps: float = 1e-15

    base_decay: int = 150

    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""

    reset_interval: int = 100000

    loss: SSIML1Loss = SSIML1Loss(ssim_lambda=0.2)

    def setup(
        self,
        model: MeshSplatter,
        dataset: MeshViewSynthesisDataset,
    ) -> ModuleOptimizers:
        assert isinstance(dataset, MeshViewSynthesisDataset)
        mesh = dataset.get_meta(split='train')
        model.reset_from_mesh(mesh)
        self._gt = mesh

        optim_dict = {
            'means': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='means'),
                lr=self.base_lr * 0.16,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
            'scales': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='scales'),
                lr=self.base_lr * 5,
                eps=self.base_eps,
            ),
            'quats': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='quats'),
                lr=self.base_lr,
                eps=self.base_eps,
            ),
            'colors': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='colors'),
                lr=self.base_lr * 2.5,
                eps=self.base_eps,
            ),
            'normals': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='normals'),
                lr=self.base_lr * 5,
                eps=self.base_eps,
            ),
        }
        if model.sh_degree > 0:
            optim_dict['shs'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='shs'),
                lr=self.base_lr * 0.125,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            )
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def step(
        self,
        model: MeshSplatter,
        inputs: Cameras,
        gt_outputs: RGBAImages,
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        gt_rgb = gt_outputs.blend(model.get_background_color()).clamp(0, 1)
        outputs = model.render_rgb(inputs)
        our_normal = model.render_normals(inputs)
        gt_normal = self._gt.render(
            inputs,
            shader=NormalShader(normal_type='flat'),
        ).visualize(model.get_background_color())
        loss = self.loss(gt_rgb, outputs) * 0.1 + self.loss(gt_normal, our_normal) * 0.9
        metrics = {
            'ssim': loss.detach(),
            'psnr-rgb': PSNRLoss()(gt_rgb, outputs.detach()),
            'psnr-normal': PSNRLoss()(gt_normal, our_normal.detach()),
            '#gaussians': model.gaussians.shape[0]
        }
        image = None
        if visual:
            row1 = torch.cat((outputs.item(), gt_rgb.item()), dim=1).clamp(0, 1)  # [H, 2W, 3]
            row2 = torch.cat((our_normal.item(), gt_normal.item()), dim=1).clamp(0, 1)  # [H, 2W, 3]
            image = torch.cat((row1, row2), dim=0)
        return loss, metrics, image

    def before_update(
        self,
        model: MeshSplatter,
        optimizers: ModuleOptimizers,
        *,
        curr_step: int,
    ) -> None:
        model.set_max_sh_degree(curr_step // self.sh_degree_interval)

    def after_update(
        self,
        model: MeshSplatter,
        optimizers: ModuleOptimizers,
        *,
        curr_step: int,
    ) -> None:
        if curr_step % self.reset_interval == 0:
            model.reset_rotations()
            optimizers['quats'].mutate_params(clear=True)
        model.set_max_sh_degree(None)
        model.update_cov3d()
