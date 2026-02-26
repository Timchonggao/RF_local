from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from kornia.filters import spatial_gradient
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, SfMDataset
from rfstudio.graphics import Cameras, RGBAImages, RGBImages
from rfstudio.loss import PSNRLoss, SSIML1Loss
from rfstudio.model import GeoSplatterS3
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


@dataclass
class GeoSplatS3Trainer(BaseTrainer):

    geometry_lr: float = 0.01
    geometry_decay: int = 500
    splat_lr: float = 1e-3
    splat_decay: int = 200

    mean_factor: float = 0.16
    scale_factor: float = 5.0
    quat_factor: float = 1.0
    opacity_factor: float = 50

    base_eps: float = 1e-8

    sdf_reg_begin: float = 0.2
    sdf_reg_end: float = 0.01
    sdf_reg_decay: Optional[int] = 450

    normal_reg: float = 0.01

    use_mask: bool = True

    def setup(
        self,
        model: GeoSplatterS3,
        dataset: Union[MultiViewDataset, SfMDataset, MeshViewSynthesisDataset],
    ) -> ModuleOptimizers:

        optim_dict = {
            'deforms': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='deforms'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                lr_decay=self.geometry_decay,
            ),
            'sdfs': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='sdfs'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                lr_decay=self.geometry_decay,
            ),
            'weights': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='weights'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                lr_decay=self.geometry_decay,
            ),
            'means': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='means'),
                lr=self.splat_lr * self.mean_factor,
                eps=self.base_eps,
                lr_decay=self.splat_decay,
            ),
            'scales': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='scales'),
                lr=self.splat_lr * self.scale_factor,
                eps=self.base_eps,
                lr_decay=self.splat_decay,
            ),
            'quats': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='quats'),
                lr=self.splat_lr * self.quat_factor,
                eps=self.base_eps,
                lr_decay=self.splat_decay,
            ),
            'opacities': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='opacities'),
                lr=self.splat_lr * self.opacity_factor,
                eps=self.base_eps,
                lr_decay=self.splat_decay,
            ),
        }
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def step(
        self,
        model: GeoSplatterS3,
        inputs: Cameras,
        gt_outputs: Union[RGBImages, RGBAImages],
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        (
            mesh_depths, splat_colors, splat_normals, splat_depths, mesh, vis, mesh_loss
        ) = model.render_report(inputs, vis=visual)
        results = []
        for depth, gt_depth in zip(mesh_depths, splat_depths, strict=True):
            gt_mask = (gt_depth[..., 1:] > 0.1).float()
            item = torch.nn.functional.l1_loss(
                depth[..., :1] * gt_mask,
                gt_depth[..., :1] * gt_mask,
            ) * 10
            if self.use_mask:
                item = item + torch.nn.functional.l1_loss(gt_mask, depth[..., 1:])
            results.append(item)

        depth_loss = torch.stack(results).mean()

        if isinstance(gt_outputs, RGBAImages):
            gt_outputs = gt_outputs.blend((1, 1, 1))

        normal_loss = 0
        for splat_normal, gt_rgb in zip(splat_normals, gt_outputs, strict=True):
            first_order_edge_aware_loss = torch.mul(
                spatial_gradient(splat_normal[None].permute(0, 3, 1, 2), order=1)[0].abs(),
                (-spatial_gradient(gt_rgb[None].permute(0, 3, 1, 2), order=1)[0].abs()).exp()
            ).sum(1).mean()
            normal_loss = normal_loss + first_order_edge_aware_loss * self.normal_reg / len(gt_outputs)

        splat_loss = SSIML1Loss()(splat_colors, gt_outputs)
        psnr = PSNRLoss()(splat_colors.detach(), gt_outputs)

        metrics = {
            'depth-loss': depth_loss.detach(),
            'splat-loss': splat_loss.detach(),
            'normal-loss': normal_loss.detach(),
            'psnr': psnr,
            '#vertex': mesh.num_vertices,
            '#triangle': mesh.num_faces,
        }
        image = None
        if visual:
            with torch.no_grad():
                mesh_pretty = vis[0].item()
                mesh_normal = vis[1].item()
                splat_rgb = vis[2].item()
                splat_normal = vis[3].item()
                light = vis[4].item()
                row0 = torch.cat((splat_rgb, gt_outputs[0].item(), light), dim=1)
                row1 = torch.cat((mesh_pretty, mesh_normal, splat_normal), dim=1)
                image = torch.cat((row0, row1)).clamp(0, 1)
        return depth_loss + splat_loss + mesh_loss, metrics, image

    def before_update(self, model: GeoSplatterS3, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        if self.sdf_reg_decay is not None:
            model.sdf_weight = (
                self.sdf_reg_begin -
                (self.sdf_reg_begin - self.sdf_reg_end) * min(1.0, curr_step / self.sdf_reg_decay)
            )
