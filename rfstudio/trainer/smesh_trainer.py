from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, SfMDataset
from rfstudio.graphics import Cameras, RGBAImages, RGBImages
from rfstudio.loss import PSNRLoss, SSIML1Loss
from rfstudio.model.mesh_based.smesh import SplattableMesh
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


def get_tv(x: Float32[Tensor, "H W 1"]) -> Float32[Tensor, "H W 1"]:
    return (x[:-1, :-1, :] - x[1:, :-1, :]).square() + (x[:-1, :-1, :] - x[:-1, 1:, :]).square()

@dataclass
class SplattableMeshTrainer(BaseTrainer):

    geometry_lr: float = 0.01
    texture_lr: float = 0.01
    light_lr: float = 0.01
    splat_lr: float = 0.01
    warm_up: Optional[int] = 50

    mean_factor: float = 0.16
    scale_factor: float = 5.0
    color_factor: float = 2.5
    quat_factor: float = 1.0
    normal_factor: float = 0.5
    opacity_factor: float = 50

    kd_smooth: float = 0.2
    ks_smooth: float = 0.05

    base_eps: float = 1e-8

    sdf_reg_begin: float = 0.2
    sdf_reg_end: float = 0.01
    sdf_reg_decay: Optional[int] = 1250

    use_mask: bool = True
    normal_weight: float = 0.5

    num_steps_per_reset: int = 200
    stop_reset_after: int = 4000

    def setup(
        self,
        model: SplattableMesh,
        dataset: Union[MultiViewDataset, SfMDataset, MeshViewSynthesisDataset],
    ) -> ModuleOptimizers:
        assert isinstance(dataset, (MultiViewDataset, SfMDataset, MeshViewSynthesisDataset))

        model.update_splats()

        optim_dict = {
            'deforms': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='deforms'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                warm_up=self.warm_up,
                lr_decay=1500,
            ),
            'sdfs': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='sdfs'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                warm_up=self.warm_up,
                lr_decay=1500,
            ),
            'weights': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='weights'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                warm_up=self.warm_up,
                lr_decay=1500,
            ),
            'texture': Optimizer(
                category=torch.optim.Adam,
                modules=model.mlp_texture,
                lr=self.texture_lr,
                eps=self.base_eps,
                warm_up=self.warm_up,
                lr_decay=1500,
            ),
            'means': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='means'),
                lr=self.splat_lr * self.mean_factor,
                eps=self.base_eps,
                lr_decay=1500,
            ),
            'scales': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='scales'),
                lr=self.splat_lr * self.scale_factor,
                eps=self.base_eps
            ),
            'normals': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='normals'),
                lr=self.splat_lr * self.normal_factor,
                eps=self.base_eps
            ),
            'quats': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='quats'),
                lr=self.splat_lr * self.quat_factor,
                eps=self.base_eps
            ),
            'colors': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='colors'),
                lr=self.splat_lr * self.color_factor,
                eps=self.base_eps
            ),
            'opacities': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='opacities'),
                lr=self.splat_lr * self.opacity_factor,
                eps=self.base_eps
            ),
            'shs': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='shs'),
                lr=self.splat_lr * self.color_factor / 20,
                eps=self.base_eps
            ),
            'envmap': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='envmap'),
                lr=self.light_lr,
                eps=self.base_eps
            ),
            'exposure': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='exposure'),
                lr=self.light_lr,
                eps=self.base_eps
            ),
        }
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def step(
        self,
        model: SplattableMesh,
        inputs: Cameras,
        gt_outputs: Union[RGBImages, RGBAImages],
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        (
            mesh_depths, splat_colors, splat_depths, mesh, vis, kd_smooth, ks_smooth, normal_reg, mesh_loss
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
        normal_loss = self.normal_weight * normal_reg
        jitter_loss = kd_smooth * self.kd_smooth + ks_smooth * self.ks_smooth

        if isinstance(gt_outputs, RGBAImages):
            gt_outputs = gt_outputs.blend((1, 1, 1))

        splat_loss = SSIML1Loss()(splat_colors, gt_outputs)
        psnr = PSNRLoss()(splat_colors.detach(), gt_outputs)

        metrics = {
            'depth-loss': depth_loss.detach(),
            'splat-loss': splat_loss.detach(),
            'normal-loss': normal_loss.detach(),
            'jitter-loss': jitter_loss.detach(),
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
                light = vis[3].item()
                kd = vis[4].item()
                ks = vis[5].item()
                splat_normal = vis[6].item()
                row0 = torch.cat((
                    splat_rgb,
                    kd,
                    ks,
                    gt_outputs[0].item()[::2, ::2, :],
                ), dim=1)
                row1 = torch.cat((
                    splat_normal,
                    mesh_normal,
                    mesh_pretty,
                    light,
                ), dim=1)
                image = torch.cat((row0, row1)).clamp(0, 1)
        return depth_loss + splat_loss + mesh_loss + jitter_loss, metrics, image

    def before_update(self, model: SplattableMesh, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        if self.sdf_reg_decay is not None:
            model.sdf_weight = (
                self.sdf_reg_begin -
                (self.sdf_reg_begin - self.sdf_reg_end) * min(1.0, curr_step / self.sdf_reg_decay)
            )
        if curr_step >= self.stop_reset_after and model.mode == 'sh':
            model.sh_degree = 3

    def after_update(self, model: SplattableMesh, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        with torch.no_grad():
            model.envmap.data.clamp_min_(1e-3)
        if curr_step < self.stop_reset_after and curr_step % self.num_steps_per_reset == 0:
            gc.collect()
            torch.cuda.empty_cache()
            model.update_splats()
            for key in ['means', 'scales', 'quats', 'colors', 'opacities', 'normals']:
                optimizers[key].mutate_params(reset=True)
