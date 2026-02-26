from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, RelightDataset
from rfstudio.graphics import Cameras, RGBAImages, RGBImages, TextureCubeMap
from rfstudio.loss import PSNRLoss, SSIML1Loss
from rfstudio.model import GeoSplatter
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


@dataclass
class GeoSplatTrainer(BaseTrainer):

    cov3d_lr: float = 3e-3
    geometry_lr: float = 1e-2
    appearance_lr: float = 3e-3
    light_lr: float = 1e-2
    sh_interval: int = 400

    base_decay: Optional[int] = 800

    base_eps: float = 1e-15

    vertex_sample_warmup: int = 50
    split_sum_warmup: int = 99999

    light_reg_begin: float = 2e-3
    light_reg_end: float = 2e-3
    light_reg_decay: int = 500

    sdf_reg_begin: float = 0.2
    sdf_reg_end: float = 0.12
    sdf_reg_decay: int = 500

    occ_reg_begin: float = 0.0
    occ_reg_end: float = 0.0
    occ_reg_decay: Optional[int] = 0

    kd_grad_reg_begin: float = 0.0
    kd_grad_reg_end: float = 0.03
    kd_grad_reg_decay: Optional[int] = 500
    kd_regualr_perturb_std: float = 0.01

    ks_grad_reg_begin: float = 0.0
    ks_grad_reg_end: float = 0.001
    ks_grad_reg_decay: Optional[int] = 500
    ks_regualr_perturb_std: float = 0.01

    normal_grad_reg_begin: float = 0.0
    normal_grad_reg_end: float = 0.5
    normal_grad_reg_decay: Optional[int] = 0

    mc_warmup: int = 0

    use_mask_loss: bool = True
    visual_mode: Literal['default', 'production'] = 'default'

    def setup(
        self,
        model: GeoSplatter,
        dataset: Union[MeshViewSynthesisDataset, RelightDataset, MultiViewDataset],
    ) -> ModuleOptimizers:
        assert isinstance(dataset, (MeshViewSynthesisDataset, RelightDataset, MultiViewDataset))

        model.setup_train_data_size(dataset.get_size(split='train'))
        model.cubemap.register_hook(lambda grad: grad * 64)
        model.latlng.register_hook(lambda grad: grad * 64)

        if model.initial_guess == 'specular':
            self.kd_grad_reg_begin = 0.5
            self.ks_grad_reg_begin = 0.1
            self.geometry_lr = self.geometry_lr * 5
            self.light_lr = self.light_lr * 3
        elif model.initial_guess == 'glossy':
            self.light_lr = self.light_lr * 3

        self.gt_mesh = None
        if isinstance(dataset, MeshViewSynthesisDataset):
            mesh = dataset.get_meta(split='train')
            if mesh is not None:
                self.gt_mesh = mesh.detach().compute_vertex_normals(fix=True)
        if model.geometry == 'gt':
            assert self.gt_mesh is not None
            model.setup_gt_mesh(self.gt_mesh.replace(normals=None).subdivide().subdivide())
        optim_dict = {
            'deforms': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='deforms'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
            'kd': Optimizer(
                category=torch.optim.Adam,
                modules=model.field.kd_enc,
                lr=self.appearance_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
            'ks': Optimizer(
                category=torch.optim.Adam,
                modules=model.field.ks_enc,
                lr=self.appearance_lr * 0.5,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
            'z': Optimizer(
                category=torch.optim.Adam,
                modules=model.field.z_enc,
                lr=self.cov3d_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
            'exposure': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='exposure'),
                lr=self.light_lr * 0.5,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            ),
        }
        if model.sdf_representation == 'mlp':
            optim_dict['sdfs'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.sdf_field,
                lr=self.geometry_lr * 0.01,
                eps=self.base_eps,
                lr_decay=self.base_decay,
                max_norm=0.5,
                warm_up=50,
            )
        elif model.sdf_representation == 'grid':
            optim_dict['sdfs'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='sdfs'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            )
        if model.geometry == 'flexicubes':
            optim_dict['weights'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='weights'),
                lr=self.geometry_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            )
        if model.gt_envmap is None:
            optim_dict['light'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='light'),
                lr=self.light_lr,
                eps=self.base_eps,
                lr_decay=self.base_decay,
            )
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def step(
        self,
        model: GeoSplatter,
        inputs: Cameras,
        gt_outputs: RGBAImages,
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        bg_color = model.get_background_color().to(model.device)
        (
            pbra,
            vis,
            mesh_normal,
            num_gaussians,
            reg_loss,
        ) = model.render_report(inputs, indices=indices if training else None, gt_outputs=gt_outputs)

        gt_rgba = gt_outputs
        rgba = pbra.rgb2srgb()

        if model.minimal_memory:
            gc.collect()
            torch.cuda.empty_cache()

        losses = []
        if isinstance(gt_rgba, RGBAImages):
            for pbra_item, gt_pbra_item in zip(pbra, gt_rgba.srgb2rgb(), strict=True):
                train_bg_color = torch.rand_like(pbra_item[..., :3]) # [H, W, 3]
                mask = gt_pbra_item[..., 3:] # [H, W, 1]
                img1 = pbra_item[..., :3] + (1 - pbra_item[..., 3:]) * train_bg_color
                img2 = gt_pbra_item[..., :3] * mask + (1 - mask) * train_bg_color
                loss = SSIML1Loss()._impl(img1, img2)
                if self.use_mask_loss:
                    loss = loss + 5 * (mask - pbra_item[..., 3:]).square().mean()
                losses.append(loss)
        else:
            for pbra_item, gt_pbra_item in zip(pbra, gt_rgba.srgb2rgb(), strict=True):
                train_bg_color = torch.rand_like(pbra_item[..., :3]) # [H, W, 3]
                img1 = pbra_item[..., :3] + (1 - pbra_item[..., 3:]) * train_bg_color
                img2 = gt_pbra_item[..., :3] * mask + (1 - mask) * train_bg_color
                loss = SSIML1Loss()._impl(img1, img2)
                losses.append(loss)
        loss = sum(losses) / len(losses)

        if model.save_memory:
            metrics = {
                'l1-ssim': loss.detach(),
                '#gaussians': num_gaussians,
                'regularization': reg_loss.detach(),
                'exposure': model.exposure_params.detach().mean().exp(),
            }
            return loss + reg_loss, metrics, torch.zeros(1, 1, 3) if visual else None

        rgb = RGBImages([
            rgba_item[..., :3] + (1 - rgba_item[..., 3:]) * bg_color
            for rgba_item in rgba.detach()
        ])
        if isinstance(gt_rgba, RGBAImages):
            gt_rgb = gt_rgba.blend(bg_color)
        else:
            gt_rgb = gt_rgba
        splat_psnr = PSNRLoss()(gt_rgb, rgb.clamp(0, 1)) # srgb space metrics
        metrics = {
            'l1-ssim': loss.detach(),
            'splat-psnr': splat_psnr,
            '#gaussians': num_gaussians,
            'regularization': reg_loss.detach(),
            'exposure': model.exposure_params.detach().mean().exp(),
        }

        image = None
        if visual:
            if self.visual_mode == 'production':
                image = torch.cat((
                    rgb[0].item(),
                    vis[0].item(),
                    vis[5].item(),
                    vis[4].item(),
                ), dim=1).clamp(0, 1)   # [H, 5W, 3]
            else:
                row1 = torch.cat((
                    rgb[0].item(),
                    mesh_normal[0].visualize(bg_color).item(),
                    gt_rgb[0].item(),
                ), dim=1)                                                  # [H, 3W, 3]
                row2 = torch.cat((
                    vis[0].item(),
                    vis[2].item(),
                    vis[1].item()[..., 2:3].expand_as(vis[0].item()),
                ), dim=1)                                                  # [H, 3W, 3]
                row3 = torch.cat((
                    vis[1].item()[..., 1:2].expand_as(vis[0].item()),
                    vis[3].item(),
                ), dim=1)                                                  # [H, 3W, 3]
                image = torch.cat((row1, row2, row3), dim=0).clamp(0, 1)   # [3H, 3W, 3]

        return loss + reg_loss, metrics, image

    def visualize(self, model: GeoSplatter, inputs: Cameras) -> Tensor:
        bg_color = model.get_background_color().to(model.device)
        pbra, vis, *_ = model.render_report(inputs[None], indices=None, gt_outputs=None)
        rgba_item = pbra.rgb2srgb().item()
        rgb = rgba_item[..., :3] + (1 - rgba_item[..., 3:]) * bg_color
        return torch.cat((
            rgb,
            vis[0].item(),
            vis[4].item(),
            vis[3].item(),
        ), dim=1).clamp(0, 1) # [H, 5W, 3]

    def before_update(
        self,
        model: GeoSplatter,
        optimizers: ModuleOptimizers,
        *,
        curr_step: int,
    ) -> None:
        if self.vertex_sample_warmup > 0 and curr_step < self.vertex_sample_warmup:
            model.sample_method = 'vertex'
        elif model.sample_method != 'face':
            gc.collect()
            torch.cuda.empty_cache()
            model.sample_method = 'face'

        if curr_step < self.split_sum_warmup:
            model.render_method = 'split-sum'
        else:
            if self.mc_warmup > 0:
                model.shadow_scale = min(1.0, (curr_step - self.split_sum_warmup) / self.mc_warmup)
            if model.render_method != 'mc':
                gc.collect()
                torch.cuda.empty_cache()
                model.render_method = 'mc'
                with torch.no_grad():
                    model.latlng.copy_(
                        TextureCubeMap(data=model.cubemap)
                            .as_latlng(width=model.latlng.shape[1], height=model.latlng.shape[0])
                            .data
                    )
                    model.field.occ_enc.reset()
                    model.field.ks_enc.reset()
                    model.field.kd_enc.reset()
                    optimizers['occ'].mutate_params(reset=True)
                    optimizers['light'].mutate_params(reset=True)
                    optimizers['exposure'].mutate_params(reset=True)
                    optimizers['z'].mutate_params(reset=True)
                    optimizers['ks'].mutate_params(reset=True)
                    optimizers['kd'].mutate_params(reset=True)
                    optimizers['deforms'].mutate_params(reset=True)
                    optimizers['sdfs'].mutate_params(reset=True)
                    optimizers['weights'].mutate_params(reset=True)

        model.light_weight = (
            self.light_reg_begin -
            (self.light_reg_begin - self.light_reg_end) * min(1.0, curr_step / self.light_reg_decay)
        )

        model.max_sh_degree = curr_step // self.sh_interval

        if self.sdf_reg_decay > 0:
            model.sdf_weight = (
                self.sdf_reg_begin -
                (self.sdf_reg_begin - self.sdf_reg_end) * min(1.0, curr_step / self.sdf_reg_decay)
            )

        if self.occ_reg_decay > 0:
            model.occ_weight = (
                self.occ_reg_begin -
                (self.occ_reg_begin - self.occ_reg_end) * min(1.0, curr_step / self.occ_reg_decay)
            )

        if self.kd_grad_reg_decay > 0:
            model.kd_grad_weight = (
                self.kd_grad_reg_begin -
                (self.kd_grad_reg_begin - self.kd_grad_reg_end) * min(1.0, curr_step / self.kd_grad_reg_decay)
            )
            model.kd_regualr_perturb_std = self.kd_regualr_perturb_std

        if self.ks_grad_reg_decay > 0:
            model.ks_grad_weight = (
                self.ks_grad_reg_begin -
                (self.ks_grad_reg_begin - self.ks_grad_reg_end) * min(1.0, curr_step / self.ks_grad_reg_decay)
            )
            model.ks_regualr_perturb_std = self.ks_regualr_perturb_std

        if self.normal_grad_reg_decay > 0:
            model.normal_grad_weight = (
                self.normal_grad_reg_begin -
                (self.normal_grad_reg_begin - self.normal_grad_reg_end) * min(1.0, max(curr_step - 200, 0) / self.normal_grad_reg_decay)
            )

    def after_update(self, model: GeoSplatter, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        if model.render_method == 'split-sum':
            if curr_step % 10 == 0:
                if model.save_memory or curr_step % 50 == 0:
                    gc.collect()
                torch.cuda.empty_cache()
            if model.gt_envmap is None and not model.exp_lighting:
                model.cubemap.data.clamp_min_(1e-2)
        else:
            if curr_step % 5 == 0:
                torch.cuda.empty_cache()
            model.latlng.data.clamp_min_(1e-2)
