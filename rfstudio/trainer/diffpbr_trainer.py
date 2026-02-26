from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, SfMDataset
from rfstudio.graphics import Cameras, PBRAImages, RGBAImages, TriangleMesh
from rfstudio.loss import HDRLoss, PSNRLoss
from rfstudio.model import DiffPBR
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


@dataclass
class DiffPBRTrainer(BaseTrainer):

    appearance_lr: float = 0.03

    light_lr: float = 0.09

    base_eps: float = 1e-8

    occ_reg_begin: float = 0.0
    occ_reg_end: float = 0.001
    occ_reg_decay: Optional[int] = 500

    shadow_scale_begin: float = 0.0
    shadow_scale_end: float = 1.0
    shadow_scale_decay: Optional[int] = 350

    light_grad_amp: Optional[float] = 64.0

    def setup(
        self,
        model: DiffPBR,
        dataset: Union[MeshViewSynthesisDataset, MultiViewDataset, SfMDataset],
    ) -> ModuleOptimizers:
        assert isinstance(dataset, MeshViewSynthesisDataset)

        optim_dict = {
            'kd': Optimizer(
                category=torch.optim.Adam,
                modules=model.mlp_texture,
                lr=self.appearance_lr,
                eps=self.base_eps,
                lr_decay=1500
            ),
            'gamma': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='gamma'),
                lr=self.light_lr,
                eps=self.base_eps,
                lr_decay=1500,
            )
        }
        if model.gt_envmap is None:
            optim_dict['light'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='light'),
                lr=self.light_lr,
                eps=self.base_eps,
                lr_decay=1500,
            )
            if self.light_grad_amp is not None:
                model.cubemap.register_hook(lambda g: g * self.light_grad_amp)
        if model.gt_mesh is None:
            gt_mesh = dataset.get_meta(split='train')
            assert gt_mesh is not None
        else:
            gt_mesh = TriangleMesh.from_file(model.gt_mesh).replace_(
                normals=None,
                face_normals=None,
                uvs=None,
                kd=None,
                ks=None,
            ).to(model.device)
        model.set_gt_geometry(gt_mesh)
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def step(
        self,
        model: DiffPBR,
        inputs: Cameras,
        gt_outputs: RGBAImages,
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        bg_color = model.get_background_color().to(inputs.device)
        outputs, kdks, mesh, reg_loss = model.render_report(inputs)
        losses = []
        pred_pbra = []
        gt_pbra = gt_outputs.srgb2rgb()
        for output, gt_output in zip(outputs, gt_pbra, strict=True):
            diffuse_luma  = output[..., :3].mean(-1, keepdim=True)
            specular_luma = output[..., 6:9].mean(-1, keepdim=True)
            ref_luma = gt_output[..., :3].max(-1, keepdim=True).values
            img = _rgb2srgb((((diffuse_luma + specular_luma) * gt_output[..., 3:]).clamp(0, 65535) + 1).log())
            target = _rgb2srgb(((ref_luma * gt_output[..., 3:]).clamp(0, 65535) + 1).log())
            error = (img - target).abs() * diffuse_luma / (diffuse_luma + specular_luma).clamp_min(1e-3)
            loss = error.mean() * 0.15 + specular_luma.mean() / diffuse_luma.mean().clamp_min(1e-3) * 0.0025
            losses.append(loss)
            pred_pbra.append(torch.cat((output[..., :3] * output[..., 3:6] + output[..., 6:9], output[..., 9:]), dim=-1))
        pred_pbra = PBRAImages(pred_pbra)
        loss = HDRLoss()(pred_pbra.blend(bg_color), gt_pbra.blend(bg_color)) + torch.stack(losses).mean()
        gt_rgb = gt_outputs.blend(bg_color)
        pred_rgb = pred_pbra.detach().rgb2srgb().blend(bg_color)
        metrics = {
            'l1': loss.detach(),
            'psnr': PSNRLoss()(gt_rgb.clamp(0, 1), pred_rgb.clamp(0, 1)),
            'reg-loss': reg_loss.detach(),
            'gamma': model.gamma.sigmoid().detach(),
            '#vertex': mesh.num_vertices,
            '#triangle': mesh.num_faces,
        }
        image = None
        if visual:
            with torch.no_grad():
                kdks = kdks.blend(bg_color)
                normal_map = kdks[2].item()
                cubemap = model.get_envmap().as_cubemap(resolution=512).visualize(
                    width=normal_map.shape[1] * 2,
                    height=normal_map.shape[0],
                ).item()
                row1 = torch.cat((kdks[0].item(), pred_rgb[0].item(), gt_rgb[0].item()), dim=1) # [H, 3W, 3]
                row2 = torch.cat((kdks[1].item(), cubemap), dim=1) # [H, 3W, 3]
                image = torch.cat((row1, row2), dim=0).clamp(0, 1)
        return loss + reg_loss, metrics, image

    def before_update(self, model: DiffPBR, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        if self.occ_reg_decay is not None:
            model.occ_weight = (
                self.occ_reg_begin -
                (self.occ_reg_begin - self.occ_reg_end) * min(1.0, curr_step / self.occ_reg_decay)
            )
        if self.shadow_scale_decay is not None:
            model.shadow_scale = (
                self.shadow_scale_begin -
                (self.shadow_scale_begin - self.shadow_scale_end) * min(1.0, curr_step / self.shadow_scale_decay)
            )

    def after_update(self, model: DiffPBR, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        with torch.no_grad():
            model.cubemap.clamp_min_(1e-3)


def _rgb2srgb(f: Tensor) -> Tensor:
    return torch.where(
        f > 0.0031308,
        torch.pow(torch.clamp(f, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
        12.92 * f,
    )
