from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, SfMDataset
from rfstudio.graphics import Cameras, RGBAImages, RGBImages
from rfstudio.loss import HDRLoss, L1Loss, L2Loss, MaskedPhotometricLoss, PSNRLoss
from rfstudio.model import NVDiffRec
from rfstudio.optim import ModuleOptimizers, Optimizer
from rfstudio.utils.lazy_module import dr

from .base_trainer import BaseTrainer


@torch.no_grad()
def vis_cubemap(cubemap: Float32[Tensor, "6 R R 3"], *, width: int, height: int) -> Float32[Tensor, "H W 3"]:
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / height, 1.0 - 1.0 / height, height, device=cubemap.device) * torch.pi,
        torch.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width, device=cubemap.device) * torch.pi,
        indexing='ij',
    )
    sin_theta = gy.sin()
    reflvec = torch.stack((sin_theta * gx.sin(), gy.cos(), -sin_theta * gx.cos()), dim=-1)
    return dr.texture(cubemap[None], reflvec[None].contiguous(), filter_mode='linear', boundary_mode='cube')[0]


@dataclass
class NVDiffRecTrainer(BaseTrainer):

    geometry_lr: float = 0.03

    appearance_lr: float = 0.01

    light_lr: float = 0.01

    base_eps: float = 1e-8

    sdf_reg_begin: float = 0.2
    sdf_reg_end: float = 0.01
    sdf_reg_decay: Optional[int] = 1250

    occ_reg_begin: float = 0.0
    occ_reg_end: float = 0.001
    occ_reg_decay: Optional[int] = 500

    kd_grad_reg_begin: float = 0.0
    kd_grad_reg_end: float = 0.03
    kd_grad_reg_decay: Optional[int] = 500

    light_reg: float = 0.005
    light_grad_amp: Optional[float] = 64.0

    use_mask: bool = True

    def setup(
        self,
        model: NVDiffRec,
        dataset: Union[MeshViewSynthesisDataset, MultiViewDataset, SfMDataset],
    ) -> ModuleOptimizers:
        assert isinstance(dataset, (MeshViewSynthesisDataset, MultiViewDataset, SfMDataset))

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
            'texture': Optimizer(
                category=torch.optim.Adam,
                modules=model.mlp_texture,
                lr=self.appearance_lr,
                eps=self.base_eps,
                lr_decay=1500
            ),
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
        if model.geometry == 'gt':
            assert isinstance(dataset, MeshViewSynthesisDataset)
            model.set_gt_geometry(dataset.get_meta(split='train'))
        self.real_geometry_lr = self.geometry_lr
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def step(
        self,
        model: NVDiffRec,
        inputs: Cameras,
        gt_outputs: Union[RGBAImages, RGBImages],
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        bg_color = model.get_background_color().to(inputs.device)
        outputs, kdks, mesh, reg_loss = model.render_report(inputs)
        if self.use_mask:
            assert isinstance(gt_outputs, RGBAImages)
            loss_fn = MaskedPhotometricLoss(
                photometric_term=HDRLoss(),
                coverage_coeff=(5.0 if model.geometry == 'flexicubes' else 1.0),
                coverage_loss=(L1Loss if model.geometry == 'flexicubes' else L2Loss)(),
            )
            loss = loss_fn(gt_outputs=gt_outputs, outputs=outputs)
            gt_rgb = gt_outputs.blend(bg_color)
        else:  # noqa: PLR5501
            if isinstance(gt_outputs, RGBAImages):
                loss = HDRLoss()(
                    gt_outputs=gt_outputs.srgb2rgb().blend(bg_color),
                    outputs=outputs.blend(bg_color),
                )
                gt_rgb = gt_outputs.blend(bg_color)
            else:
                loss = HDRLoss()(
                    gt_outputs=gt_outputs.srgb2rgb(),
                    outputs=outputs.blend(bg_color),
                )
                gt_rgb = gt_outputs
        pred_rgb = outputs.detach().rgb2srgb().blend(bg_color)
        metrics = {
            'l1': loss.detach(),
            'psnr': PSNRLoss()(gt_rgb.clamp(0, 1), pred_rgb.clamp(0, 1)),
            'reg-loss': reg_loss.detach(),
            'learning-rate': self.real_geometry_lr,
            '#vertex': mesh.num_vertices,
            '#triangle': mesh.num_faces,
        }
        image = None
        if visual:
            with torch.no_grad():
                kdks = kdks.blend(bg_color)
                normal_map = kdks[2].item()
                if model.gt_envmap is None:
                    cubemap = vis_cubemap(model.cubemap, width=normal_map.shape[1] * 2, height=normal_map.shape[0])
                else:
                    cubemap = torch.zeros(normal_map.shape[0], normal_map.shape[1] * 2, 3).to(normal_map)
                row1 = torch.cat((kdks[0].item(), pred_rgb[0].item(), gt_rgb[0].item()), dim=1) # [H, 3W, 3]
                row2 = torch.cat((kdks[1].item(), cubemap), dim=1) # [H, 3W, 3]
                # row3 = torch.cat((kdks[0].item(), kdks[1].item(), normal_map), dim=1) # [H, 3W, 3]
                image = torch.cat((row1, row2), dim=0).clamp(0, 1)
        return loss + reg_loss, metrics, image

    def before_update(self, model: NVDiffRec, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        model.light_weight = self.light_reg
        if self.sdf_reg_decay is not None:
            model.sdf_weight = (
                self.sdf_reg_begin -
                (self.sdf_reg_begin - self.sdf_reg_end) * min(1.0, curr_step / self.sdf_reg_decay)
            )
        if self.occ_reg_decay is not None:
            model.occ_weight = (
                self.occ_reg_begin -
                (self.occ_reg_begin - self.occ_reg_end) * min(1.0, curr_step / self.occ_reg_decay)
            )
        if self.kd_grad_reg_decay is not None:
            model.kd_grad_weight = (
                self.kd_grad_reg_begin -
                (self.kd_grad_reg_begin - self.kd_grad_reg_end) * min(1.0, curr_step / self.kd_grad_reg_decay)
            )

    def after_update(self, model: NVDiffRec, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        self.real_geometry_lr = optimizers.optimizers['deforms'].param_groups[0]['lr']
        with torch.no_grad():
            model.cubemap.clamp_min_(0)
