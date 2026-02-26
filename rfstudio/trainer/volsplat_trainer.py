from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, SfMDataset
from rfstudio.graphics import Cameras, RGBAImages, RGBImages
from rfstudio.graphics.shaders import PrettyShader
from rfstudio.loss import ImageL1Loss, ImageL2Loss
from rfstudio.model.density_field.volume_splatter import VolumeSplatter
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


@dataclass
class VolSplatTrainer(BaseTrainer):

    num_rays_per_batch: int = 8192

    base_lr: float = 0.05

    eps: float = 1e-8

    max_norm: Optional[float] = None

    lr_decay: Optional[int] = None

    eikonal_reg: float = 0.2
    entropy_reg: float = 0.2
    anisotropy_warmup_schedule: float = 0.75
    min_anisotropy: float = 0.1
    target_variance: Optional[float] = None
    random_bg: bool = True

    def setup(
        self,
        model: VolumeSplatter,
        dataset: Union[MultiViewDataset, MeshViewSynthesisDataset, SfMDataset],
    ) -> ModuleOptimizers:
        self._target_variance = None
        optim_dict = {
            'color': Optimizer(
                torch.optim.Adam,
                modules=model.as_module(field_name='colors'),
                lr=self.base_lr,
                eps=self.eps,
                max_norm=self.max_norm,
                lr_decay=self.lr_decay,
                lr_decay_mode='cos',
            ),
            'scalar': Optimizer(
                torch.optim.AdamW,
                modules=model.as_module(field_name='scalars'),
                lr=self.base_lr,
                eps=self.eps,
                max_norm=self.max_norm,
                lr_decay=self.lr_decay,
                lr_decay_mode='cos',
            ),
            'deviation': Optimizer(
                torch.optim.AdamW,
                modules=model.as_module(field_name='deviation'),
                lr=self.base_lr * 0.01,
                eps=self.eps,
                max_norm=self.max_norm,
                lr_decay=self.lr_decay,
                lr_decay_mode='cos',
            ),
        }
        if model.reparameterization == 'mlp':
            optim_dict['field'] = Optimizer(
                torch.optim.AdamW,
                modules=model.sdf_field,
                lr=self.base_lr * 0.1,
                eps=self.eps,
                max_norm=self.max_norm,
                lr_decay=self.lr_decay,
                lr_decay_mode='cos',
            )
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def before_update(self, model: VolumeSplatter, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        if model.field_type in ['sdf', 'vacancy']:
            model.anisotropy = self.min_anisotropy + (1.0 - self.min_anisotropy) * min(
                1,
                curr_step / (self.num_steps * self.anisotropy_warmup_schedule)
            )
            if self.target_variance is not None:
                self._target_variance = (curr_step / self.num_steps) * (self.target_variance - torch.e) + torch.e

    def step(
        self,
        model: VolumeSplatter,
        inputs: Cameras,
        gt_outputs: Union[RGBImages, RGBAImages],
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        assert not (training and visual)

        l2 = ImageL2Loss()
        l1 = ImageL1Loss()

        gt_outputs = gt_outputs.clamp(0, 1)
        if isinstance(gt_outputs, RGBAImages):
            gt_rgb = gt_outputs.blend((model.bg, model.bg, model.bg))
        else:
            gt_rgb = gt_outputs

        if training:
            assert not visual
            rays = inputs.generate_rays(self.num_rays_per_batch)
            if self.random_bg:
                rand_bg = torch.rand(inputs.shape[0], 3, device=inputs.device)[rays.pixel_indices[:, 0], :]
            else:
                rand_bg = model.bg
            outputs = model.render_rgba_along_rays(rays) # [N, 4]
            outputs = (
                outputs[:, :3] * outputs[:, 3:] +
                (1 - outputs[:, 3:]) * rand_bg
            )
            if isinstance(gt_outputs, RGBAImages):
                gt_pixels = gt_outputs.query_pixels(rays) # [N, 4]
                gt_pixels = (
                    gt_pixels[:, :3] * gt_pixels[:, 3:] +
                    (1 - gt_pixels[:, 3:]) * rand_bg
                )
            else:
                gt_pixels = gt_outputs.query_pixels(rays) # [N, 3]
            l2_loss = l2._impl(gt_pixels, outputs.detach())
            loss = l1._impl(gt_pixels, outputs)
        else:
            outputs = model.render_rgb(inputs)
            l2_loss = l2(gt_rgb, outputs.detach())
            loss = l1(gt_rgb, outputs)

        metrics = {
            'psnr': -10 * l2_loss.log10(),
        }

        if model.field_type in ['sdf', 'vacancy']:
            eikonal_reg = model.compute_eikonal_loss()
            # entropy_reg = model.compute_entropy()
            metrics['eikonal'] = eikonal_reg.detach()
            # metrics['entropy'] = entropy_reg.detach()
            metrics['variance'] = model.get_variance().detach()
            metrics['anisotropy'] = model.anisotropy
            loss = loss + eikonal_reg * self.eikonal_reg # entropy_reg * self.entropy_reg
            if self._target_variance is not None:
                loss = loss + (model.get_variance() - self._target_variance).abs() * 0.1
        image = None
        if visual:
            image = torch.cat((outputs[0].item().detach(), gt_rgb[0].item()), dim=1)      # [H 2W 3]
            if model.field_type in ['sdf', 'vacancy']:
                mesh_vis = model.extract_mesh().render(
                    inputs[0],
                    shader=PrettyShader(normal_type='flat', z_up=model.z_up),
                ).rgb2srgb().blend((model.bg, model.bg, model.bg))
                image = torch.cat((image, mesh_vis.item().clamp(0, 1)), dim=1)
        return loss, metrics, image

    @torch.no_grad()
    def visualize(self,
        model: VolumeSplatter,
        inputs: Cameras,
    ) -> Tensor:
        return model.render_rgb(inputs).item()
