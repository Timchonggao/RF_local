from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, SfMDataset
from rfstudio.graphics import Cameras, RGBAImages, TriangleMesh
from rfstudio.loss import PSNRLoss, SSIML1Loss
from rfstudio.model import DiffPBR
from rfstudio.optim import ModuleOptimizers, Optimizer

from .base_trainer import BaseTrainer


@dataclass
class DiffPBRTrainer(BaseTrainer):

    appearance_lr: float = 1e-2

    light_lr: float = 1e-2

    base_eps: float = 1e-8

    occ_reg_begin: float = 0.0
    occ_reg_end: float = 0.001
    occ_reg_decay: Optional[int] = 500

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
                modules=model.kd_texture,
                lr=self.appearance_lr,
                eps=self.base_eps,
                lr_decay=1500
            ),
            'ks': Optimizer(
                category=torch.optim.Adam,
                modules=model.ks_texture,
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
        loss = SSIML1Loss()(gt_outputs=gt_outputs.blend(bg_color), outputs=outputs.blend(bg_color))
        gt_rgb = gt_outputs.blend(bg_color)
        pred_rgb = outputs.detach().blend(bg_color)
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
                cubemap = model.get_envmap().visualize(
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
