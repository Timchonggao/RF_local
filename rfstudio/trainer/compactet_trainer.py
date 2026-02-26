from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from rfstudio.data import DepthSynthesisDataset
from rfstudio.graphics import Cameras, DepthImages, VectorImages
from rfstudio.graphics.shaders import NormalShader
from rfstudio.model import CompacTet
from rfstudio.optim import ModuleOptimizers, Optimizer
from rfstudio.trainer import BaseTrainer


@dataclass
class CompacTetTrainer(BaseTrainer):

    sdf_lr: float = 6e-3
    tet_lr: float = 1e-2
    base_decay: int = 1250
    sdf_reg_begin: float = 0.05
    sdf_reg_end: float = 0.002
    sdf_reg_decay_ratio: float = 0.25
    energy_reg: float = 0.2
    fairness_reg: float = 0.05

    resampling_schedules: List[int] = field(default_factory=lambda : [500, 750, 1000, 1500, 2000, 2500, 3000])
    reach_max_samples_at: int = 2000
    num_steps_warm_up: int = 500
    num_steps_per_delaunay: int = 1
    stop_resample_at: Optional[int] = 3500
    z_up: bool = False

    def setup(self, model: CompacTet, dataset: DepthSynthesisDataset) -> ModuleOptimizers:
        self._inputs = dataset.get_inputs(split='train')[...]
        self._gt_mesh = dataset.get_meta(split='train')
        self._gt_normals = self._gt_mesh.render(
            self._inputs,
            shader=NormalShader(force_alpha_antialias=True, normal_type='flat'),
        )
        return ModuleOptimizers(
            mixed_precision=False,
            optim_dict={
                'deform': Optimizer(
                    torch.optim.AdamW,
                    modules=model.as_module(field_name='vertices'),
                    lr=self.tet_lr,
                    warm_up=self.num_steps_warm_up,
                    lr_decay=self.base_decay,
                    lr_decay_max_ratio=0.9,
                ),
                'sdf': Optimizer(
                    torch.optim.AdamW,
                    modules=model.as_module(field_name='sdf_values'),
                    lr=self.sdf_lr,
                    warm_up=self.num_steps_warm_up,
                    lr_decay=self.base_decay,
                    lr_decay_max_ratio=0.9,
                ),
            }
        )

    def step(
        self,
        model: CompacTet,
        inputs: Cameras,
        gt_outputs: DepthImages,
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        depths, normals, mesh, reg_loss = model.render(inputs)

        if indices is None or not training:
            gt_normals = self._gt_mesh.render(inputs, shader=NormalShader(normal_type='flat'))
            loss = self.compute_loss(depths, gt_outputs, normals, gt_normals).mean() + reg_loss
        else:
            loss = self.compute_loss(depths, gt_outputs, normals, self._gt_normals[indices]).mean() + reg_loss
        metrics = {
            'geom-loss': loss.detach(),
            'reg-loss': reg_loss.detach(),
            '#points': model.vertices.shape[0],
            '#vertex': mesh.num_vertices,
            '#triangle': mesh.num_faces,
        }
        image = None
        if visual:
            with torch.no_grad():
                depth_map = depths[0].visualize().item()
                normal_map = normals[0].visualize((1, 1, 1)).item()
                gt_depth_map = gt_outputs[0].visualize().item()
                gt_normal = self._gt_mesh.render(inputs[0], shader=NormalShader(normal_type='flat'))
                gt_normal_map = gt_normal.visualize((1, 1, 1)).item()
                row1 = torch.cat((depth_map, gt_depth_map), dim=1)
                row2 = torch.cat((normal_map, gt_normal_map), dim=1)
                image = torch.cat((row1, row2))
        return loss, metrics, image

    def compute_loss(
        self,
        depths: DepthImages,
        gt_depths: DepthImages,
        normals: VectorImages,
        gt_normals: VectorImages,
        *,
        eps: float = 1e-8,
        error_map: bool = False
    ) -> Tensor:
        results = []
        for depth, gt_depth, normal, gt_normal in zip(depths, gt_depths, normals, gt_normals):
            if depth is None:
                dloss = normal.new_zeros(1)
            else:
                gt_mask = gt_depth[..., 1:]
                curr_mask = depth[..., 1:]
                dloss = (((depth[..., :1] - gt_depth[..., :1]) * gt_mask).square().sum(-1, keepdim=True) + eps).sqrt()
            if normal is None:
                nloss = depth.new_zeros(1)
            else:
                gt_mask = gt_normal[..., 1:]
                curr_mask = normal[..., 1:]
                nloss = (((normal[..., :1] - gt_normal[..., :1]) * gt_mask).square().sum(-1, keepdim=True) + eps).sqrt()
            if error_map:
                results.append(250 * dloss + nloss)
            else:
                mloss = (gt_mask - curr_mask).abs()
                results.append(250 * dloss.mean() + 10 * mloss.mean() + nloss.mean())
        return torch.stack(results)

    @torch.no_grad()
    def visualize(
        self,
        model: CompacTet,
        inputs: Cameras,
    ) -> Tensor:
        depths, normals, _, _ = model.render(inputs)
        dmtet = model.get_dmtet()
        depth_map = depths.visualize().item()
        normal_map = normals.visualize((1, 1, 1)).item()
        pretty_map = dmtet.render_pretty(
            inputs,
            z_up=self.z_up,
            point_shape='circle' if dmtet.num_vertices <= 1024 else 'square',
            point_size=0.02 if dmtet.num_vertices <= 1024 else (8e-3 / dmtet.num_vertices) ** (1 / 3),
        ).blend((1, 1, 1)).item()
        return torch.cat((depth_map, normal_map, pretty_map), dim=1).clamp(0, 1)

    def before_update(self, model: CompacTet, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        model.sdf_weight = (
            self.sdf_reg_begin -
            (self.sdf_reg_begin - self.sdf_reg_end) * min(1.0, self.sdf_reg_decay_ratio * curr_step / self.num_steps)
        )
        if curr_step < self.stop_resample_at:
            model.fairness_weight = self.fairness_reg
            model.energy_weight = self.energy_reg
        else:
            model.fairness_weight = 0.0
            model.energy_weight = 0.0
        if (
            (self.stop_resample_at is not None and curr_step >= self.stop_resample_at) or
            (
                curr_step % self.num_steps_per_delaunay > 0 and
                not (curr_step >= self.num_steps_warm_up and curr_step in self.resampling_schedules)
            )
        ):
            optimizers['deform'].skip_once()

    @torch.no_grad()
    def after_update(self, model: CompacTet, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        if (
            curr_step >= self.num_steps_warm_up and
            (self.stop_resample_at is None or curr_step < self.stop_resample_at) and
            curr_step in self.resampling_schedules
        ):
            ratio = max(0, (curr_step - self.num_steps_warm_up) / (self.reach_max_samples_at - self.num_steps_warm_up))
            target_num_points = min(
                model.num_target_points,
                round(model.num_init_points + (model.num_target_points - model.num_init_points) * ratio)
            )
            num_new_samples = target_num_points - model.vertices.shape[0]
            _, normals, mesh, _ = model.render(self._inputs)
            empty = [None] * len(normals)
            error_maps = self.compute_loss(empty, empty, normals, self._gt_normals, error_map=True) # [B, H, W, 1]
            positions, values = mesh.spatial_aggregation(cameras=self._inputs, images=error_maps)
            model.resample(num_new_samples, aggregated_positions=positions, aggregated_errors=values)
            optimizers.mutate_params(reset=True)
            model.update()
        elif curr_step % self.num_steps_per_delaunay == 0:
            model.update()

        if model.clamp_isovalue:
            if model.optimizable == 'positive':
                model.sdf_values.data.clamp_max_(1.0)
            elif model.optimizable == 'negative':
                model.sdf_values.data.clamp_min_(-1.0)
            else:
                model.sdf_values.data.clamp_(-1.0, 1.0)
