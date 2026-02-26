from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from rfstudio.data import DepthSynthesisDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import OptimizationVisualizer, TrainTask
from rfstudio.graphics import Cameras, DepthImages, DMTet, Points, TriangleMesh
from rfstudio.graphics.math import get_random_normal_from_sphere
from rfstudio.graphics.shaders import DepthShader, NormalShader
from rfstudio.nn import Module
from rfstudio.optim import ModuleOptimizers, Optimizer
from rfstudio.trainer import BaseTrainer
from rfstudio.ui import console
from rfstudio.utils.colormap import BinaryColorMap, IntensityColorMap
from rfstudio.utils.decorator import chains
from rfstudio.visualization import Visualizer


@dataclass
class TestDMTet(Task):

    viser: Visualizer = Visualizer(port=6789)
    num_points: int = 128
    radius: float = 2.5

    def run(self) -> None:
        positions = torch.randn(self.num_points, 3, device=self.device)
        radii = torch.rand(self.num_points, 1, device=self.device) ** (1/3) * self.radius
        points = Points(positions=(positions / positions.norm(dim=-1, keepdim=True)) * radii)

        dmtet = DMTet.from_delaunay(points, random_sdf=False)
        dmtet_regular = DMTet.from_predefined(resolution=32, scale=1.2, random_sdf=False, device=self.device)
        with self.viser.customize() as handle:

            sphere_sdfs = dmtet.vertices.norm(dim=-1, keepdim=True) - 0.8 # [..., 1]
            cube_sdfs = (dmtet.vertices.abs() - 0.9).max(-1, keepdim=True).values
            handle['delaunay/sphere'].show(dmtet.replace(sdf_values=sphere_sdfs).marching_tets()).configurate(normal_size=0.05)
            handle['delaunay/cube'].show(dmtet.replace(sdf_values=cube_sdfs).marching_tets()).configurate(normal_size=0.05)
            handle['delaunay/tet'].show(dmtet).configurate(point_size=0.02)

            sphere_sdfs = dmtet_regular.vertices.norm(dim=-1, keepdim=True) - 0.8 # [..., 1]
            cube_sdfs = (dmtet_regular.vertices.abs() - 0.9).max(-1, keepdim=True).values
            handle['regular/sphere'].show(dmtet_regular.replace(sdf_values=sphere_sdfs).marching_tets()).configurate(normal_size=0.05)
            handle['regular/cube'].show(dmtet_regular.replace(sdf_values=cube_sdfs).marching_tets()).configurate(normal_size=0.05)
            handle['regular/tet'].show(dmtet_regular).configurate(point_size=0.02)


@dataclass
class DMTetpp(Module):

    num_points: int = 1024
    radius: float = 1.5
    sdf_init: Literal['random', 'cube', 'sphere'] = 'random'
    optimizable: Literal['all', 'positive', 'negative', 'none'] = 'all'
    clamp_isovalue: bool = True

    def __setup__(self) -> None:
        positions = torch.randn(self.num_points, 3, device=self.device)
        radii = torch.rand(self.num_points, 1, device=self.device) ** (1/3) * self.radius
        points = Points(positions=(positions / positions.norm(dim=-1, keepdim=True)) * radii)
        dmtet = DMTet.from_delaunay(points)
        if self.sdf_init == 'cube':
            dmtet.sdf_values.copy_((dmtet.vertices.abs() - 1).max(-1, keepdim=True).values)
        elif self.sdf_init == 'sphere':
            dmtet.sdf_values.copy_(dmtet.vertices.norm(dim=-1, keepdim=True) - 1)
        self.vertices = nn.Parameter(dmtet.vertices)
        self.sdf_values = nn.Parameter(dmtet.sdf_values)
        self.indices = nn.Parameter(dmtet.indices, requires_grad=False)
        self.sdf_weight = 0.0
        self.energy_weight = 0.0
        self.fairness_weight = 0.0
        self._uncertainty = None
        self._context = None

    @torch.no_grad()
    def pop_uncertainty(self) -> Tensor:
        if self._uncertainty is None:
            self._uncertainty = torch.zeros_like(self.sdf_values) # [V, 1]
        uncertainty = self._uncertainty.clone()
        self._uncertainty.zero_()
        return uncertainty

    @torch.no_grad()
    def accumulate_uncertainty(self) -> None:
        if self._uncertainty is None:
            return
        mesh_verts, edges, dmtet = self._context
        uncertainty = dmtet.compute_uncertainty(mesh_verts.grad, edges=edges)
        self._context = None
        mesh_verts.grad = None
        self._uncertainty += uncertainty

    @torch.no_grad()
    def load_state_dict(self, state_dict) -> None:
        self.vertices.data.copy_(state_dict['vertices'])
        self.sdf_values.data.copy_(state_dict['sdf_values'])
        self.indices = nn.Parameter(state_dict['indices'].to(self.indices.device), requires_grad=False)

    @torch.no_grad()
    def update(self) -> None:
        if self.clamp_isovalue:
            if self.optimizable == 'positive':
                self.sdf_values.data.clamp_max_(1.0)
            elif self.optimizable == 'negative':
                self.sdf_values.data.clamp_min_(-1.0)
            else:
                self.sdf_values.data.clamp_(-1.0, 1.0)
        dmtet = DMTet.from_delaunay(Points(positions=self.vertices), random_sdf=False)
        self.vertices.data.copy_(dmtet.vertices)
        self.indices = nn.Parameter(dmtet.indices, requires_grad=False)

    def get_dmtet(self) -> DMTet:
        dmtet = DMTet(vertices=self.vertices, sdf_values=self.sdf_values, indices=self.indices)
        if self.optimizable == 'positive':
            dmtet.replace_(sdf_values=torch.where(self.sdf_values > 1e-6, self.sdf_values, -0.1))
        elif self.optimizable == 'negative':
            dmtet.replace_(sdf_values=torch.where(self.sdf_values < -1e-6, self.sdf_values, 0.1))
        elif self.optimizable == 'none':
            dmtet.replace_(sdf_values=torch.where(self.sdf_values < 0, -0.1, 0.1))
        return dmtet

    def render(self, cameras: Cameras) -> Tuple[DepthImages, TriangleMesh, Tensor]:
        dmtet = self.get_dmtet()
        mesh, edges = dmtet.marching_tets_with_edges()
        if self._uncertainty is not None and self.training and mesh.vertices.requires_grad:
            mesh.vertices.retain_grad()
            self._context = (mesh.vertices, edges, dmtet)
        reg = dmtet.compute_entropy() * self.sdf_weight + dmtet.compute_fairness() * self.fairness_weight
        depths = mesh.render(cameras, shader=DepthShader(antialias=False, culling=True))
        return depths, mesh, reg

    def compute_delaunay_energy(self) -> Tensor:
        dmtet = DMTet(vertices=self.vertices, sdf_values=self.sdf_values, indices=self.indices)
        return dmtet.compute_delaunay_energy() * self.energy_weight

    @chains
    def as_module(self, *, field_name: str) -> Any:

        def parameters(self) -> Any:
            return [getattr(self, field_name)]

        return parameters

@dataclass
class DMTetppTrainer(BaseTrainer):

    base_lr: float = 1e-2
    base_decay: int = 800
    sdf_reg_begin: float = 0.05
    sdf_reg_end: float = 0.001
    sdf_reg_decay: Optional[int] = 1250
    energy_reg: float = 0.01
    fairness_reg_begin: float = 0.5
    fairness_reg_end: float = 0.1
    fairness_reg_decay: int = 1250
    num_steps_per_resample: int = 250
    num_steps_warm_up: int = 200
    stop_resample_at: Optional[int] = 3000
    resample_ratio: float = 0.3
    point_init: Literal['random', 'gt'] = 'random'
    z_up: bool = False

    def setup(self, model: DMTetpp, dataset: DepthSynthesisDataset) -> ModuleOptimizers:
        self._last_inactive = torch.nan
        if self.point_init == 'gt':
            with torch.no_grad():
                gt_mesh = dataset.get_meta(split='test').compute_vertex_normals_(fix=True)
                model.vertices = torch.nn.Parameter(torch.cat((
                    gt_mesh.vertices + gt_mesh.normals * 0.05,
                    gt_mesh.vertices + gt_mesh.normals * 0.01,
                    gt_mesh.vertices - gt_mesh.normals * 0.05,
                )))
                model.vertices.data += torch.randn_like(model.vertices) * 0.005
                model.sdf_values = torch.nn.Parameter(torch.cat((
                    torch.ones_like(gt_mesh.vertices[..., :1]) * 0.1,
                    torch.ones_like(gt_mesh.vertices[..., :1]) * 0.1,
                    -torch.ones_like(gt_mesh.vertices[..., :1]) * 0.5,
                )))
                model.num_points = model.vertices.shape[0]
                model.update()

        return ModuleOptimizers(
            mixed_precision=False,
            optim_dict={
                'deform': Optimizer(
                    torch.optim.Adam,
                    modules=model.as_module(field_name='vertices'),
                    lr=self.base_lr,
                    lr_decay=self.base_decay,
                ),
                'sdf': Optimizer(
                    torch.optim.Adam,
                    modules=model.as_module(field_name='sdf_values'),
                    lr=self.base_lr * 0.6,
                    lr_decay=self.base_decay,
                ),
            }
        )

    def step(
        self,
        model: DMTetpp,
        inputs: Cameras,
        gt_outputs: DepthImages,
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        outputs, mesh, reg_loss = model.render(inputs)
        results = []
        for depth, gt_depth in zip(outputs, gt_outputs, strict=True):
            item = 10 * torch.nn.functional.l1_loss(
                depth[..., :1] * gt_depth[..., 1:],
                gt_depth[..., :1] * gt_depth[..., 1:],
            )
            item = item + torch.nn.functional.l1_loss(gt_depth[..., 1:], depth[..., 1:])
            results.append(item)

        loss = torch.stack(results).mean() + reg_loss
        metrics = {
            'geom-loss': loss.detach(),
            'reg-loss': reg_loss.detach(),
            'inactive': self._last_inactive,
            '#vertex': mesh.num_vertices,
            '#triangle': mesh.num_faces,
        }
        image = None
        if visual:
            with torch.no_grad():
                depth_map = outputs[0].visualize().item()
                normal_map = mesh.render(
                    inputs[0],
                    shader=NormalShader(normal_type='flat', culling=True),
                ).visualize((1, 1, 1)).item()
                gt_depth_map = gt_outputs[0].visualize().item()
                image = torch.cat((depth_map, normal_map, gt_depth_map), dim=1)
        return loss, metrics, image

    @torch.no_grad()
    def visualize(
        self,
        model: DMTetpp,
        inputs: Cameras,
    ) -> Tensor:
        outputs, mesh, _ = model.render(inputs)
        dmtet = model.get_dmtet()
        depth_map = outputs.visualize().item()
        normal_map = mesh.render(
            inputs,
            shader=NormalShader(normal_type='flat', culling=True),
        ).visualize((1, 1, 1)).item()
        pretty_map = dmtet.render_pretty(
            inputs,
            uncertainty=model._uncertainty,
            z_up=self.z_up,
            point_shape='circle' if model.num_points <= 1024 else 'square',
            point_size=0.02 if model.num_points <= 1024 else (8e-3 / model.num_points) ** (1 / 3),
        ).blend((1, 1, 1)).item()
        return torch.cat((depth_map, normal_map, pretty_map), dim=1).clamp(0, 1)

    def before_update(self, model: DMTetpp, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        if self.sdf_reg_decay is not None:
            model.sdf_weight = (
                self.sdf_reg_begin -
                (self.sdf_reg_begin - self.sdf_reg_end) * min(1.0, curr_step / self.sdf_reg_decay)
            )
        model.fairness_weight = (
            self.fairness_reg_begin -
            (self.fairness_reg_begin - self.fairness_reg_end) * min(1.0, curr_step / self.fairness_reg_decay)
        )
        model.energy_weight = self.energy_reg
        if curr_step == self.num_steps_warm_up:
            model.pop_uncertainty()

    def after_backward(self, model: DMTetpp, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        model.accumulate_uncertainty()
        model.compute_delaunay_energy().backward()

    @torch.no_grad()
    def after_update(self, model: DMTetpp, optimizers: ModuleOptimizers, *, curr_step: int) -> None:
        if (
            curr_step > self.num_steps_warm_up and
            (self.stop_resample_at is None or curr_step < self.stop_resample_at) and
            curr_step % self.num_steps_per_resample == 0
        ):
            uncertainty = model.pop_uncertainty().squeeze(-1)
            inactive_indices = (uncertainty == 0).nonzero(as_tuple=False).flatten() # [V']
            self._last_inactive = inactive_indices.shape[0] / uncertainty.shape[0]
            num_resamples = min(
                int(self.resample_ratio * inactive_indices.shape[0]),
                uncertainty.shape[0] - inactive_indices.shape[0],
            )
            if num_resamples > 0:
                resample_indices = torch.randperm(inactive_indices.shape[0]).to(inactive_indices)[:num_resamples]
                resample_indices = inactive_indices[resample_indices] # [S]
                split_indices = uncertainty.argsort(descending=True)[:num_resamples] # [S]
                dists = Points(positions=model.vertices).k_nearest(k=4)[0][split_indices].max(-1).values
                offsets = torch.mul(
                    0.5 * dists.unsqueeze(-1),
                    get_random_normal_from_sphere(num_resamples, device=uncertainty.device),
                ) # [S, 3]
                model.vertices.data[resample_indices] = model.vertices.data[split_indices] - offsets
                model.vertices.data[split_indices] += offsets
                model.sdf_values.data[resample_indices] = model.sdf_values.data[split_indices]

                param_indices = torch.arange(uncertainty.shape[0], device=uncertainty.device)
                param_indices[resample_indices] = split_indices
                optimizers.mutate_params(indices=param_indices.unsqueeze(-1))

        model.update()

test_cube=TrainTask(
    dataset=DepthSynthesisDataset(path=Path('data') / 'cube'),
    model=DMTetpp(num_points=12, sdf_init='cube'),
    experiment=Experiment(name='dmtetpp'),
    trainer=DMTetppTrainer(
        num_steps=2000,
        stop_resample_at=1500,
        batch_size=4,
    ),
    viser=OptimizationVisualizer(up='+y', export='video', hfov_degree=45, radius=5),
    cuda=0,
)

test_block=TrainTask(
    dataset=DepthSynthesisDataset(path=Path('data') / 'inputmodels'),
    model=DMTetpp(),
    experiment=Experiment(name='dmtetpp'),
    trainer=DMTetppTrainer(
        num_steps=4000,
        batch_size=4,
    ),
    viser=OptimizationVisualizer(up='+y', export='video'),
    cuda=0,
)

test_cube_voro=TrainTask(
    dataset=DepthSynthesisDataset(path=Path('data') / 'cube'),
    model=DMTetpp(num_points=32, optimizable='negative', sdf_init='cube'),
    experiment=Experiment(name='dmtetpp_voro'),
    trainer=DMTetppTrainer(
        num_steps=2000,
        base_lr=5e-3,
        stop_resample_at=1500,
        sdf_reg_decay=None,
        fairness_reg_begin=0,
        fairness_reg_end=0,
        energy_reg=0,
        batch_size=4,
    ),
    viser=OptimizationVisualizer(up='+y', export='video', hfov_degree=45, radius=5),
    cuda=0,
)

test_spot=TrainTask(
    dataset=DepthSynthesisDataset(path=Path('data') / 'spot'),
    model=DMTetpp(),
    experiment=Experiment(name='dmtetpp'),
    trainer=DMTetppTrainer(
        num_steps=4000,
        batch_size=4,
    ),
    viser=OptimizationVisualizer(up='+y', export='video'),
    cuda=0,
)

test_spot_voro=TrainTask(
    dataset=DepthSynthesisDataset(path=Path('data') / 'spot'),
    model=DMTetpp(optimizable='negative'),
    experiment=Experiment(name='dmtetpp_voro'),
    trainer=DMTetppTrainer(
        num_steps=4000,
        base_lr=5e-3,
        sdf_reg_decay=None,
        fairness_reg_begin=0,
        fairness_reg_end=0,
        batch_size=4,
    ),
    viser=OptimizationVisualizer(up='+y', export='video'),
    cuda=0,
)

test_block_voro=TrainTask(
    dataset=DepthSynthesisDataset(path=Path('data') / 'inputmodels'),
    model=DMTetpp(optimizable='none'),
    experiment=Experiment(name='dmtetpp_voro'),
    trainer=DMTetppTrainer(
        num_steps=4000,
        base_lr=5e-3,
        sdf_reg_decay=None,
        fairness_reg_begin=0,
        fairness_reg_end=0,
        energy_reg=0.2,
        batch_size=4,
    ),
    viser=OptimizationVisualizer(up='+y', export='video'),
    cuda=0,
)

test_lego=TrainTask(
    dataset=DepthSynthesisDataset(path=Path('data') / 'lego'),
    model=DMTetpp(num_points=32768),
    experiment=Experiment(name='dmtetpp'),
    trainer=DMTetppTrainer(
        base_lr=3e-3,
        num_steps=8000,
        num_steps_per_resample=400,
        resample_ratio=0.15,
        base_decay=1600,
        fairness_reg_decay=4000,
        sdf_reg_decay=2500,
        stop_resample_at=6000,
        batch_size=4,
        z_up=True,
    ),
    viser=OptimizationVisualizer(up='+z', export='video'),
    cuda=0,
)

test_lego_voro=TrainTask(
    dataset=DepthSynthesisDataset(path=Path('data') / 'lego'),
    model=DMTetpp(num_points=32768, optimizable='negative'),
    experiment=Experiment(name='dmtetpp_voro'),
    trainer=DMTetppTrainer(
        num_steps=8000,
        base_lr=1e-3,
        sdf_reg_decay=None,
        num_steps_per_resample=400,
        resample_ratio=0.25,
        base_decay=1600,
        stop_resample_at=6000,
        fairness_reg_begin=0,
        fairness_reg_end=0,
        batch_size=4,
        z_up=True,
    ),
    viser=OptimizationVisualizer(up='+z', export='video'),
    cuda=0,
)

test_dragon_voro=TrainTask(
    dataset=DepthSynthesisDataset(path=Path('data') / 'dragon_recon'),
    model=DMTetpp(num_points=32768, optimizable='negative'),
    experiment=Experiment(name='dmtetpp_voro'),
    trainer=DMTetppTrainer(
        num_steps=8000,
        base_lr=1e-3,
        sdf_reg_decay=None,
        num_steps_per_resample=400,
        resample_ratio=0.25,
        base_decay=1600,
        stop_resample_at=6000,
        fairness_reg_begin=0,
        fairness_reg_end=0,
        batch_size=4,
        z_up=True,
    ),
    viser=OptimizationVisualizer(up='+y', export='video'),
    cuda=0,
)

@dataclass
class Vis(Task):

    load: Path = ...
    step: Optional[int] = None
    viser: Visualizer = Visualizer()

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load, step=self.step)
        model: DMTetpp = train_task.model
        dataset: DepthSynthesisDataset = train_task.dataset

        dmtet = model.get_dmtet()
        mesh = dmtet.marching_tets()
        model.pop_uncertainty()
        inputs = dataset.get_inputs(split='train')[...]
        gt_depths = dataset.get_gt_outputs(split='train')[...]
        with console.progress('Accumulating Gradient') as ptrack:
            with torch.enable_grad():
                vertex_grad = torch.zeros(mesh.vertices.shape[0], 3, device=mesh.device)
                vertex_grad_cnt = torch.zeros(mesh.vertices.shape[0], 1, dtype=torch.long, device=mesh.device)
                for camera, gt_depth in zip(ptrack(inputs), gt_depths):
                    depth = model.render(camera)[0].item()
                    loss = 10 * torch.nn.functional.l1_loss(
                        depth[..., :1] * gt_depth[..., 1:],
                        gt_depth[..., :1] * gt_depth[..., 1:],
                    )
                    loss = loss + torch.nn.functional.l1_loss(gt_depth[..., 1:], depth[..., 1:])
                    loss.backward()
                    local_v_grad = model._context[0].grad
                    vertex_grad += local_v_grad
                    # vertex_grad += local_v_grad.norm(dim=-1, keepdim=True)
                    vertex_grad_cnt[(local_v_grad > 0).any(-1)] += 1
                    model.accumulate_uncertainty()
        vertex_grad_mag = vertex_grad.norm(dim=-1, keepdim=True) / vertex_grad_cnt.clamp_min(1)
        vertex_grad_color = IntensityColorMap(discretization=64)(vertex_grad_mag)
        pts = Points(positions=mesh.vertices, colors=vertex_grad_color)

        uncertainty = model.pop_uncertainty()
        uncertainty_color = IntensityColorMap(discretization=64)(uncertainty)

        with self.viser.customize() as handle:
            handle['mesh/vis'].show(mesh).configurate(vertex_colors=vertex_grad_color)
            handle['mesh/geometry'].show(mesh).configurate(normal_size=0.05)
            handle['v_grad'].show(pts).configurate(point_size=0.01)
            handle['tet'].show(dmtet).configurate(point_size=0.02, point_color=uncertainty_color)

@dataclass
class Vis2(Task):

    load: Path = ...
    step: Optional[int] = None
    viser: Visualizer = Visualizer()

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load, step=self.step)
        model: DMTetpp = train_task.model
        dataset: DepthSynthesisDataset = train_task.dataset

        with torch.enable_grad():
            dmtet = model.get_dmtet()
            dmtet.replace_(sdf_values=dmtet.sdf_values.detach())
            mesh = dmtet.marching_tets()
            inputs = dataset.get_inputs(split='train')[...]
            gt_depths = dataset.get_gt_outputs(split='train')[...]
            with console.progress('Accumulating Gradient') as ptrack:
                vertex_grad = torch.zeros(model.vertices.shape[0], 3, device=model.device)
                # vertex_grad = torch.zeros(model.vertices.shape[0], 1, device=model.device)
                vis_counts = torch.zeros(model.vertices.shape[0], 1, dtype=torch.long, device=model.device)
                for camera, gt_depth in zip(ptrack(inputs), gt_depths):
                    model.vertices.grad = None
                    depth = mesh.render(camera, shader=DepthShader(antialias=False, culling=True)).item()
                    loss = 10 * torch.nn.functional.l1_loss(
                        depth[..., :1] * gt_depth[..., 1:],
                        gt_depth[..., :1] * gt_depth[..., 1:],
                    )
                    loss = loss + torch.nn.functional.l1_loss(gt_depth[..., 1:], depth[..., 1:])
                    loss.backward(retain_graph=True)
                    vertex_grad += model.vertices.grad
                    # vertex_grad += torch.norm(model.vertices.grad, dim=-1, keepdim=True)
                    vis_counts[(model.vertices.grad != 0).all(-1)] += 1
        # vertex_grad = vertex_grad / vis_counts.clamp_min(1)
        vertex_grad = vertex_grad.norm(dim=-1, keepdim=True)
        vertex_grad_color = IntensityColorMap()(vertex_grad.clamp_min(1e-6))

        with self.viser.customize() as handle:
            handle['mesh'].show(mesh).configurate(normal_size=0.05)
            handle['tet'].show(dmtet).configurate(point_size=0.02, point_color=vertex_grad_color)

@dataclass
class Vis_(Task):

    load: Path = ...
    step: Optional[int] = None
    viser: Visualizer = Visualizer()

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load, step=self.step)
        model: DMTetpp = train_task.model

        dmtet = model.get_dmtet()
        mesh = dmtet.marching_tets()
        vertex_grad_color = BinaryColorMap().from_scaled((model.sdf_values > 0).float())

        with self.viser.customize() as handle:
            handle['mesh'].show(mesh).configurate(normal_size=0.05)
            handle['tet'].show(dmtet).configurate(point_size=0.01, point_color=vertex_grad_color)

if __name__ == '__main__':
    TaskGroup(
        cube=test_cube,
        block=test_block,
        vorocube=test_cube_voro,
        vorospot=test_spot_voro,
        vorolego=test_lego_voro,
        vorodragon=test_dragon_voro,
        voroblock=test_block_voro,
        spot=test_spot,
        lego=test_lego,
        vis=Vis(cuda=0),
        vis2=Vis2(cuda=0),
        test=TestDMTet(cuda=0),
    ).run()
