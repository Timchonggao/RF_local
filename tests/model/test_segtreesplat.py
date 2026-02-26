from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from rfstudio.data import SegTreeDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.graphics import Cameras, DepthImages, TriangleMesh
from rfstudio.graphics.math import spectral_clustering
from rfstudio.io import open_video_renderer
from rfstudio.model import FeatureSplatter
from rfstudio.model.density_primitives.featsplat_s2 import FeatureSplatterS2
from rfstudio.trainer import SegTreeSplatTrainer
from rfstudio.trainer.segtreesplat_s2_trainer import SegTreeSplatS2Trainer
from rfstudio.ui import console
from rfstudio.utils.pretty import P
from rfstudio.visualization import Visualizer

usb_task = TrainTask(
    dataset=SegTreeDataset(
        path=Path('data') / 'artgs' / 'usb',
    ),
    model=FeatureSplatter(
        sh_degree=3,
        feature_dim=6,
        prepare_densification=True,
    ),
    experiment=Experiment(name='segtreesplat'),
    trainer=SegTreeSplatTrainer(
        num_steps=20000,
        batch_size=1,
        num_steps_per_val=1000,
        mixed_precision=False,
        full_test_after_train=False,
    ),
    cuda=0,
    seed=1,
)

table_task = TrainTask(
    dataset=SegTreeDataset(
        path=Path('data') / 'artgs' / 'table',
    ),
    model=FeatureSplatter(
        sh_degree=3,
        feature_dim=6,
        prepare_densification=True,
    ),
    experiment=Experiment(name='segtreesplat'),
    trainer=SegTreeSplatTrainer(
        num_steps=20000,
        batch_size=1,
        num_steps_per_val=1000,
        mixed_precision=False,
        full_test_after_train=False,
    ),
    cuda=0,
    seed=1,
)

usb_s2_task = TrainTask(
    dataset=SegTreeDataset(
        path=Path('data') / 'artgs' / 'usb',
    ),
    model=FeatureSplatterS2(
        load=...,
    ),
    experiment=Experiment(name='segtreesplat_s2'),
    trainer=SegTreeSplatS2Trainer(
        num_steps=20000,
        batch_size=1,
        num_steps_per_val=1000,
        mixed_precision=False,
        full_test_after_train=False,
    ),
    cuda=0,
    seed=1,
)

@dataclass
class Test(Task):

    load: Path = ...
    step: Optional[int] = None
    cluster: Optional[int] = None
    video: bool = True
    z_up: bool = True

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load, step=self.step)
        model = train_task.model
        assert isinstance(model, FeatureSplatter)
        model.gaussians.spatially_aggregate_features(voxel_size=0.1)
        model.gaussians.colors.copy_(model.gaussians.as_splats(gamma=1.0, num_clusters=self.cluster).colors)
        model.set_max_sh_degree(0)
        cameras = Cameras.from_orbit(
            center=(0, 0, 0),
            up=(0, 0, 1) if self.z_up else (0, 1, 0),
            resolution=(800, 800),
            hfov_degree=45,
            radius=2.5,
            pitch_degree=30,
            num_samples=100,
            device=self.device
        )
        cameras = Cameras.cat((cameras, cameras), dim=0)

        ctx = open_video_renderer(train_task.experiment.dump_path / 'test.mp4', 20) if self.video else nullcontext()
        with ctx as handle:
            with console.progress('Rendering') as ptrack:
                for i, inputs in enumerate(ptrack(cameras)):
                    img = model.render_rgba(inputs.view(1)).blend((1, 1, 1)).item().clamp(0, 1)
                    if self.video:
                        handle.write(img)
                    else:
                        train_task.experiment.dump_image('test', index=i, image=img)

@dataclass
class Vis(Task):

    load: Path = ...
    step: Optional[int] = None
    cluster: Optional[int] = None

    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load, step=self.step)
        model = train_task.model
        assert isinstance(model, FeatureSplatter)
        model.gaussians.spatially_aggregate_features(voxel_size=0.1)
        with Visualizer().customize() as viser:
            clustered_splats = model.gaussians.as_clustered_splats(gamma=1.0, num_clusters=self.cluster)
            for i, splats in enumerate(clustered_splats):
                viser[f'cluster_{i}'].show(splats)

@dataclass
class Export(Task):

    load: Path = ...
    step: Optional[int] = None
    output: Optional[Path] = None
    cluster: int = ...

    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load, step=self.step)
        model = train_task.model
        assert isinstance(model, FeatureSplatter)
        model.gaussians.spatially_aggregate_features(voxel_size=0.1)
        output = self.output
        if output is None:
            output = train_task.experiment.dump_path / 'export.pkl'
        cluster_results = spectral_clustering(
            model.gaussians.features,
            downsample_to=1024,
            dim=-1,
            num_clusters=self.cluster,
            return_centers=True,
        )
        ckpt = {
            'means': model.gaussians.means,
            'colors': model.gaussians.colors,
            'shs': model.gaussians.shs,
            'opacities': model.gaussians.opacities,
            'scales': model.gaussians.scales,
            'quats': model.gaussians.quats,
            'features': model.gaussians.features,
            'cluster_centers': cluster_results.centers,
        }
        torch.save(ckpt, output)

@dataclass
class ExportMesh(Task):

    load: Path = ...
    step: Optional[int] = None
    z_up: bool = True

    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load, step=self.step)
        model = train_task.model
        assert isinstance(model, FeatureSplatter)
        cameras = Cameras.from_sphere(
            center=(0, 0, 0),
            up=(0, 0, 1) if self.z_up else (0, 1, 0),
            resolution=(800, 800),
            hfov_degree=45,
            radius=2.5,
            num_samples=128,
            device=self.device,
        )
        depths = DepthImages(model.render_depth(camera).item() for camera in cameras.view(-1, 1))
        with console.progress('Fusing') as ptrack:
            mesh = TriangleMesh.from_depth_fusion(
                depths,
                cameras=cameras,
                voxel_size=0.01,
                sdf_trunc=0.05,
                progress_handle=ptrack,
            )
        mesh.export(train_task.experiment.dump_path / 'fused.ply', only_geometry=True)
        V = mesh.num_vertices
        F = mesh.num_faces
        console.print(P@'Mesh vertices: {V}')
        console.print(P@'Mesh faces: {F}')

if __name__ == '__main__':
    TaskGroup(
        usb=usb_task,
        usb_s2=usb_s2_task,
        table=table_task,
        test=Test(cuda=0),
        vis=Vis(cuda=0),
        export=Export(cuda=0),
        export_mesh=ExportMesh(cuda=0),
    ).run()
