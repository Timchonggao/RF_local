from __future__ import annotations

import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from rfstudio.data import MeshViewSynthesisDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.model.mesh_based.smesh import SplattableMesh
from rfstudio.trainer.smesh_trainer import SplattableMeshTrainer

spot_stage_1 = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=pathlib.Path('data') / 'spot',
    ),
    model=SplattableMesh(resolution=160, mode='rgb'),
    experiment=Experiment(name='smesh', timestamp='spot_s1'),
    trainer=SplattableMeshTrainer(
        num_steps=300,
        batch_size=8,
        num_steps_per_val=50,
        mixed_precision=False,
        full_test_after_train=False,
    ),
    cuda=0,
    seed=1,
)

spot_stage_2 = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=pathlib.Path('data') / 'spot',
    ),
    model=SplattableMesh(
        resolution=160,
        mode='splitsum',
        load=Path('outputs') / 'smesh' / 'spot_s1' / 'ckpts' / '0000000300.ckpt',
        opacity_reset=0.98,
    ),
    experiment=Experiment(name='smesh', timestamp='spot_s2'),
    trainer=SplattableMeshTrainer(
        warm_up=None,
        num_steps=5000,
        batch_size=8,
        light_lr=1e-2,
        texture_lr=1e-2,
        geometry_lr=1e-3,
        splat_lr=1e-3,
        mean_factor=2,
        opacity_factor=5,
        color_factor=10,
        sdf_reg_begin=0.05,
        num_steps_per_val=5,
        num_steps_per_reset=80,
        mixed_precision=False,
    ),
    cuda=0,
    seed=1,
)

lego_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=pathlib.Path('data') / 'blender' / 'lego',
    ),
    model=SplattableMesh(z_up=True, scale=0.8, resolution=192),
    experiment=Experiment(name='smesh'),
    trainer=SplattableMeshTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        mixed_precision=False,
    ),
    cuda=0,
    seed=1,
)

ficus_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=pathlib.Path('data') / 'blender' / 'ficus',
    ),
    model=SplattableMesh(z_up=True, scale=0.85, resolution=192),
    experiment=Experiment(name='smesh'),
    trainer=SplattableMeshTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        mixed_precision=False,
    ),
    cuda=0,
    seed=1,
)

@dataclass
class MeshExport(Task):

    load: Path = ...

    output: Path = ...

    step: Optional[int] = None

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load, step=self.step)
        model = train_task.model
        assert isinstance(model, SplattableMesh)
        mesh, _ = model.get_geometry()
        mesh.export(self.output, only_geometry=True)

if __name__ == '__main__':
    TaskGroup(
        lego=lego_task,
        ficus=ficus_task,
        spot_s1=spot_stage_1,
        spot_s2=spot_stage_2,
        export=MeshExport(cuda=0),
    ).run()
