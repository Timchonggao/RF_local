from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.data import DepthSynthesisDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.model import DiffDR
from rfstudio.trainer import DiffDRTrainer

spot_task = TrainTask(
    dataset=DepthSynthesisDataset(
        path=Path('data') / 'spot',
    ),
    model=DiffDR(),
    experiment=Experiment(name='diffdr'),
    trainer=DiffDRTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=25,
        num_steps_per_save=1000,
        mixed_precision=False,
    ),
    cuda=0,
    seed=1
)

lego_task = TrainTask(
    dataset=DepthSynthesisDataset(
        path=Path('data') / 'lego',
    ),
    model=DiffDR(z_up=True, resolution=192),
    experiment=Experiment(name='diffdr'),
    trainer=DiffDRTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=25,
        num_steps_per_save=1000,
        mixed_precision=False,
    ),
    cuda=0,
    seed=1
)

@dataclass
class MeshExport(Task):

    load: Path = ...

    output: Path = ...

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, DiffDR)
        fused_mesh, _ = model.get_geometry()
        fused_mesh.export(self.output, only_geometry=True)

if __name__ == '__main__':
    TaskGroup(
        spot=spot_task,
        lego=lego_task,
        export=MeshExport(cuda=0),
    ).run()
