from __future__ import annotations

from pathlib import Path

from rfstudio.data import MeshViewSynthesisDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.model import DiffPBR
from rfstudio.trainer import DiffPBRTrainer

spot_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'spot',
    ),
    model=DiffPBR(),
    experiment=Experiment(name='diffpbr'),
    trainer=DiffPBRTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=25,
        num_steps_per_save=1000,
        mixed_precision=False,
    ),
    cuda=0,
    seed=1
)

hotdog_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'hotdog',
    ),
    model=DiffPBR(gt_mesh=Path('exports') / 'hotdog.ply', z_up=True),
    experiment=Experiment(name='diffpbr'),
    trainer=DiffPBRTrainer(
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
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'lego',
    ),
    model=DiffPBR(gt_mesh=Path('exports') / 'lego.smesh.fine.ply'),
    experiment=Experiment(name='diffpbr'),
    trainer=DiffPBRTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=25,
        num_steps_per_save=1000,
        mixed_precision=False,
    ),
    cuda=0,
    seed=1
)

if __name__ == '__main__':
    TaskGroup(
        spot=spot_task,
        hotdog=hotdog_task,
        lego=lego_task,
    ).run()
