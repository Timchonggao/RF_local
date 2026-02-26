from __future__ import annotations

import pathlib

from rfstudio.data import MultiViewDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.train import TrainTask
from rfstudio.model import TinyNeRF
from rfstudio.trainer import TinyNeRFTrainer

default_train_task = TrainTask(
    dataset=MultiViewDataset(
        path=pathlib.Path('data') / 'blender' / 'chair',
    ),
    model=TinyNeRF(),
    experiment=Experiment(name='tinynerf'),
    trainer=TinyNeRFTrainer(
        num_steps=40000,
        batch_size=4,
        lr=5e-4,
        max_norm=0.5,
        num_rays_per_batch=8192,
        num_steps_per_val=400,
        num_steps_per_save=10000,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)


if __name__ == '__main__':
    default_train_task.run()
