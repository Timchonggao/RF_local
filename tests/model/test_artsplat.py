from __future__ import annotations

import pathlib

from rfstudio.data import DynamicDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.model import ArtSplatter
from rfstudio.trainer import ArtSplatTrainer

usb_task = TrainTask(
    dataset=DynamicDataset(
        path=pathlib.Path('..') / 'ArtGS' / 'data' / 'paris' / 'sapien' / 'USB_100109',
    ),
    model=ArtSplatter(
        background_color='white',
        sh_degree=3,
        prepare_densification=True,
        connectivity=[(0, 1, 'rotating')],
    ),
    experiment=Experiment(name='artsplat'),
    trainer=ArtSplatTrainer(
        num_steps=30000,
        batch_size=1,
        num_steps_per_val=100,
        mixed_precision=False,
        full_test_after_train=False,
    ),
    cuda=0,
    seed=1,
)

if __name__ == '__main__':
    TaskGroup(
        usb=usb_task,
        placeholder=usb_task,
    ).run()
