from __future__ import annotations

import pathlib

from rfstudio.data import MeshViewSynthesisDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.train import TrainTask
from rfstudio.model import MeshSplatter
from rfstudio.trainer import MeSplatTrainer

me_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=(
            pathlib.Path('..') /
            'data' /
            'ShapeNetCore.v2' /
            '02691156' /
            '1a04e3eab45ca15dd86060f189eb133'
        ),
    ),
    model=MeshSplatter(
        background_color='black',
        sh_degree=0,
    ),
    experiment=Experiment(name='mesplat'),
    trainer=MeSplatTrainer(
        num_steps=1000,
        batch_size=1,
        mixed_precision=False,
        hold_after_train=False,
    ),
    auto_breakpoint=False,
    traceback_pretty=False,
    cuda=0,
    seed=1
)

if __name__ == '__main__':
    me_task.run()
