from __future__ import annotations

from pathlib import Path

from rfstudio.data import DepthSynthesisDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import TaskGroup
from rfstudio.engine.train import OptimizationVisualizer, TrainTask
from rfstudio.model import TetWeave
from rfstudio.trainer import TetWeaveTrainer

test_spot=TrainTask(
    dataset=DepthSynthesisDataset(path=Path('data') / 'spot'),
    model=TetWeave(),
    experiment=Experiment(name='tetweave'),
    trainer=TetWeaveTrainer(
        num_steps=4500,
        batch_size=4,
    ),
    viser=OptimizationVisualizer(up='+y', export='video'),
    cuda=0,
)

test_block=TrainTask(
    dataset=DepthSynthesisDataset(path=Path('data') / 'inputmodels'),
    model=TetWeave(),
    experiment=Experiment(name='tetweave'),
    trainer=TetWeaveTrainer(
        num_steps=4500,
        batch_size=4,
    ),
    viser=OptimizationVisualizer(up='+y', export='video'),
    cuda=0,
)

if __name__ == '__main__':
    TaskGroup(
        spot=test_spot,
        block=test_block,
    ).run()
