from __future__ import annotations

from pathlib import Path

from rfstudio.data.dataset import MultiView2DDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.train import TrainTask
from rfstudio.model.density_field.tiny_nerf2d import TinyNeRF2D
from rfstudio.trainer.tinynerf2d_trainer import TinyNeRF2DTrainer

default_train_task = TrainTask(
    dataset=MultiView2DDataset(
        path=Path('synthetic:/2d/circle'),
    ),
    model=TinyNeRF2D(),
    experiment=Experiment(name='tinynerf2d'),
    trainer=TinyNeRF2DTrainer(
        num_steps=3000,
        batch_size=32,
        lr=5e-4,
        max_norm=0.5,
        num_steps_per_val=50,
        num_steps_per_save=2500,
        mixed_precision=False,
    ),
    cuda=0,
    seed=1,
)


if __name__ == '__main__':
    default_train_task.run()
