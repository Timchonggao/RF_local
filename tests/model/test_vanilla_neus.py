from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rfstudio.data import MultiViewDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.model import VanillaNeuS
from rfstudio.trainer import VanillaNeuSTrainer
from rfstudio.visualization import Visualizer

chair_task = TrainTask(
    dataset=MultiViewDataset(
        path=Path('data') / 'blender' / 'chair',
    ),
    model=VanillaNeuS(background='none'),
    experiment=Experiment(name='vanillaneus'),
    trainer=VanillaNeuSTrainer(
        num_steps=100000,
        batch_size=4,
        lr=5e-4,
        num_rays_per_batch=1024,
        num_steps_per_val=500,
        num_steps_per_save=10000,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

hotdog_task = TrainTask(
    dataset=MultiViewDataset(
        path=Path('data') / 'blender' / 'hotdog',
    ),
    model=VanillaNeuS(background='none'),
    experiment=Experiment(name='vanillaneus'),
    trainer=VanillaNeuSTrainer(
        num_steps=100000,
        batch_size=4,
        lr=5e-4,
        num_rays_per_batch=1024,
        num_steps_per_val=500,
        num_steps_per_save=10000,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

garden_task = TrainTask(
    dataset=MultiViewDataset(
        path=Path('data') / 'mip360' / 'garden',
    ),
    model=VanillaNeuS(),
    experiment=Experiment(name='vanillaneus'),
    trainer=VanillaNeuSTrainer(
        num_steps=300000,
        batch_size=4,
        lr=5e-4,
        num_rays_per_batch=1024,
        num_steps_per_val=500,
        num_steps_per_save=10000,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)


@dataclass
class Export(Task):

    load: Path = ...
    step: Optional[int] = None
    output: Optional[Path] = None
    viser: Visualizer = Visualizer()

    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load, step=self.step)
        model = train_task.model
        assert isinstance(model, VanillaNeuS)
        mesh = model.extract_mesh()
        if self.output is not None:
            mesh.export(self.output, only_geometry=True)
        self.viser.show(mesh=mesh)


if __name__ == '__main__':
    TaskGroup(
        chair=chair_task,
        hotdog=hotdog_task,
        garden=garden_task,
        export=Export(cuda=0),
    ).run()
