from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from rfstudio.data import MeshViewSynthesisDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.model.mesh_based.neufam import NeuFAM
from rfstudio.trainer.neufam_trainer import NeuFAMTrainer
from rfstudio.ui import console
from rfstudio.visualization import Visualizer

spot_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'spot',
    ),
    model=NeuFAM(
        resolution=48,
        background_color='white',
    ),
    experiment=Experiment(name='neufam'),
    trainer=NeuFAMTrainer(
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
class Viser(Task):

    load: Path = ...

    viser: Visualizer = Visualizer(port=6789)

    num_chamfer_samples: Optional[int] = None

    output: Optional[Path] = None

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load)
            model = train_task.model
            assert isinstance(model, NeuFAM)
            mesh, _ = model.get_geometry()
        with self.viser.customize() as handle:
            dataset = train_task.dataset
            if isinstance(dataset, MeshViewSynthesisDataset):
                gt_mesh = dataset.get_meta(split='train')
                handle['gt'].show(gt_mesh).configurate(normal_size=0.02)
            handle['mesh'].show(mesh).configurate(normal_size=0.02)

if __name__ == '__main__':
    TaskGroup(
        spot=spot_task,
        vis=Viser(cuda=0),
    ).run()
