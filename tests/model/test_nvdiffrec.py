from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from rfstudio.data import MeshViewSynthesisDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.loss import ChamferDistanceMetric
from rfstudio.model import NVDiffRec
from rfstudio.trainer import NVDiffRecTrainer
from rfstudio.ui import console
from rfstudio.utils.pretty import P
from rfstudio.visualization import Visualizer

spot_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'spot',
    ),
    model=NVDiffRec(
        resolution=48,
        geometry='flexicubes',
        background_color='white',
        min_roughness=0.1,
    ),
    experiment=Experiment(name='nvdiffrec_spot', timestamp='vanilla'),
    trainer=NVDiffRecTrainer(
        light_lr=3e-3,
        appearance_lr=3e-3,
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=25,
        num_steps_per_save=1000,
        mixed_precision=False,
    ),
    cuda=0,
    seed=1
)

chair_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'chair',
    ),
    model=NVDiffRec(
        resolution=64,
        geometry='flexicubes',
        background_color='white',
        min_roughness=0.1,
    ),
    experiment=Experiment(name='nvdiffrec', timestamp='chair'),
    trainer=NVDiffRecTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=25,
        num_steps_per_save=1000,
        mixed_precision=False,
    ),
    cuda=0,
    seed=1
)

materials_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'materials',
    ),
    model=NVDiffRec(
        resolution=96,
        geometry='flexicubes',
        background_color='white',
        min_roughness=0.1,
    ),
    experiment=Experiment(name='nvdiffrec', timestamp='materials'),
    trainer=NVDiffRecTrainer(
        appearance_lr=0.003,
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
    model=NVDiffRec(
        resolution=96,
        geometry='flexicubes',
        background_color='white',
        min_roughness=0.25,
    ),
    experiment=Experiment(name='nvdiffrec_lego'),
    trainer=NVDiffRecTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=25,
        num_steps_per_save=1000,
        mixed_precision=False,
    ),
    cuda=0,
    seed=1
)


dami_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'damicornis',
    ),
    model=NVDiffRec(
        resolution=64,
        geometry='flexicubes',
        background_color='white',
        min_roughness=0.1,
    ),
    experiment=Experiment(name='nvdiffrec', timestamp='dami'),
    trainer=NVDiffRecTrainer(
        num_steps=5000,
        batch_size=4,
        num_steps_per_val=25,
        num_steps_per_save=1000,
        mixed_precision=False,
    ),
    cuda=0,
    seed=1
)

dtu_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'dtu' / 'dtu_scan24',
    ),
    model=NVDiffRec(
        resolution=48,
        geometry='flexicubes',
        background_color='black',
        min_roughness=0.1,
    ),
    experiment=Experiment(name='nvdiffrec', timestamp='dtu_scan24'),
    trainer=NVDiffRecTrainer(
        num_steps=5000,
        batch_size=4,
        num_steps_per_val=25,
        num_steps_per_save=1000,
        mixed_precision=False,
        use_mask=False,
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
            assert isinstance(model, NVDiffRec)
            mesh, _ = model.get_geometry()
        with self.viser.customize() as handle:
            dataset = train_task.dataset
            if isinstance(dataset, MeshViewSynthesisDataset):
                gt_mesh = dataset.get_meta(split='train')
                handle['gt'].show(gt_mesh).configurate(normal_size=0.02)
            handle['mesh'].show(mesh).configurate(normal_size=0.02)

@dataclass
class Evaler(Task):
    load: Path = ...

    num_chamfer_samples: Optional[int] = None

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load)
            model = train_task.model
            assert isinstance(model, NVDiffRec)
            mesh, _ = model.get_geometry()
        dataset = train_task.dataset
        if isinstance(dataset, MeshViewSynthesisDataset):
            gt_mesh = dataset.get_meta(split='train')
            with console.status(desc='Computing Chamfer Distance'):
                chamfer = ChamferDistanceMetric(target_num_points=self.num_chamfer_samples)(gt_mesh, mesh)
            console.print(P@'Chamfer Distance: {chamfer:.6f}')

@dataclass
class Exporter(Task):

    load: Path = ...

    output: Path = ...

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load)
            model = train_task.model
            assert isinstance(model, NVDiffRec)
            mesh, _ = model.get_geometry()
        if self.output is not None:
            mesh.export(self.output, only_geometry=True)


if __name__ == '__main__':
    TaskGroup(
        spot=spot_task,
        lego=lego_task,
        chair=chair_task,
        materials=materials_task,
        dami=dami_task,
        dtu=dtu_task,
        export=Exporter(cuda=0),
        eval=Evaler(cuda=0),
        vis=Viser(cuda=0),
    ).run()
