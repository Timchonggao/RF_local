from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

from rfstudio.data import SegTreeDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.model.mesh_based.featmesh import FeatureMesh
from rfstudio.trainer.segtreemesh_trainer import SegTreeMeshTrainer
from rfstudio.visualization import Visualizer

usb_task = TrainTask(
    dataset=SegTreeDataset(
        path=Path('data') / 'artgs' / 'usb',
    ),
    model=FeatureMesh(
        gt_mesh=...,
    ),
    experiment=Experiment(name='segtreemesh'),
    trainer=SegTreeMeshTrainer(
        num_steps=1000,
        batch_size=8,
        num_steps_per_val=25,
        mixed_precision=False,
        full_test_after_train=False,
    ),
    cuda=0,
    seed=1,
)

@dataclass
class Vis(Task):

    load: List[Path] = ...
    step: Optional[int] = None
    cluster: int = 10
    vis: Literal['mesh', 'points', 'both'] = 'points'

    def run(self) -> None:
        with Visualizer().customize() as viser:
            for p in self.load:
                train_task = TrainTask.load_from_script(p, step=self.step)
                model = train_task.model
                assert isinstance(model, FeatureMesh)
                if self.vis in ['mesh', 'both']:
                    mesh, vertex_colors = model.get_segmented_mesh(num_clusters=self.cluster)
                    viser['mesh/' + p.parent.name].show(mesh).configurate(vertex_colors=vertex_colors)
                if self.vis in ['points', 'both']:
                    pts = model.get_segmented_points(num_clusters=self.cluster)
                    viser['points/' + p.parent.name].show(pts).configurate(point_size=0.01)

if __name__ == '__main__':
    TaskGroup(
        usb=usb_task,
        vis=Vis(cuda=0),
    ).run()
