from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch

from rfstudio.data import MeshViewSynthesisDataset
from rfstudio.data.dataparser import MaskedBlenderDataparser
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import OptimizationVisualizer, TrainTask
from rfstudio.graphics import Points
from rfstudio.model.density_field.volume_splatter import VolumeSplatter
from rfstudio.trainer.volsplat_trainer import VolSplatTrainer
from rfstudio.utils.colormap import IntensityColorMap
from rfstudio.visualization import Visualizer

spot_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'spot',
    ),
    model=VolumeSplatter(),
    experiment=Experiment(name='volsplat'),
    trainer=VolSplatTrainer(
        num_steps=2000,
        batch_size=4,
        num_steps_per_val=25,
        mixed_precision=False,
    ),
    cuda=0,
    seed=1
)

hotdog_dens_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'hotdog',
        dataparser=MaskedBlenderDataparser(scale_factor=0.5),
    ),
    model=VolumeSplatter(scale=0.9, z_up=True, field_type='density'),
    experiment=Experiment(name='volsplat'),
    trainer=VolSplatTrainer(
        num_steps=2000,
        num_steps_per_save=1000,
        batch_size=4,
        num_steps_per_val=25,
        mixed_precision=False,
        base_lr=0.01,
    ),
    viser=OptimizationVisualizer(up='+z', export='video', num_ease_in_step=0),
    cuda=0,
    seed=1
)

hotdog_vaca_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'hotdog',
        dataparser=MaskedBlenderDataparser(scale_factor=0.5),
    ),
    model=VolumeSplatter(scale=0.9, z_up=True, field_type='vacancy'),
    experiment=Experiment(name='volsplat'),
    trainer=VolSplatTrainer(
        num_steps=2000,
        num_steps_per_save=1000,
        batch_size=4,
        num_steps_per_val=25,
        mixed_precision=False,
        base_lr=0.03,
        lr_decay=2000,
    ),
    cuda=0,
    seed=1
)

hotdog_repa_vaca_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'hotdog',
        dataparser=MaskedBlenderDataparser(scale_factor=0.5),
    ),
    model=VolumeSplatter(scale=0.9, z_up=True, field_type='vacancy', reparameterization='hierarchical'),
    experiment=Experiment(name='volsplat'),
    trainer=VolSplatTrainer(
        num_steps=2000,
        num_steps_per_save=1000,
        batch_size=4,
        num_steps_per_val=25,
        mixed_precision=False,
        base_lr=0.03,
        lr_decay=2000,
    ),
    cuda=0,
    seed=1
)

hotdog_sdf_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'hotdog',
        dataparser=MaskedBlenderDataparser(scale_factor=0.5),
    ),
    model=VolumeSplatter(scale=0.9, z_up=True, field_type='sdf'),
    experiment=Experiment(name='volsplat'),
    trainer=VolSplatTrainer(
        num_steps=2000,
        num_steps_per_save=1000,
        batch_size=4,
        num_steps_per_val=25,
        mixed_precision=False,
        eikonal_reg=0.5,
        lr_decay=2000,
    ),
    cuda=0,
    seed=1
)

hotdog_repa_sdf_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'hotdog',
        dataparser=MaskedBlenderDataparser(scale_factor=0.5),
    ),
    model=VolumeSplatter(scale=0.9, z_up=True, field_type='sdf', reparameterization='hierarchical'),
    experiment=Experiment(name='volsplat'),
    trainer=VolSplatTrainer(
        num_steps=2000,
        num_steps_per_save=1000,
        batch_size=4,
        num_steps_per_val=25,
        mixed_precision=False,
        eikonal_reg=0.5,
        lr_decay=2000,
    ),
    cuda=0,
    seed=1
)

hotdog_kernel_sdf_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'hotdog',
        dataparser=MaskedBlenderDataparser(scale_factor=0.5),
    ),
    model=VolumeSplatter(scale=0.9, z_up=True, field_type='sdf', use_kernel=True, reparameterization='hierarchical'),
    experiment=Experiment(name='volsplat'),
    trainer=VolSplatTrainer(
        num_steps=2000,
        num_steps_per_save=1000,
        batch_size=4,
        num_steps_per_val=25,
        mixed_precision=False,
        eikonal_reg=0.5,
        lr_decay=2000,
    ),
    cuda=0,
    seed=1
)

hotdog_mlp_sdf_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'hotdog',
        dataparser=MaskedBlenderDataparser(scale_factor=0.5),
    ),
    model=VolumeSplatter(scale=0.9, z_up=True, field_type='sdf', reparameterization='mlp'),
    experiment=Experiment(name='volsplat'),
    trainer=VolSplatTrainer(
        num_steps=2000,
        num_steps_per_save=1000,
        batch_size=4,
        num_steps_per_val=25,
        mixed_precision=False,
        eikonal_reg=0.5,
        lr_decay=2000,
    ),
    cuda=0,
    seed=1
)

lego_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'lego',
        dataparser=MaskedBlenderDataparser(scale_factor=0.5),
    ),
    model=VolumeSplatter(scale=0.9, z_up=True, field_type='vacancy'),
    experiment=Experiment(name='volsplat'),
    trainer=VolSplatTrainer(
        num_steps=2000,
        num_steps_per_save=1000,
        batch_size=4,
        num_steps_per_val=25,
        mixed_precision=False,
        anisotropy_warmup_schedule=0.8,
        base_lr=0.01,
    ),
    cuda=0,
    seed=1
)

@dataclass
class Vis(Task):

    load: Path = ...
    step: Optional[int] = None
    viser: Visualizer = Visualizer()
    num_bins: int = 10
    bin_type: Literal['value', 'quantile'] = 'value'
    log_scale: float = 10.

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load, step=self.step)
        model = train_task.model
        assert isinstance(model, VolumeSplatter)
        grid = model.get_grid()
        positions = grid.get_grid_centers().view(-1, 3) # [N, 3]
        scalars = grid.query(positions)[..., 3].contiguous() # [N]
        if self.bin_type == 'quantile':
            linspace = torch.linspace(0, 1, self.num_bins + 1) ** (1 / self.log_scale) # [B+1]
            quantiles = scalars.quantile(linspace.to(scalars)) # [B+1]
            indices = torch.searchsorted(quantiles, scalars, side='left').clamp(1, self.num_bins) - 1 # [N]
        elif self.bin_type == 'value':
            assert model.field_type == 'vacancy'
            linspace = torch.linspace(0, 1, self.num_bins + 1).to(scalars) * 2 - 1
            linspace = (linspace.sgn() * linspace.abs() ** self.log_scale).clamp(-1, 1) * 0.5 + 0.5 # [B+1]
            indices = torch.searchsorted(linspace, scalars.sigmoid(), side='left').clamp(1, self.num_bins) - 1 # [N]
        else:
            raise ValueError(self.bin_type)

        with self.viser.customize() as handle:
            colors = IntensityColorMap().from_scaled((
                torch.linspace(0, 1, self.num_bins)
                if self.bin_type == 'quantile'
                else ((linspace[:1] + linspace[1:]) / 2).cpu()
            ).unsqueeze(-1)) # [B, 3]
            for i in range(self.num_bins):
                range0 = linspace[i].item()
                range1 = linspace[i + 1].item()
                local_positions = positions[indices == i].cpu()
                configurater = handle[f'{model.field_type}/{range0:.1%}-{range1:.1%}'].show(
                    Points(
                        positions=local_positions,
                        colors=colors[i].expand_as(local_positions)
                    )
                )
                configurater.configurate(point_shape='square', point_size=0.01)

@dataclass
class MeshVis(Task):

    load: Path = ...
    step: Optional[int] = None
    viser: Visualizer = Visualizer()
    depth_idx: Optional[int] = None

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load, step=self.step)
        model = train_task.model
        assert isinstance(model, VolumeSplatter)
        with self.viser.customize() as handle:
            handle['mesh'].show(model.extract_mesh(depth_idx=self.depth_idx))

if __name__ == '__main__':
    TaskGroup(
        spot=spot_task,
        hotdog_dens=hotdog_dens_task,
        hotdog_vaca=hotdog_vaca_task,
        hotdog_repa_vaca=hotdog_repa_vaca_task,
        hotdog_sdf=hotdog_sdf_task,
        hotdog_repa_sdf=hotdog_repa_sdf_task,
        hotdog_kernel_sdf=hotdog_kernel_sdf_task,
        hotdog_mlp_sdf=hotdog_mlp_sdf_task,
        lego=lego_task,
        mesh=MeshVis(cuda=0),
        vis=Vis(cuda=0),
    ).run()
