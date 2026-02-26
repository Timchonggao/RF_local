from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from rfstudio.data import MeshViewSynthesisDataset, RelightDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.graphics.shaders import PrettyShader
from rfstudio.model import GeoSplatterS2
from rfstudio.trainer import GeoSplatS2Trainer
from rfstudio.ui import console

spot_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'spot',
    ),
    model=GeoSplatterS2(
        load=Path('exports') / 'spot.pkl',
        background_color='white',
    ),
    experiment=Experiment(name='geosplat_s2', timestamp='spot'),
    trainer=GeoSplatS2Trainer(
        num_steps=500,
        batch_size=8,
        num_steps_per_val=25,
        mixed_precision=False,
        full_test_after_train=False,
    ),
    cuda=0,
    seed=1
)

garden_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'mip360' / 'masked_garden',
    ),
    model=GeoSplatterS2(
        load=Path('exports') / 'garden.pkl',
        background_color='white',
    ),
    experiment=Experiment(name='geosplat_s2', timestamp='garden'),
    trainer=GeoSplatS2Trainer(
        num_steps=1000,
        batch_size=8,
        sdf_reg_begin=0.2,
        sdf_reg_decay=800,
        num_steps_per_val=25,
        mixed_precision=False,
        full_test_after_train=False,
    ),
    cuda=0,
    seed=1
)

dami_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'damicornis',
    ),
    model=GeoSplatterS2(
        load=Path('exports') / 'dami.pkl',
        background_color='white',
    ),
    experiment=Experiment(name='geosplat_s2', timestamp='dami'),
    trainer=GeoSplatS2Trainer(
        num_steps=500,
        batch_size=8,
        num_steps_per_val=25,
        mixed_precision=False,
        full_test_after_train=False,
    ),
    cuda=0,
    seed=1
)

blender_task = {}

for scene in ['chair', 'lego', 'ficus', 'mic', 'ship', 'materials', 'drums', 'hotdog']:
    blender_task[scene] = TrainTask(
        dataset=MeshViewSynthesisDataset(
            path=Path('data') / 'blender' / scene,
        ),
        model=GeoSplatterS2(
            load=Path('exports') / f'{scene}.pkl',
            # z_up=True,
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s2', timestamp=scene),
        trainer=GeoSplatS2Trainer(
            num_steps=500,
            batch_size=8,
            num_steps_per_val=25,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1
    )

blender_task['lego_highres'] = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'tensoir' / 'lego',
    ),
    model=GeoSplatterS2(
        load=Path('exports') / 'lego_highres.pkl',
        background_color='white',
    ),
    experiment=Experiment(name='geosplat_s2', timestamp='lego_highres'),
    trainer=GeoSplatS2Trainer(
        num_steps=500,
        batch_size=8,
        num_steps_per_val=25,
        mixed_precision=False,
        full_test_after_train=False,
    ),
    cuda=0,
    seed=1
)

shiny_blender_task = {}

for scene in ['car', 'coffee', 'ball', 'helmet', 'teapot', 'toaster']:
    blender_task[scene] = TrainTask(
        dataset=MeshViewSynthesisDataset(
            path=Path('data') / 'refnerf' / scene,
        ),
        model=GeoSplatterS2(
            load=Path('exports') / f'{scene}.pkl',
            # z_up=True,
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s2', timestamp=scene),
        trainer=GeoSplatS2Trainer(
            num_steps=1000,
            batch_size=8,
            num_steps_per_val=25,
            normal_grad_reg_decay=250,
            sdf_reg_decay=1000,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1
    )

sorb_task = {}

for scene in [
    'baking1',
    'ball2',
    'blocks2',
    'cactus1',
    'car4',
    'cup3',
    'curry1',
    'gnome7',
    'grogu1',
    'pepsi2',
    'pitcher1',
    'teapot1',
]:
    path = scene[:-1] + '_scene00' + scene[-1]
    scene = 'sorb_' + scene
    sorb_task[scene] = TrainTask(
        dataset=MeshViewSynthesisDataset(
            path=Path('data') / 'stanford_orb' / 'blender_LDR' / path,
        ),
        model=GeoSplatterS2(
            load=Path('exports') / f'{scene}.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s2', timestamp=scene),
        trainer=GeoSplatS2Trainer(
            num_steps=500,
            batch_size=8,
            num_steps_per_val=25,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1
    )

s4r_task = {}

for scene in ['air_baloons', 'jugs', 'chair', 'hotdog']:
    if scene == 'air_baloons':
        desc = 's4r_air'
    else:
        desc = f's4r_{scene}'
    s4r_task[desc] = TrainTask(
        dataset=RelightDataset(
            path=Path('data') / 'Synthetic4Relight' / scene,
        ),
        model=GeoSplatterS2(
            load=Path('exports') / f'{desc}.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s2', timestamp=desc),
        trainer=GeoSplatS2Trainer(
            num_steps=500,
            batch_size=8,
            num_steps_per_val=25,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1
    )

tsir_task = {}

for scene in ['lego', 'armadillo', 'ficus', 'hotdog']:
    if scene == 'armadillo':
        desc = 'tsir_arm'
    else:
        desc = f'tsir_{scene}'
    tsir_task[desc] = TrainTask(
        dataset=RelightDataset(
            path=Path('data') / 'tensoir' / scene,
        ),
        model=GeoSplatterS2(
            load=Path('exports') / f'{desc}.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s2', timestamp=desc),
        trainer=GeoSplatS2Trainer(
            num_steps=500,
            batch_size=8,
            num_steps_per_val=25,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1
    )

@dataclass
class Export(Task):

    load: Path = ...
    output: Path = ...

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, GeoSplatterS2)
        model.export_splats(self.output)

@dataclass
class PBRRenderer(Task):

    load: Path = ...

    step: Optional[int] = None

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            dataset = train_task.dataset
            assert isinstance(model, GeoSplatterS2)
            assert isinstance(dataset, (MeshViewSynthesisDataset, RelightDataset))
            mesh, _ = model.get_geometry()
        with console.progress(desc='Rendering Test View', transient=True) as ptrack:
            bg_color = model.get_background_color()
            idx = 0
            for inputs, gt_outputs, _ in ptrack(dataset.get_test_iter(1), total=dataset.get_size(split='test')):
                pbra, _, _, _, _ = model.render_report(inputs, indices=None, gt_outputs=gt_outputs)
                rgb = pbra.rgb2srgb().blend(bg_color).item().clamp(0, 1)
                mesh_rgb = mesh.render(
                    inputs[0].resize(2),
                    shader=PrettyShader(occlusion_type='none', z_up=True, wireframe=True),
                ).rgb2srgb().blend(bg_color).resize_to(800, 800).item().clamp(0, 1)
                train_task.experiment.dump_image('pbr', index=idx, image=rgb)
                train_task.experiment.dump_image('mesh', index=idx, image=mesh_rgb)
                idx += 1

if __name__ == '__main__':
    TaskGroup(
        **s4r_task,
        **tsir_task,
        **sorb_task,
        **blender_task,
        **shiny_blender_task,
        spot=spot_task,
        dami=dami_task,
        garden=garden_task,
        export=Export(cuda=0),
        pbr=PBRRenderer(cuda=0),
    ).run()
