from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch

from rfstudio.data import MeshViewSynthesisDataset, RelightDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.graphics import Points, Texture2D, TextureCubeMap, TextureLatLng
from rfstudio.graphics.shaders import MCShader, NormalShader
from rfstudio.io import dump_float32_image
from rfstudio.loss import ChamferDistanceMetric
from rfstudio.model import GeoSplatter
from rfstudio.trainer import GeoSplatTrainer
from rfstudio.ui import console
from rfstudio.utils.colormap import IntensityColorMap
from rfstudio.utils.pretty import P
from rfstudio.visualization import Visualizer, vis_3dgs

spot_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'spot',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
    ),
    experiment=Experiment(name='geosplat', timestamp='spot'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        num_steps_per_val=25,
        batch_size=8,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

garden_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'mip360' / 'masked_garden',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        initial_guess='outdoor',
    ),
    experiment=Experiment(name='geosplat', timestamp='garden'),
    trainer=GeoSplatTrainer(
        num_steps=150,
        num_steps_per_val=25,
        kd_grad_reg_decay=150,
        ks_grad_reg_decay=150,
        batch_size=8,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

car_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'refnerf' / 'car',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        scale=0.9,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='car'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        num_steps_per_val=25,
        batch_size=8,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

coffee_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'refnerf' / 'coffee',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.9,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='coffee'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        num_steps_per_val=25,
        batch_size=8,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

ball_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'refnerf' / 'ball',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.9,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='ball'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        num_steps_per_val=25,
        batch_size=8,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

helmet_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'refnerf' / 'helmet',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.9,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='helmet'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        num_steps_per_val=25,
        batch_size=8,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

teapot_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'refnerf' / 'teapot',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.7,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='teapot'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        num_steps_per_val=25,
        batch_size=8,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

toaster_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'refnerf' / 'toaster',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.9,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='toaster'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        num_steps_per_val=25,
        batch_size=8,
        full_test_after_train=False,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

chair_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'chair',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.7,
    ),
    experiment=Experiment(name='geosplat', timestamp='chair'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

materials_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'materials',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        scale=0.8,
    ),
    experiment=Experiment(name='geosplat', timestamp='materials'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

lego_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'lego',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        scale=0.8,
        initial_guess='diffuse',
    ),
    experiment=Experiment(name='geosplat', timestamp='lego'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

lego_highres_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'tensoir' / 'lego',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=128,
        scale=0.8,
        initial_guess='diffuse',
    ),
    experiment=Experiment(name='geosplat', timestamp='lego_highres'),
    trainer=GeoSplatTrainer(
        num_steps=1500,
        batch_size=8,
        full_test_after_train=False,
        sdf_reg_begin=0.4,
        sdf_reg_end=0.2,
        sdf_reg_decay=1000,
        num_steps_per_val=50,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

drums_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'drums',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.75,
    ),
    experiment=Experiment(name='geosplat', timestamp='drums'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

hotdog_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'hotdog',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.9,
    ),
    experiment=Experiment(name='geosplat', timestamp='hotdog'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

ficus_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'ficus',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.8,
    ),
    experiment=Experiment(name='geosplat', timestamp='ficus'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

mic_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'mic',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.85,
    ),
    experiment=Experiment(name='geosplat', timestamp='mic'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

ship_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'ship',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.95,
    ),
    experiment=Experiment(name='geosplat', timestamp='ship'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

dami_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'damicornis',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
    ),
    experiment=Experiment(name='geosplat', timestamp='dami'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

s4r_air_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'Synthetic4Relight' / 'air_baloons',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        scale=0.95,
    ),
    experiment=Experiment(name='geosplat', timestamp='s4r_air'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

s4r_chair_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'Synthetic4Relight' / 'chair',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        scale=0.8,
        initial_guess='diffuse',
    ),
    experiment=Experiment(name='geosplat', timestamp='s4r_chair'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

s4r_hotdog_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'Synthetic4Relight' / 'hotdog',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.9,
    ),
    experiment=Experiment(name='geosplat', timestamp='s4r_hotdog'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

s4r_jugs_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'Synthetic4Relight' / 'jugs',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        scale=0.9,
    ),
    experiment=Experiment(name='geosplat', timestamp='s4r_jugs'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=100,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

tsir_arm_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'tensoir' / 'armadillo',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        initial_guess='diffuse',
        scale=0.85,
    ),
    experiment=Experiment(name='geosplat', timestamp='tsir_arm'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

tsir_lego_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'tensoir' / 'lego',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=96,
        scale=0.8,
        initial_guess='diffuse',
    ),
    experiment=Experiment(name='geosplat', timestamp='tsir_lego'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

tsir_hotdog_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'tensoir' / 'hotdog',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.9,
    ),
    experiment=Experiment(name='geosplat', timestamp='tsir_hotdog'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

tsir_ficus_task = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'tensoir' / 'ficus',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=120,
        scale=0.8,
        initial_guess='diffuse',
    ),
    experiment=Experiment(name='geosplat', timestamp='tsir_ficus'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

sorb_baking1_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'stanford_orb' / 'blender_LDR' / 'baking_scene001',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.3,
    ),
    experiment=Experiment(name='geosplat', timestamp='sorb_baking1'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

sorb_ball2_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'stanford_orb' / 'blender_LDR' / 'ball_scene002',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.3,
        initial_guess='glossy',
    ),
    experiment=Experiment(name='geosplat', timestamp='sorb_ball2'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

sorb_blocks2_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'stanford_orb' / 'blender_LDR' / 'blocks_scene002',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.25,
    ),
    experiment=Experiment(name='geosplat', timestamp='sorb_blocks2'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

sorb_cactus1_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'stanford_orb' / 'blender_LDR' / 'cactus_scene001',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.35,
        initial_guess='glossy',
    ),
    experiment=Experiment(name='geosplat', timestamp='sorb_cactus1'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

sorb_car4_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'stanford_orb' / 'blender_LDR' / 'car_scene004',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.25,
        initial_guess='diffuse',
    ),
    experiment=Experiment(name='geosplat', timestamp='sorb_car4'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

sorb_cup3_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'stanford_orb' / 'blender_LDR' / 'cup_scene003',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.3,
        initial_guess='glossy',
    ),
    experiment=Experiment(name='geosplat', timestamp='sorb_cup3'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

sorb_curry1_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'stanford_orb' / 'blender_LDR' / 'curry_scene001',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.3,
        initial_guess='glossy',
    ),
    experiment=Experiment(name='geosplat', timestamp='sorb_curry1'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

sorb_gnome7_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'stanford_orb' / 'blender_LDR' / 'gnome_scene007',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.35,
        initial_guess='glossy',
    ),
    experiment=Experiment(name='geosplat', timestamp='sorb_gnome7'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

sorb_grogu1_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'stanford_orb' / 'blender_LDR' / 'grogu_scene001',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.3,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='sorb_grogu1'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

sorb_pepsi2_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'stanford_orb' / 'blender_LDR' / 'pepsi_scene002',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.3,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='sorb_pepsi2'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

sorb_pitcher1_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'stanford_orb' / 'blender_LDR' / 'pitcher_scene001',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.35,
        initial_guess='specular',
    ),
    experiment=Experiment(name='geosplat', timestamp='sorb_pitcher1'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

sorb_teapot1_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'stanford_orb' / 'blender_LDR' / 'teapot_scene001',
    ),
    model=GeoSplatter(
        background_color='white',
        resolution=72,
        scale=0.3,
        initial_guess='glossy',
    ),
    experiment=Experiment(name='geosplat', timestamp='sorb_teapot1'),
    trainer=GeoSplatTrainer(
        num_steps=500,
        batch_size=8,
        full_test_after_train=False,
        num_steps_per_val=25,
        mixed_precision=False
    ),
    cuda=0,
    seed=1
)

@dataclass
class Viser(Task):

    load: Path = ...

    mode: Literal['mesh', 'gs'] = 'mesh'

    viser: Visualizer = Visualizer(port=6787)

    @torch.no_grad()
    def run(self) -> None:

        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load)
            model = train_task.model
            assert isinstance(model, GeoSplatter)

            mesh, gs, _, _, _ = model.get_gsplat(sampling='face')
            cov3d = gs.gaussians.get_cov3d_shape(iso_values=2.0)
            vertices = mesh.vertices # [V, 3]
            suggested_scale = torch.quantile(vertices.abs().max(-1).values, 0.95) + 0.05
        console.print(P@'Suggested Scale: {suggested_scale:.2f}')

        if self.mode == 'mesh':
            with self.viser.customize() as handle:
                dataset = train_task.dataset
                if isinstance(dataset, MeshViewSynthesisDataset):
                    gt_mesh = dataset.get_meta(split='train')
                    if gt_mesh is not None:
                        handle['gt'].show(gt_mesh).configurate(normal_size=0.02)
                handle['mesh'].show(mesh).configurate(normal_size=0.02)
                handle['cov3d'].show(cov3d)
                handle['dataset'].show(dataset)
        elif self.mode == 'gs':
            vis_3dgs(splats=model.get_splats(), port=self.viser.port)
        else:
            raise ValueError("The argument `mode` must be one of mesh and gs.")

@dataclass
class Evaler(Task):

    load: Path = ...

    step: Optional[int] = None

    num_chamfer_samples: Optional[int] = None

    geometric: bool = True

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            assert isinstance(model, GeoSplatter)
            model.eval()
            mesh, _ = model.get_geometry()
        dataset = train_task.dataset
        assert isinstance(dataset, MeshViewSynthesisDataset)
        gt_mesh = dataset.get_meta(split='train')
        if gt_mesh is not None and self.geometric:
            with console.status(desc='Computing Chamfer Distance'):
                chamfer = ChamferDistanceMetric(target_num_points=self.num_chamfer_samples)(gt_mesh, mesh)
            console.print(P@'Chamfer Distance: {chamfer:.6f}')
        with console.progress(desc='Evaluating Test View', transient=True) as ptrack:
            psnrs = []
            test_iter = dataset.get_test_iter(1)
            for inputs, gt_outputs, indices in ptrack(test_iter, total=dataset.get_size(split='test')):
                _, metrics, _ = train_task.trainer.step(
                    model,
                    inputs,
                    gt_outputs,
                    indices=None,
                    training=False,
                    visual=False,
                )
                psnrs.append(metrics['splat-psnr'])
            psnr = torch.stack(psnrs).mean()
            console.print(P@'PSNR: {psnr:.3f}')

@dataclass
class GeoEvaler(Task):

    '''
    MAE Computation:
        https://github.com/lzhnb/GS-IR/blob/e5a030b3957c11bb70d893cbb74835065d87ee72/normal_eval.py#L12
    '''

    load: Path = ...

    step: Optional[int] = None

    num_chamfer_samples: Optional[int] = 1000000

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            assert isinstance(model, GeoSplatter)
            model.eval()
            mesh, _ = model.get_geometry()
        dataset = train_task.dataset
        assert isinstance(dataset, MeshViewSynthesisDataset)
        with console.progress(desc='Evaluating Normal Quality', transient=True) as ptrack:
            maes = []
            normal_bg = torch.tensor([0, 0, 1]).float().to(self.device)
            test_iter = dataset.get_test_iter(1)
            gt_mesh = dataset.get_meta(split='train')
            assert gt_mesh is not None
            for inputs, gt_outputs, indices in ptrack(test_iter, total=dataset.get_size(split='test')):
                pred = model.render_report(inputs, indices=indices)[2].item() # [H, W, 4]
                pred = torch.add(
                    pred[..., :3] / pred[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-6) * pred[..., 3:],
                    normal_bg * (1 - pred[..., 3:]),
                ) # [H, W, 3]
                gt = gt_mesh.render(inputs, shader=NormalShader()).item()
                gt = torch.add(
                    gt[..., :3] / gt[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-6) * gt[..., 3:],
                    normal_bg * (1 - gt[..., 3:]),
                ) # [H, W, 3]
                ae = (pred * gt).sum(-1, keepdim=True).clamp(-1, 1)
                maes.append(ae.arccos().rad2deg().mean())
            maes = torch.stack(maes).mean()
            console.print(P@'MAE: {maes:.3f}')
        with console.status(desc='Computing Chamfer Distance'):
            chamfer = ChamferDistanceMetric(target_num_points=self.num_chamfer_samples)(gt_mesh, mesh)
            console.print(P@'Chamfer Distance: {chamfer:.6f}')

@dataclass
class ExportMesh(Task):
    load: Path = ...

    output: Optional[Path] = None

    denoising_threshold: float = 0.001

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load)
            model = train_task.model
            assert isinstance(model, GeoSplatter)
            assert isinstance(train_task.dataset, MeshViewSynthesisDataset)
            mesh, gsplat, _, _ = model.get_gsplat_from_face()
            mask = gsplat.gaussians.scales.exp().max(-1).values > self.denoising_threshold
            pcd = Points(positions=gsplat.gaussians.means[mask, :])
            gt_mesh = train_task.dataset.get_meta(split='train')
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
            mesh.export(self.output / 'recon.ply', only_geometry=True)
            pcd.export(self.output / 'pcd.ply')
            if gt_mesh is not None:
                gt_mesh.export(self.output / 'gt.ply', only_geometry=True)
        else:
            mesh.export(train_task.experiment.dump_path / 'recon.ply', only_geometry=True)
            if gt_mesh is not None:
                gt_mesh.export(train_task.experiment.dump_path / 'gt.ply', only_geometry=True)
            pcd.export(train_task.experiment.dump_path / 'pcd.ply')

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
            assert isinstance(model, GeoSplatter)
            assert isinstance(dataset, (MeshViewSynthesisDataset, RelightDataset))
        with console.progress(desc='Rendering Test View', transient=True) as ptrack:
            bg_color = model.get_background_color()
            idx = 0
            for inputs, gt_outputs, _ in ptrack(dataset.get_test_iter(1), total=dataset.get_size(split='test')):
                pbra, vis, normal, _, _ = model.render_report(inputs, indices=None)
                rgb = pbra.rgb2srgb().blend(bg_color).item().clamp(0, 1)
                albedo = vis[0].item().clamp(0, 1)
                occlusion = vis[1].item()[..., 0:1].expand_as(rgb).clamp(0, 1)
                roughness = vis[1].item()[..., 1:2].expand_as(rgb).clamp(0, 1)
                metallic = vis[1].item()[..., 2:3].expand_as(rgb).clamp(0, 1)
                train_task.experiment.dump_image('pbr', index=idx, image=rgb)
                train_task.experiment.dump_image('normal', index=idx, image=normal.item().clamp(0, 1))
                train_task.experiment.dump_image('albedo', index=idx, image=albedo)
                train_task.experiment.dump_image('occlusion', index=idx, image=occlusion)
                train_task.experiment.dump_image('roughness', index=idx, image=roughness)
                train_task.experiment.dump_image('metallic', index=idx, image=metallic)
                train_task.experiment.dump_image('reference', index=idx, image=gt_outputs.item().clamp(0, 1))
                idx += 1
            envmap = TextureCubeMap(data=model.cubemap, transform=None).visualize(
                width=rgb.shape[1],
                height=rgb.shape[0],
            ).item().clamp(0, 1)
            train_task.experiment.dump_image('light', index=0, image=envmap)

@dataclass
class GeoNVS(Task):

    load: Path = ...

    step: Optional[int] = None

    view: int = ...

    output: Path = ...

    with_gt: bool = False

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            assert isinstance(model, GeoSplatter)
        dataset = train_task.dataset
        camera = dataset.get_inputs(split='test')[self.view]
        camera_pos = camera.c2w[:3, 3] # [3]
        mesh, gsplat, _, _ = model.get_gsplat_from_face()
        mesh_normal_map = mesh.render(camera, shader=NormalShader()).visualize((1, 1, 1))
        visibilities = ((camera_pos - gsplat.gaussians.means) * gsplat.gaussians.colors).sum(-1) > 0
        gsplat.gaussians = gsplat.gaussians[visibilities]
        normals = gsplat.gaussians.colors
        gsplat.gaussians.replace_(colors=normals * 0.5 + 0.5)
        gs_normal_map = gsplat.render_rgb(camera.view(-1)).item()
        image = torch.cat((mesh_normal_map, gs_normal_map), dim=1)
        dump_float32_image(self.output, image.clamp(0, 1))

@dataclass
class Export(Task):

    load: Path = ...
    output: Path = ...

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, GeoSplatter)
        model.export_splats(self.output)

@dataclass
class RenderShadowEffect(Task):

    load: Path = ...

    step: Optional[int] = None

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            dataset = train_task.dataset
            assert isinstance(model, GeoSplatter)
            assert isinstance(dataset, (MeshViewSynthesisDataset, RelightDataset))
            mesh, _ = model.get_geometry()
            mesh.replace_(
                kd=Texture2D.from_constants((119/255, 150/255, 154/255), device=mesh.device),
                ks=Texture2D.from_constants((0.6, 0.25, 0.05), device=mesh.device),
                uvs=mesh.vertices.new_zeros(mesh.num_faces, 3, 2)
            )
            envmap: TextureLatLng = model.get_envmap('latlng')[0]
            envmap.data.mul_(model.exposure_params.exp().mean())
        with console.progress(desc='Rendering Test View', transient=True) as ptrack:
            bg_color = model.get_background_color()
            idx = 0
            for inputs in ptrack(dataset.get_inputs(split='test')):
                mc_render = mesh.render(inputs, shader=MCShader(envmap=envmap)).blend(bg_color).item()
                train_task.experiment.dump_image('shadow', index=idx, image=mc_render.clamp(0, 1))
                idx += 1
            train_task.experiment.dump_image('light', index=0, image=envmap.visualize().item().clamp(0, 1))
            train_task.experiment.dump_image(
                'light-intensity',
                index=0,
                image=envmap.replace(data=IntensityColorMap()(envmap.data.mean(-1, keepdim=True))).visualize().item(),
            )

if __name__ == '__main__':
    TaskGroup(
        # spot dataset
        spot=spot_task,
        garden=garden_task,

        # Shiny Blender
        ball=ball_task,
        car=car_task,
        coffee=coffee_task,
        helmet=helmet_task,
        teapot=teapot_task,
        toaster=toaster_task,

        # Nerf sysnhetic dataset
        mic=mic_task,
        lego=lego_task,
        ship=ship_task,
        drums=drums_task,
        chair=chair_task,
        ficus=ficus_task,
        hotdog=hotdog_task,
        materials=materials_task,
        lego_highres=lego_highres_task,

        # Damicornis dataset
        dami=dami_task,

        # Synthetic4Relight dataset
        s4r_air=s4r_air_task,
        s4r_chair=s4r_chair_task,
        s4r_hotdog=s4r_hotdog_task,
        s4r_jugs=s4r_jugs_task,

        # TensoIR dataset
        tsir_arm=tsir_arm_task,
        tsir_lego=tsir_lego_task,
        tsir_ficus=tsir_ficus_task,
        tsir_hotdog=tsir_hotdog_task,

        # Stanford ORB dataset
        sorb_baking1=sorb_baking1_task,
        sorb_ball2=sorb_ball2_task,
        sorb_blocks2=sorb_blocks2_task,
        sorb_cactus1=sorb_cactus1_task,
        sorb_car4=sorb_car4_task,
        sorb_cup3=sorb_cup3_task,
        sorb_curry1=sorb_curry1_task,
        sorb_gnome7=sorb_gnome7_task,
        sorb_grogu1=sorb_grogu1_task,
        sorb_pepsi2=sorb_pepsi2_task,
        sorb_pitcher1=sorb_pitcher1_task,
        sorb_teapot1=sorb_teapot1_task,

        # Visualization and evaluation
        eval=Evaler(cuda=0),
        shadow=RenderShadowEffect(cuda=0),
        geoeval=GeoEvaler(cuda=0),
        export=Export(cuda=0),
        geoexport=ExportMesh(cuda=0),
        pbr=PBRRenderer(cuda=0),
        vis=Viser(cuda=0),
    ).run()
