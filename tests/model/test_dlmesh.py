from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from rfstudio.data import MeshViewSynthesisDataset, RelightDataset
from rfstudio.data.dataparser import MaskedBlenderDataparser
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.graphics import PBRAImages, RGBAImages, Texture2D, TextureLatLng, TriangleMesh
from rfstudio.graphics.shaders import FlatShader, MCShader
from rfstudio.io import dump_float32_image
from rfstudio.loss import ImageL2Loss
from rfstudio.model import DLMesh
from rfstudio.trainer import DLMeshTrainer
from rfstudio.ui import console

spot_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'spot',
    ),
    model=DLMesh(),
    experiment=Experiment(name='dlmesh'),
    trainer=DLMeshTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=25,
        num_steps_per_save=1000,
        mixed_precision=False,
    ),
    cuda=0,
    seed=1
)

hotdog_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'blender' / 'hotdog',
    ),
    model=DLMesh(gt_mesh=Path('exports') / 'hotdog.ply', z_up=True),
    experiment=Experiment(name='dlmesh'),
    trainer=DLMeshTrainer(
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
        path=Path('data') / 'blender' / 'lego_mc',
    ),
    model=DLMesh(gt_mesh=Path('exports') / 'lego.ply'),
    experiment=Experiment(name='dlmesh'),
    trainer=DLMeshTrainer(
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
class DatasetSynthesizer(Task):

    dataset: MeshViewSynthesisDataset = MeshViewSynthesisDataset(path=...)

    model: Path = ...

    albedo: Path = ...

    output: Path = ...

    envmap: Path = ...

    z_up: bool = False

    @torch.no_grad()
    def run(self) -> None:
        self.dataset.to(self.device)
        albedo: Tensor = torch.load(self.albedo, map_location='cpu').to(self.device)
        envmap = TextureLatLng.from_image_file(self.envmap, device=self.device)
        if self.z_up:
            envmap.z_up_to_y_up_()
        shader = MCShader(normal_type='flat', envmap=envmap)
        mesh = TriangleMesh.from_file(self.model).to(self.device)
        mesh.build_texture_from_tensors_(albedo, attrs='flat')
        mesh.replace_(ks=Texture2D.from_constants((0, 1, 0), device=self.device))
        for split in ['train', 'val', 'test']:
            inputs = self.dataset.get_inputs(split=split)
            with console.progress(f'Rendering {split} set', transient=True) as ptrack:
                gt_outputs = []
                for camera in ptrack(inputs):
                    gt_outputs.append(mesh.render(camera, shader=shader).rgb2srgb().clamp(0, 1).item())
                    # dump_float32_image(Path('temp.png'), gt_outputs[-1])
            with console.status(f'Dumping {split} set'):
                MaskedBlenderDataparser.dump(inputs, RGBAImages(gt_outputs), None, path=self.output, split=split)

@dataclass
class AlbedoBaker(Task):

    dataset: RelightDataset = RelightDataset(path=...)

    model: Path = ...

    num_epoch: int = 5

    lr: float = 0.01

    output: Path = ...

    vis: bool = False

    z_up_to_y_up: bool = False

    def run(self) -> None:
        self.dataset.to(self.device)
        inputs = self.dataset.get_inputs(split='test')[...]
        gt_albedo = self.dataset.get_meta(split='test')[0][...].blend((0.5, 0.5, 0.5))
        mesh = TriangleMesh.from_file(self.model).to(self.device)
        if self.z_up_to_y_up:
            transform = torch.tensor([
                [0, -1, 0],
                [0, 0, 1],
                [-1, 0, 0],
            ]).float().to(self.device)
            mesh.replace_(vertices=(transform @ mesh.vertices.unsqueeze(-1)).squeeze(-1))
        fcolor = torch.empty(mesh.num_faces, 3, device=self.device).fill_(0.5).requires_grad_()
        shader = FlatShader(face_colors=fcolor)
        optimizer = torch.optim.Adam([fcolor], lr=self.lr)
        for e in range(self.num_epoch):
            psnr = 0
            perm = torch.randperm(len(inputs))
            with console.progress(desc=f'Epoch #{e+1}', transient=True) as ptrack:
                for i in ptrack(perm):
                    optimizer.zero_grad()
                    albedo = PBRAImages([mesh.render(inputs[i], shader=shader).item()]).rgb2srgb().blend((0.5, 0.5, 0.5))
                    loss = ImageL2Loss()(albedo, gt_albedo[i])
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        fcolor.clamp_(0, 1)
                        psnr += -10 * loss.log10().item()
                if self.vis:
                    vis = torch.cat((gt_albedo[i].item(), albedo.item()), dim=1)
                    assert vis.min() >= 0.0 and vis.max() <= 1.0
                    dump_float32_image(Path(f'temp{e+1}.png'), vis)
            print(f'Epoch #{e+1}: PSNR={psnr/len(inputs):.2f}')
        torch.save(fcolor.detach().clone(), self.output)

if __name__ == '__main__':
    TaskGroup(
        spot=spot_task,
        hotdog=hotdog_task,
        lego=lego_task,
        albedo=AlbedoBaker(cuda=0),
        dataset=DatasetSynthesizer(cuda=0),
    ).run()
