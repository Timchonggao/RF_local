from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, List

from rfstudio.engine.task import Task
from rfstudio.io import load_float32_image, open_video_renderer, dump_float32_image
from rfstudio.ui import console

from natsort import natsorted
from tqdm import tqdm

# import modulues
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt

import torch
from torch import Tensor
import numpy as np

# import rfstudio modules
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.ui import console
from rfstudio.utils.pretty import P
from rfstudio.loss import ChamferDistanceMetric,PSNRLoss,LPIPSLoss,SSIMLoss
from rfstudio.graphics import Points, TriangleMesh
from rfstudio.graphics.shaders import PrettyShader, DepthShader, NormalShader
from rfstudio.graphics import DepthImages, VectorImages, PBRAImages

# import rfstudio_ds modules
from rfstudio_ds.engine.experiment import DS_Experiment
from rfstudio_ds.engine.train import DS_TrainTask
from rfstudio_ds.data import SyntheticDynamicMultiViewBlenderRGBADataset
from rfstudio_ds.model import D_Joint # rgb image loss optimization model
from rfstudio_ds.trainer import D_JointTrainer, JointRegularizationConfig # rgb image loss optimization trainer, dynamic nvdiffrec
from rfstudio_ds.graphics.spatial_sampler import NearestGridSampler
from rfstudio_ds.visualization import Visualizer
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh
from rfstudio_ds.model.density_field.components.encoding_4d import Grid4d_HashEncoding
from rfstudio_ds.nn import MLPDecoderNetwork, Grid4DDecoderNetwork




@dataclass
class Eval_Synthetic_results(Task):

    step: Optional[int] = None

    load: Optional[Path] = None

    baseline_visual: Optional[Path] = None

    baseline_mesh: Optional[Path] = None
    
    method_name: Optional[str] = None

    def process_loader(self):
        full_batch_size = self.dataset.get_size(split='fix_vis') # full time size
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)

        # 初始化extract存储容器
        pred_images, pred_normals, pred_pretty_meshes = [], [], []
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )
        normal_shader = NormalShader(
            antialias=True, 
            normal_type='vertex'
        )

        iter_count = 0
        with tqdm(total=full_batch_size * 6, desc="Evaluating and Extracting results", unit="frame") as pbar:
            for inputs, gt_outputs, _ in loader:

                for i in range(len(inputs)):
                    camera = inputs[i]
                    count_id = iter_count * 50 + i
                    view_num = count_id // full_batch_size
                    frame_id = count_id % full_batch_size

                    if self.method_name == "dgmesh":
                        pred_image_path = self.baseline_visual / f'{count_id}.png'
                    elif self.method_name == "neus2":
                        pred_image_path = self.baseline_visual / f'frame_{frame_id}' / f'psnr_{view_num:04d}.png'
                    else:
                        pred_image_path = self.baseline_visual / f'{count_id:05d}.png'
                    if self.method_name == "sc-gs":
                        bg_color = self.model.get_background_color().to(self.model.device)
                        pred_image = load_float32_image(pred_image_path,alpha_color=bg_color.cpu()).clamp(0, 1)
                    else:
                        pred_image = load_float32_image(pred_image_path).clamp(0, 1)
                    pred_images.append(pred_image.cpu())

                    if self.method_name == "neus2":
                        pred_mesh_path = self.baseline_mesh / f'frame_{frame_id:04d}.obj'
                    else:
                        pred_mesh_path = self.baseline_mesh / f'frame_{frame_id}.ply'
                    pred_mesh_pkl_path = pred_mesh_path.with_suffix('.pkl')
                    if pred_mesh_pkl_path.exists():
                        pred_mesh = DS_TriangleMesh.deserialize(pred_mesh_pkl_path).to(self.device)
                    else:
                        pred_mesh = DS_TriangleMesh.from_file(pred_mesh_path, read_mtl=False).to(self.device)
                        pred_mesh.serialize(pred_mesh_pkl_path)
                    pred_mesh = pred_mesh.replace(vertices=((pred_mesh.vertices * (2/3)).unsqueeze(-1)).squeeze(-1))
                    pred_pretty_meshes.append(
                        pred_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                    )
                    pred_normals.append(
                        pred_mesh.render(camera, shader=normal_shader).visualize((1,1,1)).item().clamp(0, 1).cpu()
                    )

                    # 保存 Pred 图像
                    for idx, (img, sub) in enumerate([
                        (pred_images[-1], 'pred_image'), 
                        (pred_normals[-1], 'pred_normal'), 
                        (pred_pretty_meshes[-1], 'pred_mesh')
                    ]):
                        self.experiment.dump_image(
                            f'{self.load.parent.parent}/gather_image/test_view{view_num}/frame_{frame_id}',
                            image=img, 
                            name=f'{self.method_name}_{sub}'
                        )
                    pbar.update(1)
                iter_count += 1
                torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self) -> None:
        print(f'Processing: {self.load}...')
        print(f'Baseline: {self.baseline_visual}')
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step, load_checkpoint=False)
            self.experiment: DS_Experiment = task.experiment
            self.model: D_Joint = task.model
            self.model.background_color = 'white'
            self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
            self.trainer: D_JointTrainer = task.trainer
            self.trainer.setup(model=self.model, dataset=self.dataset)
            self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)

        self.process_loader()
        torch.cuda.empty_cache()


@dataclass
class Eval_Diva_results(Task):

    step: Optional[int] = None

    load: Optional[Path] = None

    baseline_visual: Optional[Path] = None

    baseline_mesh: Optional[Path] = None
    
    method_name: Optional[str] = None

    def process_loader(self):
        full_batch_size = self.dataset.dataparser.model.num_frames
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)

        # 初始化extract存储容器
        pred_images, pred_normals, pred_pretty_meshes = [], [], []
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )
        normal_shader = NormalShader(
            antialias=True, 
            normal_type='vertex'
        )

        iter_count = 0
        with tqdm(total=full_batch_size * 6, desc="Evaluating and Extracting results", unit="frame") as pbar:
            for inputs, gt_outputs, _ in loader:
                for i in range(len(inputs)):
                    camera = inputs[i]
                    count_id = iter_count * 50 + i
                    view_num = count_id // full_batch_size
                    frame_id = count_id % full_batch_size

                    if self.method_name == "dgmesh":
                        pred_image_path = self.baseline_visual / f'{count_id}.png'
                    else:
                        pred_image_path = self.baseline_visual / f'{count_id:05d}.png'
                    if self.method_name == "sc-gs":
                        bg_color = self.model.get_background_color().to(self.model.device)
                        pred_image = load_float32_image(pred_image_path,alpha_color=bg_color.cpu()).clamp(0, 1)
                    else:
                        pred_image = load_float32_image(pred_image_path).clamp(0, 1)
                    pred_images.append(pred_image.cpu())
                    
                    pred_mesh_path = self.baseline_mesh / f'frame_{frame_id}.ply'
                    pred_mesh_pkl_path = pred_mesh_path.with_suffix('.pkl')
                    if pred_mesh_pkl_path.exists():
                        pred_mesh = DS_TriangleMesh.deserialize(pred_mesh_pkl_path).to(self.device)
                    else:
                        pred_mesh = DS_TriangleMesh.from_file(pred_mesh_path, read_mtl=False).to(self.device)
                        pred_mesh.serialize(pred_mesh_pkl_path)
                    if self.method_name == "dgmesh":
                        pred_mesh = pred_mesh.replace(vertices=((pred_mesh.vertices * (1/3)).unsqueeze(-1)).squeeze(-1))
                    else:
                        pred_mesh = pred_mesh.replace(vertices=((pred_mesh.vertices * (4/3)).unsqueeze(-1)).squeeze(-1))
                    pred_pretty_meshes.append(
                        pred_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                    )
                    pred_normals.append(
                        pred_mesh.render(camera, shader=normal_shader).visualize((1,1,1)).item().clamp(0, 1).cpu()
                    )

                    # 保存 Pred 图像
                    for idx, (img, sub) in enumerate([
                        (pred_images[-1], 'pred_image'), 
                        (pred_normals[-1], 'pred_normal'), 
                        (pred_pretty_meshes[-1], 'pred_mesh')
                    ]):
                        self.experiment.dump_image(
                            f'{self.load.parent.parent}/gather_image/test_view{view_num}/frame_{frame_id}',
                            image=img, 
                            name=f'{self.method_name}_{sub}'
                        )
                    
                    if frame_id == full_batch_size - 1:
                        del pred_images, pred_normals, pred_pretty_meshes
                        torch.cuda.empty_cache()
                        pred_images, pred_normals, pred_pretty_meshes = [], [], []
                    
                    pbar.update(1)
                iter_count += 1
                torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self) -> None:
        print(f'Processing: {self.load}...')
        print(f'Baseline: {self.baseline_visual}')
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step, load_checkpoint=False)
            self.experiment: DS_Experiment = task.experiment
            self.model: D_Joint = task.model
            self.model.background_color = 'white'
            self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
            self.trainer: D_JointTrainer = task.trainer
            self.trainer.setup(model=self.model, dataset=self.dataset)
            self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)

        self.process_loader()
        torch.cuda.empty_cache()

@dataclass
class Eval_CMU_results(Task):

    step: Optional[int] = None

    load: Optional[Path] = None

    baseline_visual: Optional[Path] = None

    baseline_mesh: Optional[Path] = None
    
    method_name: Optional[str] = None

    def process_loader(self):
        full_batch_size = self.dataset.dataparser.model.num_frames
        loader = self.dataset.get_test_iter(batch_size=20, shuffle=False, infinite=False)

        # 初始化extract存储容器
        pred_images, pred_normals, pred_pretty_meshes = [], [], []
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )
        normal_shader = NormalShader(
            antialias=True, 
            normal_type='vertex'
        )

        iter_count = 0
        with tqdm(total=full_batch_size * 6, desc="Evaluating and Extracting results", unit="frame") as pbar:
            for inputs, gt_outputs, _ in loader:
                for i in range(len(inputs)):
                    camera = inputs[i]
                    count_id = iter_count * 20 + i
                    view_num = count_id // full_batch_size
                    if view_num > 5:
                        break
                    frame_id = count_id % full_batch_size

                    if self.method_name == "dgmesh":
                        pred_image_path = self.baseline_visual / f'{count_id}.png'
                    else:
                        pred_image_path = self.baseline_visual / f'{count_id:05d}.png'
                    if self.method_name == "sc-gs":
                        bg_color = self.model.get_background_color().to(self.model.device)
                        pred_image = load_float32_image(pred_image_path,alpha_color=bg_color.cpu()).clamp(0, 1)
                    else:
                        pred_image = load_float32_image(pred_image_path).clamp(0, 1)
                    pred_images.append(pred_image.cpu())
                    pred_image = pred_image.unsqueeze(0).to(self.device)
                    
                    pred_mesh_path = self.baseline_mesh / f'frame_{frame_id}.ply'
                    pred_mesh_pkl_path = pred_mesh_path.with_suffix('.pkl')
                    if pred_mesh_pkl_path.exists():
                        pred_mesh = DS_TriangleMesh.deserialize(pred_mesh_pkl_path).to(self.device)
                    else:
                        pred_mesh = DS_TriangleMesh.from_file(pred_mesh_path, read_mtl=False).to(self.device)
                        pred_mesh.serialize(pred_mesh_pkl_path)
                    pred_pretty_meshes.append(
                        pred_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                    )
                    pred_normals.append(
                        pred_mesh.render(camera, shader=normal_shader).visualize((1,1,1)).item().clamp(0, 1).cpu()
                    )

                    # 保存 Pred 图像
                    for idx, (img, sub) in enumerate([
                        (pred_images[-1], 'pred_image'), 
                        (pred_normals[-1], 'pred_normal'), 
                        (pred_pretty_meshes[-1], 'pred_mesh')
                    ]):
                        self.experiment.dump_image(
                            f'{self.load.parent.parent}/gather_image/test_view{view_num}/frame_{frame_id}',
                            image=img, 
                            name=f'{self.method_name}_{sub}'
                        )
                    
                    if frame_id == full_batch_size - 1:
                        del pred_images, pred_normals, pred_pretty_meshes
                        torch.cuda.empty_cache()
                        pred_images, pred_normals, pred_pretty_meshes = [], [], []
                    
                    pbar.update(1)
                iter_count += 1
                torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self) -> None:
        print(f'Processing: {self.load}...')
        print(f'Baseline: {self.baseline_visual}')
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step, load_checkpoint=False)
            self.experiment: DS_Experiment = task.experiment
            self.model: D_Joint = task.model
            self.model.background_color = 'white'
            self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
            self.trainer: D_JointTrainer = task.trainer
            self.trainer.setup(model=self.model, dataset=self.dataset)
            self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)

        self.process_loader()
        torch.cuda.empty_cache()



if __name__ == '__main__':
    TaskGroup(
        evalsyn = Eval_Synthetic_results(cuda=0),
        evaldiva = Eval_Diva_results(cuda=0),
        evalcmu = Eval_CMU_results(cuda=0)
    ).run()

