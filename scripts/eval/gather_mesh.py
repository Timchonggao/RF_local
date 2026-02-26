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

        iter_count = 0
        with tqdm(total=full_batch_size * 6, desc="Evaluating and Extracting results", unit="frame") as pbar:
            for inputs, gt_outputs, _ in loader:

                for i in range(len(inputs)):
                    count_id = iter_count * 50 + i
                    view_num = count_id // full_batch_size
                    if view_num >= 1:
                        break
                    frame_id = count_id % full_batch_size

                    pred_mesh_path = self.baseline_mesh / f'frame_{frame_id}.ply'
                    pred_mesh_pkl_path = pred_mesh_path.with_suffix('.pkl')
                    if pred_mesh_pkl_path.exists():
                        pred_mesh = DS_TriangleMesh.deserialize(pred_mesh_pkl_path).to(self.device)
                    else:
                        pred_mesh = DS_TriangleMesh.from_file(pred_mesh_path, read_mtl=False).to(self.device)
                        pred_mesh.serialize(pred_mesh_pkl_path)

                    inv_trans = torch.tensor([[1,0,0],[0,0,1],[0,-1,0]]).to(pred_mesh.vertices)
                    pred_mesh.replace_(vertices=(inv_trans @ (pred_mesh.vertices * 2).unsqueeze(-1)).squeeze(-1)) # * 2 是把0.5的范围映射到1
                    pred_mesh.export(
                        path=self.experiment.dump_file_path(subfolder=f'{self.load.parent.parent}/gather_mesh/frame_{frame_id}', file_name=f'{self.method_name}.obj'),
                        only_geometry=True
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

        iter_count = 0
        with tqdm(total=full_batch_size * 6, desc="Evaluating and Extracting results", unit="frame") as pbar:
            for inputs, gt_outputs, _ in loader:
                for i in range(len(inputs)):
                    count_id = iter_count * 50 + i
                    view_num = count_id // full_batch_size
                    if view_num >= 1:
                        break
                    frame_id = count_id % full_batch_size

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
                        pred_mesh = pred_mesh.replace(vertices=((pred_mesh.vertices * (4/3)).unsqueeze(-1)).squeeze(-1)) # 1/3 是为了符合camera， 4 是因为原本方法train的时候缩放了4倍
                    
                    inv_trans = torch.tensor([[1,0,0],[0,0,1],[0,-1,0]]).to(pred_mesh.vertices)
                    pred_mesh.replace_(vertices=(inv_trans @ (pred_mesh.vertices * 2).unsqueeze(-1)).squeeze(-1)) # * 2 是把0.5的范围映射到1
                    pred_mesh.export(
                        path=self.experiment.dump_file_path(subfolder=f'{self.load.parent.parent}/gather_mesh/frame_{frame_id}', file_name=f'{self.method_name}.obj'),
                        only_geometry=True
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
class Eval_CMU_results(Task):

    step: Optional[int] = None

    load: Optional[Path] = None

    baseline_visual: Optional[Path] = None

    baseline_mesh: Optional[Path] = None
    
    method_name: Optional[str] = None

    def process_loader(self):
        full_batch_size = self.dataset.dataparser.model.num_frames
        loader = self.dataset.get_test_iter(batch_size=20, shuffle=False, infinite=False)

        iter_count = 0
        with tqdm(total=full_batch_size * 6, desc="Evaluating and Extracting results", unit="frame") as pbar:
            for inputs, gt_outputs, _ in loader:
                for i in range(len(inputs)):
                    count_id = iter_count * 20 + i
                    view_num = count_id // full_batch_size
                    if view_num >= 1:
                        break
                    frame_id = count_id % full_batch_size
                    
                    pred_mesh_path = self.baseline_mesh / f'frame_{frame_id}.ply'
                    pred_mesh_pkl_path = pred_mesh_path.with_suffix('.pkl')
                    if pred_mesh_pkl_path.exists():
                        pred_mesh = DS_TriangleMesh.deserialize(pred_mesh_pkl_path).to(self.device)
                    else:
                        pred_mesh = DS_TriangleMesh.from_file(pred_mesh_path, read_mtl=False).to(self.device)
                        pred_mesh.serialize(pred_mesh_pkl_path)
                    inv_trans = torch.tensor([[1,0,0],[0,-1,0],[0,0,-1]]).to(pred_mesh.vertices)
                    pred_mesh.replace_(vertices=(inv_trans@(pred_mesh.vertices * 2).unsqueeze(-1)).squeeze(-1)) # * 2 是把0.5的范围映射到1
                    pred_mesh.export(
                        path=self.experiment.dump_file_path(subfolder=f'{self.load.parent.parent}/gather_mesh/frame_{frame_id}', file_name=f'{self.method_name}.obj'),
                        only_geometry=True
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



if __name__ == '__main__':
    TaskGroup(
        evalsyn = Eval_Synthetic_results(cuda=0),
        evaldiva = Eval_Diva_results(cuda=0),
        evalcmu = Eval_CMU_results(cuda=0)
    ).run()

