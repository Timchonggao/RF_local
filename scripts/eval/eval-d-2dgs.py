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

    def process_loader(self):
        full_batch_size = self.dataset.get_size(split='fix_vis') # full time size
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='fix_vis')

        # 初始化eval存储容器
        psnrs = []
        ssims = []
        lpipss = []
        chamfer_dists = []
        normal_maes = []

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
        bg_color = self.model.get_background_color().to(self.model.device)
        with tqdm(total=full_batch_size * 6, desc="Evaluating and Extracting results", unit="frame") as pbar:
            for inputs, gt_outputs, _ in loader:
                # gt_rgbs = gt_outputs.blend(bg_color)
                for i in range(len(inputs)):
                    camera = inputs[i]
                    count_id = iter_count * 50 + i
                    view_num = count_id // full_batch_size
                    frame_id = count_id % full_batch_size

                    pred_image_path = self.baseline_visual / f'{count_id:05d}.png'
                    pred_image = load_float32_image(pred_image_path).clamp(0, 1)
                    pred_images.append(pred_image.cpu())
                    
                    # gt_image = gt_rgbs[i].clamp(0, 1).item().unsqueeze(0).to(self.device)
                    # pred_image = pred_image.unsqueeze(0).to(self.device)
                    # psnr = PSNRLoss()(gt_image, pred_image)
                    # ssim = (1 - SSIMLoss()(gt_image, pred_image))
                    # lpips = LPIPSLoss()(gt_image, pred_image)
                    # psnrs.append(psnr)
                    # ssims.append(ssim)
                    # lpipss.append(lpips)

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

                    # gt_mesh = gt_meshes[frame_id].clone()
                    # normal_bg = torch.tensor([0, 0, 1]).float().to(self.device)
                    # gt_normal_ = gt_mesh.render(camera, shader=NormalShader()).item()
                    # gt_normal = torch.add(
                    #     gt_normal_[..., :3] / gt_normal_[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-6) * gt_normal_[..., 3:],
                    #     normal_bg * (1 - gt_normal_[..., 3:]),
                    # ).to(self.device) # [H, W, 3]
                    # pred_normal_ = pred_mesh.render(camera, shader=NormalShader()).item()
                    # pred_normal = torch.add(
                    #     pred_normal_[..., :3] / pred_normal_[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-6) * pred_normal_[..., 3:],
                    #     normal_bg * (1 - pred_normal_[..., 3:]),
                    # ).to(self.device) # [H, W, 3]
                    # ae = (pred_normal * gt_normal).sum(-1, keepdim=True).clamp(-1, 1)
                    # mae = ae.arccos().rad2deg().mean()
                    # normal_maes.append(mae)

                    # self.experiment.log(f"Test view {view_num}, Frame {frame_id}: PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}, MAE: {mae}", new_logfile=f'{self.load.parent.parent}/gather_image/dynamic-2dgs_eval.txt')

                    # if view_num == 0:
                    #     inv_trans = torch.tensor([[1,0,0],[0,0,1],[0,-1,0]]).to(pred_mesh.vertices)
                    #     pred_mesh = pred_mesh.replace(vertices=(inv_trans @ (pred_mesh.vertices * (3/2) * 2).unsqueeze(-1)).squeeze(-1)).to(self.device)
                    #     gt_mesh = gt_mesh.replace(vertices=(inv_trans @ (gt_mesh.vertices * (3/2) * 2).unsqueeze(-1)).squeeze(-1)).to(self.device)
                    #     chamfer_dist = ChamferDistanceMetric(target_num_points=1000000)(gt_mesh, pred_mesh)
                    #     chamfer_dists.append(chamfer_dist)

                    # 保存 Pred 图像
                    for idx, (img, sub) in enumerate([
                        (pred_images[-1], 'pred_image'), 
                        (pred_normals[-1], 'pred_normal'), 
                        (pred_pretty_meshes[-1], 'pred_mesh')
                    ]):
                        self.experiment.dump_image(
                            f'{self.load.parent.parent}/gather_image/test_view{view_num}/frame_{frame_id}',
                            image=img, 
                            name=f'dynamic-2dgs_{sub}'
                        )
                    pbar.update(1)
                iter_count += 1
                torch.cuda.empty_cache()

        # for i in range(len(chamfer_dists)):
        #     self.experiment.log(f"Frame {i}: Chamfer Distance: {chamfer_dists[i]}")
        # mean_chamfer_dist = torch.stack(chamfer_dists).mean().item()
        # self.experiment.log(f"Mean Chamfer Distance: {mean_chamfer_dist}")
        # mean_psnr = torch.stack(psnrs).mean().item()
        # mean_ssim = torch.stack(ssims).mean().item()
        # mean_lpips = torch.stack(lpipss).mean().item()
        # mean_normal_mae = torch.stack(normal_maes).mean().item()
        # self.experiment.log(f"Test view, Mean PSNR: {mean_psnr}, Mean SSIM: {mean_ssim}, Mean LPIPS: {mean_lpips}, Mean Normal MAE: {mean_normal_mae}")

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

    def process_loader(self):
        full_batch_size = self.dataset.dataparser.model.num_frames
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)
        
        # 初始化eval存储容器
        # psnrs = []
        # ssims = []
        # lpipss = []

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
        # bg_color = self.model.get_background_color().to(self.model.device)
        with tqdm(total=full_batch_size * 6, desc="Evaluating and Extracting results", unit="frame") as pbar:
            for inputs, gt_outputs, _ in loader:
                # gt_rgbs = gt_outputs.blend(bg_color)
                for i in range(len(inputs)):
                    camera = inputs[i]
                    count_id = iter_count * 50 + i
                    view_num = count_id // full_batch_size
                    frame_id = count_id % full_batch_size

                    pred_image_path = self.baseline_visual / f'{count_id:05d}.png'
                    pred_image = load_float32_image(pred_image_path).clamp(0, 1)
                    pred_images.append(pred_image.cpu())
                    
                    # gt_image = gt_rgbs[i].clamp(0, 1).item().unsqueeze(0).to(self.device)
                    # pred_image = pred_image.unsqueeze(0).to(self.device)
                    # psnr = PSNRLoss()(gt_image, pred_image)
                    # ssim = (1 - SSIMLoss()(gt_image, pred_image))
                    # lpips = LPIPSLoss()(gt_image, pred_image)
                    # psnrs.append(psnr)
                    # ssims.append(ssim)
                    # lpipss.append(lpips)

                    
                    pred_mesh_path = self.baseline_mesh / f'frame_{frame_id}.ply'
                    pred_mesh_pkl_path = pred_mesh_path.with_suffix('.pkl')
                    if pred_mesh_pkl_path.exists():
                        pred_mesh = DS_TriangleMesh.deserialize(pred_mesh_pkl_path).to(self.device)
                    else:
                        pred_mesh = DS_TriangleMesh.from_file(pred_mesh_path, read_mtl=False).to(self.device)
                        pred_mesh.serialize(pred_mesh_pkl_path)
                    pred_mesh = pred_mesh.replace(vertices=((pred_mesh.vertices * (4/3)).unsqueeze(-1)).squeeze(-1))
                    pred_pretty_meshes.append(
                        pred_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                    )
                    pred_normals.append(
                        pred_mesh.render(camera, shader=normal_shader).visualize((1,1,1)).item().clamp(0, 1).cpu()
                    )

                    # self.experiment.log(f"Test view {view_num}, Frame {frame_id}: PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}", new_logfile=f'{self.load.parent.parent}/baselines_eval_full/dynamic-2dgs/eval.txt')

                    # 保存 Pred 图像
                    for idx, (img, sub) in enumerate([
                        (pred_images[-1], 'pred_image'), 
                        (pred_normals[-1], 'pred_normal'), 
                        (pred_pretty_meshes[-1], 'pred_mesh')
                    ]):
                        self.experiment.dump_image(
                            f'{self.load.parent.parent}/gather_image/test_view{view_num}/frame_{frame_id}',
                            image=img, 
                            name=f'dynamic-2dgs_{sub}'
                        )
                    
                    if frame_id == full_batch_size - 1:
                        # video_dict = {
                        #     'pred_mesh': pred_pretty_meshes,
                        #     'pred_image': pred_images,
                        #     'pred_normal': pred_normals,
                        # }
                        # for name, imgs in video_dict.items():
                        #     self.experiment.dump_images2video(
                        #         f'{self.load.parent.parent}/baselines_eval_full/dynamic-2dgs/extract/test_view{view_num}',
                        #         name=name, 
                        #         images=imgs, 
                        #         downsample=1, 
                        #         fps=48, 
                        #         duration=5
                        #     )
                        del pred_images, pred_normals, pred_pretty_meshes
                        torch.cuda.empty_cache()
                        pred_images, pred_normals, pred_pretty_meshes = [], [], []
                    
                    pbar.update(1)
                iter_count += 1
                torch.cuda.empty_cache()
        # mean_psnr = torch.stack(psnrs).mean().item()
        # mean_ssim = torch.stack(ssims).mean().item()
        # mean_lpips = torch.stack(lpipss).mean().item()
        # self.experiment.log(f"Test view, Mean PSNR: {mean_psnr}, Mean SSIM: {mean_ssim}, Mean LPIPS: {mean_lpips}")

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

    def process_loader(self):
        full_batch_size = self.dataset.dataparser.model.num_frames
        loader = self.dataset.get_test_iter(batch_size=20, shuffle=False, infinite=False)
        # gt_meshes = self.dataset.get_meshes(split='test')

        # 初始化eval存储容器
        # chamfer_dists = []

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

                    pred_image_path = self.baseline_visual / f'{count_id:05d}.png'
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

                    # if view_num == 0:
                    #     gt_mesh = gt_meshes[frame_id].clone()
                    #     chamfer_dist = ChamferDistanceMetric(target_num_points=1000000)(gt_mesh, pred_mesh)
                    #     chamfer_dists.append(chamfer_dist)

                    # 保存 Pred 图像
                    for idx, (img, sub) in enumerate([
                        (pred_images[-1], 'pred_image'), 
                        (pred_normals[-1], 'pred_normal'), 
                        (pred_pretty_meshes[-1], 'pred_mesh')
                    ]):
                        self.experiment.dump_image(
                            f'{self.load.parent.parent}/gather_image/test_view{view_num}/frame_{frame_id}',
                            image=img, 
                            name=f'dynamic-2dgs_{sub}'
                        )
                    
                    if frame_id == full_batch_size - 1:
                        # video_dict = {
                        #     'pred_mesh': pred_pretty_meshes,
                        #     'pred_image': pred_images,
                        #     'pred_normal': pred_normals,
                        # }
                        # for name, imgs in video_dict.items():
                        #     self.experiment.dump_images2video(
                        #         f'{self.load.parent.parent}/baselines_eval_full/dynamic-2dgs/extract/test_view{view_num}',
                        #         name=name, 
                        #         images=imgs, 
                        #         downsample=1, 
                        #         fps=48, 
                        #         duration=5
                        #     )
                        del pred_images, pred_normals, pred_pretty_meshes
                        torch.cuda.empty_cache()
                        pred_images, pred_normals, pred_pretty_meshes = [], [], []
                    
                    pbar.update(1)
                iter_count += 1
                torch.cuda.empty_cache()
        # for i in range(len(chamfer_dists)):
        #     self.experiment.log(f"Frame {i}: Chamfer Distance: {chamfer_dists[i]}", new_logfile=f'{self.load.parent.parent}/baselines_eval_full/dynamic-2dgs/eval.txt')
        # mean_chamfer_dist = torch.stack(chamfer_dists).mean().item()
        # self.experiment.log(f"Mean Chamfer Distance: {mean_chamfer_dist}")

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

