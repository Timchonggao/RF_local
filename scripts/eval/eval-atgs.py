from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, List

from rfstudio.engine.task import Task
from rfstudio.io import load_float32_image, open_video_renderer
from rfstudio.ui import console

from natsort import natsorted

# import modulues
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt

import torch
from torch import Tensor
import numpy as np
import gc

import os
import glob
import json

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
class Extract_results(Task):

    step: Optional[int] = None

    transform_root: Path = Path('/data3/gaochong/project/RadianceFieldStudio/data/ObjSel-Dyn-multiview')

    def process_loader(self):
        """
        处理一个数据迭代器：
        - 推理模型
        - 渲染网格
        - 保存图像和 PBR 属性
        - 导出网格（可选）
        """
        case_name = self.load.parent.parts[-2]
        test_transform_path = self.transform_root / case_name / 'transforms_test.json'
        test_transform_path = Path(str(test_transform_path).replace('footballplayer', 'football_player'))
        test_transform_path = Path(str(test_transform_path).replace('spidermanfight', 'spiderman_fight'))
        with open(str(test_transform_path), 'r') as f:
            meta = json.loads(f.read())
        testcamera_names = [frame["camera_name"] for frame in meta["frames"]]
        unique_testcamera_names = list(dict.fromkeys(testcamera_names))        

        full_batch_size = self.dataset.get_size(split='fix_vis') # full time size
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='orbit_vis')
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )
        normal_shader = NormalShader(
            antialias=True, 
            normal_type='flat'
        )

        # 初始化存储容器
        gt_images, gt_normals, gt_pretty_meshes = [], [], []
        pred_images, pred_normals, pred_pretty_meshes = [], [], []

        iter_count = 0
        for inputs, gt_outputs, indices in loader:
            bg_color = self.model.get_background_color().to(self.model.device)
            gt_rgbs = gt_outputs.blend(bg_color)
            for i in range(len(inputs)):
                count_id = iter_count * 50 + i
                view_num = count_id // full_batch_size
                split_name = f'test_view{view_num}'
                frame_id = count_id % full_batch_size

                camera = inputs[i]

                gt_mesh = gt_meshes[frame_id].clone()
                gt_pretty_meshes.append(
                    gt_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                )
                gt_normals.append(
                    gt_mesh.render(camera, shader=normal_shader).visualize((1,1,1)).item().clamp(0, 1).cpu()
                )

                pred_mesh_path = self.baseline / 'meshes' / f'Frame_{frame_id:06d}.ply'
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

                gt_image = gt_rgbs[i].clamp(0, 1).item()
                pred_image_root_1 = self.baseline / f'frame_{frame_id}' / 'test'
                pred_image_root_2 = glob.glob(os.path.join(pred_image_root_1, "ours_*"))
                pred_image_root = pred_image_root_1 / pred_image_root_2[0]
                pred_image_path = pred_image_root / 'renders' / f'{unique_testcamera_names[view_num]}.png'
                pred_image = load_float32_image(pred_image_path).clamp(0, 1)
                # 切分可视化图像
                gt_images.append(gt_image.cpu())
                pred_images.append(pred_image.cpu())

                # 保存 GT 图像
                for idx, (img, sub) in enumerate([
                    (gt_images[-1], 'gt_image'), 
                    (gt_normals[-1], 'gt_normal'), 
                    (gt_pretty_meshes[-1], 'gt_mesh')
                ]):
                    self.experiment.dump_image(f'{self.load.parent}/baselines_eval/atgs/extract/{split_name}/gt/{sub}', image=img, name=f'{frame_id}')

                # 保存 Pred 图像
                for idx, (img, sub) in enumerate([
                    (pred_images[-1], 'pred_image'), 
                    (pred_normals[-1], 'pred_normal'), 
                    (pred_pretty_meshes[-1], 'pred_mesh')
                ]):
                    self.experiment.dump_image(f'{self.load.parent}/baselines_eval/atgs/extract/{split_name}/pred/{sub}', image=img, name=f'{frame_id}')

                if frame_id == full_batch_size - 1:
                    video_dict = {
                        'gt_mesh': gt_pretty_meshes,
                        'pred_mesh': pred_pretty_meshes,
                        'gt_image': gt_images,
                        'gt_normal': gt_normals,
                        'pred_image': pred_images,
                        'pred_normal': pred_normals,
                    }
                    for name, imgs in video_dict.items():
                        split_name = f'test_view{view_num}'
                        self.experiment.dump_images2video(
                            f'{self.load.parent}/baselines_eval/atgs/extract/{split_name}', 
                            name=name, 
                            images=imgs, 
                            downsample=1, 
                            fps=48, 
                            duration=5
                        )
                    del gt_images, gt_normals, gt_pretty_meshes, pred_images, pred_normals, pred_pretty_meshes
                    torch.cuda.empty_cache()
                    gc.collect()
                    # 初始化存储容器
                    gt_images, gt_normals, gt_pretty_meshes = [], [], []
                    pred_images, pred_normals, pred_pretty_meshes = [], [], []
            
            iter_count += 1
            torch.cuda.empty_cache()

        del gt_images, gt_normals, gt_pretty_meshes, pred_images, pred_normals, pred_pretty_meshes, loader, gt_meshes
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def run(self) -> None:
        # categories = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        categories = ['deer', 'spidermanfight']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint')
        cases = []
        baselines = []
        for category in categories:
            # case路径
            case = root / category / 'test' / 'task.py'
            cases.append(case)
            # baseline路径 (递归搜索)
            baseline_path = root / category / 'test' / 'baselines' / 'atgs'
            baselines.append(baseline_path)

        for case, baseline in zip(cases, baselines):
            self.load = case
            self.baseline = baseline
            print(f'Processing: {self.load}...')
            print(f'Baseline: {self.baseline}')
            with console.status(desc='Loading Model'):
                task = DS_TrainTask.load_from_script(self.load, step=self.step, load_checkpoint=False)
                self.experiment: DS_Experiment = task.experiment
                self.model: D_Joint = task.model
                self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
                self.trainer: D_JointTrainer = task.trainer
                self.trainer.setup(model=self.model, dataset=self.dataset)
                self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)

            self.process_loader()
            del self.model, self.dataset, self.trainer, self.experiment
            torch.cuda.empty_cache()
            gc.collect()


@dataclass
class Eval_results(Task):

    step: Optional[int] = None

    transform_root: Path = Path('/data3/gaochong/project/RadianceFieldStudio/data/ObjSel-Dyn-multiview')

    def process_loader(self):
        """
        处理一个数据迭代器：
        - 推理模型
        - 渲染网格
        - 保存图像和 PBR 属性
        - 导出网格（可选）
        """
        case_name = self.load.parent.parts[-2]
        test_transform_path = self.transform_root / case_name / 'transforms_test.json'
        test_transform_path = Path(str(test_transform_path).replace('footballplayer', 'football_player'))
        test_transform_path = Path(str(test_transform_path).replace('spidermanfight', 'spiderman_fight'))
        with open(str(test_transform_path), 'r') as f:
            meta = json.loads(f.read())
        testcamera_names = [frame["camera_name"] for frame in meta["frames"]]
        unique_testcamera_names = list(dict.fromkeys(testcamera_names)) 
        
        full_batch_size = self.dataset.get_size(split='fix_vis') # full time size
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='orbit_vis')
        
        # 初始化存储容器
        psnrs = []
        ssims = []
        lpipss = []
        chamfer_dists = []
        normal_maes = []

        iter_count = 0
        for inputs, gt_outputs, indices in loader:
            bg_color = self.model.get_background_color().to(self.model.device)
            gt_rgbs = gt_outputs.blend(bg_color)
            for i in range(len(inputs)):
                count_id = iter_count * 50 + i
                view_num = count_id // full_batch_size
                frame_id = count_id % full_batch_size

                gt_image = gt_rgbs[i].clamp(0, 1).item().unsqueeze(0).to(self.device)
                pred_image_root_1 = self.baseline / f'frame_{frame_id}' / 'test'
                pred_image_root_2 = glob.glob(os.path.join(pred_image_root_1, "ours_*"))
                pred_image_root = pred_image_root_1 / pred_image_root_2[0]
                pred_image_path = pred_image_root / 'renders' / f'{unique_testcamera_names[view_num]}.png'
                pred_image = load_float32_image(pred_image_path).clamp(0, 1).unsqueeze(0).to(self.device)
                psnr = PSNRLoss()(gt_image, pred_image)
                ssim = (1 - SSIMLoss()(gt_image, pred_image))
                lpips = LPIPSLoss()(gt_image, pred_image)

                camera = inputs[i]
                normal_bg = torch.tensor([0, 0, 1]).float().to(self.device)
                gt_mesh = gt_meshes[frame_id].clone()
                pred_mesh_path = self.baseline / 'meshes' / f'Frame_{frame_id:06d}.ply'
                pred_mesh_pkl_path = pred_mesh_path.with_suffix('.pkl')
                if pred_mesh_pkl_path.exists():
                    pred_mesh = DS_TriangleMesh.deserialize(pred_mesh_pkl_path).to(self.device)
                else:
                    pred_mesh = DS_TriangleMesh.from_file(pred_mesh_path, read_mtl=False).to(self.device)
                    pred_mesh.serialize(pred_mesh_pkl_path)
                pred_mesh = pred_mesh.replace(vertices=((pred_mesh.vertices * (2/3)).unsqueeze(-1)).squeeze(-1))
                gt_normal_ = gt_mesh.render(camera, shader=NormalShader()).item()
                gt_normal = torch.add(
                    gt_normal_[..., :3] / gt_normal_[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-6) * gt_normal_[..., 3:],
                    normal_bg * (1 - gt_normal_[..., 3:]),
                ).to(self.device) # [H, W, 3]
                pred_normal_ = pred_mesh.render(camera, shader=NormalShader()).item()
                pred_normal = torch.add(
                    pred_normal_[..., :3] / pred_normal_[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-6) * pred_normal_[..., 3:],
                    normal_bg * (1 - pred_normal_[..., 3:]),
                ).to(self.device) # [H, W, 3]
                
                ae = (pred_normal * gt_normal).sum(-1, keepdim=True).clamp(-1, 1)
                mae = ae.arccos().rad2deg().mean()

                self.experiment.log(f"Test view {view_num}, Frame {frame_id}: PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}, MAE: {mae}", new_logfile=f'{self.load.parent}/baselines_eval/atgs/eval.txt')
                psnrs.append(psnr)
                ssims.append(ssim)
                lpipss.append(lpips)
                normal_maes.append(mae)

                if view_num == 0:
                    inv_trans = torch.tensor([[1,0,0],[0,0,1],[0,-1,0]]).to(pred_mesh.vertices)
                    pred_mesh = pred_mesh.replace(vertices=(inv_trans @ (pred_mesh.vertices * (3/2) * 2).unsqueeze(-1)).squeeze(-1)).to(self.device)
                    gt_mesh = gt_mesh.replace(vertices=(inv_trans @ (gt_mesh.vertices * (3/2) * 2).unsqueeze(-1)).squeeze(-1)).to(self.device)
                    chamfer_dist = ChamferDistanceMetric(target_num_points=1000000)(gt_mesh, pred_mesh)
                    chamfer_dists.append(chamfer_dist)
                    pred_mesh.export(
                        path=self.experiment.dump_file_path(subfolder=f'{self.load.parent}/baselines_eval/atgs/scale_pred_mesh', file_name=f'frame{frame_id}.obj'),
                        only_geometry=True
                    )
                    gt_mesh.export(
                        path=self.experiment.dump_file_path(subfolder=f'{self.load.parent}/baselines_eval/atgs/scale_gt_mesh', file_name=f'frame{frame_id}.obj'),
                        only_geometry=True
                    )
            iter_count += 1
            torch.cuda.empty_cache()
        for i in range(len(chamfer_dists)):
            self.experiment.log(f"Frame {i}: Chamfer Distance: {chamfer_dists[i]}")
        mean_chamfer_dist = torch.stack(chamfer_dists).mean().item()
        self.experiment.log(f"Mean Chamfer Distance: {mean_chamfer_dist}")
        mean_psnr = torch.stack(psnrs).mean().item()
        mean_ssim = torch.stack(ssims).mean().item()
        mean_lpips = torch.stack(lpipss).mean().item()
        mean_normal_mae = torch.stack(normal_maes).mean().item()
        self.experiment.log(f"Test view, Mean PSNR: {mean_psnr}, Mean SSIM: {mean_ssim}, Mean LPIPS: {mean_lpips}, Mean Normal MAE: {mean_normal_mae}")

        del psnrs, ssims, lpipss, chamfer_dists, normal_maes, loader, gt_meshes
        torch.cuda.empty_cache()
        gc.collect()
    
    @torch.no_grad()
    def run(self) -> None:
        # categories = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        categories = ['spidermanfight']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint')
        cases = []
        baselines = []
        for category in categories:
            # case路径
            case = root / category / 'test' / 'task.py'
            cases.append(case)
            # baseline路径 (递归搜索)
            baseline_path = root / category / 'test' / 'baselines' / 'atgs'
            baselines.append(baseline_path)

        for case, baseline in zip(cases, baselines):
            self.load = case
            self.baseline = baseline
            print(f'Processing: {self.load}...')
            print(f'Baseline: {self.baseline}')
            with console.status(desc='Loading Model'):
                task = DS_TrainTask.load_from_script(self.load, step=self.step, load_checkpoint=False)
                self.experiment: DS_Experiment = task.experiment
                self.model: D_Joint = task.model
                self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
                self.trainer: D_JointTrainer = task.trainer
                self.trainer.setup(model=self.model, dataset=self.dataset)
                self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)

            self.process_loader()
            del self.model, self.dataset, self.trainer, self.experiment
            torch.cuda.empty_cache()
            gc.collect()




if __name__ == '__main__':
    TaskGroup(
        extract = Extract_results(cuda=0), 
        eval = Eval_results(cuda=0)
    ).run()

