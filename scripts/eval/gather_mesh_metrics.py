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

import trimesh
from scipy.spatial import cKDTree
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes, knn_points

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



def normalize(v):
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)

@dataclass
class Eval_Synthetic_results(Task):

    step: Optional[int] = None

    load: Optional[Path] = None
    
    @torch.no_grad()
    def evaluate_mesh_metrics(self, pred_mesh: Meshes, gt_mesh: Meshes, point_sample_num: int = 1000000, distance_thresh: float = 0.01):
        """
        计算 mesh 评估指标：
        - Normal Angle Difference (IN>5°)
        - F1 score (基于点距离阈值)
        - Edge Chamfer Distance (ECD)
        - Edge F1 score (EF1)
        - #V / #F
        """

        device = pred_mesh.device

        metrics = {}

        # ---------------------
        # 1. 顶点/面数量
        metrics['#V'] = pred_mesh.verts_packed().shape[0]
        metrics['#F'] = pred_mesh.faces_packed().shape[0]

        # ---------------------
        # 2. Normal Angle Difference (IN>5°)
        # 获取面法线
        pred_face_normals = normalize(pred_mesh.faces_normals_packed())
        gt_face_normals = normalize(gt_mesh.faces_normals_packed())

        # 对齐方式：使用顶点重心 nearest neighbor 匹配
        pred_faces = pred_mesh.faces_packed()
        gt_faces = gt_mesh.faces_packed()
        pred_verts = pred_mesh.verts_packed()
        gt_verts = gt_mesh.verts_packed()

        pred_face_centers = pred_verts[pred_faces].mean(dim=1, keepdim=False)
        gt_face_centers = gt_verts[gt_faces].mean(dim=1, keepdim=False)

        # 用 KNN 匹配最近 gt face
        knn = knn_points(pred_face_centers[None, ...], gt_face_centers[None, ...], K=1)
        matched_gt_normals = gt_face_normals[knn.idx[0].squeeze(1)]

        # 计算夹角
        cos_theta = (pred_face_normals * matched_gt_normals).sum(dim=-1).clamp(-1, 1)
        theta_deg = torch.acos(cos_theta) * 180.0 / 3.1415926
        metrics['IN>5°'] = (theta_deg > 5.0).float().mean().item()

        # ---------------------
        # 3. Chamfer Distance & F1 Score
        pred_points = sample_points_from_meshes(pred_mesh, point_sample_num)
        gt_points = sample_points_from_meshes(gt_mesh, point_sample_num)

        # 双向距离
        knn_pred2gt = knn_points(pred_points, gt_points, K=1)
        knn_gt2pred = knn_points(gt_points, pred_points, K=1)

        dist_pred2gt = knn_pred2gt.dists[..., 0].sqrt()
        dist_gt2pred = knn_gt2pred.dists[..., 0].sqrt()

        # Chamfer Distance
        metrics['CD'] = (dist_pred2gt.mean() + dist_gt2pred.mean()).item()

        # F1 Score
        precision = (dist_pred2gt < distance_thresh).float().mean()
        recall = (dist_gt2pred < distance_thresh).float().mean()
        metrics['F1'] = (2 * precision * recall / (precision + recall + 1e-8)).item()

        # ---------------------
        # 4. Edge Chamfer Distance & Edge F1
        pred_edges = pred_mesh.edges_packed()   # [E_pred, 2]
        gt_edges = gt_mesh.edges_packed()       # [E_gt, 2]

        pred_edge_points = (pred_verts[pred_edges[:, 0]] + pred_verts[pred_edges[:, 1]]) / 2.0
        gt_edge_points = (gt_verts[gt_edges[:, 0]] + gt_verts[gt_edges[:, 1]]) / 2.0

        # 双向距离
        knn_edge_pred2gt = knn_points(pred_edge_points[None,...], gt_edge_points[None,...], K=1)
        knn_edge_gt2pred = knn_points(gt_edge_points[None,...], pred_edge_points[None,...], K=1)

        edge_dist_pred2gt = knn_edge_pred2gt.dists[...,0].sqrt()
        edge_dist_gt2pred = knn_edge_gt2pred.dists[...,0].sqrt()

        metrics['ECD'] = (edge_dist_pred2gt.mean() + edge_dist_gt2pred.mean()).item()

        # Edge F1
        precision_edge = (edge_dist_pred2gt < distance_thresh).float().mean()
        recall_edge = (edge_dist_gt2pred < distance_thresh).float().mean()
        metrics['EF1'] = (2 * precision_edge * recall_edge / (precision_edge + recall_edge + 1e-8)).item()

        return metrics

    def process_loader(self):
        full_batch_size = self.dataset.get_size(split='fix_vis')  # full time size
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)

        mesh_root = Path(f'{self.load.parent.parent}/gather_mesh')
        metrics_root = Path(f'{self.load.parent.parent}/gather_metrics')
        metrics_root.mkdir(parents=True, exist_ok=True)
        iter_count = 0

        # 初始化 summary 字典
        summary_metrics = {method: {'#V': [], '#F': [], 'IN>5°': [], 'CD': [], 'F1': [], 'ECD': [], 'EF1': []}
                        for method in ['psdf','d2dgs','dgmesh','grid4d','sc-gs','deformable-3dgs']}

        with tqdm(total=full_batch_size * 6, desc="Evaluating and Extracting results", unit="frame") as pbar:
            for inputs, gt_outputs, _ in loader:
                for i in range(len(inputs)):
                    count_id = iter_count * 50 + i
                    view_num = count_id // full_batch_size
                    if view_num >= 1:
                        break
                    frame_id = count_id % full_batch_size

                    gt_mesh_path = mesh_root / f'frame_{frame_id}' / 'gt.obj'
                    gt_mesh = load_objs_as_meshes([gt_mesh_path], device=self.device)

                    frame_metrics = {}  # 每个frame的详细结果

                    for method in ['psdf','d2dgs','dgmesh','grid4d','sc-gs','deformable-3dgs']:
                        pred_mesh_path = mesh_root / f'frame_{frame_id}' / f'{method}.obj'
                        pred_mesh = load_objs_as_meshes([pred_mesh_path], device=self.device)
                        
                        # 计算指标
                        metrics = self.evaluate_mesh_metrics(pred_mesh, gt_mesh)
                        frame_metrics[method] = metrics

                        # 累积到 summary
                        for k, v in metrics.items():
                            summary_metrics[method][k].append(v)

                        # 写入每帧详细日志
                        log_line = f"Frame {frame_id} {method} Metrics: " + \
                            " ".join([f"{k} {v:.6f}" if isinstance(v,float) else f"{k} {v}" for k,v in metrics.items()])
                        self.experiment.log(log_line, new_logfile=metrics_root / 'mesh_detail.txt')

                    pbar.update(1)

                iter_count += 1
                torch.cuda.empty_cache()

        # 保存 summary metrics（平均值）
        summary_file = metrics_root / 'mesh_summary.txt'
        with open(summary_file, 'w') as f:
            for method, method_metrics in summary_metrics.items():
                avg_metrics = {k: sum(v_list)/len(v_list) if len(v_list)>0 else 0.0 for k, v_list in method_metrics.items()}
                line = f"{method} Average Metrics: " + " ".join([f"{k} {v:.6f}" for k,v in avg_metrics.items()])
                f.write(line + "\n")

    @torch.no_grad()
    def run(self) -> None:
        print(f'Processing: {self.load}...')
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
    
    @torch.no_grad()
    def evaluate_mesh_metrics(self, pred_mesh: Meshes):
        """
        计算 mesh 几何指标：
        - 顶点数 (#V)
        - 面数 (#F)
        - Aspect Ratio > 4 (%)
        - Radius Ratio > 4 (%)
        - Min Angle < 10 (%)
        """
        device = pred_mesh.device
        metrics = {}

        verts = pred_mesh.verts_packed()   # [V,3]
        faces = pred_mesh.faces_packed()   # [F,3]
        metrics['#V'] = verts.shape[0]
        metrics['#F'] = faces.shape[0]

        # -------------------------
        # 计算面三角形几何属性
        v0 = verts[faces[:,0]]
        v1 = verts[faces[:,1]]
        v2 = verts[faces[:,2]]

        # 边向量
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        # 边长度
        l0 = e0.norm(dim=1)
        l1 = e1.norm(dim=1)
        l2 = e2.norm(dim=1)

        # 面面积
        face_areas = 0.5 * torch.cross(e0, -e2, dim=1).norm(dim=1)

        # 面周长
        face_perimeters = l0 + l1 + l2

        # Aspect Ratio = longest_edge / shortest_altitude
        # 计算三角形半高 = 2*area / edge_length
        alt0 = 2*face_areas / l0.clamp(min=1e-8)
        alt1 = 2*face_areas / l1.clamp(min=1e-8)
        alt2 = 2*face_areas / l2.clamp(min=1e-8)
        min_alt = torch.stack([alt0, alt1, alt2], dim=1).min(dim=1)[0]
        max_edge = torch.stack([l0,l1,l2], dim=1).max(dim=1)[0]
        aspect_ratio = max_edge / min_alt
        metrics['Aspect Ratio > 4 (%)'] = (aspect_ratio > 4.0).float().mean().item() * 100

        # Radius Ratio = circumscribed radius / inscribed radius
        # inscribed radius r = 2*area / perimeter
        # circumscribed radius R = (l0*l1*l2) / (4*area)
        r_in = 2*face_areas / face_perimeters.clamp(min=1e-8)
        r_circ = (l0*l1*l2) / (4*face_areas.clamp(min=1e-8))
        radius_ratio = r_circ / r_in
        metrics['Radius Ratio > 4 (%)'] = (radius_ratio > 4.0).float().mean().item() * 100

        # Min Angle
        # cos(angle) = (b^2 + c^2 - a^2)/(2bc)
        def angle(a,b,c):
            cos_theta = (b**2 + c**2 - a**2) / (2*b*c).clamp(min=1e-8)
            cos_theta = cos_theta.clamp(-1.0,1.0)
            return torch.acos(cos_theta) * 180.0 / 3.1415926

        angle0 = angle(l0, l2, l1)
        angle1 = angle(l1, l0, l2)
        angle2 = angle(l2, l1, l0)
        min_angle = torch.stack([angle0, angle1, angle2], dim=1).min(dim=1)[0]
        metrics['Min Angle < 10 (%)'] = (min_angle < 10.0).float().mean().item() * 100

        return metrics

    def process_loader(self):
        full_batch_size = self.dataset.dataparser.model.num_frames
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)

        mesh_root = Path(f'{self.load.parent.parent}/gather_mesh')
        metrics_root = Path(f'{self.load.parent.parent}/gather_metrics')
        metrics_root.mkdir(parents=True, exist_ok=True)
        
        iter_count = 0
        summary_metrics = {method: {'#V': [], '#F': [], 
                                    'Aspect Ratio > 4 (%)': [], 
                                    'Radius Ratio > 4 (%)': [], 
                                    'Min Angle < 10 (%)': []}
                        for method in ['psdf','d2dgs','dgmesh','grid4d','sc-gs','deformable-3dgs']}
        
        with tqdm(total=full_batch_size * 6, desc="Evaluating and Extracting results", unit="frame") as pbar:
            for inputs, _, _ in loader:
                for i in range(len(inputs)):
                    count_id = iter_count * 50 + i
                    view_num = count_id // full_batch_size
                    if view_num >= 1:
                        break
                    frame_id = count_id % full_batch_size

                    frame_metrics = {}

                    for method in ['psdf','d2dgs','dgmesh','grid4d','sc-gs','deformable-3dgs']:
                        pred_mesh_path = mesh_root / f'frame_{frame_id}' / f'{method}.obj'
                        pred_mesh = load_objs_as_meshes([pred_mesh_path], device=self.device)

                        # 计算指标
                        metrics = self.evaluate_mesh_metrics(pred_mesh)
                        frame_metrics[method] = metrics

                        # 累积到 summary
                        for k, v in metrics.items():
                            summary_metrics[method][k].append(v)

                        # 写入每帧详细日志
                        log_line = f"Frame {frame_id} {method} Metrics: " + \
                            " ".join([f"{k} {v:.6f}" if isinstance(v,float) else f"{k} {v}" for k,v in metrics.items()])
                        self.experiment.log(log_line, new_logfile=metrics_root / 'mesh_detail.txt')

                    pbar.update(1)

                iter_count += 1
                torch.cuda.empty_cache()

        # 保存 summary metrics（平均值）
        summary_file = metrics_root / 'mesh_summary.txt'
        with open(summary_file, 'w') as f:
            for method, method_metrics in summary_metrics.items():
                avg_metrics = {k: sum(v_list)/len(v_list) if len(v_list)>0 else 0.0 for k, v_list in method_metrics.items()}
                line = f"{method} Average Metrics: " + " ".join([f"{k} {v:.6f}" for k,v in avg_metrics.items()])
                f.write(line + "\n")

    @torch.no_grad()
    def run(self) -> None:
        print(f'Processing: {self.load}...')
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
    
    @torch.no_grad()
    def evaluate_mesh_metrics(self, pred_mesh: Meshes):
        """
        计算 mesh 几何指标：
        - 顶点数 (#V)
        - 面数 (#F)
        - Aspect Ratio > 4 (%)
        - Radius Ratio > 4 (%)
        - Min Angle < 10 (%)
        """
        device = pred_mesh.device
        metrics = {}

        verts = pred_mesh.verts_packed()   # [V,3]
        faces = pred_mesh.faces_packed()   # [F,3]
        metrics['#V'] = verts.shape[0]
        metrics['#F'] = faces.shape[0]

        # -------------------------
        # 计算面三角形几何属性
        v0 = verts[faces[:,0]]
        v1 = verts[faces[:,1]]
        v2 = verts[faces[:,2]]

        # 边向量
        e0 = v1 - v0
        e1 = v2 - v1
        e2 = v0 - v2

        # 边长度
        l0 = e0.norm(dim=1)
        l1 = e1.norm(dim=1)
        l2 = e2.norm(dim=1)

        # 面面积
        face_areas = 0.5 * torch.cross(e0, -e2, dim=1).norm(dim=1)

        # 面周长
        face_perimeters = l0 + l1 + l2

        # Aspect Ratio = longest_edge / shortest_altitude
        # 计算三角形半高 = 2*area / edge_length
        alt0 = 2*face_areas / l0.clamp(min=1e-8)
        alt1 = 2*face_areas / l1.clamp(min=1e-8)
        alt2 = 2*face_areas / l2.clamp(min=1e-8)
        min_alt = torch.stack([alt0, alt1, alt2], dim=1).min(dim=1)[0]
        max_edge = torch.stack([l0,l1,l2], dim=1).max(dim=1)[0]
        aspect_ratio = max_edge / min_alt
        metrics['Aspect Ratio > 4 (%)'] = (aspect_ratio > 4.0).float().mean().item() * 100

        # Radius Ratio = circumscribed radius / inscribed radius
        # inscribed radius r = 2*area / perimeter
        # circumscribed radius R = (l0*l1*l2) / (4*area)
        r_in = 2*face_areas / face_perimeters.clamp(min=1e-8)
        r_circ = (l0*l1*l2) / (4*face_areas.clamp(min=1e-8))
        radius_ratio = r_circ / r_in
        metrics['Radius Ratio > 4 (%)'] = (radius_ratio > 4.0).float().mean().item() * 100

        # Min Angle
        # cos(angle) = (b^2 + c^2 - a^2)/(2bc)
        def angle(a,b,c):
            cos_theta = (b**2 + c**2 - a**2) / (2*b*c).clamp(min=1e-8)
            cos_theta = cos_theta.clamp(-1.0,1.0)
            return torch.acos(cos_theta) * 180.0 / 3.1415926

        angle0 = angle(l0, l2, l1)
        angle1 = angle(l1, l0, l2)
        angle2 = angle(l2, l1, l0)
        min_angle = torch.stack([angle0, angle1, angle2], dim=1).min(dim=1)[0]
        metrics['Min Angle < 10 (%)'] = (min_angle < 10.0).float().mean().item() * 100

        return metrics

    def process_loader(self):
        full_batch_size = self.dataset.dataparser.model.num_frames
        loader = self.dataset.get_test_iter(batch_size=20, shuffle=False, infinite=False)

        mesh_root = Path(f'{self.load.parent.parent}/gather_mesh')
        metrics_root = Path(f'{self.load.parent.parent}/gather_metrics')
        metrics_root.mkdir(parents=True, exist_ok=True)
        
        iter_count = 0
        summary_metrics = {method: {'#V': [], '#F': [], 
                                    'Aspect Ratio > 4 (%)': [], 
                                    'Radius Ratio > 4 (%)': [], 
                                    'Min Angle < 10 (%)': []}
                        for method in ['psdf','d2dgs','dgmesh','grid4d','sc-gs','deformable-3dgs']}
        
        with tqdm(total=full_batch_size * 6, desc="Evaluating and Extracting results", unit="frame") as pbar:
            for inputs, _, _ in loader:
                for i in range(len(inputs)):
                    count_id = iter_count * 50 + i
                    view_num = count_id // full_batch_size
                    if view_num >= 1:
                        break
                    frame_id = count_id % full_batch_size

                    frame_metrics = {}

                    for method in ['psdf','d2dgs','dgmesh','grid4d','sc-gs','deformable-3dgs']:
                        pred_mesh_path = mesh_root / f'frame_{frame_id}' / f'{method}.obj'
                        pred_mesh = load_objs_as_meshes([pred_mesh_path], device=self.device)

                        # 计算指标
                        metrics = self.evaluate_mesh_metrics(pred_mesh)
                        frame_metrics[method] = metrics

                        # 累积到 summary
                        for k, v in metrics.items():
                            summary_metrics[method][k].append(v)

                        # 写入每帧详细日志
                        log_line = f"Frame {frame_id} {method} Metrics: " + \
                            " ".join([f"{k} {v:.6f}" if isinstance(v,float) else f"{k} {v}" for k,v in metrics.items()])
                        self.experiment.log(log_line, new_logfile=metrics_root / 'mesh_detail.txt')

                    pbar.update(1)

                iter_count += 1
                torch.cuda.empty_cache()

        # 保存 summary metrics（平均值）
        summary_file = metrics_root / 'mesh_summary.txt'
        with open(summary_file, 'w') as f:
            for method, method_metrics in summary_metrics.items():
                avg_metrics = {k: sum(v_list)/len(v_list) if len(v_list)>0 else 0.0 for k, v_list in method_metrics.items()}
                line = f"{method} Average Metrics: " + " ".join([f"{k} {v:.6f}" for k,v in avg_metrics.items()])
                f.write(line + "\n")

    @torch.no_grad()
    def run(self) -> None:
        print(f'Processing: {self.load}...')
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

