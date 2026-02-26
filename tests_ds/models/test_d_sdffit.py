from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import gc

import torch
from torch import Tensor
import numpy as np
import os
import pandas as pd
import re
import gc
from tqdm import tqdm
import cv2
from rfstudio.utils.lazy_module import dr

from rfstudio.io import load_float32_image, open_video_renderer, dump_float32_image
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.ui import console
from rfstudio.utils.pretty import P
from rfstudio.loss import ChamferDistanceMetric
from rfstudio.graphics.shaders import PrettyShader
from rfstudio.graphics import Points, TriangleMesh

from rfstudio_ds.loss import L2Loss
from rfstudio_ds.data import DynamicSDFDataset
from rfstudio_ds.engine.experiment import DS_Experiment
from rfstudio_ds.engine.train_4dsdf import DSDF_TrainTask
from rfstudio_ds.model import D_SDFFit
from rfstudio_ds.trainer import D_SDFFitTrainer, SDFFitRegularizationConfig
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh
from rfstudio_ds.visualization import Visualizer
from rfstudio_ds.visualization._plot_cube_curve import plot_cube_curve, plot_curve

from pytorch3d.ops import knn_points
import heapq


horse_task = DSDF_TrainTask(
    dataset=DynamicSDFDataset(
        path=Path("data/dg-mesh/horse/"),
    ),
    model=D_SDFFit(
        geometry='DD_flexicubes',
        geometry_resolution=127, # extracted GT mesh sdf resolution is 127 + 1， so we set cubes' resolution to 127
        time_resolution=240,
    ),
    experiment=DS_Experiment(name='d_sdffit/horse', timestamp='test'),
    trainer=D_SDFFitTrainer(
        num_steps=2500,
        batch_size=2,
        num_steps_per_val=500,
        num_steps_per_save=1000,
        mixed_precision=False,
        full_test_after_train=True,
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)


toy_task = DSDF_TrainTask(
    dataset=DynamicSDFDataset(
        path=Path("data/multiview_dynamic_blender/toy/"),
    ),
    model=D_SDFFit(
        geometry='DD_isocubes',
        geometry_resolution=127, # extracted GT mesh sdf resolution is 127 + 1， so we set cubes' resolution to 127
        geometry_scale=1.0,
        time_resolution=111,
    ),
    experiment=DS_Experiment(name='d_sdffit/toy', timestamp='test'),
    trainer=D_SDFFitTrainer(
        num_steps=2500,
        batch_size=2,
        num_steps_per_val=500,
        num_steps_per_save=1000,
        mixed_precision=False,
        full_test_after_train=True,
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)


@dataclass
class EvalOmegas(Task):
    load: Path = Path('outputs/d_sdffit_horse/debug/task.py')

    step: Optional[int] = None

    @torch.no_grad()
    def run(self) -> None:
        # === 模型加载部分 ===
        with console.status(desc='Loading Model'):
            task = DSDF_TrainTask.load_from_script(self.load, step=self.step)
            model: D_SDFFit = task.model


        model.eval()
        if model.geometric_repr.device != self.device:
            model.geometric_repr.swap_(model.geometric_repr.to(self.device))
        print(model.geometric_repr.sdf_curve_low_freq_fourier_omega)
        print(model.sdf_curve_low_freq_fourier_omega)
        print(model.geometric_repr.sdf_curve_mid_freq_fourier_omega)
        print(model.sdf_curve_mid_freq_fourier_omega)
        print(model.geometric_repr.sdf_curve_high_freq_fourier_omega)
        print(model.sdf_curve_high_freq_fourier_omega)



@dataclass
class EvalSDFCurve(Task):

    load: Path = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/d_sdffit_horse/test/task.py')

    step: Optional[int] = None

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            task = DSDF_TrainTask.load_from_script(self.load, step=self.step)
            model: D_SDFFit = task.model
            experiment: DS_Experiment = task.experiment
            dataset: DynamicSDFDataset = task.dataset
            trainer: D_SDFFitTrainer = task.trainer

            loader = dataset.get_all_iter(infinite=False)
            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)

        model.eval()
        if model.geometric_repr.device != self.device:
            model.geometric_repr.swap_(model.geometric_repr.to(self.device))
        DD_isocubes = model.geometric_repr.replace(
            static_sdf_values = model.static_sdf_params,
            sdf_curve_poly_coefficient = model.sdf_curve_poly_coefficient,
            sdf_curve_low_freq_fourier_coefficient = model.sdf_curve_low_freq_fourier_coefficient,
            sdf_curve_mid_freq_fourier_coefficient = model.sdf_curve_mid_freq_fourier_coefficient,
            sdf_curve_high_freq_fourier_coefficient = model.sdf_curve_high_freq_fourier_coefficient,
            sdf_curve_wavelet_coefficient = model.sdf_curve_wavelet_coefficient,
        )
        
        inputs, gt_sdf_all, _ = next(loader)
        t = inputs
        batch_size = t.shape[0]
        t_expanded = t.unsqueeze(1)  # [batch_size, 1]

        sdf_static_component = DD_isocubes.static_sdf_values.T.expand(batch_size, -1)
        sdf_poly_component = DD_isocubes._compute_polynomial(t_expanded, batch_size)
        sdf_lowfreq_component = DD_isocubes._compute_fourier(
            t_expanded,
            batch_size,
            DD_isocubes.sdf_curve_low_freq_fourier_omega,
            DD_isocubes.sdf_curve_low_freq_fourier_coefficient,
            DD_isocubes.sdf_curve_low_frequency_fourier_degree,
        )
        sdf_midfreq_component = DD_isocubes._compute_fourier(
            t_expanded,
            batch_size,
            DD_isocubes.sdf_curve_mid_freq_fourier_omega,
            DD_isocubes.sdf_curve_mid_freq_fourier_coefficient,
            DD_isocubes.sdf_curve_mid_frequency_fourier_degree,
        )
        sdf_highfreq_component = DD_isocubes._compute_fourier(
            t_expanded,
            batch_size,
            DD_isocubes.sdf_curve_high_freq_fourier_omega,
            DD_isocubes.sdf_curve_high_freq_fourier_coefficient,
            DD_isocubes.sdf_curve_high_frequency_fourier_degree,
        )
        sdf_wavelet_component = DD_isocubes._compute_wavelet(t_expanded)
        pred_sdf_all = sdf_static_component + sdf_poly_component + sdf_lowfreq_component + sdf_midfreq_component + sdf_highfreq_component + sdf_wavelet_component
        gt_sdf_all = gt_sdf_all.reshape(pred_sdf_all.shape)

        times = inputs.detach().cpu()

        def process_and_visualize(indices: Tensor, prefix: str) -> None:
            """处理并可视化（最大或最小误差或者指定的点）。"""
            points_infos = model._get_info_for_points(indices)
            if indices.ndim == 2:
                indices = points_infos['global_indices']
            points_sdfs = pred_sdf_all[:, indices]  # [times_batch, k]
            points_sdfs_static_component = sdf_static_component[:, indices]
            points_sdfs_poly_component = sdf_poly_component[:, indices]
            points_sdfs_low_freq_component = sdf_lowfreq_component[:, indices]
            points_sdfs_mid_freq_component = sdf_midfreq_component[:, indices]
            points_sdfs_high_freq_component = sdf_highfreq_component[:, indices]
            points_sdfs_wavelet_component = sdf_wavelet_component[:, indices]
            points_gt_sdfs = gt_sdf_all[:, indices]  # [times_batch, k]

            for i in range(indices.shape[0]):
                sdf_pred = points_sdfs[:, i].detach().cpu()
                sdf_pred_static_component = points_sdfs_static_component[:, i].detach().cpu()
                sdf_pred_poly_component = points_sdfs_poly_component[:, i].detach().cpu()
                sdf_pred_low_freq_component = points_sdfs_low_freq_component[:, i].detach().cpu()
                sdf_pred_mid_freq_component = points_sdfs_mid_freq_component[:, i].detach().cpu()
                sdf_pred_high_freq_component = points_sdfs_high_freq_component[:, i].detach().cpu()
                sdf_pred_wavelet_component = points_sdfs_wavelet_component[:, i].detach().cpu()
                sdf_gt = points_gt_sdfs[:, i].detach().cpu()
                single_info_dict = {
                    key: value[i].detach().cpu().item() if value[i].ndim == 0 else value[i].detach().cpu()
                    for key, value in points_infos.items()
                }
                grid_indices_str = f"({single_info_dict['grid_indices'][0]}, {single_info_dict['grid_indices'][1]}, {single_info_dict['grid_indices'][2]})"

                path = experiment.dump_image(
                    subfolder=f'pointssdfanalysis',
                    name=f'{prefix}_{i+1}_error_points_{grid_indices_str}_predsdf',
                )
                plot_curve(
                    times=times, 
                    pred_data=sdf_pred,  gt_data=sdf_gt,
                    info_dict=single_info_dict,
                    save_path=path, figsize=(10, 8)
                )

                path = experiment.dump_image(
                    subfolder=f'pointssdfanalysis',
                    name=f'{prefix}_{i+1}_error_points_{grid_indices_str}_predsdf_vs_predsdfcomponents',
                )
                plot_curve(
                    times=times, 
                    pred_data=sdf_pred, 
                    pred_data_static_component = sdf_pred_static_component,
                    pred_data_poly_component = sdf_pred_poly_component,
                    pred_data_low_freq_component = sdf_pred_low_freq_component,
                    pred_data_mid_freq_component = sdf_pred_mid_freq_component,
                    pred_data_high_freq_component = sdf_pred_high_freq_component,
                    pred_data_wavelet_component = sdf_pred_wavelet_component,
                    info_dict=single_info_dict, 
                    save_path=path, figsize=(10, 8)
                )

        # 误差分析
        pointwise_error = L2Loss()(pred_sdf_all, gt_sdf_all, reduction='none').mean(dim=0)
        _, max_error_idx = torch.topk(pointwise_error, k=5)
        _, min_error_idx = torch.topk(pointwise_error, k=5, largest=False)
        track_indice = torch.tensor(
            [
                [56, 69, 96],
                [97, 79, 81],

                [67, 33, 64],
                [20, 67, 64],

                [20, 67, 65],
                [70, 64, 16]

            ]
        )
        process_and_visualize(max_error_idx, 'max')
        process_and_visualize(min_error_idx,'min')
        process_and_visualize(track_indice, 'track')

        torch.cuda.empty_cache()
        gc.collect()



@dataclass
class EvalCubeCurve(Task):

    load: Path = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/d_sdffit/toy/test_iso/task.py')

    step: Optional[int] = None

    def analyze_cubes_curve_info(
        self,
        trainer: D_SDFFitTrainer,
        experiment: DS_Experiment,
        save_subfolder: str,
        cubes_info: Dict[str, Any],
        times: torch.Tensor,
        pred_data: Optional[torch.Tensor] = None,
        gt_data: Optional[torch.Tensor] = None,
        highlight_indices: Optional[List[int]] = None
    ):
        assert pred_data is not None or gt_data is not None
        num_cubes = cubes_info["cube_positions"].shape[0]
        for cube_idx in range(num_cubes):
            single_info_dict = {
                key: value[cube_idx].detach().cpu().item() if value[cube_idx].ndim == 0 else value[cube_idx].detach().cpu()
                for key, value in cubes_info.items()
            }
            cube_indices = single_info_dict["flatten_indices"]
            cube_sdfs = pred_data[:, cube_indices] if pred_data is not None else None
            cube_gt_sdfs = gt_data[:, cube_indices] if gt_data is not None else None
            plot_cube_curve(
                times=times.detach().cpu(),
                info_dict=single_info_dict,
                pred_data=cube_sdfs.detach().cpu() if cube_sdfs is not None else None,
                gt_data=cube_gt_sdfs.detach().cpu() if cube_gt_sdfs is not None else None,
                save_path=experiment.dump_file_path(subfolder=save_subfolder, file_name=f'cube_{cube_idx}_curve.png'),
                highlight_indices=highlight_indices
            )

    def compute_sdf_components(
        self,
    ):
        pass

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            task = DSDF_TrainTask.load_from_script(self.load, step=self.step)
            model: D_SDFFit = task.model
            experiment: DS_Experiment = task.experiment
            dataset: DynamicSDFDataset = task.dataset
            trainer: D_SDFFitTrainer = task.trainer

            loader = dataset.get_all_iter(infinite=False)
            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)

            model.eval()
            if model.geometric_repr.device != self.device:
                model.geometric_repr.swap_(model.geometric_repr.to(self.device))
            if model.geometry == 'DD_isocubes':
                geometric_repr = model.geometric_repr.replace(
                    static_sdf_values = model.static_sdf_params,
                    sdf_curve_poly_coefficient = model.sdf_curve_poly_coefficient,
                    sdf_curve_low_freq_fourier_coefficient = model.sdf_curve_low_freq_fourier_coefficient,
                    sdf_curve_mid_freq_fourier_coefficient = model.sdf_curve_mid_freq_fourier_coefficient,
                    sdf_curve_high_freq_fourier_coefficient = model.sdf_curve_high_freq_fourier_coefficient,
                )
            elif model.geometry == 'DD_flexicubes':
                geometric_repr = model.geometric_repr.replace(
                    static_sdf_values = model.static_sdf_params,
                    sdf_curve_poly_coefficient = model.sdf_curve_poly_coefficient,
                    sdf_curve_low_freq_fourier_coefficient = model.sdf_curve_low_freq_fourier_coefficient,
                    sdf_curve_mid_freq_fourier_coefficient = model.sdf_curve_mid_freq_fourier_coefficient,
                    sdf_curve_high_freq_fourier_coefficient = model.sdf_curve_high_freq_fourier_coefficient,
                )

            inputs, gt_sdf_all, _ = next(loader)
            times = inputs

            pred_sdf, pred_sdf_flow = geometric_repr.query_sdf_at_times(t=times, model_stage=model.dynamic_model_stage, compute_sdf_flow=True)

        fix_positions = torch.tensor([
            [0.1, 0.0, 0.0], # mesh inside
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [0.5, 0.0, 0.0], # mesh outside
            [0.0, 0.7, 0.0],
            [0.0, 0.0, 0.9],
            [-0.03149604797363281, -0.0787401795387268, 0.4409449100494385], # on mesh
            [-0.015748023986816406, -0.12598425149917603, 0.3307086229324341], 
            [0.1574803590774536, 0.03149604797363281, 0.015748023986816406],
            [-0.04724407196044922, -0.4094488024711609, 0.42519688606262207], # high freq cube
            [-0.06299212574958801, -0.3937007784843445, 0.5196850299835205], 
            [-0.06299212574958801, 0.3149605989456177, 0.6614173650741577], 
            [0.03149604797363281, -0.3779527544975281, -0.12598425149917603],
        ], device=self.device)
        fix_points_cube_info = geometric_repr.get_cube_curve_info(positions=fix_positions)
        self.analyze_cubes_curve_info(
            trainer=trainer,
            experiment=experiment,
            save_subfolder='eval_fix_points_curve',
            cubes_info=fix_points_cube_info,
            times=times[20:-20],
            # pred_data=pred_sdf.squeeze(-1),
            gt_data=gt_sdf_all.reshape(pred_sdf.shape)[20:-20],
            highlight_indices=[15],
        )

        fix_cube_indices = torch.tensor(
            [
                [56, 69, 96],
                [97, 79, 81],

                [67, 33, 64],
                [20, 67, 64],

                [20, 67, 65],

                [70, 64, 16]
            ]
        )
        fix_cube_curve_info = geometric_repr.get_cube_curve_info(indices=fix_cube_indices)
        self.analyze_cubes_curve_info(
            trainer=trainer,
            experiment=experiment,
            save_subfolder='eval_fix_cubes_curve',
            cubes_info=fix_cube_curve_info,
            times=times[20:-20],
            # pred_data=pred_sdf.squeeze(-1),
            gt_data=gt_sdf_all.reshape(pred_sdf.shape)[20:-20],
            highlight_indices=[15],
        )

        max_error_points = torch.tensor([
            [-0.18897637724876404, -0.34645670652389526, -0.17322832345962524],
            [0.07874011993408203, -0.34645670652389526, 0.20472443103790283],
            [0.06299209594726562, -0.34645670652389526, 0.22047245502471924],
            [0.17322838306427002, -0.34645670652389526, 0.06299209594726562], 
            [0.1574803590774536, -0.34645670652389526, -0.11023622751235962],
            [0.07874011993408203, -0.34645670652389526, 0.22047245502471924],
            [0.1417323350906372, -0.34645670652389526, -0.14173227548599243],
            [0.28346455097198486, -0.36220473051071167, -0.015748023986816406],
            [0.23622047901153564, 0.09448814392089844, -0.15748029947280884],
            [0.12598425149917603, -0.34645670652389526, -0.15748029947280884]
        ], device=self.device)
        max_error_cube_curve_info = geometric_repr.get_cube_curve_info(positions=max_error_points)
        self.analyze_cubes_curve_info(
            trainer=trainer,
            experiment=experiment,
            save_subfolder='eval_max_error_cube_curve',
            cubes_info=max_error_cube_curve_info,
            times=times[20:-20],
            # pred_data=pred_sdf.squeeze(-1),
            gt_data=gt_sdf_all.reshape(pred_sdf.shape)[20:-20],
            highlight_indices=[15],
        )



@dataclass
class EvalSceneFlow(Task):
    """为训练好的 D_SDFFit 模型评估场景流，生成网格并计算场景流用在mesh上看预测效果。"""

    load: Path = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/d_sdffit/toy/test_flexi/task.py')

    step: Optional[int] = None

    @torch.no_grad()
    def compute_flow_diff(self, data: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """使用中心差分法计算数据的一阶时间导数"""
        flow_diff = torch.zeros_like(data)
        flow_diff[1:-1] = (data[2:] - data[:-2]) / (times[2:] - times[:-2])
        flow_diff[0] = (data[1] - data[0]) / (times[1] - times[0])
        flow_diff[-1] = (data[-1] - data[-2]) / (times[-1] - times[-2])
        return flow_diff

    @torch.no_grad()
    def run(self) -> None:
        # === 模型加载部分 ===
        with console.status(desc='Loading Model'):
            task = DSDF_TrainTask.load_from_script(self.load, step=self.step)
            model: D_SDFFit = task.model
            dataset: DynamicSDFDataset = task.dataset
            trainer: D_SDFFitTrainer = task.trainer
            experiment: DS_Experiment = task.experiment

            loader = dataset.get_all_iter(infinite=False)
            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)

        model.eval()
        if model.geometric_repr.device != self.device:
            model.geometric_repr.swap_(model.geometric_repr.to(self.device))
        if model.geometry == 'DD_isocubes':
            geometric_repr = model.geometric_repr.replace(
                static_sdf_values = model.static_sdf_params,
                sdf_curve_poly_coefficient = model.sdf_curve_poly_coefficient,
                sdf_curve_low_freq_fourier_coefficient = model.sdf_curve_low_freq_fourier_coefficient,
                sdf_curve_mid_freq_fourier_coefficient = model.sdf_curve_mid_freq_fourier_coefficient,
                sdf_curve_high_freq_fourier_coefficient = model.sdf_curve_high_freq_fourier_coefficient,
            )
        elif model.geometry == 'DD_flexicubes':
            geometric_repr = model.geometric_repr.replace(
                static_sdf_values = model.static_sdf_params,
                sdf_curve_poly_coefficient = model.sdf_curve_poly_coefficient,
                sdf_curve_low_freq_fourier_coefficient = model.sdf_curve_low_freq_fourier_coefficient,
                sdf_curve_mid_freq_fourier_coefficient = model.sdf_curve_mid_freq_fourier_coefficient,
                sdf_curve_high_freq_fourier_coefficient = model.sdf_curve_high_freq_fourier_coefficient,
            )
        
        inputs, gt_sdf_all, _ = next(loader)
        pred_sdf_all, pred_sdf_flow_all, _ = model.render_report(inputs)
        gt_sdf_all = gt_sdf_all.reshape(pred_sdf_all.shape)
        gt_sdf_flow_all = self.compute_flow_diff(gt_sdf_all.squeeze(-1), inputs.unsqueeze(-1)).unsqueeze(-1)

        def render_and_export_mesh(mesh, camera, shader, export_path, vis_path, frame_name):
            mesh.export(path=export_path, only_geometry=True)
            image = mesh.render(camera, shader=shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1)
            experiment.dump_image(subfolder=vis_path, name=f"{frame_name}.png", image=image)
            return image
        def predict_next_frame_mesh(mesh: DS_TriangleMesh, dt: Tensor):
            return mesh.get_next_frame_mesh(dt.item())
        def compute_and_log_chamfer(frame, name, mesh_a, mesh_b, record_list):
            chamfer = ChamferDistanceMetric(target_num_points=1000000)(mesh_a, mesh_b)
            experiment.log(f"Frame {frame}: Chamfer Distance {name}: {chamfer}")
            record_list.append(chamfer)

        start = dataset.dataparser.model.sdf_sequence_padding_size
        end = dataset.dataparser.model.sdf_sequence_padding_size + dataset.dataparser.model.num_frames
        vis_cameras = DS_Cameras.from_orbit(
            center=(0, 0, 0),
            up=(0, 1, 0),
            radius=2,
            pitch_degree=15.0,
            num_samples=dataset.dataparser.model.num_frames,
            resolution=(800, 800),
            near=1e-2,
            far=1e2,
            hfov_degree=60,
            device=self.device,
        )
        vis_cameras = vis_cameras.reset_c2w_to_ref_camera_pose(ref_camera_idx=int(dataset.dataparser.model.num_frames*(3/4)))
        chamfer_list1, chamfer_list2, chamfer_list3 = [], [], []
        vis_list1, vis_list2, vis_list3, vis_list4 = [], [], [], []
        
        for frame in range(start, end):
            dt = (inputs[frame + 1] - inputs[frame]) if frame < inputs.shape[0] - 1 else (inputs[frame] - inputs[frame - 1])
            camera = vis_cameras[frame-dataset.dataparser.model.sdf_sequence_padding_size]

            # --- Ground Truth Mesh ---
            geometric_repr.replace_(sdf_values=gt_sdf_all[frame], sdf_flow_values=gt_sdf_flow_all[frame])
            if model.geometry == 'DD_isocubes':
                gt_mesh = geometric_repr.marching_cubes(sdf_eps=1e-6, get_mesh_sdf_flow=True)
            elif model.geometry == 'DD_flexicubes':
                gt_mesh, _ = geometric_repr.dual_marching_cubes(sdf_eps=1e-6, get_mesh_sdf_flow=True)
            vis_list1.append(render_and_export_mesh(gt_mesh, camera, PrettyShader(z_up=True),
                                                    experiment.dump_file_path(subfolder='eval_scene_flow/gt_mesh', file_name=f'{frame}.obj'),
                                                    'eval_scene_flow/gt_mesh_vis', frame))

            # --- 预测下一帧GT Mesh ---
            next_gt_mesh_calc = predict_next_frame_mesh(gt_mesh, dt)
            vis_list2.append(render_and_export_mesh(next_gt_mesh_calc, camera, PrettyShader(z_up=True),
                                                    experiment.dump_file_path(subfolder='eval_scene_flow/gt_mesh_by_sceneflow', file_name=f'{frame+1}.obj'),
                                                    'eval_scene_flow/gt_mesh_by_sceneflow_vis', frame+1))

            # if frame < inputs.shape[0] - dataset.dataparser.model.sdf_sequence_padding_size - 1:
            #     geometric_repr.replace_(sdf_values=gt_sdf_all[frame+1])
            #     next_gt_mesh = geometric_repr.marching_cubes(sdf_eps=1e-6, get_mesh_sdf_flow=False)
            #     compute_and_log_chamfer(frame, "GT next frame vs gt flow predicted", next_gt_mesh, next_gt_mesh_calc, chamfer_list2)

            # --- Predicted Mesh ---
            geometric_repr.replace_(sdf_values=pred_sdf_all[frame], sdf_flow_values=pred_sdf_flow_all[frame])
            if model.geometry == 'DD_isocubes':
                pred_mesh = geometric_repr.marching_cubes(sdf_eps=1e-6, get_mesh_sdf_flow=True)
            elif model.geometry == 'DD_flexicubes':
                pred_mesh, _ = geometric_repr.dual_marching_cubes(sdf_eps=1e-6, get_mesh_sdf_flow=True)
            vis_list3.append(render_and_export_mesh(pred_mesh, camera, PrettyShader(z_up=True),
                                                    experiment.dump_file_path(subfolder='eval_scene_flow/pred_mesh', file_name=f'{frame}.obj'),
                                                    'eval_scene_flow/pred_mesh_vis', frame))

            # --- Predicted 下一帧 Mesh ---
            next_pred_mesh_calc = predict_next_frame_mesh(pred_mesh, dt)
            vis_list4.append(render_and_export_mesh(next_pred_mesh_calc, camera, PrettyShader(z_up=True),
                                                    experiment.dump_file_path(subfolder='eval_scene_flow/pred_mesh_by_sceneflow', file_name=f'{frame+1}.obj'),
                                                    'eval_scene_flow/pred_mesh_by_sceneflow_vis', frame+1))

            # if frame < inputs.shape[0] - 1:
            #     geometric_repr.replace_(sdf_values=pred_sdf_all[frame+1])
            #     next_pred_mesh = geometric_repr.marching_cubes(sdf_eps=1e-6)
            #     compute_and_log_chamfer(frame, "Pred next frame vs flow predicted next", next_pred_mesh, next_pred_mesh_calc, chamfer_list3)

            # --- 当前帧 GT vs Pred ---
            # compute_and_log_chamfer(frame, "GT vs Pred", gt_mesh, pred_mesh, chamfer_list1)

        # === 日志和可视化 ===
        # experiment.log(f"Avg Chamfer GT vs Pred: {sum(chamfer_list1)/len(chamfer_list1)}")
        # experiment.log(f"Avg Chamfer GT next vs gt flow pred next: {sum(chamfer_list2)/len(chamfer_list2)}")
        # experiment.log(f"Avg Chamfer Pred next vs flow pred next: {sum(chamfer_list3)/len(chamfer_list3)}")

        experiment.dump_images2video(subfolder='eval_scene_flow', images=vis_list1, name='gt_mesh_vis.mp4',target_mb=2,duration=5)
        experiment.dump_images2video(subfolder='eval_scene_flow', images=vis_list2, name='gt_mesh_by_sceneflow_vis.mp4',target_mb=2,duration=5)
        experiment.dump_images2video(subfolder='eval_scene_flow', images=vis_list3, name='pred_mesh_vis.mp4',target_mb=2,duration=5)
        experiment.dump_images2video(subfolder='eval_scene_flow', images=vis_list4,name='pred_mesh_by_sceneflow_vis.mp4',target_mb=2,duration=5)


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_uv = flow_uv.detach().cpu().numpy() if torch.is_tensor(flow_uv) else flow_uv
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return torch.from_numpy(flow_image).float() / 255.0

@dataclass
class ExtractSceneFlowMap(Task):
    """为训练好的 D_DiffDR 模型评估场景流，生成网格并计算场景流用在mesh上看预测效果。"""

    load: Path = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/d_sdffit/toy/test_flexi/task.py')

    step: Optional[int] = None

    def compute_optical_flow_from_scene_flow(self, camera, pred_mesh,dt):
        # ===================== [修改] =====================
        #  我们将完全使用MVP矩阵和nvdiffrast的坐标系来计算一切
        # ================================================
        
        ctx = dr.RasterizeCudaContext(camera.device)
        H, W = camera.height.item(), camera.width.item()

        # --- 步骤 1: 将当前帧和下一帧的顶点都转换到裁剪空间 ---
        # 准备当前帧的顶点
        verts_t0 = pred_mesh.vertices
        verts_t0_hom = torch.cat([verts_t0, torch.ones_like(verts_t0[..., :1])], dim=-1) # 转齐次坐标 (V, 4)

        # 准备下一帧的顶点 (基于3D场景流)
        verts_t1 = verts_t0 + pred_mesh.vertices_scene_flow * dt
        verts_t1_hom = torch.cat([verts_t1, torch.ones_like(verts_t1[..., :1])], dim=-1) # 转齐次坐标 (V, 4)
        
        # 获取MVP矩阵
        mvp = camera.projection_matrix @ camera.view_matrix
        
        # 将两组顶点都投影到裁剪空间 (Clip Space)
        clip_t0 = (mvp @ verts_t0_hom.T).T.contiguous() # (V, 4)
        clip_t1 = (mvp @ verts_t1_hom.T).T.contiguous() # (V, 4)

        # --- 步骤 2: 从裁剪空间计算像素坐标和光流 ---
        # 透视除法，得到NDC坐标 (-1, 1)
        ndc_t0 = clip_t0[..., :2] / clip_t0[..., 3:4]
        ndc_t1 = clip_t1[..., :2] / clip_t1[..., 3:4]
        
        # 从NDC坐标转换到屏幕像素坐标 (0, W) 和 (0, H)
        # 公式: screen = (ndc + 1) * 0.5 * image_size
        # nvdiffrast的Y轴是向下的, 和常规图像坐标系一致，通常不需要反转
        viewport = torch.tensor([W, H], dtype=torch.float32, device=camera.device)
        uv0 = (ndc_t0 + 1) * 0.5 * viewport
        uv1 = (ndc_t1 + 1) * 0.5 * viewport
        
        # 计算每个顶点的2D光流 (单位：像素)
        flow = uv1 - uv0 # (V, 2)

        # --- 步骤 3: 光栅化和插值 (这部分和你原来的一样) ---
        indices = pred_mesh.indices.int()
        
        # 注意：光栅化器需要 (B, V, 4) 的输入，B=1
        with dr.DepthPeeler(ctx, clip_t0[None, ...], indices, resolution=(H, W)) as peeler:
            rast, _ = peeler.rasterize_next_layer()
        
        alphas = (rast[..., -1:] > 0).float()

        # 插值我们刚刚正确计算出的光流
        flow_attr = flow[None, ...]  # (1, V, 2)
        flow_image, _ = dr.interpolate(flow_attr, rast, indices)
        flow_image = flow_image[0]

        return flow_image, alphas[0]

    @torch.no_grad()
    def compute_flow_diff(self, data: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """使用中心差分法计算数据的一阶时间导数"""
        flow_diff = torch.zeros_like(data)
        flow_diff[1:-1] = (data[2:] - data[:-2]) / (times[2:] - times[:-2])
        flow_diff[0] = (data[1] - data[0]) / (times[1] - times[0])
        flow_diff[-1] = (data[-1] - data[-2]) / (times[-1] - times[-2])
        return flow_diff
    
    @torch.no_grad()
    def run(self) -> None:
        # === 模型加载部分 ===
        with console.status(desc='Loading Model'):
            task = DSDF_TrainTask.load_from_script(self.load, step=self.step)
            model: D_SDFFit = task.model
            dataset: DynamicSDFDataset = task.dataset
            trainer: D_SDFFitTrainer = task.trainer
            experiment: DS_Experiment = task.experiment

            loader = dataset.get_all_iter(infinite=False)
            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)

        model.eval()
        if model.geometric_repr.device != self.device:
            model.geometric_repr.swap_(model.geometric_repr.to(self.device))
        if model.geometry == 'DD_isocubes':
            geometric_repr = model.geometric_repr.replace(
                static_sdf_values = model.static_sdf_params,
                sdf_curve_poly_coefficient = model.sdf_curve_poly_coefficient,
                sdf_curve_low_freq_fourier_coefficient = model.sdf_curve_low_freq_fourier_coefficient,
                sdf_curve_mid_freq_fourier_coefficient = model.sdf_curve_mid_freq_fourier_coefficient,
                sdf_curve_high_freq_fourier_coefficient = model.sdf_curve_high_freq_fourier_coefficient,
            )
        elif model.geometry == 'DD_flexicubes':
            geometric_repr = model.geometric_repr.replace(
                static_sdf_values = model.static_sdf_params,
                sdf_curve_poly_coefficient = model.sdf_curve_poly_coefficient,
                sdf_curve_low_freq_fourier_coefficient = model.sdf_curve_low_freq_fourier_coefficient,
                sdf_curve_mid_freq_fourier_coefficient = model.sdf_curve_mid_freq_fourier_coefficient,
                sdf_curve_high_freq_fourier_coefficient = model.sdf_curve_high_freq_fourier_coefficient,
            )
        
        inputs, gt_sdf_all, _ = next(loader)
        pred_sdf_all, pred_sdf_flow_all, _ = model.render_report(inputs)
        gt_sdf_all = gt_sdf_all.reshape(pred_sdf_all.shape)
        gt_sdf_flow_all = self.compute_flow_diff(gt_sdf_all.squeeze(-1), inputs.unsqueeze(-1)).unsqueeze(-1)

        def render_and_export_mesh(mesh, camera, shader, export_path, vis_path, frame_name):
            mesh.export(path=export_path, only_geometry=True)
            image = mesh.render(camera, shader=shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1)
            experiment.dump_image(subfolder=vis_path, name=f"{frame_name}.png", image=image)
            return image
        def predict_next_frame_mesh(mesh: DS_TriangleMesh, dt: Tensor):
            return mesh.get_next_frame_mesh(dt.item(),compute_scene_flow=True)

        start = dataset.dataparser.model.sdf_sequence_padding_size
        end = dataset.dataparser.model.sdf_sequence_padding_size + dataset.dataparser.model.num_frames
        vis_cameras = DS_Cameras.from_orbit(
            center=(0, 0, 0),
            up=(0, 1, 0),
            radius=2,
            pitch_degree=15.0,
            num_samples=dataset.dataparser.model.num_frames,
            resolution=(800, 800),
            near=1e-2,
            far=1e2,
            hfov_degree=60,
            device=self.device,
        )
        vis_cameras = vis_cameras.reset_c2w_to_ref_camera_pose(ref_camera_idx=int(dataset.dataparser.model.num_frames*(3/4)))
        
        for frame in range(start, end):
            dt = (inputs[frame + 1] - inputs[frame]) if frame < inputs.shape[0] - 1 else (inputs[frame] - inputs[frame - 1])
            camera = vis_cameras[frame-dataset.dataparser.model.sdf_sequence_padding_size]

            # --- Predicted Mesh ---
            geometric_repr.replace_(sdf_values=pred_sdf_all[frame], sdf_flow_values=pred_sdf_flow_all[frame])
            if model.geometry == 'DD_isocubes':
                pred_mesh = geometric_repr.marching_cubes(sdf_eps=1e-6, get_mesh_sdf_flow=True)
            elif model.geometry == 'DD_flexicubes':
                pred_mesh, _ = geometric_repr.dual_marching_cubes(sdf_eps=1e-6, get_mesh_sdf_flow=True)

            # --- Predicted 下一帧 Mesh ---
            next_pred_mesh_calc = predict_next_frame_mesh(pred_mesh, dt)
            optical_flow_map, alphas = self.compute_optical_flow_from_scene_flow(camera, pred_mesh,dt)
            optical_flow_flow_vis = flow_to_image(optical_flow_map.cpu())
            dump_float32_image(Path(f'temp{frame}.png'),optical_flow_flow_vis)

@dataclass
class ExtractGlobalMesh(Task):
    """为训练好的 D_SDFFit 模型评估场景流，生成网格并计算场景流用在mesh上看预测效果。"""

    load: Path = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/d_sdffit/toy/test_flexi/task.py')

    step: Optional[int] = None

    @torch.no_grad()
    def compute_flow_diff(self, data: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """使用中心差分法计算数据的一阶时间导数"""
        flow_diff = torch.zeros_like(data)
        flow_diff[1:-1] = (data[2:] - data[:-2]) / (times[2:] - times[:-2])
        flow_diff[0] = (data[1] - data[0]) / (times[1] - times[0])
        flow_diff[-1] = (data[-1] - data[-2]) / (times[-1] - times[-2])
        return flow_diff


    @torch.no_grad()
    def run(self) -> None:
        def render_and_export_mesh(mesh, camera, shader, export_path, vis_path, frame_name):
            mesh.export(path=export_path, only_geometry=True)
            image = mesh.render(camera, shader=shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1)
            experiment.dump_image(subfolder=vis_path, name=f"{frame_name}.png", image=image)
            return image
        def predict_next_frame_mesh(mesh: DS_TriangleMesh, dt: Tensor, compute_scene_flow=True):
            return mesh.get_next_frame_mesh(dt.item(),compute_scene_flow=compute_scene_flow)
        def compute_and_log_chamfer(frame, name, mesh_a, mesh_b, record_list):
            chamfer = ChamferDistanceMetric(target_num_points=1000000)(mesh_a, mesh_b)
            experiment.log(f"Frame {frame}: Chamfer Distance {name}: {chamfer}")
            record_list.append(chamfer)
        def knn_average_weighted(vertices, features, K=10, max_dist=None, eps=1e-8, sigma=None):
            """
            对 features 做 knn 加权平均
            vertices: [N, 3]
            features: [N, C]
            K: 邻居数量
            max_dist: 最大邻居半径，超出则忽略
            eps: 防止除零
            sigma: 高斯权重的尺度（如果为 None 就用 1/d 权重）
            """
            vertices_b = vertices.unsqueeze(0)  # [1, N, 3]
            
            knn_result = knn_points(vertices_b, vertices_b, K=K, return_sorted=True)
            indices = knn_result.idx[0]      # [N, K]
            dists = knn_result.dists[0].sqrt()  # [N, K] 欧氏距离
            
            neighbor_features = features[indices]  # [N, K, C]
            
            # 权重计算
            if sigma is None:
                weights = 1.0 / (dists + eps)  # [N, K]
            else:
                weights = torch.exp(- (dists ** 2) / (2 * sigma ** 2))  # 高斯核
            
            # 距离阈值过滤
            if max_dist is not None:
                mask = (dists <= max_dist).float()  # [N, K]
                weights = weights * mask
            
            # 归一化权重
            weights_sum = weights.sum(dim=1, keepdim=True) + eps  # [N, 1]
            weights = weights / weights_sum  # [N, K]
            
            # 加权平均
            features_smoothed = torch.sum(neighbor_features * weights.unsqueeze(-1), dim=1)  # [N, C]
            return features_smoothed
        def knn_update_vertices(tracked_vertices, pred_vertices, K=30, max_dist=None, eps=1e-8, sigma=None):
            """
            用 pred_vertices 的 knn 来更新 tracked_vertices 的位置
            tracked_vertices: [N, 3]
            pred_vertices: [M, 3]
            """

            tracked_b = tracked_vertices.unsqueeze(0)  # [1, N, 3]
            pred_b = pred_vertices.unsqueeze(0)        # [1, M, 3]

            knn_result = knn_points(tracked_b, pred_b, K=K, return_sorted=True)
            indices = knn_result.idx[0]       # [N, K]
            dists = knn_result.dists[0].sqrt()  # [N, K]

            neighbor_vertices = pred_vertices[indices]  # [N, K, 3]

            # 权重计算
            if sigma is None:
                weights = 1.0 / (dists + eps)  # [N, K]
            else:
                weights = torch.exp(- (dists ** 2) / (2 * sigma ** 2))

            if max_dist is not None:
                mask = (dists <= max_dist).float()
                weights = weights * mask

            # 归一化
            weights_sum = weights.sum(dim=1, keepdim=True) + eps
            weights = weights / weights_sum

            # 加权平均 -> 更新 tracked_vertices
            updated_tracked_vertices = torch.sum(neighbor_vertices * weights.unsqueeze(-1), dim=1)  # [N, 3]
            return updated_tracked_vertices

        # === 模型加载部分 ===
        with console.status(desc='Loading Model'):
            task = DSDF_TrainTask.load_from_script(self.load, step=self.step)
            model: D_SDFFit = task.model
            dataset: DynamicSDFDataset = task.dataset
            trainer: D_SDFFitTrainer = task.trainer
            experiment: DS_Experiment = task.experiment

            loader = dataset.get_all_iter(infinite=False)
            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)

            model.eval()
            if model.geometric_repr.device != self.device:
                model.geometric_repr.swap_(model.geometric_repr.to(self.device))
            if model.geometry == 'DD_isocubes':
                geometric_repr = model.geometric_repr.replace(
                    static_sdf_values = model.static_sdf_params,
                    sdf_curve_poly_coefficient = model.sdf_curve_poly_coefficient,
                    sdf_curve_low_freq_fourier_coefficient = model.sdf_curve_low_freq_fourier_coefficient,
                    sdf_curve_mid_freq_fourier_coefficient = model.sdf_curve_mid_freq_fourier_coefficient,
                    sdf_curve_high_freq_fourier_coefficient = model.sdf_curve_high_freq_fourier_coefficient,
                )
            elif model.geometry == 'DD_flexicubes':
                geometric_repr = model.geometric_repr.replace(
                    static_sdf_values = model.static_sdf_params,
                    sdf_curve_poly_coefficient = model.sdf_curve_poly_coefficient,
                    sdf_curve_low_freq_fourier_coefficient = model.sdf_curve_low_freq_fourier_coefficient,
                    sdf_curve_mid_freq_fourier_coefficient = model.sdf_curve_mid_freq_fourier_coefficient,
                    sdf_curve_high_freq_fourier_coefficient = model.sdf_curve_high_freq_fourier_coefficient,
                )
        
        inputs, gt_sdf_all, _ = next(loader)
        ori_times = inputs[dataset.dataparser.model.sdf_sequence_padding_size:dataset.dataparser.model.sdf_sequence_padding_size+dataset.dataparser.model.num_frames]
        ori_times_start = ori_times[0]
        ori_times_end = ori_times[-1]
        ori_times_interpolated = torch.linspace(ori_times_start, ori_times_end, 500)
        pred_sdf_all, pred_sdf_flow_all, _ = model.render_report(ori_times_interpolated)
        # gt_sdf_all = gt_sdf_all.reshape(pred_sdf_all.shape)
        # gt_sdf_flow_all = self.compute_flow_diff(gt_sdf_all.squeeze(-1), ori_times_interpolated.unsqueeze(-1)).unsqueeze(-1)
        
        # start = dataset.dataparser.model.sdf_sequence_padding_size
        # end = dataset.dataparser.model.sdf_sequence_padding_size + dataset.dataparser.model.num_frames
        vis_cameras = DS_Cameras.from_orbit(
            center=(0, 0, 0),
            up=(0, 1, 0),
            radius=2,
            pitch_degree=15.0,
            num_samples=ori_times_interpolated.shape[0],
            resolution=(800, 800),
            near=1e-2,
            far=1e2,
            hfov_degree=60,
            device=self.device,
        )
        vis_cameras = vis_cameras.reset_c2w_to_ref_camera_pose(ref_camera_idx=int(ori_times_interpolated.shape[0]*(3/4)))
        chamfer_list1, chamfer_list2, chamfer_list3 = [], [], []
        vis_list1, vis_list2, vis_list3, vis_list4 = [], [], [], []
        
        dt = ori_times_interpolated[1] - ori_times_interpolated[0]
        # for frame in range(start, end):
        for frame in range(ori_times_interpolated.shape[0]):
            camera = vis_cameras[frame]

            # --- Predicted Mesh ---
            geometric_repr.replace_(sdf_values=pred_sdf_all[frame], sdf_flow_values=pred_sdf_flow_all[frame])
            if frame == 0:
                if model.geometry == 'DD_isocubes':
                    init_pred_mesh = geometric_repr.marching_cubes(sdf_eps=1e-6, get_mesh_sdf_flow=True)
                elif model.geometry == 'DD_flexicubes':
                    init_pred_mesh, _ = geometric_repr.dual_marching_cubes(sdf_eps=1e-6, get_mesh_sdf_flow=True)
                vis_list2.append(render_and_export_mesh(init_pred_mesh, camera, PrettyShader(z_up=True),
                    experiment.dump_file_path(subfolder='eval_scene_flow/tracked_pred_mesh', file_name=f'init_tracked{frame}.obj'),
                    'eval_scene_flow/tracked_pred_mesh_vis', frame))
                tracked_pred_mesh = predict_next_frame_mesh(init_pred_mesh, dt)
                tracked_pred_mesh = tracked_pred_mesh.simplify(target_num_faces=int(tracked_pred_mesh.num_faces*0.3))
            else:
                vis_list2.append(render_and_export_mesh(tracked_pred_mesh, camera, PrettyShader(z_up=True),
                    experiment.dump_file_path(subfolder='eval_scene_flow/tracked_pred_mesh', file_name=f'tracked{frame}.obj'),
                    'eval_scene_flow/tracked_pred_mesh_vis', frame))
                if model.geometry == 'DD_isocubes':
                    pred_mesh = geometric_repr.marching_cubes(sdf_eps=1e-6, get_mesh_sdf_flow=True)
                elif model.geometry == 'DD_flexicubes':
                    pred_mesh, _ = geometric_repr.dual_marching_cubes(sdf_eps=1e-6, get_mesh_sdf_flow=True)
                
                pred_mesh.compute_scene_flow()
                pred_mesh_vertices = pred_mesh.vertices.clone()
                pred_mesh_vertices_scene_flow = pred_mesh.vertices_scene_flow

                tracked_mesh_vertices = tracked_pred_mesh.vertices.clone()
                knn_result = knn_points(
                    tracked_mesh_vertices.unsqueeze(0)  ,  # 查询点 [1, Vt, 3]
                    pred_mesh_vertices.unsqueeze(0)  ,     # 检索点 [1, Vp, 3]
                    K=30, 
                    return_sorted=True
                )
                indices = knn_result.idx[0]   # [Vt, 30] 每个 tracked 顶点对应 pred 的 K 个索引
                scene_flow_knn = pred_mesh_vertices_scene_flow[indices]  # [Vt, 30, 3]
                tracked_scene_flow = scene_flow_knn.mean(dim=1)  # [Vt, 3]
                # tracked_scene_flow_final = knn_average_weighted(tracked_mesh_vertices, tracked_scene_flow, K=30, max_dist=0.2)

                tracked_pred_mesh.replace_(vertices_scene_flow=tracked_scene_flow)
                # tracked_pred_mesh.replace_(vertices_scene_flow=tracked_scene_flow_final)
                tracked_pred_mesh = tracked_pred_mesh.get_next_frame_mesh(dt.item(),compute_scene_flow=False)

        # experiment.dump_images2video(subfolder='eval_scene_flow', images=vis_list1, name='tracked_gt_mesh_vis.mp4',target_mb=2,duration=5)
        experiment.dump_images2video(subfolder='eval_scene_flow', images=vis_list2, name='tracked_pred_mesh_vis.mp4',target_mb=2,duration=5)



if __name__ == '__main__':
    TaskGroup(
        # train task
        horse = horse_task,
        toy = toy_task,

        # eval task
        evalomegas = EvalOmegas(cuda=0),
        evalsdfcurve = EvalSDFCurve(cuda=0),
        evalcubecurve = EvalCubeCurve(cuda=0), # python tests_ds/models/test_d_sdffit.py evalcubecurve --load outputs/d_sdffit_horse/test/task.py
        evalsceneflow = EvalSceneFlow(cuda=0),
        extractsceneflowmap = ExtractSceneFlowMap(cuda=0),
        extrackglobalmesh = ExtractGlobalMesh(cuda=0),
    ).run()

