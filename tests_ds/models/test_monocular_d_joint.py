from __future__ import annotations

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
from rfstudio.loss import ChamferDistanceMetric
from rfstudio.graphics import Points, TriangleMesh
from rfstudio.graphics.shaders import PrettyShader, DepthShader, NormalShader
from rfstudio.graphics import DepthImages, VectorImages

# import rfstudio_ds modules
from rfstudio_ds.engine.experiment import DS_Experiment
from rfstudio_ds.engine.train import DS_TrainTask
from rfstudio_ds.data import SyntheticDynamicMonocularBlenderRGBADataset
from rfstudio_ds.model import D_Joint # rgb image loss optimization model
from rfstudio_ds.trainer import D_JointTrainer, JointRegularizationConfig # rgb image loss optimization trainer, dynamic nvdiffrec
from rfstudio_ds.graphics.spatial_sampler import NearestGridSampler
from rfstudio_ds.visualization import Visualizer
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh
from rfstudio_ds.model.density_field.components.encoding_4d import Grid4d_HashEncoding
from rfstudio_ds.nn import MLPDecoderNetwork, Grid4DDecoderNetwork



jumpingjacks_task = DS_TrainTask(
    dataset=SyntheticDynamicMonocularBlenderRGBADataset(
        path=Path("data/DNerf/jumpingjacks/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=1,
        low_freq_fourier_bands=[1, 10],
        mid_freq_fourier_bands=[11, 20],
        high_freq_fourier_bands=[21,30],
        dynamic_texture=Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 96, 32, 6],
                activation='sigmoid',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[16,16,8],
            deform_desired_resolution=[4096,4096,64],
            deform_num_levels=32,
        ),
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-3],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-3],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='dnerf_d_joint/jumpingjacks', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
        num_steps_per_save=1000,
        hold_after_train=False,

        geometry_lr=1e-3, # 用于flexicube的 deform 和 weight 参数
        geometry_lr_decay=800,
        
        static_sdf_params_learning_rate=1e-2,
        static_sdf_params_decay=800, # 控制学习率的半衰期，在trainer中默认是800
        static_sdf_params_max_norm=None, # 控制系数优化的最大step，在trainer中默认没有开启
        
        poly_coeffs_learning_rate=1e-2,
        poly_coeffs_decay=400,
        poly_coeffs_max_norm=None,

        fourier_low_learning_rate = 5e-3,
        fourier_low_decay=800,
        fourier_low_max_norm=0.005,

        fourier_mid_learning_rate = 1e-3,
        fourier_mid_decay=1200,
        fourier_mid_max_norm=0.005,

        fourier_high_learning_rate = 5e-4, 
        fourier_high_decay=1600,
        fourier_high_max_norm=0.005,

        appearance_learning_rate=1e-3,
        appearance_decay=1250,

        light_learning_rate=5e-4, # 用于light的参数
        light_learning_rate_decay=1250,

        regularization=JointRegularizationConfig(
            ssim_weight_begin = 1., # ssim loss
            ssim_weight_end  = 10.,
            ssim_weight_decay_steps = 2500,
            ssim_weight_start_step = 0,

            mask_weight_begin = 10., # mask supervision loss
            mask_weight_end = 10.,
            mask_weight_decay_steps = 1250,
            mask_weight_start_step = 0,

            # sdf entropy loss
            sdf_entropy_weight_begin = 0.2, # 控制sdf entropy loss的权重
            sdf_entropy_weight_end = 0.01,
            sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
            sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
            sdf_entropy_end_step = -1, # 控制sdf entropy loss的终止步数

            # curve derivative smooth loss
            time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
            time_tv_weight_end = 0.05,
            time_tv_decay_steps = 1250, # 控制curve derivative smooth loss的线性衰减步数
            time_tv_start_step = 0, # 控制curve derivative smooth loss的起始步数
            time_tv_end_step=-1, # 控制curve derivative smooth loss的终止步数

            # sdf eikonal loss，开启后，大概需要占用1G显存
            sdf_eikonal_weight_begin = 0.01, # 控制sdf eikonal loss的权重
            sdf_eikonal_weight_end = 0.001,
            sdf_eikonal_decay_steps = 1250, # 控制sdf eikonal loss的线性衰减步数
            sdf_eikonal_start_step = 0, # 控制sdf eikonal loss的起始步数
            sdf_eikonal_end_step=-1, # 控制sdf eikonal loss的终止步数

            reg_temporal_hashgrid_begin = 0.1,
            reg_temporal_hashgrid_end = 0.01,
            reg_temporal_hashgrid_decay_steps = 1250,
            reg_temporal_hashgrid_start_step = 0,
            reg_temporal_hashgrid_end_step=-1,

            reg_occ_begin=0.0,
            reg_occ_end=0.001,
            reg_occ_decay_steps=1250,
            reg_occ_start_step=0,
            reg_occ_end_step=-1,

            reg_light_begin=0.001,
            reg_light_end=0.01,
            reg_light_decay_steps=1250,
            reg_light_start_step=0,
            reg_light_end_step=-1,
        ),

        mixed_precision=False,
        full_test_after_train=True,
        full_fix_vis_after_train=True,
        full_orbit_vis_after_train=True,
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)


beagle_task = DS_TrainTask(
    dataset=SyntheticDynamicMonocularBlenderRGBADataset(
        path=Path("data/dg-mesh/beagle/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=1,
        low_freq_fourier_bands=[1, 10],
        mid_freq_fourier_bands=[11, 20],
        high_freq_fourier_bands=[21,30],
        dynamic_texture=Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 96, 32, 6],
                activation='sigmoid',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[16,16,8],
            deform_desired_resolution=[4096,4096,64],
            deform_num_levels=32,
        ),
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-3],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-3],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='dgmesh_d_joint/beagle', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
        num_steps_per_save=1000,
        hold_after_train=False,

        geometry_lr=1e-3, # 用于flexicube的 deform 和 weight 参数
        geometry_lr_decay=800,
        
        static_sdf_params_learning_rate=1e-2,
        static_sdf_params_decay=800, # 控制学习率的半衰期，在trainer中默认是800
        static_sdf_params_max_norm=None, # 控制系数优化的最大step，在trainer中默认没有开启
        
        poly_coeffs_learning_rate=1e-2,
        poly_coeffs_decay=400,
        poly_coeffs_max_norm=None,

        fourier_low_learning_rate = 5e-3,
        fourier_low_decay=800,
        fourier_low_max_norm=0.005,

        fourier_mid_learning_rate = 1e-3,
        fourier_mid_decay=1200,
        fourier_mid_max_norm=0.005,

        fourier_high_learning_rate = 5e-4, 
        fourier_high_decay=1600,
        fourier_high_max_norm=0.005,

        appearance_learning_rate=1e-3,
        appearance_decay=1250,

        light_learning_rate=5e-4, # 用于light的参数
        light_learning_rate_decay=1250,

        regularization=JointRegularizationConfig(
            ssim_weight_begin = 1., # ssim loss
            ssim_weight_end  = 10.,
            ssim_weight_decay_steps = 2500,
            ssim_weight_start_step = 0,

            mask_weight_begin = 10., # mask supervision loss
            mask_weight_end = 10.,
            mask_weight_decay_steps = 1250,
            mask_weight_start_step = 0,

            # sdf entropy loss
            sdf_entropy_weight_begin = 0.2, # 控制sdf entropy loss的权重
            sdf_entropy_weight_end = 0.01,
            sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
            sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
            sdf_entropy_end_step = -1, # 控制sdf entropy loss的终止步数

            # curve derivative smooth loss
            time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
            time_tv_weight_end = 0.05,
            time_tv_decay_steps = 1250, # 控制curve derivative smooth loss的线性衰减步数
            time_tv_start_step = 0, # 控制curve derivative smooth loss的起始步数
            time_tv_end_step=-1, # 控制curve derivative smooth loss的终止步数

            # sdf eikonal loss，开启后，大概需要占用1G显存
            sdf_eikonal_weight_begin = 0.01, # 控制sdf eikonal loss的权重
            sdf_eikonal_weight_end = 0.001,
            sdf_eikonal_decay_steps = 1250, # 控制sdf eikonal loss的线性衰减步数
            sdf_eikonal_start_step = 0, # 控制sdf eikonal loss的起始步数
            sdf_eikonal_end_step=-1, # 控制sdf eikonal loss的终止步数

            reg_temporal_hashgrid_begin = 0.1,
            reg_temporal_hashgrid_end = 0.01,
            reg_temporal_hashgrid_decay_steps = 1250,
            reg_temporal_hashgrid_start_step = 0,
            reg_temporal_hashgrid_end_step=-1,

            reg_occ_begin=0.0,
            reg_occ_end=0.001,
            reg_occ_decay_steps=1250,
            reg_occ_start_step=0,
            reg_occ_end_step=-1,

            reg_light_begin=0.001,
            reg_light_end=0.01,
            reg_light_decay_steps=1250,
            reg_light_start_step=0,
            reg_light_end_step=-1,
        ),

        mixed_precision=False,
        full_test_after_train=True,
        full_fix_vis_after_train=True,
        full_orbit_vis_after_train=True,
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)


toy_task = DS_TrainTask(
    dataset=SyntheticDynamicMonocularBlenderRGBADataset(
        path=Path("data/ObjSel-Dyn-monocular/toy/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=0.5,
        low_freq_fourier_bands=[1, 10],
        mid_freq_fourier_bands=[11, 20],
        high_freq_fourier_bands=[21,30],
        dynamic_texture=Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 96, 32, 6],
                activation='sigmoid',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[16,16,8],
            deform_desired_resolution=[4096,4096,64],
            deform_num_levels=32,
        ),
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-3],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-3],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='ObjSel-Dyn-monocular_d_joint/toy', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
        num_steps_per_save=1000,
        hold_after_train=False,

        geometry_lr=1e-3, # 用于flexicube的 deform 和 weight 参数
        geometry_lr_decay=800,
        
        static_sdf_params_learning_rate=1e-2,
        static_sdf_params_decay=800, # 控制学习率的半衰期，在trainer中默认是800
        static_sdf_params_max_norm=None, # 控制系数优化的最大step，在trainer中默认没有开启
        
        poly_coeffs_learning_rate=1e-2,
        poly_coeffs_decay=400,
        poly_coeffs_max_norm=None,

        fourier_low_learning_rate = 5e-3,
        fourier_low_decay=800,
        fourier_low_max_norm=0.005,

        fourier_mid_learning_rate = 1e-3,
        fourier_mid_decay=1200,
        fourier_mid_max_norm=0.005,

        fourier_high_learning_rate = 5e-4, 
        fourier_high_decay=1600,
        fourier_high_max_norm=0.005,

        appearance_learning_rate=1e-4,
        appearance_decay=1250,

        light_learning_rate=1e-4, # 用于light的参数
        light_learning_rate_decay=1250,

        regularization=JointRegularizationConfig(
            ssim_weight_begin = 1., # ssim loss
            ssim_weight_end  = 10.,
            ssim_weight_decay_steps = 2500,
            ssim_weight_start_step = 0,

            mask_weight_begin = 10., # mask supervision loss
            mask_weight_end = 10.,
            mask_weight_decay_steps = 1250,
            mask_weight_start_step = 0,

            # sdf entropy loss
            sdf_entropy_weight_begin = 0.2, # 控制sdf entropy loss的权重
            sdf_entropy_weight_end = 0.01,
            sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
            sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
            sdf_entropy_end_step = -1, # 控制sdf entropy loss的终止步数

            # curve derivative smooth loss
            time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
            time_tv_weight_end = 0.2,
            time_tv_decay_steps = 1250, # 控制curve derivative smooth loss的线性衰减步数
            time_tv_start_step = 0, # 控制curve derivative smooth loss的起始步数
            time_tv_end_step=-1, # 控制curve derivative smooth loss的终止步数

            # sdf eikonal loss，开启后，大概需要占用1G显存
            sdf_eikonal_weight_begin = 0.01, # 控制sdf eikonal loss的权重
            sdf_eikonal_weight_end = 0.001,
            sdf_eikonal_decay_steps = 1250, # 控制sdf eikonal loss的线性衰减步数
            sdf_eikonal_start_step = 0, # 控制sdf eikonal loss的起始步数
            sdf_eikonal_end_step=-1, # 控制sdf eikonal loss的终止步数

            reg_temporal_hashgrid_begin = 0.1,
            reg_temporal_hashgrid_end = 0.01,
            reg_temporal_hashgrid_decay_steps = 1250,
            reg_temporal_hashgrid_start_step = 0,
            reg_temporal_hashgrid_end_step=-1,

            reg_occ_begin=0.0,
            reg_occ_end=0.001,
            reg_occ_decay_steps=1250,
            reg_occ_start_step=0,
            reg_occ_end_step=-1,

            reg_light_begin=0.001,
            reg_light_end=0.01,
            reg_light_decay_steps=1250,
            reg_light_start_step=0,
            reg_light_end_step=-1,
        ),

        mixed_precision=False,
        full_test_after_train=True,
        full_fix_vis_after_train=True,
        full_orbit_vis_after_train=True,
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)



@dataclass
class RenderAfterTrain(Task):
    """渲染训练好的模型"""
    load: Path = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/dnerf_d_joint/jumpingjacks/test/task.py')
    step: Optional[int] = None

    def render_and_save_video(self, model: D_Joint, trainer: D_JointTrainer, experiment: DS_Experiment, loader, name: str, fix_camera: bool = False, fix_camera_order: int = 0):
        inputs, gt_outputs, indices = next(loader)

        if fix_camera:
            inputs = inputs.reset_c2w_to_ref_camera_pose(ref_camera_pose=inputs.c2w[fix_camera_order])

        model.set_batch_gt_geometry([getattr(trainer, 'gt_geometry_test_vis')[i] for i in indices.tolist()])

        (
            color_outputs,
            depth_outputs, gt_depth_outputs,
            normal_outputs, gt_normal_outputs,
            reg_loss_dict
        ) = model.render_report(camera_inputs=inputs)

        images = []
        for i in range(len(inputs)):
            if model.geometry == 'gt':
                continue  # 若只使用 GT 几何，则无需渲染对比

            col0 = None
            col1 = None
            image = None

            if model.texture_able:
                bg_color = model.get_background_color().to(inputs.device)
                pred_rgb = color_outputs.blend(bg_color)
                col0 = torch.cat((
                    pred_rgb[i].detach().item(), # no ref gt view
                    pred_rgb[i].detach().item(),
                ), dim=0).clamp(0, 1)
                image = col0
            if model.geometry != 'gt':
                col1 = torch.cat((
                    gt_normal_outputs[i].visualize((1, 1, 1)).item(),
                    normal_outputs[i].visualize((1, 1, 1)).item(),
                ), dim=0).clamp(0, 1)
                if image is not None:
                    image = torch.cat((image, col1), dim=1)
                else:
                    image = col1
            images.append(image)

        experiment.dump_images2video('test_vis', name=name, images=images, fps=48,duration=5)
        del inputs, gt_outputs, indices, color_outputs, depth_outputs, gt_depth_outputs, normal_outputs, gt_normal_outputs, reg_loss_dict, images
        torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            experiment: DS_Experiment = task.experiment
            model: D_Joint = task.model
            dataset: SyntheticDynamicMonocularBlenderRGBADataset = task.dataset
            trainer: D_JointTrainer = task.trainer
            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)
        model.eval()
        # test_vis_size = dataset.get_size(split='test_vis')
        # loader_orbit = dataset.get_test_vis_iter(batch_size=test_vis_size, shuffle=False, infinite=False)
        # loader_fixed = dataset.get_test_vis_iter(batch_size=test_vis_size, shuffle=False, infinite=True)

        # self.render_and_save_video(model, trainer, experiment, loader_orbit, name='orbit_camera_fps48') # 渲染轨道视角
        # self.render_and_save_video(model, trainer, experiment, loader_fixed, name='fix_camera_order0_fps48', fix_camera=True, fix_camera_order=0) # 渲染固定视角，固定为第49个相机
        # self.render_and_save_video(model, trainer, experiment, loader_fixed, name='fix_camera_order1_fps48', fix_camera=True, fix_camera_order=49) # 渲染固定视角，固定为第49个相机
        # self.render_and_save_video(model, trainer, experiment, loader_fixed, name='fix_camera_order2_fps48', fix_camera=True, fix_camera_order=99) # 渲染固定视角，固定为第99个相机
        # self.render_and_save_video(model, trainer, experiment, loader_fixed, name='fix_camera_order3_fps48', fix_camera=True, fix_camera_order=149) # 渲染固定视角，固定为第149个相机
        # experiment.parse_log_auto(experiment.log_path) # 分析日志

        train_batch_size = dataset.get_size(split='train') // 4
        train_data_iterator = dataset.get_train_iter(
            batch_size=train_batch_size,
            shuffle=False,
            infinite=False,
        )
        count = 0
        train_metrics = {}
        for inputs, gt_outputs, indices in train_data_iterator:
            _, metrics, visualization = trainer.step(
                model,
                inputs,
                gt_outputs,
                indices=indices,
                mode='train',
                visual=True,
                val_pbr_attr=True,
            )
            for i, image in enumerate(visualization):
                experiment.dump_image('train', index=i+count*train_batch_size, image=image)
            for key, val in metrics.items():
                train_metrics.setdefault(key, []).append(val)
            count += 1
        train_metrics = { key: sum(val) / len(val) for key, val in train_metrics.items() }
        experiment.log(P@'Train Metrics: {train_metrics}')



@dataclass
class EvalSceneFlow(Task):
    """为训练好的 D_DiffDR 模型评估场景流，生成网格并计算场景流用在mesh上看预测效果。"""

    load: Path = Path('outputs/d_diffdr_horse/test.2.1/task.py')

    step: Optional[int] = None

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            experiment: DS_Experiment = task.experiment
            model: D_Joint = task.model
            dataset: SyntheticDynamicMonocularBlenderRGBADataset = task.dataset
            trainer: D_JointTrainer = task.trainer
            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)
        model.eval()
        test_vis_size = dataset.get_size(split='test_vis')
        loader = dataset.get_test_vis_iter(batch_size=test_vis_size, shuffle=False, infinite=False)
        inputs, gt_outputs, indices = next(loader)
        model.set_batch_gt_geometry([getattr(trainer, f'gt_geometry_test_vis')[i] for i in indices.tolist()])
        pred_meshes, _, _, _, _, _ = model.get_geometry(times=inputs.times)

        chamfer_list1, chamfer_list2, chamfer_list3 = [], [], []
        vis_list1, vis_list3, vis_list4 = [], [], []

        def render_and_export_mesh(mesh, camera, shader, export_path, vis_path, frame_name):
            mesh.export(path=export_path, only_geometry=True)
            image = mesh.render(camera, shader=shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1)
            experiment.dump_image(subfolder=vis_path, name=f"{frame_name}.png", image=image)
            return image

        def compute_and_log_chamfer(frame, name, mesh_a, mesh_b, record_list):
            chamfer = ChamferDistanceMetric(target_num_points=1000000)(mesh_a, mesh_b)
            experiment.log(f"Frame {frame}: Chamfer Distance {name}: {chamfer}")
            record_list.append(chamfer)

        for frame in range(inputs.shape[0]):
            dt = (inputs.times[frame + 1] - inputs.times[frame]) if frame < inputs.times.shape[0] - 1 else (inputs.times[frame] - inputs.times[frame - 1])
            camera = inputs[frame]
            
            # --- Ground Truth Mesh ---
            gt_mesh = model.get_gt_geometry(indice=frame)
            vis_list1.append(render_and_export_mesh(
                gt_mesh, camera, PrettyShader(True),
                experiment.dump_file_path(subfolder='eval_scene_flow_gt_mesh', file_name=f'{frame}.obj'),
                'eval_scene_flow_gt_mesh_vis', frame)
            )
            # --- Predicted Mesh ---
            pred_mesh = pred_meshes[frame]
            vis_list3.append(render_and_export_mesh(
                pred_mesh, camera, PrettyShader(True),
                experiment.dump_file_path(subfolder='eval_scene_flow_pred_mesh', file_name=f'{frame}.obj'),
                'eval_scene_flow_pred_mesh_vis', frame)
            )
            # --- Predicted 下一帧 Mesh use Scene Flow ---
            next_pred_mesh_calc = pred_mesh.get_next_frame_mesh(dt.item())
            vis_list4.append(render_and_export_mesh(
                next_pred_mesh_calc, camera, PrettyShader(True),
                experiment.dump_file_path(subfolder='eval_scene_flow_pred_mesh_next_frame_calc_by_scene_flow', file_name=f'{frame+1}.obj'),
                'eval_scene_flow_pred_mesh_next_frame_calc_by_scene_flow_vis', frame+1)
            )
            if frame < inputs.shape[0] - 1:
                next_gt_mesh = model.get_gt_geometry(indice=frame+1)
                next_pred_mesh = pred_meshes[frame+1]
                compute_and_log_chamfer(frame, "Pred next frame vs scene flow predicted next frame", next_pred_mesh, next_pred_mesh_calc, chamfer_list3)
                compute_and_log_chamfer(frame, "GT next frame vs scene flow predicted next frame", next_gt_mesh, next_pred_mesh_calc, chamfer_list2)
            
            compute_and_log_chamfer(frame, "GT vs Pred", gt_mesh, pred_mesh, chamfer_list1)

        # === 日志和可视化 ===
        experiment.log(f"Avg Chamfer GT vs Pred: {sum(chamfer_list1)/len(chamfer_list1)}")
        experiment.log(f"Avg Chamfer GT next vs scene flow pred next: {sum(chamfer_list2)/len(chamfer_list2)}")
        experiment.log(f"Avg Chamfer pred next vs scene flow pred next: {sum(chamfer_list3)/len(chamfer_list3)}")

        experiment.dump_images2video(subfolder='eval_scene_flow_gt_mesh_vis', images=vis_list1, name='eval_scene_flow_gt_mesh_vis.mp4',target_mb=2,duration=5)
        experiment.dump_images2video(subfolder='eval_scene_flow_pred_mesh_vis', images=vis_list3, name='eval_scene_flow_pred_mesh_vis.mp4',target_mb=2,duration=5)
        experiment.dump_images2video(subfolder='eval_scene_flow_pred_mesh_next_frame_calc_by_scene_flow_vis', images=vis_list4,name='eval_scene_flow_pred_mesh_next_frame_calc_by_scene_flow_vis.mp4',target_mb=2,duration=5)



if __name__ == '__main__':
    TaskGroup(
        # train task

        jumpingjacks = jumpingjacks_task,

        beagle = beagle_task,

        toy = toy_task,
        
        # eval task
        rendermodel = RenderAfterTrain(cuda=0), 
        evalsceneflow = EvalSceneFlow(cuda=0),
    ).run()
