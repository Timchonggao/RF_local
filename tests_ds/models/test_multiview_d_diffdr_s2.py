from __future__ import annotations

# import modulues
from dataclasses import dataclass
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
from rfstudio_ds.data import (
    SyntheticDynamicMonocularBlenderDepthDataset,
    SyntheticDynamicMonocularCostumeDepthDataset,
    SyntheticDynamicMultiViewBlenderDepthDataset,
    SyntheticDynamicMultiViewCostumeDepthDataset,
)
from rfstudio_ds.model import D_DiffDR_S2
from rfstudio_ds.trainer import D_DiffDRTrainer_S2, DiffDRRegularizationConfig_S2
from rfstudio_ds.graphics.spatial_sampler import NearestGridSampler
from rfstudio_ds.visualization import Visualizer
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh
from rfstudio_ds.visualization._plot_cube_curve import plot_cube_curve, plot_curve
from rfstudio_ds.model.density_field.components.encoding_4d import Grid4d_HashEncoding
from rfstudio_ds.nn import MLPDecoderNetwork, Grid4DDecoderNetwork


# costume mv dataset tasks


# blender mv dataset tasks
multi_view_cat_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderDepthDataset(
        path=Path("data/multiview_dynamic_blender/cat/"), # ori frames: 240 + 2* 20 = 280, high freq frames: [[120,170], [190,220]]
        # agumentation_sample_frames=[[120,170], [190,220]],
        # costume_sample_frames=[0,100],
        # costume_padding_size=10
    ),
    model=D_DiffDR_S2(
        load=Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_diffdr/cat/test/dump/model.pth'),
        sdf_residual_enc = Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 32, 1],
                activation='sigmoid',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[8,8,8],
            deform_desired_resolution=[128,128,256],
            deform_num_levels=16,
        ),
        wavelet_name='db1',
        wavelet_level=3,
        wavelet_min_step=1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_diffdr/cat', timestamp='test_s2'),
    trainer=D_DiffDRTrainer_S2(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_fix_vis=500,
        # num_steps_per_analyze_cube_curve=500,
        num_steps_per_save=1000,

        wavelet_params_learning_rate = 0.003,
        walelet_params_decay=800,
        wavelet_params_max_norm=0.2,

        use_multi_model_stage = True,
        model_start_steps=[0,0,0,0,0,0],

        regularization=DiffDRRegularizationConfig_S2(
            sdf_entropy_weight_begin = 0.01, # sdf entropy loss
            sdf_entropy_weight_end = 0.005,
            sdf_entropy_decay_steps = 1250,
            sdf_entropy_start_step = -1,
            sdf_entropy_end_step = -1,

            time_tv_weight_begin = 0.01, # curve derivative smooth loss
            time_tv_weight_end = 0.001,
            time_tv_decay_steps = 1000,
            time_tv_start_step = -1,
            time_tv_end_step = -1,

            curve_wavelet_sparse_weight_begin = 0.01, # cruve coefficient tv smooth loss
            curve_wavelet_sparse_weight_end = 0.001,
            curve_wavelet_sparse_decay_steps = 1000,
            curve_wavelet_sparse_start_step = 0, # 开启后，大概需要占用4G显存
            curve_wavelet_sparse_end_step = 5000,
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

multi_view_deer_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderDepthDataset(
        path=Path("data/multiview_dynamic_blender/deer/"),
    ),
    model=D_DiffDR_S2(
        load=Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_diffdr/deer/test/dump/model.pth'),
        sdf_residual_enc = Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 32, 1],
                activation='sigmoid',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[8,8,8],
            deform_desired_resolution=[128,128,256],
            deform_num_levels=16,
        ),
        wavelet_name='db1',
        wavelet_level=3,
        wavelet_min_step=1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_diffdr/deer', timestamp='test_s2'),
    trainer=D_DiffDRTrainer_S2(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_fix_vis=500,
        # num_steps_per_analyze_cube_curve=500,
        num_steps_per_save=1000,

        wavelet_params_learning_rate = 0.003,
        use_multi_model_stage = True,
        model_start_steps=[0,0,0,0,0,0],
        regularization=DiffDRRegularizationConfig_S2(
            sdf_entropy_weight_begin = 0.01, # sdf entropy loss
            sdf_entropy_weight_end = 0.005,
            sdf_entropy_decay_steps = 1250,
            sdf_entropy_start_step = 0,

            curve_wavelet_sparse_weight_begin = 0.01, # cruve coefficient tv smooth loss
            curve_wavelet_sparse_weight_end = 0.01,
            curve_wavelet_sparse_decay_steps = 1000,
            curve_wavelet_sparse_start_step = -1, # 开启后，大概需要占用4G显存
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

multi_view_lego_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderDepthDataset(
        path=Path("data/multiview_dynamic_blender/lego/"),
    ),
    model=D_DiffDR_S2(
        load=Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_diffdr/lego/test_flexi/dump/model.pth'),
        sdf_residual_enc = Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 32, 1],
                activation='sigmoid',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[8,8,8],
            deform_desired_resolution=[128,128,256],
            deform_num_levels=16,
        ),
        wavelet_name='db1',
        wavelet_level=3,
        wavelet_min_step=1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_diffdr/lego', timestamp='test_s2'),
    trainer=D_DiffDRTrainer_S2(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_fix_vis=500,
        # num_steps_per_analyze_cube_curve=500,
        num_steps_per_save=1000,

        wavelet_params_learning_rate = 0.003,
        use_multi_model_stage = True,
        model_start_steps=[0,0,0,0,0,0],
        regularization=DiffDRRegularizationConfig_S2(
            sdf_entropy_weight_begin = 0.01, # sdf entropy loss
            sdf_entropy_weight_end = 0.005,
            sdf_entropy_decay_steps = 1250,
            sdf_entropy_start_step = -1,

            time_tv_weight_begin = 0.01, # curve derivative smooth loss
            time_tv_weight_end = 0.001,
            time_tv_decay_steps = 1000,
            time_tv_start_step = -1,

            curve_wavelet_sparse_weight_begin = 0.01, # cruve coefficient tv smooth loss
            curve_wavelet_sparse_weight_end = 0.001,
            curve_wavelet_sparse_decay_steps = 1000,
            curve_wavelet_sparse_start_step = 0, # 开启后，大概需要占用4G显存
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
class ExportMesh(Task):
    """为训练好的 D_DiffDR_S2 模型评估场景流，生成网格并计算场景流用在mesh上看预测效果。"""

    load: Path = Path('outputs/d_diffdr_horse/test.2.1/task.py')

    step: Optional[int] = None

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            experiment: DS_Experiment = task.experiment
            model: D_DiffDR_S2 = task.model
            dataset: SyntheticDynamicMultiViewCostumeDepthDataset = task.dataset
            trainer: D_DiffDRTrainer_S2 = task.trainer
            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)
        model.eval()
        vis_size = dataset.get_size(split='fix_vis')
        loader = dataset.get_orbit_vis_iter(batch_size=vis_size, shuffle=False, infinite=True)
        inputs, gt_outputs, indices = next(loader)
        model.set_batch_gt_geometry([getattr(trainer, f'gt_geometry_fix_vis')[i] for i in indices.tolist()])
        pred_meshes, _ = model.get_geometry(times=inputs.times)
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )

        vis_list1, vis_list2 = [], []
        for frame in range(inputs.shape[0]):
            camera = inputs[frame]
            
            # --- Ground Truth Mesh ---
            gt_mesh = model.get_gt_geometry(indice=frame)
            gt_mesh.export(
                path=experiment.dump_file_path(subfolder='exportmesh/gt_mesh', file_name=f'{frame}.obj'),
                only_geometry=True
            )
            vis_list1.append(
                gt_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1)
            )

            # --- Predicted Mesh ---
            pred_mesh = pred_meshes[frame]
            pred_mesh.export(
                path=experiment.dump_file_path(subfolder='exportmesh/pred_mesh', file_name=f'{frame}.obj'),
                only_geometry=True
            )
            vis_list2.append(
                pred_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1)
            )

        experiment.dump_images2video(subfolder='exportmesh', images=vis_list1, name='gt_mesh',target_mb=2,duration=5)
        experiment.dump_images2video(subfolder='exportmesh', images=vis_list2, name='pred_mesh',target_mb=2,duration=5)



@dataclass
class RenderAfterTrain(Task):
    """渲染训练好的模型"""
    load: Path = Path('outputs/d_diffdr_horse/test.2.1/task.py')
    step: Optional[int] = None

    def render_and_save_video(self, model: D_DiffDR_S2, trainer: D_DiffDRTrainer_S2, experiment: DS_Experiment, loader, name: str, fix_camera: bool = False, fix_camera_order: int = 0):
        inputs, gt_outputs, indices = next(loader)

        if fix_camera:
            inputs = inputs.reset_c2w_to_ref_camera_pose(ref_camera_pose=inputs.c2w[fix_camera_order])

        model.set_batch_gt_geometry([getattr(trainer, 'gt_geometry_fix_vis')[i] for i in indices.tolist()])

        (
            pred_meshes,
            depth_outputs, gt_depth_outputs,
            normal_outputs, gt_normal_outputs,
            reg_loss_dict,
        ) = model.render_report(camera_inputs=inputs)

        images = []
        for i in range(len(inputs)):
            if model.geometry == 'gt':
                continue  # 若只使用 GT 几何，则无需渲染对比

            # col1 = torch.cat((
            #     gt_normal_outputs[i].visualize((1, 1, 1)).item(),
            #     normal_outputs[i].visualize((1, 1, 1)).item(),
            # ), dim=0).clamp(0, 1)

            # col2 = torch.cat((
            #     gt_depth_outputs[i].visualize().item(),
            #     depth_outputs[i].visualize().item(),
            # ), dim=0).clamp(0, 1)
            # image = torch.cat((col1, col2), dim=1)

            row1 = gt_normal_outputs[i].visualize((1, 1, 1)).item().clamp(0, 1)
            row2 = normal_outputs[i].visualize((1, 1, 1)).item().clamp(0, 1)

            image = torch.cat((row1, row2), dim=0)
            images.append(image)

        experiment.dump_images2video('render_model', name=name, images=images, fps=48,duration=5)
        del inputs, gt_outputs, indices, pred_meshes, depth_outputs, gt_depth_outputs, normal_outputs, gt_normal_outputs, reg_loss_dict, images
        torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            experiment: DS_Experiment = task.experiment
            model: D_DiffDR_S2 = task.model
            dataset: SyntheticDynamicMultiViewCostumeDepthDataset = task.dataset
            trainer: D_DiffDRTrainer_S2 = task.trainer
            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)
        model.eval()
        vis_size = dataset.get_size(split='fix_vis')
        loader_orbit = dataset.get_orbit_vis_iter(batch_size=vis_size, shuffle=False, infinite=True)
        loader_fix = dataset.get_fix_vis_iter(batch_size=vis_size, shuffle=False, infinite=False)

        self.render_and_save_video(model, trainer, experiment, loader_orbit, name='fix_camera_order0_fps48', fix_camera=True, fix_camera_order=0) # 渲染固定视角，固定为第49个相机
        self.render_and_save_video(model, trainer, experiment, loader_orbit, name='fix_camera_order1_fps48', fix_camera=True, fix_camera_order=49) # 渲染固定视角，固定为第49个相机
        self.render_and_save_video(model, trainer, experiment, loader_orbit, name='fix_camera_order2_fps48', fix_camera=True, fix_camera_order=99) # 渲染固定视角，固定为第99个相机
        self.render_and_save_video(model, trainer, experiment, loader_orbit, name='fix_camera_order3_fps48', fix_camera=True, fix_camera_order=149) # 渲染固定视角，固定为第149个相机
        
        # self.render_and_save_video(model, trainer, experiment, loader_orbit, name='orbit_camera_fps48') # 渲染轨道视角
        # self.render_and_save_video(model, trainer, experiment, loader_fix, name='fix_camera_fps48') # 渲染固定视角        
        
        # experiment.parse_log_auto(experiment.log_path) # 分析日志



@dataclass
class EvalTrain(Task):
    """在训练好的模型上渲染 train data的拟合效果"""

    load: Path = Path('outputs/d_diffdr_horse/test.2.1/task.py')

    step: Optional[int] = None

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            experiment: DS_Experiment = task.experiment
            model: D_DiffDR_S2 = task.model
            dataset: SyntheticDynamicMonocularBlenderDepthDataset = task.dataset
            trainer: D_DiffDRTrainer_S2 = task.trainer

            loader = dataset.get_train_iter(batch_size=8, shuffle=False, infinite=False)
            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)

        model.eval()
        for inputs, gt_outputs, indices in loader:
            model.set_batch_gt_geometry([getattr(trainer, f'gt_geometry_train')[i] for i in indices.tolist()])
            _, metrics, visualization = trainer.step(
                model,
                inputs,
                gt_outputs,
                indices=indices,
                mode='train',
                visual='all'
            )
            for i, image in enumerate(visualization):
                experiment.dump_image('train', index=indices[i], image=image)



@dataclass
class EvalSceneFlow(Task):
    """为训练好的 D_DiffDR_S2 模型评估场景流，生成网格并计算场景流用在mesh上看预测效果。"""

    load: Path = Path('outputs/d_diffdr_horse/test.2.1/task.py')

    step: Optional[int] = None

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            experiment: DS_Experiment = task.experiment
            model: D_DiffDR_S2 = task.model
            dataset: SyntheticDynamicMonocularBlenderDepthDataset = task.dataset
            trainer: D_DiffDRTrainer_S2 = task.trainer

            loader = dataset.get_fix_vis_iter(batch_size=dataset.get_size(split='fix_vis'), shuffle=False, infinite=False)
            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)

        model.eval()
        inputs, gt_outputs, indices = next(loader)
        model.set_batch_gt_geometry([getattr(trainer, f'gt_geometry_fix_vis')[i] for i in indices.tolist()])
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



@dataclass
class EvalCubeCurve(Task):

    load: Path = Path('outputs/d_diffdr_horse/test.2.3/task.py')

    step: Optional[int] = None

    compute_max_error_frame: bool = False

    def compute_loss(
        self,
        depths: DepthImages,
        gt_depths: DepthImages,
        normals: VectorImages,
        gt_normals: VectorImages,
        *,
        eps: float = 1e-8,
        error_map: bool = True
    ) -> Tensor:
        results = []
        for depth, gt_depth, normal, gt_normal in zip(depths, gt_depths, normals, gt_normals):
            if depth is None:
                dloss = normal.new_zeros(1)
            else:
                gt_mask = gt_depth[..., 1:]
                curr_mask = depth[..., 1:]
                dloss = (((depth[..., :1] - gt_depth[..., :1]) * gt_mask).square().sum(-1, keepdim=True) + eps).sqrt()
            if normal is None:
                nloss = depth.new_zeros(1)
            else:
                gt_mask = gt_normal[..., 1:]
                curr_mask = normal[..., 1:]
                nloss = (((normal[..., :1] - gt_normal[..., :1]) * gt_mask).square().sum(-1, keepdim=True) + eps).sqrt()
            if error_map:
                results.append(250 * dloss + nloss)
            else:
                mloss = (gt_mask - curr_mask).abs()
                results.append(250 * dloss.mean() + 10 * mloss.mean() + nloss.mean())
        return torch.stack(results)        

    def get_max_error_frmae(
        self,
        pred_meshes: List[DS_TriangleMesh],
        gt_meshes: List[DS_TriangleMesh],
        times: torch.Tensor,
        cameras: List[DS_Cameras],
        experiment: DS_Experiment
    ):
        mesh_errors = []
        meshes_errors = []
        depth_shader = DepthShader(antialias=True, culling=False)
        normal_shader = NormalShader(antialias=True, normal_type='flat')

        for i in range(0, len(times)):
            pred_mesh = pred_meshes[i]
            gt_mesh = gt_meshes[i]
            for camera_idx in range(0, len(cameras), 2):
                camera = cameras[camera_idx]
                depth_image = pred_mesh.render(camera, shader=depth_shader)
                normal_image = pred_mesh.render(camera, shader=normal_shader)
                gt_depth_image = gt_mesh.render(camera, shader=depth_shader)
                gt_normal_image = gt_mesh.render(camera, shader=normal_shader)

                loss = self.compute_loss(    
                    depths=depth_image,
                    gt_depths=gt_depth_image,
                    normals=normal_image,
                    gt_normals=gt_normal_image,
                    error_map=False
                )
                mesh_errors.append(loss)
            meshes_errors.append(torch.stack(mesh_errors).mean().item())
            experiment.log(f"Frame {i} Error: {meshes_errors[-1]}")
        self.plot_depth_error_curve(times, meshes_errors, save_path=experiment.dump_file_path(subfolder='eval_max_error_frame', file_name='eval_max_error_frame.png'))
        experiment.log(f"Mean Error: {np.mean(meshes_errors)}")
        experiment.log(f"Max Error Frame: {np.argmax(meshes_errors)}, Min Error Frame: {np.argmin(meshes_errors)}")
        return np.argmax(meshes_errors)

    def plot_depth_error_curve(self, times: torch.Tensor, errors: List[float], save_path: str):
        times_np = times.cpu().numpy()
        errors_np = np.array(errors)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_axes([0.1, 0.35, 0.8, 0.6])
        text_ax = fig.add_axes([0.1, 0.05, 0.8, 0.25])

        ax.plot(
            times_np,
            errors_np,
            label='Depth Error',
            linestyle='-',
            color='blue',
            linewidth=2.0,
            marker='o',
            markersize=4
        )

        ax.set_title('Depth Error Over Time', fontsize=12, pad=10)
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Depth Error', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', fontsize=9)
        ax.tick_params(axis='both', labelsize=8)

        # 文本信息展示
        info_lines = [
            f"Mean Error: {errors_np.mean():.4f}",
            f"Max Error: {errors_np.max():.4f}, Frame: {np.argmax(errors_np)}",
            f"Min Error: {errors_np.min():.4f}, Frame: {np.argmin(errors_np)}"
        ]
        y_pos = 0.9
        for line in info_lines:
            text_ax.text(0.5, y_pos, line, fontsize=10, ha='center', va='top', color='black')
            y_pos -= 0.3

        text_ax.axis('off')

        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
        except Exception as e:
            print(f"[Plot Error] Failed to save plot: {e}")
        finally:
            plt.close(fig)
    
    def get_max_error_cubes(
        self,
        pred_mesh: DS_TriangleMesh,
        gt_mesh: DS_TriangleMesh,
        cameras: DS_Cameras,
        sampler: NearestGridSampler,
        topk: int = 1
    ):
        error_maps = []
        for camera in cameras:
            pred_depth_image = pred_mesh.render(camera, shader=DepthShader(antialias=True, culling=False))
            pred_normal_image = pred_mesh.render(camera, shader=NormalShader(antialias=True, normal_type='flat'))
            gt_depth_image = gt_mesh.render(camera, shader=DepthShader(antialias=True, culling=False))
            gt_normal_image = gt_mesh.render(camera, shader=NormalShader(antialias=True, normal_type='flat'))

            error_map = self.compute_loss(    
                depths=pred_depth_image,
                gt_depths=gt_depth_image,
                normals=pred_normal_image,
                gt_normals=gt_normal_image,
                error_map=True
            )
            error_maps.append(error_map)
        error_maps = torch.stack(error_maps)

        positions, values = pred_mesh.spatial_aggregation(cameras=cameras, images=error_maps)
        sampler.aggregate(positions=positions, importances=values, average_aggregate=True)
        max_errors_cube_values, max_error_cube_indices = sampler.get_max_density(topk=topk)

        return max_errors_cube_values, max_error_cube_indices

    def analyze_cubes_curve_info(
        self,
        trainer: D_DiffDRTrainer_S2,
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
                pred_data=cube_sdfs.detach().cpu(),
                gt_data=cube_gt_sdfs,
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
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            experiment: DS_Experiment = task.experiment
            model: D_DiffDR_S2 = task.model
            dataset: SyntheticDynamicMultiViewCostumeDepthDataset = task.dataset
            trainer: D_DiffDRTrainer_S2 = task.trainer
            sampler = NearestGridSampler(
                resolution=model.geometry_resolution,
                scale=model.geometry_scale,    
            )

            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)
            sampler.__setup__()
            sampler.to(self.device)
            sampler.reset()
            if model.geometric_repr.device != self.device:
                model.geometric_repr.swap_(model.geometric_repr.to(self.device))
            DD_isocubes = model.geometric_repr.replace(
                static_sdf_values = model.static_sdf_params,
                sdf_curve_poly_coefficient = model.sdf_curve_poly_coefficient,
                sdf_curve_low_freq_fourier_coefficient = model.sdf_curve_low_freq_fourier_coefficient,
                sdf_curve_mid_freq_fourier_coefficient = model.sdf_curve_mid_freq_fourier_coefficient,
                sdf_curve_high_freq_fourier_coefficient = model.sdf_curve_high_freq_fourier_coefficient,
            )

            loader = dataset.get_val_iter(batch_size=dataset.get_size(split='val'), shuffle=False, infinite=False)
            inputs, gt_outputs, indices = next(loader)
            times = inputs.times

            gt_meshes = [getattr(trainer, f'gt_geometry_val')[i] for i in indices.tolist()]
            pred_meshes, _, _, _, _, _ = model.get_geometry(times=times)
            pred_sdf, _ = DD_isocubes.query_sdf_at_times(t=times)

        max_error_fame = 185
        if self.compute_max_error_frame:
            # 在mesh sequence中找到 error 最大的 frame，也可以手动指定某一个 frame（自己认为有明显错误的某一帧）
            max_error_fame = self.get_max_error_frmae(
                pred_meshes=pred_meshes,
                gt_meshes=gt_meshes,
                times=times,
                cameras=inputs,
                experiment=experiment
            )

        max_errors_cube_values, max_error_cube_indices = self.get_max_error_cubes(
            pred_mesh=pred_meshes[max_error_fame],
            gt_mesh=gt_meshes[max_error_fame],
            cameras=inputs,
            sampler=sampler,
            topk=10  
        ) # 在最大误差帧或者指定帧中找到最大 error 的 cube，并分析其曲线信息，这样可以避免手动找cube的坐标
        max_error_cube_curve_info = DD_isocubes.get_cube_curve_info(indices=max_error_cube_indices)
        self.analyze_cubes_curve_info(
            trainer=trainer,
            experiment=experiment,
            save_subfolder='eval_max_error_cube_curve',
            cubes_info=max_error_cube_curve_info,
            times=times,
            pred_data=pred_sdf.squeeze(-1),
            highlight_indices=[max_error_fame],
        )
        experiment.log(f"Max Error Cube Mean positions: {max_error_cube_curve_info['mean_cube_positions'].tolist()}") 

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
            [0.03149604797363281, -0.3779527544975281, -0.12598425149917603]
        ], device=self.device)
        fix_points_cube_info = DD_isocubes.get_cube_curve_info(positions=fix_positions)
        self.analyze_cubes_curve_info(
            trainer=trainer,
            experiment=experiment,
            save_subfolder='eval_fix_points_cubes_curve',
            cubes_info=fix_points_cube_info,
            times=times,
            pred_data=pred_sdf.squeeze(-1),
            highlight_indices=[max_error_fame],
        )

        Visualizer().show(
            pred_mesh=pred_meshes[max_error_fame],
            gt_mesh=gt_meshes[max_error_fame],
            
            max_error_cube_vertices_mean_pos=Points(positions=max_error_cube_curve_info["mean_cube_positions"]),
            max_error_cube_vertices=Points(positions=max_error_cube_curve_info["cube_positions"].reshape(-1, 3)),

            fix_points=Points(positions=fix_positions),
            fix_points_cube_vertices=Points(positions=fix_points_cube_info["cube_positions"].reshape(-1, 3)),
        )



if __name__ == '__main__':
    TaskGroup(
        # train task
        mvcat = multi_view_cat_task, # python tests_ds/models/test_d_diffdr.py mv_cat --experiment.timestamp test1
        mvdeer = multi_view_deer_task, # python tests_ds/models/test_d_diffdr.py mv_deer --experiment.timestamp test1
        mvlego = multi_view_lego_task, # python tests_ds/models/test_d_diffdr.py mv_lego --experiment.timestamp test1

        # export task
        exportmesh = ExportMesh(cuda=0), # python tests_ds/models/test_d_diffdr.py exportmesh --load outputs/d_diffdr_horse/test1.1/task.py

        # eval task
        rendermodel = RenderAfterTrain(cuda=0), # python tests_ds/models/test_d_diffdr.py rendermodel --load outputs/d_diffdr_horse/test1.1/task.py
        evaltrain = EvalTrain(cuda=0), # python tests_ds/models/test_d_diffdr.py evaltrain --load outputs/d_diffdr_horse/test1.1/task.py
        evalsceneflow = EvalSceneFlow(cuda=0), # python tests_ds/models/test_d_diffdr.py evalsceneflow --load outputs/d_diffdr_horse/test1.1/task.py
        evalcubecurve = EvalCubeCurve(cuda=0), # python tests_ds/models/test_d_diffdr.py evalcubecurve --load outputs/d_diffdr_horse/test1.1/task.py --no_compute_max_error_frame
    ).run()
