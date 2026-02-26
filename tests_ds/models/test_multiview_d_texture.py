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
from rfstudio_ds.data import SyntheticDynamicMultiViewBlenderRGBADataset
from rfstudio_ds.model import D_Texture # rgb image loss optimization model
from rfstudio_ds.trainer import D_TextureTrainer, TextureRegularizationConfig # rgb image loss optimization trainer, dynamic nvdiffrec
from rfstudio_ds.graphics.spatial_sampler import NearestGridSampler
from rfstudio_ds.visualization import Visualizer
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh
from rfstudio_ds.model.density_field.components.encoding_4d import Grid4d_HashEncoding
from rfstudio_ds.nn import MLPDecoderNetwork, Grid4DDecoderNetwork




cat_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/multiview_dynamic_blender/cat/"),
        costume_sample_frames=[0,239],
        costume_padding_size=1
    ),
    model=D_Texture(
        geometry_scale=0.5,
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
            deform_desired_resolution=[4096,4096,256],
            deform_num_levels=32,
        ),
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-3],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-3],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-3],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_texture/cat', timestamp='test'),
    trainer=D_TextureTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
        num_steps_per_save=1000,

        appearance_learning_rate=3e-2,
        appearance_decay=800,

        light_learning_rate=5e-3, # 用于light的参数
        light_learning_rate_decay=800,

        regularization=TextureRegularizationConfig(
            ssim_weight_begin = 1., # ssim loss
            ssim_weight_end  = 1.,
            ssim_weight_decay_steps = 1250,
            ssim_weight_start_step = 0,

            psnr_weight_begin = 1., # psnr loss
            psnr_weight_end = 1.,
            psnr_weight_decay_steps = 1250,
            psnr_weight_start_step = -1,

            reg_spatial_hashgrid_begin = 0.0,
            reg_spatial_hashgrid_end = 0.0,
            reg_spatial_hashgrid_decay_steps = 1250,
            reg_spatial_hashgrid_start_step  = -1,
            reg_spatial_hashgrid_end_step= -1,

            reg_temporal_hashgrid_begin = 0.2,
            reg_temporal_hashgrid_end = 0.01,
            reg_temporal_hashgrid_decay_steps = 1250,
            reg_temporal_hashgrid_start_step = -1,
            reg_temporal_hashgrid_end_step= 0,

            reg_kd_enc_begin = 0.01,
            reg_kd_enc_end = 0.2,
            reg_kd_enc_decay_steps = 1250,
            reg_kd_enc_start_step = -1,
            reg_kd_enc_end_step = -1,

            reg_occ_begin=0.0,
            reg_occ_end=0.001,
            reg_occ_decay_steps=1250,
            reg_occ_start_step=0,
            reg_occ_end_step=500,

            reg_light_begin=0.001,
            reg_light_end=0.005,
            reg_light_decay_steps=1250,
            reg_light_start_step=0,
            reg_light_end_step=5000,
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

deer_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/multiview_dynamic_blender/deer/"),
    ),
    model=D_Texture(
        geometry_scale=0.5,
    ),
    experiment=DS_Experiment(name='blender_mv_d_texture/deer', timestamp='test'),
    trainer=D_TextureTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_fix_vis=500,
        num_steps_per_save=1000,

        regularization=TextureRegularizationConfig(
            reg_spatial_hashgrid_begin = 0.0,
            reg_spatial_hashgrid_end = 0.0,
            reg_spatial_hashgrid_decay_steps = 1250,
            reg_spatial_hashgrid_start_step  = -1,

            reg_temporal_hashgrid_begin = 0.01,
            reg_temporal_hashgrid_end = 0.5,
            reg_temporal_hashgrid_decay_steps = 1250,
            reg_temporal_hashgrid_start_step = 0,

            reg_kd_enc_begin = 0.01,
            reg_kd_enc_end = 0.2,
            reg_kd_enc_decay_steps = 1250,
            reg_kd_enc_start_step = -1,
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

girlwalk_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/multiview_dynamic_blender/girlwalk/"),
    ),
    model=D_Texture(
        geometry_scale=0.5,
    ),
    experiment=DS_Experiment(name='blender_mv_d_texture/girlwalk', timestamp='test'),
    trainer=D_TextureTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_fix_vis=500,
        num_steps_per_save=1000,

        regularization=TextureRegularizationConfig(
            reg_spatial_hashgrid_begin = 0.0,
            reg_spatial_hashgrid_end = 0.0,
            reg_spatial_hashgrid_decay_steps = 1250,
            reg_spatial_hashgrid_start_step  = -1,

            reg_temporal_hashgrid_begin = 0.01,
            reg_temporal_hashgrid_end = 0.5,
            reg_temporal_hashgrid_decay_steps = 1250,
            reg_temporal_hashgrid_start_step = 0,

            reg_kd_enc_begin = 0.01,
            reg_kd_enc_end = 0.2,
            reg_kd_enc_decay_steps = 1250,
            reg_kd_enc_start_step = -1,
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

lego_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/multiview_dynamic_blender/lego/"),
    ),
    model=D_Texture(
        geometry_scale=0.5,
    ),
    experiment=DS_Experiment(name='blender_mv_d_texture/lego', timestamp='test'),
    trainer=D_TextureTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_fix_vis=500,
        num_steps_per_save=1000,

        regularization=TextureRegularizationConfig(
            reg_spatial_hashgrid_begin = 0.0,
            reg_spatial_hashgrid_end = 0.0,
            reg_spatial_hashgrid_decay_steps = 1250,
            reg_spatial_hashgrid_start_step  = -1,

            reg_temporal_hashgrid_begin = 0.01,
            reg_temporal_hashgrid_end = 0.5,
            reg_temporal_hashgrid_decay_steps = 1250,
            reg_temporal_hashgrid_start_step = 0,

            reg_kd_enc_begin = 0.01,
            reg_kd_enc_end = 0.2,
            reg_kd_enc_decay_steps = 1250,
            reg_kd_enc_start_step = -1,
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

spidermanwalk_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/multiview_dynamic_blender/spidermanwalk/"),
    ),
    model=D_Texture(
        geometry_scale=0.5,
    ),
    experiment=DS_Experiment(name='blender_mv_d_texture/spidermanwalk', timestamp='test'),
    trainer=D_TextureTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_fix_vis=500,
        num_steps_per_save=1000,

        regularization=TextureRegularizationConfig(
            reg_spatial_hashgrid_begin = 0.0,
            reg_spatial_hashgrid_end = 0.0,
            reg_spatial_hashgrid_decay_steps = 1250,
            reg_spatial_hashgrid_start_step  = -1,

            reg_temporal_hashgrid_begin = 0.01,
            reg_temporal_hashgrid_end = 0.5,
            reg_temporal_hashgrid_decay_steps = 1250,
            reg_temporal_hashgrid_start_step = 0,

            reg_kd_enc_begin = 0.01,
            reg_kd_enc_end = 0.2,
            reg_kd_enc_decay_steps = 1250,
            reg_kd_enc_start_step = -1,
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
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/multiview_dynamic_blender/toy/"),
        costume_sample_frames=[0,70],
        costume_padding_size=1
    ),
    model=D_Texture(
        geometry_scale=0.5,
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
            deform_desired_resolution=[4096,4096,128],
            deform_num_levels=32,
        ),
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_texture/toy', timestamp='test'),
    trainer=D_TextureTrainer(
       num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
        num_steps_per_save=1000,

        appearance_learning_rate=3e-2,
        appearance_decay=800,

        light_learning_rate=5e-3, # 用于light的参数
        light_learning_rate_decay=800,

        regularization=TextureRegularizationConfig(
            ssim_weight_begin = 1., # ssim loss
            ssim_weight_end  = 1.,
            ssim_weight_decay_steps = 1250,
            ssim_weight_start_step = 0,

            psnr_weight_begin = 1., # psnr loss
            psnr_weight_end = 1.,
            psnr_weight_decay_steps = 1250,
            psnr_weight_start_step = -1,

            reg_spatial_hashgrid_begin = 0.0,
            reg_spatial_hashgrid_end = 0.0,
            reg_spatial_hashgrid_decay_steps = 1250,
            reg_spatial_hashgrid_start_step  = -1,
            reg_spatial_hashgrid_end_step= -1,

            reg_temporal_hashgrid_begin = 0.2,
            reg_temporal_hashgrid_end = 0.01,
            reg_temporal_hashgrid_decay_steps = 1250,
            reg_temporal_hashgrid_start_step = -1,
            reg_temporal_hashgrid_end_step= 0,

            reg_kd_enc_begin = 0.01,
            reg_kd_enc_end = 0.2,
            reg_kd_enc_decay_steps = 1250,
            reg_kd_enc_start_step = -1,
            reg_kd_enc_end_step = -1,

            reg_occ_begin=0.0,
            reg_occ_end=0.001,
            reg_occ_decay_steps=1250,
            reg_occ_start_step=0,
            reg_occ_end_step=500,

            reg_light_begin=0.001,
            reg_light_end=0.005,
            reg_light_decay_steps=1250,
            reg_light_start_step=0,
            reg_light_end_step=5000,
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
    load: Path = Path('outputs/d_diffdr_horse/test.2.1/task.py')
    step: Optional[int] = None

    def render_and_save_video(self, model: D_Texture, trainer: D_TextureTrainer, experiment: DS_Experiment, loader, name: str, fix_camera: bool = False, fix_camera_order: int = 0):
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
            model: D_Texture = task.model
            dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
            trainer: D_TextureTrainer = task.trainer
            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)
        model.eval()
        test_vis_size = dataset.get_size(split='test_vis')
        loader_orbit = dataset.get_test_vis_iter(batch_size=test_vis_size, shuffle=False, infinite=False)
        loader_fixed = dataset.get_test_vis_iter(batch_size=test_vis_size, shuffle=False, infinite=True)

        self.render_and_save_video(model, trainer, experiment, loader_orbit, name='orbit_camera_fps48') # 渲染轨道视角
        self.render_and_save_video(model, trainer, experiment, loader_fixed, name='fix_camera_order0_fps48', fix_camera=True, fix_camera_order=0) # 渲染固定视角，固定为第49个相机
        self.render_and_save_video(model, trainer, experiment, loader_fixed, name='fix_camera_order1_fps48', fix_camera=True, fix_camera_order=49) # 渲染固定视角，固定为第49个相机
        self.render_and_save_video(model, trainer, experiment, loader_fixed, name='fix_camera_order2_fps48', fix_camera=True, fix_camera_order=99) # 渲染固定视角，固定为第99个相机
        self.render_and_save_video(model, trainer, experiment, loader_fixed, name='fix_camera_order3_fps48', fix_camera=True, fix_camera_order=149) # 渲染固定视角，固定为第149个相机
        experiment.parse_log_auto(experiment.log_path) # 分析日志



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
            model: D_Texture = task.model
            dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
            trainer: D_TextureTrainer = task.trainer
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
        cat = cat_task,
        deer = deer_task,
        girlwalk = girlwalk_task,
        lego = lego_task,
        spidermanwalk = spidermanwalk_task,
        toy = toy_task,

        # eval task
        rendermodel = RenderAfterTrain(cuda=0), 
        evalsceneflow = EvalSceneFlow(cuda=0),
    ).run()
