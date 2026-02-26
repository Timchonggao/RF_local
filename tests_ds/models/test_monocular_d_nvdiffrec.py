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
from rfstudio_ds.data import SyntheticDynamicMonocularBlenderRGBADataset # rgb image dataset, random monocular view
from rfstudio_ds.model import D_NVDiffRec # rgb image loss optimization model
from rfstudio_ds.trainer import D_NVDiffRecTrainer # rgb image loss optimization trainer, dynamic nvdiffrec
from rfstudio_ds.graphics.spatial_sampler import NearestGridSampler
from rfstudio_ds.visualization import Visualizer
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh



beagle_task = DS_TrainTask(
    dataset=SyntheticDynamicMonocularBlenderRGBADataset(
        path=Path("data/dg-mesh/beagle/"),
    ),
    model=D_NVDiffRec(
        geometry='DD_isocubes',
        geometry_resolution=128,
        geometry_scale=1,
        background_color='white',
        texture_able=True,
    ),
    experiment=DS_Experiment(name='d_nvdiffrec_blender_sv/beagle', timestamp='test'),
    trainer=D_NVDiffRecTrainer(
        num_steps=5000,
        batch_size=4,
        num_steps_per_val=25,
        num_steps_per_orbit_vis=500,
        # num_steps_per_analyze_cube_curve=500,
        num_steps_per_save=1000,
        mixed_precision=False,
        full_test_after_train=True,
        full_fix_vis_after_train=True,
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)

bird_task = DS_TrainTask(
    dataset=SyntheticDynamicMonocularBlenderRGBADataset(
        path=Path("data/dg-mesh/bird/"),
    ),
    model=D_NVDiffRec(
        geometry='DD_isocubes',
        geometry_resolution=128,
        geometry_scale=1.5,
        background_color='white',
        texture_able=True,
    ),
    experiment=DS_Experiment(name='d_nvdiffrec_blender_sv/bird', timestamp='test'),
    trainer=D_NVDiffRecTrainer(
        num_steps=5000,
        batch_size=4,
        num_steps_per_val=25,
        num_steps_per_orbit_vis=500,
        # num_steps_per_analyze_cube_curve=500,
        num_steps_per_save=1000,
        mixed_precision=False,
        full_test_after_train=True,
        full_fix_vis_after_train=True,
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)

duck_task = DS_TrainTask(
    dataset=SyntheticDynamicMonocularBlenderRGBADataset(
        path=Path("data/dg-mesh/duck/"),
    ),
    model=D_NVDiffRec(
        geometry='DD_isocubes',
        geometry_resolution=128,
        geometry_scale=1,
        background_color='white',
        texture_able=True,
    ),
    experiment=DS_Experiment(name='d_nvdiffrec_blender_sv/duck', timestamp='test'),
    trainer=D_NVDiffRecTrainer(
        num_steps=5000,
        batch_size=4,
        num_steps_per_val=25,
        num_steps_per_orbit_vis=500,
        # num_steps_per_analyze_cube_curve=500,
        num_steps_per_save=1000,
        mixed_precision=False,
        full_test_after_train=True,
        full_fix_vis_after_train=True,
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)

girlwalk_task = DS_TrainTask(
    dataset=SyntheticDynamicMonocularBlenderRGBADataset(
        path=Path("data/dg-mesh/girlwalk/"),
    ),
    model=D_NVDiffRec(
        geometry='DD_isocubes',
        geometry_resolution=128,
        geometry_scale=1.5,
        background_color='white',
        texture_able=True,
    ),
    experiment=DS_Experiment(name='d_nvdiffrec_blender_sv/girlwalk', timestamp='test'),
    trainer=D_NVDiffRecTrainer(
        num_steps=5000,
        batch_size=4,
        num_steps_per_val=25,
        num_steps_per_orbit_vis=500,
        # num_steps_per_analyze_cube_curve=500,
        num_steps_per_save=1000,
        mixed_precision=False,
        full_test_after_train=True,
        full_fix_vis_after_train=True,
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)

horse_task = DS_TrainTask(
    dataset=SyntheticDynamicMonocularBlenderRGBADataset(
        path=Path("data/dg-mesh/horse/"),
    ),
    model=D_NVDiffRec(
        geometry='DD_isocubes',
        geometry_resolution=128,
        geometry_scale=1.8,
        background_color='white',
        texture_able=True,
    ),
    experiment=DS_Experiment(name='d_nvdiffrec_blender_sv/horse', timestamp='test'),
    trainer=D_NVDiffRecTrainer(
        num_steps=5000,
        batch_size=4,
        num_steps_per_val=25,
        num_steps_per_orbit_vis=500,
        # num_steps_per_analyze_cube_curve=500,
        num_steps_per_save=1000,
        mixed_precision=False,
        full_test_after_train=True,
        full_fix_vis_after_train=True,
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)

torus2sphere_task = DS_TrainTask(
    dataset=SyntheticDynamicMonocularBlenderRGBADataset(
        path=Path("data/dg-mesh/torus2sphere/"),
    ),
    model=D_NVDiffRec(
        geometry='DD_isocubes',
        geometry_resolution=128,
        geometry_scale=1,
        background_color='white',
        texture_able=True,
    ),
    experiment=DS_Experiment(name='d_nvdiffrec_blender_sv/torus2sphere', timestamp='test'),
    trainer=D_NVDiffRecTrainer(
        num_steps=5000,
        batch_size=4,
        num_steps_per_val=25,
        num_steps_per_orbit_vis=500,
        # num_steps_per_analyze_cube_curve=500,
        num_steps_per_save=1000,
        mixed_precision=False,
        full_test_after_train=True,
        full_fix_vis_after_train=True,
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

    def render_and_save_video(self, model: D_NVDiffRec, trainer: D_NVDiffRecTrainer, experiment: DS_Experiment, loader, name: str, fix_camera: bool = False, fix_camera_order: int = 0):
        inputs, gt_outputs, indices = next(loader)

        if fix_camera:
            inputs = inputs.reset_c2w_to_ref_camera_pose(ref_camera_pose=inputs.c2w[fix_camera_order])

        model.set_batch_gt_geometry([getattr(trainer, 'gt_geometry_fix_vis')[i] for i in indices.tolist()])

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

        experiment.dump_images2video('fix_vis', name=name, images=images, fps=48,duration=5)
        del inputs, gt_outputs, indices, color_outputs, depth_outputs, gt_depth_outputs, normal_outputs, gt_normal_outputs, reg_loss_dict, images
        torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            experiment: DS_Experiment = task.experiment
            model: D_NVDiffRec = task.model
            dataset: SyntheticDynamicMonocularBlenderRGBADataset = task.dataset
            trainer: D_NVDiffRecTrainer = task.trainer
            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)
        model.eval()
        fix_vis_size = dataset.get_size(split='fix_vis')
        loader_orbit = dataset.get_fix_vis_iter(batch_size=fix_vis_size, shuffle=False, infinite=False)
        loader_fixed = dataset.get_fix_vis_iter(batch_size=fix_vis_size, shuffle=False, infinite=True)

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
            model: D_NVDiffRec = task.model
            dataset: SyntheticDynamicMonocularBlenderRGBADataset = task.dataset
            trainer: D_NVDiffRecTrainer = task.trainer
            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)
        model.eval()
        fix_vis_size = dataset.get_size(split='fix_vis')
        loader = dataset.get_fix_vis_iter(batch_size=fix_vis_size, shuffle=False, infinite=False)
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



if __name__ == '__main__':
    TaskGroup(
        # train task
        beagle = beagle_task,
        bird = bird_task,
        duck = duck_task,
        girlwalk = girlwalk_task,
        horse = horse_task,
        torus2sphere = torus2sphere_task,

        # eval task
        rendermodel = RenderAfterTrain(cuda=0), 
        evalsceneflow = EvalSceneFlow(cuda=0),
    ).run()
