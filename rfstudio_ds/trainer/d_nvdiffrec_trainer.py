from __future__ import annotations

# import modules
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Literal, List, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor

# import rfstudio modules
from rfstudio.optim import Optimizer
from rfstudio.graphics import DepthImages, RGBAImages, PBRAImages, VectorImages
from rfstudio.graphics.shaders import DepthShader, NormalShader
from rfstudio.loss import MaskedPhotometricLoss, PSNRLoss, SSIML1Loss
from rfstudio.io import dump_float32_image

# import rfstudio_ds modules
from .base_trainer import DS_BaseTrainer
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh

from rfstudio_ds.model import D_NVDiffRec
from rfstudio_ds.data import SyntheticDynamicMonocularBlenderRGBADataset, SyntheticDynamicMultiViewBlenderRGBADataset
from rfstudio_ds.optim import DS_ModuleOptimizers

from rfstudio_ds.visualization._plot_cube_curve import plot_cube_curve, plot_curve


@dataclass
class RegularizationConfig:

    ssim_weight_begin: Optional[float] = 1. # ssim loss
    ssim_weight_end: Optional[float] = 1.
    ssim_weight_decay_steps: Optional[int] = 1250

    depth_weight_begin: Optional[float] = 100. # depth supervision loss
    depth_weight_end: Optional[float] = 1. 
    depth_weight_decay_steps: Optional[int] = 1250

    mask_weight_begin: Optional[float] = 10. # mask supervision loss
    mask_weight_end: Optional[float] = 1.
    mask_weight_decay_steps: Optional[int] = 1250

    normal_weight_begin: Optional[float] = 0. # normal supervision loss
    normal_weight_end: Optional[float] = 0.
    normal_weight_decay_steps: Optional[int] = 0

    """正则化参数配置"""
    curve_coeff_tv_weight_begin: Optional[float] = 0.1 # cruve coefficient tv smooth loss
    curve_coeff_tv_weight_end: Optional[float] = 10
    curve_coeff_tv_decay_steps: Optional[int] = 1000
    curve_coeff_tv_start_step: Optional[int] = 0 

    time_tv_weight_begin: Optional[float] = 0.1 # cruve coefficient tv smooth loss
    time_tv_weight_end: Optional[float] = 0.01
    time_tv_decay_steps: Optional[int] = 1000
    time_tv_start_step: Optional[int] = 0

    sdf_entropy_weight_begin: Optional[float] = 0.2 # sdf entropy loss
    sdf_entropy_weight_end: Optional[float] = 0.01 
    sdf_entropy_decay_steps: Optional[int] = 1250
    sdf_entropy_start_step: Optional[int] = 0

    sdf_eikonal_weight_begin: Optional[float] = 0.01 # sdf eikonal loss
    sdf_eikonal_weight_end: Optional[float] = 0.001
    sdf_eikonal_decay_steps: Optional[int] = 1250
    sdf_eikonal_start_step: Optional[int] = 0

    scene_flow_smooth_weight_begin: Optional[float] = 0.0001 # scene flow smooth loss
    scene_flow_smooth_weight_end: Optional[float] = 0.001
    scene_flow_smooth_decay_steps: Optional[int] = 1500
    scene_flow_smooth_start_step: Optional[int] = -1

    reg_spatial_hashgrid_begin: float = 0.0
    reg_spatial_hashgrid_end: float = 0.0
    reg_spatial_hashgrid_decay_steps: Optional[int] = 1250
    reg_spatial_hashgrid_start_step: Optional[int] = -1

    reg_temporal_hashgrid_begin: float = 0.01
    reg_temporal_hashgrid_end: float = 0.5
    reg_temporal_hashgrid_decay_steps: Optional[int] = 1250
    reg_temporal_hashgrid_start_step: Optional[int] = 0

    reg_kd_enc_begin: float = 0.01
    reg_kd_enc_end: float = 0.2
    reg_kd_enc_decay_steps: Optional[int] = 1250
    reg_kd_enc_start_step: Optional[int] = -1


@dataclass
class D_NVDiffRecTrainer(DS_BaseTrainer):

    static_sdf_params_learning_rate: float = 0.01
    poly_coeffs_learning_rate: float = 0.01
    fourier_low_learning_rate: float = 0.005
    fourier_mid_learning_rate: float = 0.001
    fourier_high_learning_rate: float = 0.0005
    optimizer_epsilon: float = 1e-8
    geometry_decay: Optional[int] = 800

    appearance_lr: float = 0.01
    appearance_epsilon: float = 1e-8
    appearance_decay: Optional[int] = 800
    
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)

    def setup(
        self,
        model: D_NVDiffRec,
        dataset: Union[SyntheticDynamicMonocularBlenderRGBADataset, SyntheticDynamicMultiViewBlenderRGBADataset],
    ) -> DS_ModuleOptimizers:
        assert isinstance(dataset, (
            SyntheticDynamicMonocularBlenderRGBADataset, 
            SyntheticDynamicMultiViewBlenderRGBADataset
        ))
        self.gt_geometry_train = dataset.get_meshes(split='train')
        self.gt_geometry_val = dataset.get_meshes(split='val')
        self.gt_geometry_test = dataset.get_meshes(split='test')
        self.gt_geometry_orbit_vis = dataset.get_meshes(split='orbit_vis')
        self.gt_geometry_fix_vis = dataset.get_meshes(split='fix_vis')
        
        optim_dict = {}
        self._real_learning_rates = {}
        if model.geometry != 'gt':
            optim_dict['static_sdf_params'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='static_sdf_params'),
                lr=self.static_sdf_params_learning_rate,
                lr_decay=self.geometry_decay,
                max_norm=None,
                eps=self.optimizer_epsilon      
            )

            optim_dict['poly_coeffs'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='dddm_poly_coeffs'),
                lr=self.poly_coeffs_learning_rate,
                lr_decay=int(self.geometry_decay * 0.5),
                max_norm=None,
                eps=self.optimizer_epsilon      
            )

            optim_dict['fourier_low_coeffs'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='dddm_fourier_low_coeffs'), 
                lr=self.fourier_low_learning_rate,
                lr_decay=int(self.geometry_decay),
                max_norm=0.005,
                eps=self.optimizer_epsilon      
            )

            optim_dict['fourier_mid_coeffs'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='dddm_fourier_mid_coeffs'), 
                lr=self.fourier_mid_learning_rate,
                lr_decay=int(self.geometry_decay * 1.5),
                max_norm=0.005,
                eps=self.optimizer_epsilon      
            )

            optim_dict['fourier_high_coeffs'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='dddm_fourier_high_coeffs'), 
                lr=self.fourier_high_learning_rate,
                lr_decay=int(self.geometry_decay * 2.0),
                max_norm=0.005,
                eps=self.optimizer_epsilon      
            )

            # 记录实际学习率
            self._real_learning_rates = {
                'static_sdf_params': self.static_sdf_params_learning_rate,
                'poly_coeffs': self.poly_coeffs_learning_rate,
                'fourier_low_coeffs': self.fourier_low_learning_rate,
                'fourier_mid_coeffs': self.fourier_mid_learning_rate,
                'fourier_high_coeffs': self.fourier_high_learning_rate
            }

        if model.texture_able:
            optim_dict['dynamic_appearance'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.dynamic_texture,
                lr=self.appearance_lr,
                lr_decay=int(self.appearance_decay),
                max_norm=None,
                eps=self.appearance_epsilon,
            )

            self._real_learning_rates['dynamic_appearance'] = self.appearance_lr

        assert optim_dict, "Optimizer dictionary cannot be empty"
        return DS_ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
            start_steps=None,
        )

    def step(
        self,
        model: 'D_NVDiffRec',
        inputs: 'DS_Cameras',
        gt_outputs: Union['RGBAImages', None],
        *,
        indices: Optional[Tensor] = None,
        mode: Literal['train', 'val', 'test', 'orbit_vis', 'fix_vis', 'analyse_curve'] = 'train',
        vis_downsample_factor: Optional[int] = None,
        visual: bool = False,
        analyse_curve_save_path: Optional[str] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[List[Tensor]]]:
        if mode == 'analyse_curve':
            if model.geometry != 'gt':
                return self._run_analyse_curve(model, inputs, analyse_curve_save_path)
            else:
                return torch.tensor(0.0), {}, None
        
        if vis_downsample_factor is not None:
            indices = indices[::vis_downsample_factor]
            inputs = inputs[::vis_downsample_factor]
            if gt_outputs is not None:
                gt_outputs = gt_outputs[::vis_downsample_factor]
        model.set_batch_gt_geometry([getattr(self, f'gt_geometry_{mode}')[i] for i in indices.tolist()])
        (
            color_outputs,
            depth_outputs, gt_depth_outputs,
            normal_outputs, gt_normal_outputs,
            reg_loss_dict
        ) = model.render_report(
            camera_inputs=inputs, 
            frame_batch=self.frame_batch_size if mode == 'train' else None, # only train use multi-view data
            camera_batch=self.camera_batch_size if mode == 'train' else None, # only train use multi-view data
            trainer_mode=mode
        )
        
        if mode in ['train', 'val', 'test']:
            def safe_detach(x):
                return x.detach() if isinstance(x, Tensor) and x.requires_grad else x

            total_loss = torch.tensor(0.0, device=inputs.device)
            metrics = {}

            reg_loss = sum(reg_loss_dict.values())
            total_loss += reg_loss
            metrics.update({
                'full-reg-loss': safe_detach(reg_loss)
            })

            if model.geometry != 'gt':
                depth_loss, mask_loss = self._compute_losses(depth_outputs, gt_depth_outputs)
                total_loss += (depth_loss + mask_loss)
                metrics.update({
                    'real-depth-loss': depth_loss.detach() / self.depth_weight,
                    'real-mask-loss': mask_loss.detach() / self.mask_weight,

                    'real_reg_sdf_entropy_loss': (safe_detach(reg_loss_dict['sdf_entropy_loss']) / model.reg_sdf_entropy_weight) if model.reg_sdf_entropy_weight > 0 else 0.0,
                    'real_reg_sdf_eikonal_loss': (safe_detach(reg_loss_dict['sdf_eikonal_loss']) / model.reg_sdf_eikonal_weight) if model.reg_sdf_eikonal_weight > 0 else 0.0,
                    'real_reg_scene_flow_smoothness_loss': (safe_detach(reg_loss_dict['scene_flow_smoothness_loss']) / model.reg_scene_flow_smoothness_weight) if model.reg_scene_flow_smoothness_weight > 0 else 0.0,
                    'real_reg_curve_coeff_tv_loss': (safe_detach(reg_loss_dict['coeff_tv_loss']) / model.reg_coeff_tv_weight) if model.reg_coeff_tv_weight > 0 else 0.0,
                    'reg_time_tv_loss': (safe_detach(reg_loss_dict['time_tv_loss']) / model.reg_time_tv_weight) if model.reg_time_tv_weight > 0 else 0.0,
                })
            if model.texture_able:
                bg_color = model.get_background_color().to(inputs.device)
                pred_rgb = color_outputs.blend(bg_color)
                gt_rgb = gt_outputs.blend(bg_color)

                ssim_loss = SSIML1Loss()(gt_outputs=gt_rgb, outputs=pred_rgb)
                psnr_loss = PSNRLoss()(gt_rgb.detach().clamp(0, 1), pred_rgb.detach().clamp(0, 1))
                total_loss += ssim_loss * self.ssim_weight

                metrics.update({
                    'ssim-l1': ssim_loss.detach(),
                    'psnr': psnr_loss.detach(),

                    'real_reg_spatial_hashgrid_loss': (safe_detach(reg_loss_dict['reg_spatial_hashgrid']) / model.reg_spatial_hashgrid_weight) if model.reg_spatial_hashgrid_weight > 0 else 0.0,
                    'real_reg_temporal_hashgrid_loss': (safe_detach(reg_loss_dict['reg_temporal_hashgrid']) / model.reg_temporal_hashgrid_weight) if model.reg_temporal_hashgrid_weight > 0 else 0.0,
                    'real_reg_kd_enc_loss': (safe_detach(reg_loss_dict['reg_kd_enc']) / model.reg_kd_enc_weight) if model.reg_kd_enc_weight > 0 else 0.0,
                })

            metrics['total-loss'] = total_loss.detach()
            
            images = []
            if visual:
                with torch.no_grad():
                    for i in range(len(inputs)):
                        col0 = None # pred color vs gt color
                        col1 = None # pred normal vs gt normal
                        col2 = None # pred depth vs gt depth
                        image = None
                        if model.texture_able:
                            col0 = torch.cat((
                                gt_rgb[i].item(),
                                pred_rgb[i].detach().item(),
                            ), dim=0).clamp(0, 1)
                            image = col0
                        if model.geometry != 'gt':
                            col1 = torch.cat((
                                gt_normal_outputs[i].visualize((1, 1, 1)).item(),
                                normal_outputs[i].visualize((1, 1, 1)).item(),
                            ), dim=0).clamp(0, 1)
                            col2 = torch.cat((
                                gt_depth_outputs[i].visualize().item(),
                                depth_outputs[i].visualize().item(),
                            ), dim=0).clamp(0, 1)
                            if image is not None:
                                image = torch.cat((image, col1, col2), dim=1)
                            else:
                                image = torch.cat((col1, col2), dim=1)
                        images.append(image)
            
            return total_loss, metrics, images
        elif mode in ['orbit_vis', 'fix_vis']:
            images = []
            if model.texture_able:
                bg_color = model.get_background_color().to(inputs.device)
                pred_rgb = color_outputs.blend(bg_color)
                gt_rgb = gt_outputs.blend(bg_color) if gt_outputs is not None else pred_rgb
            with torch.no_grad():
                for i in range(len(inputs)):
                    col0 = None
                    col1 = None
                    col2 = None
                    image = None

                    if model.texture_able:
                        col0 = torch.cat((
                            gt_rgb[i].item(),
                            pred_rgb[i].detach().item(),
                        ), dim=0).clamp(0, 1)
                        image = col0
                    if model.geometry != 'gt':
                        col1 = torch.cat((
                            gt_normal_outputs[i].visualize((1, 1, 1)).item(),
                            normal_outputs[i].visualize((1, 1, 1)).item(),
                        ), dim=0).clamp(0, 1)
                        col2 = torch.cat((
                            gt_depth_outputs[i].visualize().item(),
                            depth_outputs[i].visualize().item(),
                        ), dim=0).clamp(0, 1)
                        if image is not None:
                            image = torch.cat((image, col1, col2), dim=1)
                        else:
                            image = torch.cat((col1, col2), dim=1)
                    images.append(image)
            return 0.0, {}, images

    def _compute_losses(
        self,
        depths: DepthImages,
        gt_depths: DepthImages,
        normas: Optional[VectorImages] = None,
        gt_normas: Optional[VectorImages] = None,
    ) -> Tuple[Tensor, Tensor]:
        
        def compute_depth_loss(depth: Tensor, gt_depth: Tensor) -> Tensor:
            """计算深度损失"""
            pred_depth_masked = depth[..., :1] * gt_depth[..., 1:]
            gt_depth_masked = gt_depth[..., :1] * gt_depth[..., 1:]
            return torch.nn.functional.l1_loss(pred_depth_masked, gt_depth_masked)
    
        def compute_mask_loss(depth: Tensor, gt_depth: Tensor) -> Tensor:
            """计算掩码损失"""
            return torch.nn.functional.l1_loss(gt_depth[..., 1:], depth[..., 1:])

        def compute_normal_loss(normal: Tensor, gt_normal: Tensor) -> Tensor:
            """计算法线损失（基于余弦相似度）"""
            cos_sim = (normal[..., :3] * gt_normal[..., :3]).sum(-1, keepdim=True) * gt_normal[..., 3:]
            return (1 - cos_sim).square().mean()
    
        depth_loss = 0.0
        mask_loss = 0.0
        for depth, gt_depth in zip(depths, gt_depths):
            # 深度损失
            depth_loss += compute_depth_loss(depth, gt_depth)

            # 掩码损失
            mask_loss += compute_mask_loss(depth, gt_depth)

        return (self.depth_weight * depth_loss) / len(depths), (self.mask_weight * mask_loss) / len(depths)

    @torch.no_grad()
    def _run_analyse_curve(self, model: D_NVDiffRec, inputs: DS_Cameras, save_path: str) -> Tuple[Tensor, Dict[str, Tensor], Optional[List[Tensor]]]:
        
        def split_cube_info(cubes_info):
            return [
                {
                    key: value[cube_idx].detach().cpu().item() if value[cube_idx].ndim == 0 else value[cube_idx].detach().cpu()
                    for key, value in cubes_info.items()
                }
                for cube_idx in range(cubes_info["cube_positions"].shape[0])
            ]
        
        def get_fixed_probe_positions(device):
            return torch.tensor([
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
            ], device=device)

        positions = get_fixed_probe_positions(model.device)
        geometric_repr = model.geometric_repr.replace(
            static_sdf_values=model.static_sdf_params,
            sdf_curve_poly_coefficient=model.sdf_curve_poly_coefficient,
            sdf_curve_low_freq_fourier_coefficient=model.sdf_curve_low_freq_fourier_coefficient,
            sdf_curve_mid_freq_fourier_coefficient=model.sdf_curve_mid_freq_fourier_coefficient,
            sdf_curve_high_freq_fourier_coefficient=model.sdf_curve_high_freq_fourier_coefficient,
        )
        cubes_info = geometric_repr.get_cube_curve_info(positions=positions)

        times = inputs.times
        pred_data, _ = geometric_repr.query_sdf_at_times(t=times)

        for idx, info in enumerate(split_cube_info(cubes_info)):
            sdf_values = pred_data[:, info["flatten_indices"]] if pred_data is not None else None
            plot_cube_curve(
                times=times.cpu(),
                info_dict=info,
                pred_data=sdf_values.cpu(),
                gt_data=None,
                save_path=f'{save_path}/cube_{idx}_curve.png'
            )
        return torch.tensor(0.0), {}, None

    def before_update(self, model: D_NVDiffRec, optimizers: DS_ModuleOptimizers, *, curr_step: int) -> None:
       
        def linear_decay(begin, end, step, decay_steps):
            if decay_steps > 0:
                progress = min(1.0, step / decay_steps)
                return begin - (begin - end) * progress
            return begin

        def set_reg_weight(enable_flag_name: str, weight_attr: str, begin: float, end: float, start_step: int, decay_steps: int):
            enabled = curr_step > start_step and start_step >= 0
            setattr(model, enable_flag_name, enabled)
            if enabled:
                weight = linear_decay(begin, end, curr_step - start_step, decay_steps)
            else:
                weight = 0.0
            setattr(model, weight_attr, weight)

        reg = self.regularization

        # 主要损失函数权重的线性衰减
        self.ssim_weight = linear_decay(reg.ssim_weight_begin, reg.ssim_weight_end, curr_step, reg.ssim_weight_decay_steps)
        self.depth_weight = linear_decay(reg.depth_weight_begin, reg.depth_weight_end, curr_step, reg.depth_weight_decay_steps)
        self.mask_weight = linear_decay(reg.mask_weight_begin, reg.mask_weight_end, curr_step, reg.mask_weight_decay_steps)
        self.normal_weight = linear_decay(reg.normal_weight_begin, reg.normal_weight_end, curr_step, reg.normal_weight_decay_steps)
        
        # 正则化项权重的线性衰减
        set_reg_weight(
            "reg_coeff_tv_able", "reg_coeff_tv_weight",
            reg.curve_coeff_tv_weight_begin, reg.curve_coeff_tv_weight_end,
            reg.curve_coeff_tv_start_step, reg.curve_coeff_tv_decay_steps
        )

        set_reg_weight(
            "reg_time_tv_able", "reg_time_tv_weight",
            reg.time_tv_weight_begin, reg.time_tv_weight_end,
            reg.time_tv_start_step, reg.time_tv_decay_steps
        )

        set_reg_weight(
            "reg_sdf_entropy_able", "reg_sdf_entropy_weight",
            reg.sdf_entropy_weight_begin, reg.sdf_entropy_weight_end,
            reg.sdf_entropy_start_step, reg.sdf_entropy_decay_steps
        )

        set_reg_weight(
            "reg_sdf_eikonal_able", "reg_sdf_eikonal_weight",
            reg.sdf_eikonal_weight_begin, reg.sdf_eikonal_weight_end,
            reg.sdf_eikonal_start_step, reg.sdf_eikonal_decay_steps
        )

        set_reg_weight(
            "reg_scene_flow_smoothness_able", "reg_scene_flow_smoothness_weight",
            reg.scene_flow_smooth_weight_begin, reg.scene_flow_smooth_weight_end,
            reg.scene_flow_smooth_start_step, reg.scene_flow_smooth_decay_steps
        )

        set_reg_weight(
            "reg_spatial_hashgrid_able", "reg_spatial_hashgrid_weight",
            reg.reg_spatial_hashgrid_begin, reg.reg_spatial_hashgrid_end,
            reg.reg_spatial_hashgrid_start_step, reg.reg_spatial_hashgrid_decay_steps
        )
        
        set_reg_weight(
            "reg_temporal_hashgrid_able", "reg_temporal_hashgrid_weight",
            reg.reg_temporal_hashgrid_begin, reg.reg_temporal_hashgrid_end,
            reg.reg_temporal_hashgrid_start_step, reg.reg_temporal_hashgrid_decay_steps
        )

        set_reg_weight(
            "reg_kd_enc_able", "reg_kd_enc_weight",
            reg.reg_kd_enc_begin, reg.reg_kd_enc_end,
            reg.reg_kd_enc_start_step, reg.reg_kd_enc_decay_steps
        )

    def after_update(self, model: D_NVDiffRec, optimizers: DS_ModuleOptimizers, *, curr_step: int) -> None:
        if model.texture_able:
            # self.real_appearance_lr = optimizers.optimizers['dynamic_appearance'].param_groups[0]['lr']
            self._real_learning_rates.update({
                'dynamic_appearance': optimizers.optimizers['dynamic_appearance'].param_groups[0]['lr'],
            })
        if model.geometry != 'gt':
            self._real_learning_rates.update({
                'static_sdf_params': optimizers.optimizers['static_sdf_params'].param_groups[0]['lr'],
                'poly_coeffs': optimizers.optimizers['poly_coeffs'].param_groups[0]['lr'],
                'fourier_low': optimizers.optimizers['fourier_low_coeffs'].param_groups[0]['lr'],
                'fourier_mid': optimizers.optimizers['fourier_mid_coeffs'].param_groups[0]['lr'],
                'fourier_high': optimizers.optimizers['fourier_high_coeffs'].param_groups[0]['lr']
            })

