from __future__ import annotations

# import modules
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Literal, List, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from jaxtyping import Float32

import torch
from torch import Tensor

# import rfstudio modules
from rfstudio.optim import Optimizer
from rfstudio.graphics import DepthImages, RGBAImages, PBRAImages, VectorImages
from rfstudio.graphics.shaders import DepthShader, NormalShader
from rfstudio.loss import MaskedPhotometricLoss, PSNRLoss, SSIML1Loss, ImageL2Loss
from rfstudio.io import dump_float32_image
from rfstudio.utils.lazy_module import dr
import torch.nn.functional as F

# import rfstudio_ds modules
from .base_trainer import DS_BaseTrainer
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh

from rfstudio_ds.model import D_Joint_S2
from rfstudio_ds.data import SyntheticDynamicMonocularBlenderRGBADataset, SyntheticDynamicMultiViewBlenderRGBADataset, RealDynamicMultiviewObjectRGBADataset, CMUPanonicRGBADataset, SyntheticTemporalDynamicMultiviewBlenderRGBADataset
from rfstudio_ds.optim import DS_ModuleOptimizers

from rfstudio_ds.visualization._plot_cube_curve import plot_cube_curve, plot_curve

@torch.no_grad()
def vis_cubemap(cubemap: Float32[Tensor, "6 R R 3"], *, width: int, height: int) -> Float32[Tensor, "H W 3"]:
    gy, gx = torch.meshgrid(
        torch.linspace(0.0 + 1.0 / height, 1.0 - 1.0 / height, height, device=cubemap.device) * torch.pi,
        torch.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width, device=cubemap.device) * torch.pi,
        indexing='ij',
    )
    sin_theta = gy.sin()
    reflvec = torch.stack((sin_theta * gx.sin(), gy.cos(), -sin_theta * gx.cos()), dim=-1)
    return dr.texture(cubemap[None], reflvec[None].contiguous(), filter_mode='linear', boundary_mode='cube')[0]


@dataclass
class RegularizationConfig:

    """监督信号权重配置"""
    ssim_weight_begin: Optional[float] = 1. # ssim loss
    ssim_weight_end: Optional[float] = 1.
    ssim_weight_decay_steps: Optional[int] = 1250
    ssim_weight_start_step: Optional[int] = 0

    mask_weight_begin: Optional[float] = 10. # mask supervision loss
    mask_weight_end: Optional[float] = 10.
    mask_weight_decay_steps: Optional[int] = 1250   
    mask_weight_start_step: Optional[int] = 0

    """正则化参数配置"""
    # appearence regularization
    reg_appearance_temporal_hashgrid_begin: float = 0.1
    reg_appearance_temporal_hashgrid_end: float = 0.01
    reg_appearance_temporal_hashgrid_decay_steps: Optional[int] = 1250
    reg_appearance_temporal_hashgrid_start_step: Optional[int] = 0
    reg_appearance_temporal_hashgrid_end_step: Optional[int] = -1

    reg_occ_begin: float = 0.1
    reg_occ_end: float = 0.01
    reg_occ_decay_steps: Optional[int] = 1250
    reg_occ_start_step: Optional[int] = 0
    reg_occ_end_step: Optional[int] = -1

    reg_light_begin: float = 0.1
    reg_light_end: float = 0.01
    reg_light_decay_steps: Optional[int] = 1250
    reg_light_start_step: Optional[int] = -1
    reg_light_end_step: Optional[int] = 0
    reset_light_step: Optional[int] = -1

    # geometry regularization
    reg_geometry_residual_temporal_hashgrid_begin: float = 0.1
    reg_geometry_residual_temporal_hashgrid_end: float = 0.01
    reg_geometry_residual_temporal_hashgrid_decay_steps: Optional[int] = 1250
    reg_geometry_residual_temporal_hashgrid_start_step: Optional[int] = 0
    reg_geometry_residual_temporal_hashgrid_end_step: Optional[int] = -1
    
    geometry_residual_weight_begin: Optional[float] = 0.1 # geometry residual loss
    geometry_residual_weight_end: Optional[float] = 1
    geometry_residual_decay_steps: Optional[int] = 200
    geometry_residual_start_step: Optional[int] = 0
    geometry_residual_end_step: Optional[int] = -1
    
    sdf_entropy_weight_begin: Optional[float] = 0.2 # sdf entropy loss
    sdf_entropy_weight_end: Optional[float] = 0.01 # todo 由于是很多个mesh 的entropy ，感觉要把这个参数调大一些
    sdf_entropy_decay_steps: Optional[int] = 1250
    sdf_entropy_start_step: Optional[int] = 0
    sdf_entropy_end_step: Optional[int] = -1

    time_tv_weight_begin: Optional[float] = 0.1 # curve derivative smooth loss
    time_tv_weight_end: Optional[float] = 0.01
    time_tv_decay_steps: Optional[int] = 1000
    time_tv_start_step: Optional[int] = -1
    time_tv_end_step: Optional[int] = -1
    
    sdf_eikonal_weight_begin: Optional[float] = 0.01 # sdf eikonal loss
    sdf_eikonal_weight_end: Optional[float] = 0.001
    sdf_eikonal_decay_steps: Optional[int] = 1250
    sdf_eikonal_start_step: Optional[int] = -1
    sdf_eikonal_end_step: Optional[int] = -1

    curve_coeff_tv_weight_begin: Optional[float] = 0.1 # curve coefficient tv smooth loss
    curve_coeff_tv_weight_end: Optional[float] = 10
    curve_coeff_tv_decay_steps: Optional[int] = 1000
    curve_coeff_tv_start_step: Optional[int] = 0
    curve_coeff_tv_end_step: Optional[int] = -1


@dataclass
class D_Joint_S2Trainer(DS_BaseTrainer):

    geometry_residual_lr: float = 0.003 # 用于flexicube 的 sdf, deform 和 weight residual 的学习率
    geometry_residual_lr_decay: Optional[int] = None

    static_sdf_params_learning_rate: float = 0.01
    static_sdf_params_decay: Optional[int] = None
    static_sdf_params_max_norm: Optional[float] = None

    poly_coeffs_learning_rate: float = 0.01
    poly_coeffs_decay: Optional[int] = None
    poly_coeffs_max_norm: Optional[float] = None

    fourier_low_learning_rate: float = 0.005
    fourier_low_decay: Optional[int] = None
    fourier_low_max_norm: Optional[float] = None

    fourier_mid_learning_rate: float = 0.001
    fourier_mid_decay: Optional[int] = None
    fourier_mid_max_norm: Optional[float] = None

    fourier_high_learning_rate: float = 0.0005
    fourier_high_decay: Optional[int] = None
    fourier_high_max_norm: Optional[float] = None

    appearance_learning_rate: float = 0.01
    appearance_decay: Optional[int] = None

    light_learning_rate: float = 0.01 # 用于light的参数
    light_learning_rate_decay: Optional[int] = None
    light_grad_amp: Optional[float] = 64.0
    
    optimizer_epsilon: float = 1e-8
    learning_rate_decay_steps: Optional[int] = 800

    use_multi_model_stage: bool = True
    model_start_steps: Optional[List[int]] = field(default_factory=lambda: [0,0,0,0,0]) # static, poly, fourier_low, fourier_mid, fourier_high
    
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)

    def setup(
        self,
        model: D_Joint_S2,
        dataset: Union[SyntheticDynamicMonocularBlenderRGBADataset, SyntheticDynamicMultiViewBlenderRGBADataset, RealDynamicMultiviewObjectRGBADataset, CMUPanonicRGBADataset, SyntheticTemporalDynamicMultiviewBlenderRGBADataset],
    ) -> DS_ModuleOptimizers:
        assert isinstance(dataset, (
            SyntheticDynamicMonocularBlenderRGBADataset, 
            SyntheticDynamicMultiViewBlenderRGBADataset,
            RealDynamicMultiviewObjectRGBADataset,
            CMUPanonicRGBADataset,
            SyntheticTemporalDynamicMultiviewBlenderRGBADataset
        ))
        self.gt_geometry_train = dataset.get_meshes(split='train')
        self.gt_geometry_val = dataset.get_meshes(split='val')
        self.gt_geometry_test = dataset.get_meshes(split='test')
        self.gt_geometry_orbit_vis = dataset.get_meshes(split='orbit_vis')
        self.gt_geometry_fix_vis = dataset.get_meshes(split='fix_vis')
        
        optim_dict = {}
        self._real_learning_rates = {}
        
        # geometry optimizers
        optim_dict['geometry_sdf_residual'] = Optimizer(
            category=torch.optim.Adam,
            modules=model.geometry_sdf_residual,
            lr=self.geometry_residual_lr,
            lr_decay=self.learning_rate_decay_steps if self.geometry_residual_lr_decay is None else self.geometry_residual_lr_decay,
            max_norm=None,
            eps=self.optimizer_epsilon      
        )  
        optim_dict['geometry_deform_residual'] = Optimizer(
            category=torch.optim.Adam,
            modules=model.geometry_deform_residual,
            lr=self.geometry_residual_lr,
            lr_decay=self.learning_rate_decay_steps if self.geometry_residual_lr_decay is None else self.geometry_residual_lr_decay,
            max_norm=None,
            eps=self.optimizer_epsilon      
        )  
        optim_dict['geometry_weight_residual'] = Optimizer(
            category=torch.optim.Adam,
            modules=model.geometry_weight_residual,
            lr=self.geometry_residual_lr,
            lr_decay=self.learning_rate_decay_steps if self.geometry_residual_lr_decay is None else self.geometry_residual_lr_decay,
            max_norm=None,
            eps=self.optimizer_epsilon      
        ) 

        optim_dict['static_sdf_params'] = Optimizer(
            category=torch.optim.Adam,
            modules=model.as_module(field_name='static_sdf_params'),
            lr=self.static_sdf_params_learning_rate,
            lr_decay=self.learning_rate_decay_steps if self.static_sdf_params_decay is None else self.static_sdf_params_decay,
            max_norm=self.static_sdf_params_max_norm,
            eps=self.optimizer_epsilon      
        )

        optim_dict['poly_coeffs'] = Optimizer(
            category=torch.optim.Adam,
            modules=model.as_module(field_name='dddm_poly_coeffs'),
            lr=self.poly_coeffs_learning_rate,
            lr_decay=self.learning_rate_decay_steps if self.poly_coeffs_decay is None else self.poly_coeffs_decay,
            max_norm=self.poly_coeffs_max_norm,
            eps=self.optimizer_epsilon      
        )

        optim_dict['fourier_low_coeffs'] = Optimizer(
            category=torch.optim.Adam,
            modules=model.as_module(field_name='dddm_fourier_low_coeffs'), 
            lr=self.fourier_low_learning_rate,
            lr_decay=self.learning_rate_decay_steps if self.fourier_low_decay is None else self.fourier_low_decay,
            max_norm=self.fourier_low_max_norm,
            eps=self.optimizer_epsilon      
        )

        optim_dict['fourier_mid_coeffs'] = Optimizer(
            category=torch.optim.Adam,
            modules=model.as_module(field_name='dddm_fourier_mid_coeffs'), 
            lr=self.fourier_mid_learning_rate,
            lr_decay=self.learning_rate_decay_steps if self.fourier_mid_decay is None else self.fourier_mid_decay,
            max_norm=self.fourier_mid_max_norm,
            eps=self.optimizer_epsilon      
        )

        optim_dict['fourier_high_coeffs'] = Optimizer(
            category=torch.optim.Adam,
            modules=model.as_module(field_name='dddm_fourier_high_coeffs'), 
            lr=self.fourier_high_learning_rate,
            lr_decay=self.learning_rate_decay_steps if self.fourier_high_decay is None else self.fourier_high_decay,
            max_norm=self.fourier_high_max_norm,
            eps=self.optimizer_epsilon      
        )

        # 记录实际学习率
        self._real_learning_rates = {
            'geometry_sdf_residual': self.geometry_residual_lr,
            'geometry_deform_residual': self.geometry_residual_lr,
            'geometry_weight_residual': self.geometry_residual_lr,
            'static_sdf_params': self.static_sdf_params_learning_rate,
            'poly_coeffs': self.poly_coeffs_learning_rate,
            'fourier_low_coeffs': self.fourier_low_learning_rate,
            'fourier_mid_coeffs': self.fourier_mid_learning_rate,
            'fourier_high_coeffs': self.fourier_high_learning_rate,
        }
        # 记录模型优化的开始步数
        if self.use_multi_model_stage:
            model_optimize_start_steps = {
                'geometry_sdf_residual': 0,
                'geometry_deform_residual': 0,
                'geometry_weight_residual': 0,
                'static_sdf_params': self.model_start_steps[0],
                'poly_coeffs': self.model_start_steps[1],
                'fourier_low_coeffs': self.model_start_steps[2],
                'fourier_mid_coeffs': self.model_start_steps[3],
                'fourier_high_coeffs': self.model_start_steps[4],
            }
        else:
            model_optimize_start_steps = {
                'dynamic_appearance': 0,
                'static_sdf_params': 0,
                'poly_coeffs': 0,
                'fourier_low_coeffs': 0,
                'fourier_mid_coeffs': 0,
                'fourier_high_coeffs': 0,
            }

        
        # texture optimizers
        optim_dict['dynamic_appearance'] = Optimizer(
            category=torch.optim.Adam,
            modules=model.dynamic_texture,
            lr=self.appearance_learning_rate,
            lr_decay=self.learning_rate_decay_steps if self.appearance_decay is None else self.appearance_decay,
            max_norm=None,
            eps=self.optimizer_epsilon,
        )
        # 记录实际学习率
        self._real_learning_rates.update({
            'dynamic_appearance': self.appearance_learning_rate,
        })
        # 记录模型优化的开始步数
        model_optimize_start_steps.update({
            'dynamic_appearance': 0,
        })
        
        if model.shader_type == "split_sum_pbr":
            optim_dict['light'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='light'),
                lr=self.light_learning_rate,
                lr_decay=self.learning_rate_decay_steps if self.light_learning_rate_decay is None else self.light_learning_rate_decay,
                max_norm=None,
                eps=self.optimizer_epsilon      
            ) 

            if self.light_grad_amp is not None:
                model.cubemap.register_hook(lambda g: g * self.light_grad_amp)
            # 记录实际学习率
            self._real_learning_rates.update({
                'light': self.light_learning_rate,
            })
            # 记录模型优化的开始步数
            model_optimize_start_steps.update({
                'light': 0,
            })

        assert optim_dict, "Optimizer dictionary cannot be empty"
        return DS_ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
            start_steps=model_optimize_start_steps,
        )

    def step(
        self,
        model: 'D_Joint_S2',
        inputs: 'DS_Cameras',
        gt_outputs: Union['RGBAImages', None],
        *,
        indices: Optional[Tensor] = None,
        mode: Literal['train', 'val', 'test', 'orbit_vis', 'fix_vis', 'analyse_curve'] = 'train',
        val_pbr_attr: bool = False,
        vis_downsample_factor: Optional[int] = None,
        visual: bool = False,
        analyse_curve_save_path: Optional[str] = None,
        return_pred_mesh: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[List[Tensor]]]:
        
        assert indices is not None, "indices should not be None"

        if vis_downsample_factor is not None: # for eval vis, use downsample batch, to avoid oom
            indices = indices[::vis_downsample_factor]
            inputs = inputs[::vis_downsample_factor]
            if gt_outputs is not None:
                gt_outputs = gt_outputs[::vis_downsample_factor]

        if getattr(self, f'gt_geometry_{mode}') is None:
            model.set_batch_gt_geometry(None)
        else:
            model.set_batch_gt_geometry([getattr(self, f'gt_geometry_{mode}')[i] for i in indices.tolist()])
        
        outputs = model.render_report(
            camera_inputs=inputs, 
            trainer_mode=mode,
            return_pred_mesh=return_pred_mesh,
            return_pbr_attrs=val_pbr_attr,
        )
        
        if mode in ['train', 'val', 'test']:
            return self._get_results(model, outputs, gt_outputs, mode, visual, return_pred_mesh)
        elif mode in ['orbit_vis', 'fix_vis']:
            (
                pred_outputs,
                reg_loss_dict,
                pred_meshes,
                pred_depths, gt_depths,
                pred_normals, gt_normals,
                pbr_attrs
            ) = outputs
            bg_color = model.get_background_color().to(model.device)
            if isinstance(pred_outputs, PBRAImages):
                pred_rgbs = pred_outputs.rgb2srgb().blend(bg_color)
            else:
                pred_rgbs = pred_outputs.blend(bg_color)
            gt_rgbs = gt_outputs.blend(bg_color) if gt_outputs is not None else pred_rgbs
            
            images = self._generate_visualizations(pred_rgbs, gt_rgbs, pred_depths, gt_depths, pred_normals, gt_normals, pbr_attrs=pbr_attrs)
            if return_pred_mesh:
                return 0.0, {}, images, pred_meshes
            else:
                return 0.0, {}, images, None
        raise ValueError(f"Unsupported mode: {mode}")
    
    def _get_results(self, model: D_Joint_S2, outputs: Tuple, gt_outputs: Union[RGBAImages, None], mode: str, visual: bool, return_pred_mesh: bool = False):
        def safe_detach(x):
            return x.detach() if isinstance(x, Tensor) and x.requires_grad else x

        total_loss = 0.0
        metrics = {}

        (
            pred_outputs,
            reg_loss_dict,
            pred_meshes,
            pred_depths, gt_depths,
            pred_normals, gt_normals,
            pbr_attrs
        ) = outputs
        bg_color = model.get_background_color().to(model.device)
        if isinstance(pred_outputs, PBRAImages):
            pred_rgbs = pred_outputs.rgb2srgb().blend(bg_color)
        else:
            pred_rgbs = pred_outputs.blend(bg_color)
        gt_rgbs = gt_outputs.blend(bg_color) if gt_outputs is not None else pred_rgbs

        if self.ssim_weight > 0:
            ssim_loss = SSIML1Loss()(gt_outputs=gt_rgbs, outputs=pred_rgbs) * self.ssim_weight
            total_loss += ssim_loss
            metrics.update({
                'full_ssim_l1_loss': safe_detach(ssim_loss),
                'real_ssim_l1_loss': safe_detach(ssim_loss) / self.ssim_weight,
            })
        if self.mask_weight > 0:
            mask_loss = self._compute_mask_loss(pred_outputs, gt_outputs)
            mask_loss = mask_loss * 5.0
            total_loss += mask_loss
            metrics.update({
                'full_mask_loss': safe_detach(mask_loss),
                'real_mask_loss': safe_detach(mask_loss) / (self.mask_weight * 5.0),
            })

        reg_loss = sum(reg_loss_dict.values())
        total_loss += reg_loss
        metrics.update({
            'real_reg_appearance_temporal_hashgrid_loss': (safe_detach(reg_loss_dict['reg_appearance_temporal_hashgrid']) / model.reg_appearance_temporal_hashgrid_weight) if model.reg_appearance_temporal_hashgrid_weight > 0 else 0.0,
            'real_reg_occ_loss': (safe_detach(reg_loss_dict['reg_occ']) / model.reg_occ_weight) if model.reg_occ_weight > 0 else 0.0,
            'real_reg_light_loss': (safe_detach(reg_loss_dict['reg_light']) / model.reg_light_weight) if model.reg_light_weight > 0 else 0.0,
            'real_reg_geometry_residual_temporal_hashgrid_loss': (safe_detach(reg_loss_dict['reg_geometry_residual_temporal_hashgrid']) / model.reg_geometry_residual_temporal_hashgrid_weight) if model.reg_geometry_residual_temporal_hashgrid_weight > 0 else 0.0,
            'real_corse_L_dev_loss': (safe_detach(reg_loss_dict['corse_L_dev']) / 0.25) if 'corse_L_dev' in reg_loss_dict else 0.0,
            'real_detail_L_dev_loss': (safe_detach(reg_loss_dict['detail_L_dev']) / 0.25) if 'detail_L_dev' in reg_loss_dict else 0.0,
            # 'real_corse_sdf_entropy_loss': (safe_detach(reg_loss_dict['corse_sdf_entropy_loss']) / model.reg_sdf_entropy_weight) if model.reg_sdf_entropy_weight > 0 else 0.0,
            'real_detail_sdf_entropy_loss': (safe_detach(reg_loss_dict['detail_sdf_entropy_loss']) / model.reg_sdf_entropy_weight) if model.reg_sdf_entropy_weight > 0 else 0.0,
            # 'real_corse_sdf_eikonal_loss': (safe_detach(reg_loss_dict['corse_sdf_eikonal_loss']) / model.reg_sdf_eikonal_weight) if model.reg_sdf_eikonal_weight > 0 else 0.0,
            'real_detail_sdf_eikonal_loss': (safe_detach(reg_loss_dict['detail_sdf_eikonal_loss']) / model.reg_sdf_eikonal_weight) if model.reg_sdf_eikonal_weight > 0 else 0.0,
            'real_reg_curve_coeff_tv_loss': (safe_detach(reg_loss_dict['coeff_tv_loss']) / model.reg_coeff_tv_weight) if model.reg_coeff_tv_weight > 0 else 0.0,
            'reg_time_tv_loss': (safe_detach(reg_loss_dict['time_tv_loss']) / model.reg_time_tv_weight) if model.reg_time_tv_weight > 0 else 0.0,

            'full-reg-loss': safe_detach(reg_loss),
            'total_loss': safe_detach(total_loss),
            'psnr': PSNRLoss()(gt_rgbs.clamp(0, 1), pred_rgbs.clamp(0, 1)),
        })

        images = self._generate_visualizations(pred_rgbs, gt_rgbs, pred_depths, gt_depths, pred_normals, gt_normals, pbr_attrs=pbr_attrs) if visual else []

        if return_pred_mesh:
            return total_loss, metrics, images, pred_meshes
        else:
            return total_loss, metrics, images, None

    def _compute_mask_loss(self, pred_outputs, gt_outputs):
        mask_loss = 0.0
        for pred_output, gt_output in zip(pred_outputs, gt_outputs):
            mask_loss += torch.nn.functional.l1_loss(gt_output[..., -1], pred_output[..., -1])
        mask_loss = (self.mask_weight * mask_loss) / len(pred_outputs)
        return mask_loss

    def _compute_depth_loss(self, pred_depths, gt_depths):
        depth_loss = 0.0
        for pred_depth, gt_depth in zip(pred_depths, gt_depths):
            pred_depth_masked = pred_depth[..., :1] * gt_depth[..., 1:]
            gt_depth_masked = gt_depth[..., :1] * gt_depth[..., 1:]
            depth_loss += torch.nn.functional.l1_loss(pred_depth_masked, gt_depth_masked)
        depth_loss = (self.depth_weight * depth_loss) / len(pred_depths)
        return depth_loss

    @torch.no_grad()
    def _generate_visualizations(self, pred_images, gt_images, pred_depths, gt_depths, pred_normals, gt_normals, pbr_attrs=None):
        images = []
        if pbr_attrs is not None:
            for i in range(len(pred_depths)):
                row0 = torch.cat((
                    gt_images[i].item(),
                    gt_normals[i].visualize((1,1,1)).item(),
                    gt_depths[i].visualize().item(),
                ), dim=1).clamp(0, 1)
                row1 = torch.cat((
                    pred_images[i].item(),
                    pred_normals[i].visualize((1,1,1)).item(),
                    pred_depths[i].visualize().item(),
                ), dim=1).clamp(0, 1)
                row2 = torch.cat((
                    pbr_attrs['kd'][i].item(), 
                    pbr_attrs['light'].item(),
                ), dim=1).clamp(0, 1)
                row3 = torch.cat((
                    pbr_attrs['roughness'][i].item(), 
                    pbr_attrs['metallic'][i].item(), 
                    pbr_attrs['occ'][i].item()
                ), dim=1).clamp(0, 1)
                images.append(torch.cat((row0, row1, row2, row3), dim=0))
        else:
            for i in range(len(pred_depths)):
                col0 = torch.cat([gt_images[i].item(), pred_images[i].item()], dim=0).clamp(0, 1)
                col1 = torch.cat([gt_normals[i].visualize((1,1,1)).item(), pred_normals[i].visualize((1,1,1)).item()], dim=0).clamp(0, 1)
                col2 = torch.cat([gt_depths[i].visualize().item(), pred_depths[i].visualize().item()], dim=0).clamp(0, 1)
                images.append(torch.cat((col0, col1, col2), dim=1))

        return images

    def before_update(self, model: D_Joint_S2, optimizers: DS_ModuleOptimizers, *, curr_step: int) -> None:
       
        def linear_decay(begin_weight, end_weight, run_steps, decay_steps):
            if decay_steps > 0:
                progress = min(1.0, run_steps / decay_steps)
                return begin_weight - (begin_weight - end_weight) * progress
            return begin_weight
        
        def exponential_decay(begin_weight, end_weight, run_steps, half_life):
            if half_life > 0:
                progress = run_steps / half_life
                decayed = begin_weight * (0.5 ** progress)
                return max(end_weight, decayed)
            return begin_weight

        def set_reg_weight(
            enable_flag_name: str, weight_attr: str, decay_type: str, 
            begin_weight: float, end_weight: float, 
            start_step: int, end_step: int, decay_steps: int
        ):
            if end_step < 0:
                end_step = self.num_steps
            enabled = (start_step >= 0 and end_step >= start_step and curr_step >= start_step and curr_step <= end_step)
            if enable_flag_name is not None:
                assert hasattr(model, enable_flag_name), f"Model does not have attribute {enable_flag_name}"
                setattr(model, enable_flag_name, enabled)
            run_steps = curr_step - start_step
            if enabled:
                if decay_type == 'linear':
                    weight = linear_decay(begin_weight, end_weight, run_steps, decay_steps)
                elif decay_type == 'exponential':
                    weight = exponential_decay(begin_weight, end_weight, run_steps, decay_steps)
            else:
                weight = 0.0
            if weight_attr is not None:
                assert hasattr(model, weight_attr), f"Model does not have attribute {weight_attr}"
                setattr(model, weight_attr, weight)

        reg = self.regularization

        # 主要损失函数权重的线性衰减
        if reg.ssim_weight_start_step >= 0 and curr_step >= reg.ssim_weight_start_step:
            self.ssim_weight = linear_decay(reg.ssim_weight_begin, reg.ssim_weight_end, curr_step-reg.ssim_weight_start_step, reg.ssim_weight_decay_steps)
        else:
            self.ssim_weight = -1.
        if (reg.mask_weight_start_step >= 0 and curr_step >= reg.mask_weight_start_step):
            self.mask_weight = linear_decay(reg.mask_weight_begin, reg.mask_weight_end, curr_step-reg.mask_weight_start_step, reg.mask_weight_decay_steps)
        else:
            self.mask_weight = -1.

        # 正则化项权重的线性衰减
        set_reg_weight(
            "reg_appearance_temporal_hashgrid_able", "reg_appearance_temporal_hashgrid_weight", 'linear',
            reg.reg_appearance_temporal_hashgrid_begin, reg.reg_appearance_temporal_hashgrid_end,
            reg.reg_appearance_temporal_hashgrid_start_step, reg.reg_appearance_temporal_hashgrid_end_step, reg.reg_appearance_temporal_hashgrid_decay_steps
        )
        
        set_reg_weight(
            None, "geometry_residual_weight", 'linear',
            reg.geometry_residual_weight_begin, reg.geometry_residual_weight_end,
            reg.geometry_residual_start_step, reg.geometry_residual_end_step, reg.geometry_residual_decay_steps
        )

        set_reg_weight(
            "reg_occ_able", "reg_occ_weight", 'linear',
            reg.reg_occ_begin, reg.reg_occ_end, 
            reg.reg_occ_start_step, reg.reg_occ_end_step, reg.reg_occ_decay_steps
        )

        set_reg_weight(
            "reg_light_able", "reg_light_weight", 'linear',
            reg.reg_light_begin, reg.reg_light_end,
            reg.reg_light_start_step, reg.reg_light_end_step, reg.reg_light_decay_steps
        )
        
        set_reg_weight(
            "reg_geometry_residual_temporal_hashgrid_able", "reg_geometry_residual_temporal_hashgrid_weight", 'linear',
            reg.reg_geometry_residual_temporal_hashgrid_begin, reg.reg_geometry_residual_temporal_hashgrid_end,
            reg.reg_geometry_residual_temporal_hashgrid_start_step, reg.reg_geometry_residual_temporal_hashgrid_end_step, reg.reg_geometry_residual_temporal_hashgrid_decay_steps
        )

        set_reg_weight(
            "reg_sdf_entropy_able", "reg_sdf_entropy_weight", 'linear',
            reg.sdf_entropy_weight_begin, reg.sdf_entropy_weight_end,
            reg.sdf_entropy_start_step, reg.sdf_entropy_end_step, reg.sdf_entropy_decay_steps
        )

        set_reg_weight(
            "reg_time_tv_able", "reg_time_tv_weight", 'linear',
            reg.time_tv_weight_begin, reg.time_tv_weight_end,
            reg.time_tv_start_step, reg.time_tv_end_step, reg.time_tv_decay_steps
        )

        set_reg_weight(
            "reg_coeff_tv_able", "reg_coeff_tv_weight", 'linear',
            reg.curve_coeff_tv_weight_begin, reg.curve_coeff_tv_weight_end,
            reg.curve_coeff_tv_start_step, reg.curve_coeff_tv_end_step, reg.curve_coeff_tv_decay_steps
        )

        set_reg_weight(
            "reg_sdf_eikonal_able", "reg_sdf_eikonal_weight", 'linear',
            reg.sdf_eikonal_weight_begin, reg.sdf_eikonal_weight_end,
            reg.sdf_eikonal_start_step, reg.sdf_eikonal_end_step, reg.sdf_eikonal_decay_steps
        )
        
        # 是否采用指定curve componets推理
        dynamic_model_stage_dict = {}
        dynamic_model_stage_dict['static_sdf_params'] = True
        dynamic_model_stage_dict['sdf_curve_poly_coefficient'] = True
        dynamic_model_stage_dict['sdf_curve_low_freq_fourier_coefficient'] = True
        dynamic_model_stage_dict['sdf_curve_mid_freq_fourier_coefficient'] = True
        dynamic_model_stage_dict['sdf_curve_high_freq_fourier_coefficient'] = True
        dynamic_model_stage_dict['sdf_curve_wavelet_coefficient'] = False
        model.dynamic_model_stage = dynamic_model_stage_dict

        if reg.reset_light_step >= 0 and curr_step == reg.reset_light_step:
            model.cubemap = torch.nn.Parameter(torch.empty(6, 512, 512, 3, device=model.device).fill_(0.5))

    def after_update(self, model: D_Joint_S2, optimizers: DS_ModuleOptimizers, *, curr_step: int) -> None:
        self._real_learning_rates.update({
            'dynamic_appearance': optimizers.optimizers['dynamic_appearance'].param_groups[0]['lr'],
            # 'geometry_residual': optimizers.optimizers['geometry_residual'].param_groups[0]['lr'],
            'geometry_sdf_residual': optimizers.optimizers['geometry_sdf_residual'].param_groups[0]['lr'],
            'geometry_deform_residual': optimizers.optimizers['geometry_deform_residual'].param_groups[0]['lr'],
            'geometry_weight_residual': optimizers.optimizers['geometry_weight_residual'].param_groups[0]['lr'],
            'static_sdf_params': optimizers.optimizers['static_sdf_params'].param_groups[0]['lr'],
            'poly_coeffs': optimizers.optimizers['poly_coeffs'].param_groups[0]['lr'],
            'fourier_low': optimizers.optimizers['fourier_low_coeffs'].param_groups[0]['lr'],
            'fourier_mid': optimizers.optimizers['fourier_mid_coeffs'].param_groups[0]['lr'],
            'fourier_high': optimizers.optimizers['fourier_high_coeffs'].param_groups[0]['lr'],
        })

        if model.shader_type == "split_sum_pbr":
            self._real_learning_rates.update({
                'light': optimizers.optimizers['light'].param_groups[0]['lr'],
            })
            with torch.no_grad():
                model.cubemap.clamp_min_(0)
