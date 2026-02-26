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

from rfstudio_ds.model import D_Texture, D_Texture
from rfstudio_ds.data import SyntheticDynamicMonocularBlenderRGBADataset, SyntheticDynamicMultiViewBlenderRGBADataset
from rfstudio_ds.optim import DS_ModuleOptimizers

from rfstudio_ds.visualization._plot_cube_curve import plot_cube_curve, plot_curve


@dataclass
class RegularizationConfig:

    ssim_weight_begin: Optional[float] = 1. # ssim loss
    ssim_weight_end: Optional[float] = 1.
    ssim_weight_decay_steps: Optional[int] = 1250
    ssim_weight_start_step: Optional[int] = 0

    psnr_weight_begin: Optional[float] = 1. # psnr loss
    psnr_weight_end: Optional[float] = 1.
    psnr_weight_decay_steps: Optional[int] = 1250
    psnr_weight_start_step: Optional[int] = -1

    """正则化参数配置"""
    reg_spatial_hashgrid_begin: float = 0.1
    reg_spatial_hashgrid_end: float = 0.01
    reg_spatial_hashgrid_decay_steps: Optional[int] = 1250
    reg_spatial_hashgrid_start_step: Optional[int] = -1
    reg_spatial_hashgrid_end_step: Optional[int] = -1

    reg_temporal_hashgrid_begin: float = 0.1
    reg_temporal_hashgrid_end: float = 0.01
    reg_temporal_hashgrid_decay_steps: Optional[int] = 1250
    reg_temporal_hashgrid_start_step: Optional[int] = -1
    reg_temporal_hashgrid_end_step: Optional[int] = -1

    reg_kd_enc_begin: float = 0.1
    reg_kd_enc_end: float = 0.01
    reg_kd_enc_decay_steps: Optional[int] = 1250
    reg_kd_enc_start_step: Optional[int] = -1
    reg_kd_enc_end_step: Optional[int] = -1

    reg_occ_begin: float = 0.1
    reg_occ_end: float = 0.01
    reg_occ_decay_steps: Optional[int] = 1250
    reg_occ_start_step: Optional[int] = -1
    reg_occ_end_step: Optional[int] = -1

    reg_light_begin: float = 0.1
    reg_light_end: float = 0.01
    reg_light_decay_steps: Optional[int] = 1250
    reg_light_start_step: Optional[int] = -1
    reg_light_end_step: Optional[int] = -1


@dataclass
class D_TextureTrainer(DS_BaseTrainer):

    appearance_learning_rate: float = 0.01
    appearance_decay: Optional[int] = None

    light_learning_rate: float = 0.01 # 用于light的参数
    light_learning_rate_decay: Optional[int] = None
    light_grad_amp: Optional[float] = 64.0

    optimizer_epsilon: float = 1e-8
    learning_rate_decay_steps: Optional[int] = 800
    
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)

    def setup(
        self,
        model: D_Texture,
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
        
        optim_dict['dynamic_appearance'] = Optimizer(
            category=torch.optim.Adam,
            modules=model.dynamic_texture,
            lr=self.appearance_learning_rate,
            lr_decay=self.learning_rate_decay_steps if self.appearance_decay is None else self.appearance_decay,
            max_norm=None,
            eps=self.optimizer_epsilon,
        )

        self._real_learning_rates['dynamic_appearance'] = self.appearance_learning_rate

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
            
            self._real_learning_rates.update({
                'light': self.light_learning_rate,
            })

        assert optim_dict, "Optimizer dictionary cannot be empty"
        return DS_ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
            start_steps=None,
        )

    def step(
        self,
        model: 'D_Texture',
        inputs: 'DS_Cameras',
        gt_outputs: Union['RGBAImages', None],
        *,
        indices: Optional[Tensor] = None,
        mode: Literal['train', 'val', 'test', 'orbit_vis', 'fix_vis'] = 'train',
        val_pbr_attr: bool = False,
        vis_downsample_factor: Optional[int] = None,
        visual: bool = False,
        analyse_curve_save_path: Optional[str] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[List[Tensor]]]:
        if vis_downsample_factor is not None:
            indices = indices[::vis_downsample_factor]
            inputs = inputs[::vis_downsample_factor]
            if gt_outputs is not None:
                gt_outputs = gt_outputs[::vis_downsample_factor]
        model.set_batch_gt_geometry([getattr(self, f'gt_geometry_{mode}')[i] for i in indices.tolist()])
        (
            pred_outputs,
            reg_loss_dict,
            pbr_attrs,
        ) = model.render_report(
            camera_inputs=inputs, 
            trainer_mode=mode,
            return_pbr_attrs=val_pbr_attr,
        )
        
        if mode in ['train', 'val', 'test']:
            def safe_detach(x):
                return x.detach() if isinstance(x, Tensor) and x.requires_grad else x

            total_loss = torch.tensor(0.0, device=inputs.device)
            metrics = {}

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
            if self.psnr_weight > 0:
                psnr_loss = PSNRLoss()(gt_outputs=gt_rgbs, outputs=pred_rgbs) * self.psnr_weight
                total_loss += psnr_loss
                metrics.update({
                    'full_psnr_loss': safe_detach(psnr_loss),
                    'real_psnr_loss': safe_detach(psnr_loss) / self.psnr_weight,
                })
            
            reg_loss = sum(reg_loss_dict.values())
            total_loss += reg_loss

            reg_loss = sum(reg_loss_dict.values())
            total_loss += reg_loss
            metrics.update({
                'real_reg_spatial_hashgrid_loss': (safe_detach(reg_loss_dict['reg_spatial_hashgrid']) / model.reg_spatial_hashgrid_weight) if model.reg_spatial_hashgrid_weight > 0 else 0.0,
                'real_reg_temporal_hashgrid_loss': (safe_detach(reg_loss_dict['reg_temporal_hashgrid']) / model.reg_temporal_hashgrid_weight) if model.reg_temporal_hashgrid_weight > 0 else 0.0,
                'real_reg_kd_enc_loss': (safe_detach(reg_loss_dict['reg_kd_enc']) / model.reg_kd_enc_weight) if model.reg_kd_enc_weight > 0 else 0.0,
                'real_reg_occ_loss': (safe_detach(reg_loss_dict['reg_occ']) / model.reg_occ_weight) if model.reg_occ_weight > 0 else 0.0,
                'real_reg_light_loss': (safe_detach(reg_loss_dict['reg_light']) / model.reg_light_weight) if model.reg_light_weight > 0 else 0.0,

                'full-reg-loss': safe_detach(reg_loss),
                'total_loss': safe_detach(total_loss),
            })

            images = []
            if visual:
                with torch.no_grad():
                    if val_pbr_attr:
                        for i in range(len(inputs)):
                            row0 = torch.cat((
                                gt_rgbs[i].detach().item(),
                                pred_rgbs[i].detach().item(),
                                pbr_attrs['light'].detach().item(),
                            ), dim=1).clamp(0, 1)
                            row1 = torch.cat((
                                pbr_attrs['kd'][i].detach().item(),
                                pbr_attrs['roughness'][i].detach().item(),
                                pbr_attrs['metallic'][i].detach().item(),
                                pbr_attrs['occ'][i].detach().item(),
                            ), dim=1).clamp(0, 1)
                            image = torch.cat((row0, row1), dim=0)
                            images.append(image)
                    else:
                        for i in range(len(inputs)):
                            col0 = torch.cat((
                                gt_rgbs[i].item(),
                                pred_rgbs[i].detach().item(),
                            ), dim=0).clamp(0, 1)
                            image = col0
                            images.append(image)
            return total_loss, metrics, images
        elif mode in ['orbit_vis', 'fix_vis']: # todo support vis light and kdks
            images = []
            bg_color = model.get_background_color().to(model.device)
            if isinstance(pred_outputs, PBRAImages):
                pred_rgbs = pred_outputs.rgb2srgb().blend(bg_color)
            else:
                pred_rgbs = pred_outputs.blend(bg_color)
            gt_rgbs = gt_outputs.blend(bg_color) if gt_outputs is not None else pred_rgbs
            with torch.no_grad():
                if val_pbr_attr:
                    for i in range(len(inputs)):
                        row0 = torch.cat((
                            gt_rgbs[i].detach().item(),
                            pred_rgbs[i].detach().item(),
                            pbr_attrs['light'].detach().item(),
                        ), dim=1).clamp(0, 1)
                        row1 = torch.cat((
                            pbr_attrs['kd'][i].detach().item(),
                            pbr_attrs['roughness'][i].detach().item(),
                            pbr_attrs['metallic'][i].detach().item(),
                            pbr_attrs['occ'][i].detach().item(),
                        ), dim=1).clamp(0, 1)
                        image = torch.cat((row0, row1), dim=0)
                        images.append(image)
                else:
                    for i in range(len(inputs)):
                        col0 = torch.cat((
                            gt_rgbs[i].item(),
                            pred_rgbs[i].detach().item(),
                        ), dim=0).clamp(0, 1)
                        image = col0
                        images.append(image)
            return 0.0, {}, images

    def before_update(self, model: D_Texture, optimizers: DS_ModuleOptimizers, *, curr_step: int) -> None:
       
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
            enabled = (start_step >= 0 and curr_step >= start_step and end_step >= 0 and curr_step <= end_step)
            if enable_flag_name is not None:
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
                setattr(model, weight_attr, weight)

        reg = self.regularization

        # 主要损失函数权重的线性衰减
        if reg.ssim_weight_start_step >= 0 and curr_step >= reg.ssim_weight_start_step:
            self.ssim_weight = linear_decay(reg.ssim_weight_begin, reg.ssim_weight_end, curr_step-reg.ssim_weight_start_step, reg.ssim_weight_decay_steps)
        else:
            self.ssim_weight = -1.
        if reg.psnr_weight_start_step >= 0 and curr_step >= reg.psnr_weight_start_step:
            self.psnr_weight = linear_decay(reg.psnr_weight_begin, reg.psnr_weight_end, curr_step-reg.psnr_weight_start_step, reg.psnr_weight_decay_steps)
        else:
            self.psnr_weight = -1.
       
        # 正则化项权重的线性衰减
        set_reg_weight(
            "reg_spatial_hashgrid_able", "reg_spatial_hashgrid_weight", 'linear',
            reg.reg_spatial_hashgrid_begin, reg.reg_spatial_hashgrid_end,
            reg.reg_spatial_hashgrid_start_step, reg.reg_spatial_hashgrid_end_step, reg.reg_spatial_hashgrid_decay_steps
        )
        
        set_reg_weight(
            "reg_temporal_hashgrid_able", "reg_temporal_hashgrid_weight", 'linear',
            reg.reg_temporal_hashgrid_begin, reg.reg_temporal_hashgrid_end,
            reg.reg_temporal_hashgrid_start_step, reg.reg_temporal_hashgrid_end_step, reg.reg_temporal_hashgrid_decay_steps
        )

        set_reg_weight(
            "reg_kd_enc_able", "reg_kd_enc_weight", 'linear',
            reg.reg_kd_enc_begin, reg.reg_kd_enc_end,
            reg.reg_kd_enc_start_step, reg.reg_kd_enc_end_step, reg.reg_kd_enc_decay_steps
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

    def after_update(self, model: D_Texture, optimizers: DS_ModuleOptimizers, *, curr_step: int) -> None:
        self._real_learning_rates.update({
            'dynamic_appearance': optimizers.optimizers['dynamic_appearance'].param_groups[0]['lr'],
        })
        if model.shader_type == "split_sum_pbr":
            self._real_learning_rates.update({
                'light': optimizers.optimizers['light'].param_groups[0]['lr'],
            })
            with torch.no_grad():
                model.cubemap.clamp_min_(0)
