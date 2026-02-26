from __future__ import annotations

# import modules
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Literal, List, Union
import matplotlib.pyplot as plt

import torch
from torch import Tensor

# import rfstudio modules
from rfstudio.optim import Optimizer
from rfstudio.graphics import DepthImages, VectorImages
from rfstudio.graphics.shaders import DepthShader, NormalShader

# import rfstudio_ds modules
from .base_trainer import DS_BaseTrainer
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh

from rfstudio_ds.model import D_DiffDR_S2
from rfstudio_ds.data import (
    SyntheticDynamicMonocularBlenderDepthDataset,
    SyntheticDynamicMonocularCostumeDepthDataset,
    SyntheticDynamicMultiViewBlenderDepthDataset,
    SyntheticDynamicMultiViewCostumeDepthDataset,
)
from rfstudio_ds.optim import DS_ModuleOptimizers

from rfstudio_ds.visualization._plot_cube_curve import plot_cube_curve, plot_curve


@dataclass
class RegularizationConfig():

    """监督信号权重配置"""
    depth_weight_begin: Optional[float] = 1. # depth supervision loss
    depth_weight_end: Optional[float] = 1.
    depth_weight_decay_steps: Optional[int] = 1250
    depth_weight_start_step: Optional[int] = 0 # -1 代表“不启用”

    mask_weight_begin: Optional[float] = 1. # mask supervision loss
    mask_weight_end: Optional[float] = 1.
    mask_weight_decay_steps: Optional[int] = 1250   
    mask_weight_start_step: Optional[int] = 0
    
    normal_weight_begin: Optional[float] = 0. # normal supervision loss
    normal_weight_end: Optional[float] = 0.
    normal_weight_decay_steps: Optional[int] = 1250
    normal_weight_start_step: Optional[int] = -1

    """正则化参数配置"""
    sdf_entropy_weight_begin: Optional[float] = 0.2 # sdf entropy loss
    sdf_entropy_weight_end: Optional[float] = 0.01 # todo 由于是很多个mesh 的entropy ，感觉要把这个参数调大一些
    sdf_entropy_decay_steps: Optional[int] = 1250
    sdf_entropy_start_step: Optional[int] = -1
    sdf_entropy_end_step: Optional[int] = 5000

    time_tv_weight_begin: Optional[float] = 0.1 # curve derivative smooth loss
    time_tv_weight_end: Optional[float] = 0.01
    time_tv_decay_steps: Optional[int] = 1000
    time_tv_start_step: Optional[int] = -1
    time_tv_end_step: Optional[int] = 5000

    curve_wavelet_sparse_weight_begin: float = 0.01 # curve wavelet sparse loss
    curve_wavelet_sparse_weight_end: float = 0.1
    curve_wavelet_sparse_decay_steps: Optional[int] = 1000
    curve_wavelet_sparse_start_step: int = 0
    curve_wavelet_sparse_end_step: int = 5000

    sdf_residual_enc_weight_begin: Optional[float] = 0.01 # sdf eikonal loss
    sdf_residual_enc_weight_end: Optional[float] = 1
    sdf_residual_enc_decay_steps: Optional[int] = 200
    sdf_residual_enc_start_step: Optional[int] = -1
    sdf_residual_enc_end_step: Optional[int] = -1


@dataclass
class D_DiffDRTrainer_S2(DS_BaseTrainer):

    sdf_residual_enc_learning_rate: float = 0.003

    wavelet_params_learning_rate: float = 0.003
    walelet_params_decay: Optional[int] = None
    wavelet_params_max_norm: Optional[float] = 0.05

    optimizer_epsilon: float = 1e-8
    learning_rate_decay_steps: Optional[int] = 800

    use_multi_model_stage: bool = True
    model_start_steps: Optional[List[int]] = field(default_factory=lambda: [0,0,0,0,0,0]) # static, poly, fourier_low, fourier_mid, fourier_high, wavelet

    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)

    def setup(
        self,
        model: D_DiffDR_S2,
        dataset: Union[SyntheticDynamicMonocularBlenderDepthDataset, SyntheticDynamicMonocularCostumeDepthDataset, SyntheticDynamicMultiViewCostumeDepthDataset, SyntheticDynamicMultiViewBlenderDepthDataset],
    ) -> DS_ModuleOptimizers:
        assert isinstance(dataset, (
            SyntheticDynamicMonocularBlenderDepthDataset, 
            SyntheticDynamicMonocularCostumeDepthDataset, 
            SyntheticDynamicMultiViewCostumeDepthDataset,
            SyntheticDynamicMultiViewBlenderDepthDataset,
        ))
        self.gt_geometry_train = dataset.get_meshes(split='train')
        self.gt_geometry_val = dataset.get_meshes(split='val')
        self.gt_geometry_test = dataset.get_meshes(split='test')
        self.gt_geometry_orbit_vis = dataset.get_meshes(split='orbit_vis')
        self.gt_geometry_fix_vis = dataset.get_meshes(split='fix_vis')

        optim_dict = {}
        if model.geometry != 'gt':
            optim_dict['sdf_residual_enc'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.sdf_residual_enc,
                lr=self.sdf_residual_enc_learning_rate,
                lr_decay=int(self.learning_rate_decay_steps),
                max_norm=None,
                eps=self.optimizer_epsilon,
            )
            optim_dict['wavelet_coeffs'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='dddm_wavelet_coeffs'), 
                lr=self.wavelet_params_learning_rate,
                lr_decay=self.learning_rate_decay_steps if self.walelet_params_decay is None else self.walelet_params_decay,
                max_norm=self.wavelet_params_max_norm,
                warm_up=100,
                eps=self.optimizer_epsilon      
            )

            # 记录实际学习率
            self._real_learning_rates = {
                'sdf_residual_enc': self.sdf_residual_enc_learning_rate,
                'wavelet_coeffs': self.wavelet_params_learning_rate,
            }

            model_optimize_start_steps = None
            if self.use_multi_model_stage:
                model_optimize_start_steps = {
                    'wavelet_coeffs': self.model_start_steps[5],
                    'sdf_residual_enc': self.regularization.sdf_residual_enc_start_step,
                }

        assert optim_dict, "Optimizer dictionary cannot be empty"
        return DS_ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
            start_steps=model_optimize_start_steps,
        )
    
    def step(
        self,
        model: 'D_DiffDR_S2',
        inputs: 'DS_Cameras',
        gt_depths: 'DepthImages',
        *,
        indices: Tensor = None,
        mode: Literal['train', 'val', 'test', 'orbit_vis', 'fix_vis', 'analyse_curve'] = 'train',
        vis_downsample_factor: Optional[int] = None,
        visual:  bool = False,
        analyse_curve_save_path: Optional[str] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[List[Tensor]]]:
        if mode == 'analyse_curve':
            # 在特定空间点上探测 SDF 值的时序变化并将其可视化
            return self._run_analyse_curve(model, inputs, analyse_curve_save_path)
        
        assert indices is not None, "indices should not be None"
        if vis_downsample_factor is not None:
            indices = indices[::vis_downsample_factor]
            inputs = inputs[::vis_downsample_factor]
        model.set_batch_gt_geometry([getattr(self, f'gt_geometry_{mode}')[i] for i in indices.tolist()])
        outputs = model.render_report(
            camera_inputs=inputs, 
        )

        if mode in ['train', 'val', 'test']:
            return self._get_results(model, outputs, mode, visual)
        elif mode in ['orbit_vis', 'fix_vis']:
            images = self._generate_visualizations(outputs)
            return 0.0, {}, images
        raise ValueError(f"Unsupported mode: {mode}")

    def _get_results(self, model: D_DiffDR_S2, outputs: Tuple, mode: str, visual: bool):
        def safe_detach(x):
            return x.detach() if isinstance(x, Tensor) and x.requires_grad else x

        total_loss = 0.0
        metrics = {}

        (
            pred_meshes,
            depth_outputs, gt_depth_outputs,
            normal_outputs, gt_normal_outputs,
            reg_loss_dict,
        ) = outputs

        # for key, values in reg_loss_dict.items():
        #     if key != 'coeff_tv_loss':
        #         total_loss += values
        reg_loss = sum(reg_loss_dict.values())
        total_loss += reg_loss
        depth_loss, mask_loss = self._compute_losses(depth_outputs, gt_depth_outputs)
        total_loss += (depth_loss + mask_loss)
        
        metrics.update({
            'full-reg-loss': safe_detach(reg_loss),
            'real-depth-loss': depth_loss.detach() / self.depth_weight,
            'real-mask-loss': mask_loss.detach() / self.mask_weight,
        })

        metrics.update({
            'real_reg_sdf_entropy_loss': (safe_detach(reg_loss_dict['sdf_entropy_loss']) / model.reg_sdf_entropy_weight) if model.reg_sdf_entropy_weight > 0 else 0.0,
            'reg_time_tv_loss': (safe_detach(reg_loss_dict['time_tv_loss']) / model.reg_time_tv_weight) if model.reg_time_tv_weight > 0 else 0.0,
            'real-wavelet-sparse-loss': (safe_detach(reg_loss_dict['wavelet_sparse_loss']) / model.reg_wavelet_sparse_weight) if model.reg_wavelet_sparse_weight > 0 else 0.0,
        })

        images = self._generate_visualizations(outputs) if visual else []
        metrics['total-loss'] = total_loss.detach()

        return total_loss, metrics, images

    def _compute_losses(
        self,
        depths: DepthImages,
        gt_depths: DepthImages,
        normals: Optional[VectorImages] = None,
        gt_normals: Optional[VectorImages] = None,
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

    def _generate_visualizations(self, outputs):
        images = []
        depth_outputs, gt_depth_outputs = outputs[1], outputs[2]
        normal_outputs, gt_normal_outputs = outputs[3], outputs[4]

        for i in range(len(depth_outputs)):
            col1 = torch.cat([gt_normal_outputs[i].visualize((1,1,1)).item(), normal_outputs[i].visualize((1,1,1)).item()], dim=0).clamp(0, 1)
            col2 = torch.cat([gt_depth_outputs[i].visualize().item(), depth_outputs[i].visualize().item()], dim=0).clamp(0, 1)
            images.append(torch.cat((col1, col2), dim=1))

        return images

    @torch.no_grad()
    def _run_analyse_curve(self, model: D_DiffDR_S2, inputs: DS_Cameras, save_path: str) -> Tuple[Tensor, Dict[str, Tensor], Optional[List[Tensor]]]:
        
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
            sdf_curve_wavelet_coefficient = model.sdf_curve_wavelet_coefficient,
        )
        cubes_info = geometric_repr.get_cube_curve_info(positions=positions)

        times = inputs.times
        pred_data, _ = geometric_repr.query_sdf_at_times(t=times,model_stage=model.dynamic_model_stage)
        
        xyz = geometric_repr.vertices
        xyz_normal = xyz / model.geometry_scale # normalize to [-1, 1]
        t = (times * 2 - 1).unsqueeze(0).expand(xyz_normal.shape[0], -1) # normalize to [-1, 1]
        sdfs_residual = []
        for i in range(len(times)):
            t = times[i].unsqueeze(0).expand(xyz.shape[0],-1) # [V, 1]
            xyzt = torch.cat((xyz_normal, t), dim=-1) # [V, 4]
            sdf_residual, spatial_h, temporal_h = self.sdf_residual_enc(xyzt)
            sdfs_residual.append(sdf_residual)
        sdfs_residual = torch.stack(sdfs_residual, dim=0)
        pred_data = pred_data + sdfs_residual * model.sdfs_residual_weight

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

    def before_update(self, model: D_DiffDR_S2, optimizers: DS_ModuleOptimizers, *, curr_step: int) -> None:
        """在优化器更新前调整模型参数和正则化权重。训练在初期可以专注于拟合基本形状，然后逐步引入正则化项来保证模型的平滑性和物理真实性"""
        assert isinstance(model, D_DiffDR_S2), f"Expected D_DiffDR_S2, got {type(model)}"

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
        if (reg.depth_weight_start_step >= 0 and curr_step >= reg.depth_weight_start_step):
            self.depth_weight = linear_decay(reg.depth_weight_begin, reg.depth_weight_end, curr_step-reg.depth_weight_start_step, reg.depth_weight_decay_steps)
        else:
            self.depth_weight = 0.0
        if (reg.mask_weight_start_step >= 0 and curr_step >= reg.mask_weight_start_step):
            self.mask_weight = linear_decay(reg.mask_weight_begin, reg.mask_weight_end, curr_step-reg.mask_weight_start_step, reg.mask_weight_decay_steps)
        else:
            self.mask_weight = 0.0
        if (reg.normal_weight_start_step >= 0 and curr_step >= reg.normal_weight_start_step):
            self.normal_weight = linear_decay(reg.normal_weight_begin, reg.normal_weight_end, curr_step-reg.normal_weight_start_step, reg.normal_weight_decay_steps)
        else:
            self.normal_weight = 0.0

        # 正则化项权重的线性衰减
        set_reg_weight(
            "reg_sdf_entropy_able", "reg_sdf_entropy_weight", 'linear',
            reg.sdf_entropy_weight_begin, reg.sdf_entropy_weight_end,
            reg.sdf_entropy_start_step, reg.sdf_entropy_end_step, reg.sdf_entropy_decay_steps
        )

        set_reg_weight(
            "reg_wavelet_sparse_able", "reg_wavelet_sparse_weight",
            reg.curve_wavelet_sparse_weight_begin, reg.curve_wavelet_sparse_weight_end,
            reg.curve_wavelet_sparse_start_step, reg.curve_wavelet_sparse_end_step, reg.curve_wavelet_sparse_decay_steps
        )

        set_reg_weight(
            "reg_time_tv_able", "reg_time_tv_weight",
            reg.time_tv_weight_begin, reg.time_tv_weight_end,
            reg.time_tv_start_step, reg.time_tv_end_step, reg.time_tv_decay_steps
        )

        set_reg_weight(
            'use_sdf_residual_enc', "sdf_residual_enc_weight",
            reg.sdf_residual_enc_weight_begin, reg.sdf_residual_enc_weight_end,
            reg.sdf_residual_enc_start_step, reg.sdf_residual_enc_end_step, reg.sdf_residual_enc_decay_steps
        ) # 控制是否使用sdf_residual_enc

        # 多阶段训练，控制是否使用wavelet_coeffs
        if self.use_multi_model_stage:
            dynamic_model_stage_dict = {}
            dynamic_model_stage_dict['static_sdf_params'] = True
            dynamic_model_stage_dict['sdf_curve_poly_coefficient'] = True
            dynamic_model_stage_dict['sdf_curve_low_freq_fourier_coefficient'] = True
            dynamic_model_stage_dict['sdf_curve_mid_freq_fourier_coefficient'] = True
            dynamic_model_stage_dict['sdf_curve_high_freq_fourier_coefficient'] = True
            dynamic_model_stage_dict['sdf_curve_wavelet_coefficient'] = (curr_step >= self.model_start_steps[5] and self.model_start_steps[5] >= 0)
            model.dynamic_model_stage = dynamic_model_stage_dict

    def after_update(self, model: D_DiffDR_S2, optimizers: DS_ModuleOptimizers, *, curr_step: int) -> None:
        """在优化器更新后记录实际学习率。
        """
        if model.geometry != 'gt':
            self._real_learning_rates.update({
                'wavelet': optimizers.optimizers['wavelet_coeffs'].param_groups[0]['lr'],
            })
