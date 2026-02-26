from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Literal, List
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from rfstudio.optim import Optimizer
from rfstudio_ds.optim import DS_ModuleOptimizers
from rfstudio_ds.data import DynamicSDFDataset
from rfstudio_ds.model import D_SDFFit
from rfstudio_ds.loss import L2Loss
from .base_dsdf_trainer import DSDF_BaseTrainer
from rfstudio_ds.engine.experiment import DS_Experiment

from rfstudio_ds.visualization._plot_cube_curve import plot_cube_curve, plot_curve



@dataclass
class RegularizationConfig:
    """正则化参数配置"""
    time_tv_weight_begin: Optional[float] = 0.1 # curve derivative smooth loss
    time_tv_weight_end: Optional[float] = 0.01
    time_tv_decay_steps: Optional[int] = 1000
    time_tv_start_step: Optional[int] = -1
    time_tv_end_step: Optional[int] = -1

    curve_coeff_tv_weight_begin: Optional[float] = 0.01 # curve coefficient tv smooth loss
    curve_coeff_tv_weight_end: Optional[float] = 0.01
    curve_coeff_tv_decay_steps: Optional[int] = 1000
    curve_coeff_tv_start_step: Optional[int] = 0
    curve_coeff_tv_end_step: Optional[int] = -1

    curve_wavelet_sparse_weight_begin: float = 0.01 # cruve coefficient tv smooth loss
    curve_wavelet_sparse_weight_end: float = 0.01
    curve_wavelet_sparse_decay_steps: Optional[int] = 1000
    curve_wavelet_sparse_start_step: int = -1
    curve_wavelet_sparse_end_step: int = -1


@dataclass
class D_SDFFitTrainer(DSDF_BaseTrainer):

    optimizer_epsilon: float = 1e-15
    learning_rate_decay_steps: Optional[int] = 200

    static_sdf_params_learning_rate: float = 0.5
    static_sdf_params_decay: Optional[int] = None
    static_sdf_params_max_norm: Optional[float] = None

    poly_coeffs_learning_rate: float = 0.5
    poly_coeffs_decay: Optional[int] = None
    poly_coeffs_max_norm: Optional[float] = None

    fourier_low_learning_rate: float = 0.05
    fourier_low_decay: Optional[int] = None
    fourier_low_max_norm: Optional[float] = None

    fourier_mid_learning_rate: float = 0.001
    fourier_mid_decay: Optional[int] = None
    fourier_mid_max_norm: Optional[float] = None

    fourier_high_learning_rate: float = 0.001
    fourier_high_decay: Optional[int] = None
    fourier_high_max_norm: Optional[float] = None

    learn_omega: bool = False

    fourier_low_omega_learning_rate: float = 0.05
    fourier_low_omega_decay: Optional[int] = None
    fourier_low_omega_max_norm: Optional[float] = None

    fourier_mid_omega_learning_rate: float = 0.05
    fourier_mid_omega_decay: Optional[int] = None
    fourier_mid_omega_max_norm: Optional[float] = None

    fourier_high_omega_learning_rate: float = 0.05
    fourier_high_omega_decay: Optional[int] = None
    fourier_high_omega_max_norm: Optional[float] = None
    
    learn_wavelet: bool = False

    wavelet_learning_rate: float = 0.005
    wavelet_decay: Optional[int] = None
    wavelet_max_norm: Optional[float] = None
    
    use_multi_model_stage: bool = True
    model_start_steps: Optional[List[int]] = field(default_factory=lambda: [0,0,200,800,1400,-1]) # static, poly, fourier_low, fourier_mid, fourier_high, wavelet
    
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)

    def setup(self, model: D_SDFFit, dataset: DynamicSDFDataset) -> DS_ModuleOptimizers:
        """初始化优化器集合。

        Args:
            model: D_SDFFit 模型实例。
            dataset: 动态SDF数据集。

        Returns:
            DS_ModuleOptimizers: 优化器集合。
        """
        assert isinstance(dataset, DynamicSDFDataset), "Dataset must be DynamicSDFDataset"

        optim_dict = {}
        self._real_learning_rates = {}

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
            'static_sdf_params': self.static_sdf_params_learning_rate,
            'poly_coeffs': self.poly_coeffs_learning_rate,
            'fourier_low_coeffs': self.fourier_low_learning_rate,
            'fourier_mid_coeffs': self.fourier_mid_learning_rate,
            'fourier_high_coeffs': self.fourier_high_learning_rate,
        }

        if self.use_multi_model_stage:
            model_optimize_start_steps = {
                'static_sdf_params': self.model_start_steps[0],
                'poly_coeffs': self.model_start_steps[1],
                'fourier_low_coeffs': self.model_start_steps[2],
                'fourier_mid_coeffs': self.model_start_steps[3],
                'fourier_high_coeffs': self.model_start_steps[4],
            }

        if self.learn_omega:
            optim_dict['fourier_low_omega'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='dddm_fourier_low_omega'), 
                lr=self.fourier_low_omega_learning_rate,
                lr_decay=self.learning_rate_decay_steps if self.fourier_low_omega_decay is None else self.fourier_low_omega_decay,
                max_norm=self.fourier_low_omega_max_norm,
                eps=self.optimizer_epsilon      
            )

            optim_dict['fourier_mid_omega'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='dddm_fourier_mid_omega'), 
                lr=self.fourier_mid_omega_learning_rate,
                lr_decay=self.learning_rate_decay_steps if self.fourier_mid_omega_decay is None else self.fourier_mid_omega_decay,
                max_norm=self.fourier_mid_omega_max_norm,
                eps=self.optimizer_epsilon      
            )

            optim_dict['fourier_high_omega'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='dddm_fourier_high_omega'), 
                lr=self.fourier_high_omega_learning_rate,
                lr_decay=self.learning_rate_decay_steps if self.fourier_high_omega_decay is None else self.fourier_high_omega_decay,
                max_norm=self.fourier_high_omega_max_norm,
                eps=self.optimizer_epsilon      
            )

            # 记录实际学习率
            self._real_learning_rates.update({
                'fourier_low_omega': self.fourier_low_omega_learning_rate,
                'fourier_mid_omega': self.fourier_mid_omega_learning_rate,
                'fourier_high_omega': self.fourier_high_omega_learning_rate,
            })

            if self.use_multi_model_stage:
                model_optimize_start_steps.update({
                    'fourier_low_omega': 0,
                    'fourier_mid_omega': 0,
                    'fourier_high_omega': 0,
                })

        if self.learn_wavelet:
            optim_dict['wavelet_coeffs'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='dddm_wavelet_coeffs'), 
                lr=self.wavelet_learning_rate,
                lr_decay=self.learning_rate_decay_steps if self.wavelet_decay is None else self.wavelet_decay,
                max_norm=self.wavelet_max_norm,
                eps=self.optimizer_epsilon      
            )

            # 记录实际学习率
            self._real_learning_rates.update({
                'wavelet_coeffs': self.wavelet_learning_rate,
            })

            if self.use_multi_model_stage:
                model_optimize_start_steps.update({
                    'wavelet_coeffs': self.model_start_steps[-1],
                })

        assert optim_dict, "Optimizer dictionary cannot be empty"
        return DS_ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
            start_steps=model_optimize_start_steps if self.use_multi_model_stage else None,
        )

    def safe_detach(self, x):
        return x.detach() if isinstance(x, Tensor) and x.requires_grad else x

    def step(
        self,
        model: 'D_SDFFit',
        inputs: Tensor,
        gt_outputs: Tensor,
        *,
        indices: Optional[Tensor] = None,
        mode: Literal['train', 'val', 'test', 'orbit_vis', 'fix_vis'] = 'train',
        visual: Literal['pred', 'gt', 'all', 'none'] = 'none',
        experiment: Optional[DS_Experiment] = None,
        curr_step: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[List[Tensor]]]:
        (
            pred_outputs,
            pred_outputs_flows,
            reg_loss_dict
        ) = model.render_report(inputs, trainer_mode=mode)
        pred_outputs = pred_outputs.squeeze(-1)
        loss = L2Loss()(pred_outputs, gt_outputs.reshape(pred_outputs.shape))
        reg_loss = sum(reg_loss_dict.values())

        metrics = {
            'l2-loss': self.safe_detach(loss),
            'real-time-tv-loss': (self.safe_detach(reg_loss_dict['reg_time_tv']) / model.reg_time_tv_weight) if model.reg_time_tv_weight > 0 else 0.0,
            'real-coeff-tv-loss': (self.safe_detach(reg_loss_dict['reg_coeff_tv']) / model.reg_coeff_tv_weight) if model.reg_coeff_tv_weight > 0 else 0.0,
            'real-wavelet-sparse-loss': (self.safe_detach(reg_loss_dict['reg_wavelet_sparse']) / model.reg_wavelet_sparse_weight) if model.reg_wavelet_sparse_weight > 0 else 0.0,
        }

        # 可视化（仅在需要时执行）
        if visual == 'none':
            return loss + reg_loss, metrics, None
        elif visual != 'none':
            times = inputs.detach().cpu()
            times = times
            # pred_outputs_flows = pred_outputs_flows.squeeze(-1)
            pointwise_error = L2Loss()(pred_outputs, gt_outputs.reshape(pred_outputs.shape), reduction='none').mean(dim=0)  # [V]
            
            def process_and_visualize(indices: Tensor, prefix: str) -> None:
                """处理并可视化（最大或最小误差或者指定的点）。"""
                points_infos = model._get_info_for_points(indices)
                if indices.ndim == 2:
                    indices = points_infos['global_indices']
                points_sdfs = pred_outputs[:, indices]  # [times_batch, k]
                points_gt_sdfs = gt_outputs.reshape(pred_outputs.shape)[:, indices]  # [times_batch, k]
                
                for i in range(indices.shape[0]):
                    sdf_pred = points_sdfs[:, i].detach().cpu()
                    sdf_gt = points_gt_sdfs[:, i].detach().cpu()
                    single_info_dict = {
                        key: value[i].detach().cpu().item() if value[i].ndim == 0 else value[i].detach().cpu()
                        for key, value in points_infos.items()
                    }
                    grid_indices_str = f"({single_info_dict['grid_indices'][0]}, {single_info_dict['grid_indices'][1]}, {single_info_dict['grid_indices'][2]})"

                    path = experiment.dump_image(
                        subfolder=f'vispointsanalysis_step{curr_step}',
                        name=f'{prefix}_{i+1}_error_points_{grid_indices_str}',
                    )
                    plot_curve(
                        times=times, 
                        pred_data=sdf_pred, gt_data=sdf_gt, 
                        info_dict=single_info_dict, 
                        save_path=path, figsize=(10, 8)
                    )

            _, top_max_indices = torch.topk(pointwise_error, k=5)
            process_and_visualize(top_max_indices, 'top_max')
            _, top_min_indices = torch.topk(pointwise_error, k=5, largest=False)
            process_and_visualize(top_min_indices, 'top_min')
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
            process_and_visualize(track_indice, 'track')

            return loss, metrics, None

    def before_update(self, model: D_SDFFit, optimizers: DS_ModuleOptimizers, *, curr_step: int) -> None:
        """在优化器更新前调整模型参数和正则化权重。

        Args:
            model: D_SDFFit 模型。
            optimizers: 优化器集合。
            curr_step: 当前训练步数。
        """
        assert isinstance(model, D_SDFFit), f"Expected D_SDFFit, got {type(model)}"

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

        # 正则化项权重的线性衰减
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
            "reg_wavelet_sparse_able", "reg_wavelet_sparse_weight", 'linear',
            reg.curve_wavelet_sparse_weight_begin, reg.curve_wavelet_sparse_weight_end,
            reg.curve_wavelet_sparse_start_step, reg.curve_wavelet_sparse_end_step, reg.curve_wavelet_sparse_decay_steps
        )
        
        # 多阶段训练
        if self.use_multi_model_stage:
            dynamic_model_stage_dict = {}
            dynamic_model_stage_dict['static_sdf_params'] = (curr_step >= self.model_start_steps[0] and self.model_start_steps[0] >= 0)
            dynamic_model_stage_dict['sdf_curve_poly_coefficient'] = (curr_step >= self.model_start_steps[1] and self.model_start_steps[1] >= 0)
            dynamic_model_stage_dict['sdf_curve_low_freq_fourier_coefficient'] = (curr_step >= self.model_start_steps[2] and self.model_start_steps[2] >= 0)
            dynamic_model_stage_dict['sdf_curve_mid_freq_fourier_coefficient'] = (curr_step >= self.model_start_steps[3] and self.model_start_steps[3] >= 0)
            dynamic_model_stage_dict['sdf_curve_high_freq_fourier_coefficient'] = (curr_step >= self.model_start_steps[4] and self.model_start_steps[4] >= 0)
            dynamic_model_stage_dict['sdf_curve_wavelet_coefficient'] = (curr_step >= self.model_start_steps[5] and self.model_start_steps[5] >= 0)
            model.dynamic_model_stage = dynamic_model_stage_dict

    def after_update(self, model: D_SDFFit, optimizers: DS_ModuleOptimizers, *, curr_step: int) -> None:
        """在优化器更新后记录实际学习率。

        Args:
            model: D_SDFFit 模型。
            optimizers: 优化器集合。
            curr_step: 当前训练步数。
        """
        self._real_learning_rates.update({
            'static_sdf_params': optimizers.optimizers['static_sdf_params'].param_groups[0]['lr'],
            'poly_coeffs': optimizers.optimizers['poly_coeffs'].param_groups[0]['lr'],
            'fourier_low': optimizers.optimizers['fourier_low_coeffs'].param_groups[0]['lr'],
            'fourier_mid': optimizers.optimizers['fourier_mid_coeffs'].param_groups[0]['lr'],
            'fourier_high': optimizers.optimizers['fourier_high_coeffs'].param_groups[0]['lr'],
        })
        if self.learn_wavelet:
            self._real_learning_rates.update({
                'wavelet_coeffs': optimizers.optimizers['wavelet_coeffs'].param_groups[0]['lr'],
            })
        if self.learn_omega:
            self._real_learning_rates.update({
                'fourier_low_omega': optimizers.optimizers['fourier_low_omega'].param_groups[0]['lr'],
                'fourier_mid_omega': optimizers.optimizers['fourier_mid_omega'].param_groups[0]['lr'],  
                'fourier_high_omega': optimizers.optimizers['fourier_high_omega'].param_groups[0]['lr'],
            })
