from __future__ import annotations

# import modulues
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal
import matplotlib.pyplot as plt

import torch
from torch import Tensor
import numpy as np
import os
import pandas as pd
import re
import gc
from tqdm import tqdm
# import rfstudio modules
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.ui import console
from rfstudio.utils.pretty import P
from rfstudio.loss import ChamferDistanceMetric,PSNRLoss,LPIPSLoss,SSIMLoss
from rfstudio.graphics import Points, TriangleMesh
from rfstudio.graphics.shaders import PrettyShader, DepthShader, NormalShader
from rfstudio.graphics import DepthImages, VectorImages, PBRAImages
from rfstudio.utils.lazy_module import dr

# import rfstudio_ds modules
from rfstudio_ds.engine.experiment import DS_Experiment
from rfstudio_ds.engine.train import DS_TrainTask
from rfstudio_ds.data import SyntheticDynamicMultiViewBlenderRGBADataset, RealDynamicMultiviewObjectRGBADataset, CMUPanonicRGBADataset
from rfstudio_ds.model import D_Joint # rgb image loss optimization model
from rfstudio_ds.trainer import D_JointTrainer, JointRegularizationConfig # rgb image loss optimization trainer, dynamic nvdiffrec
from rfstudio_ds.graphics.spatial_sampler import NearestGridSampler
from rfstudio_ds.visualization import Visualizer
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh
from rfstudio_ds.model.density_field.components.encoding_4d import Grid4d_HashEncoding
from rfstudio_ds.nn import MLPDecoderNetwork, Grid4DDecoderNetwork
from rfstudio.io import load_float32_image, open_video_renderer, dump_float32_image
import natsort
from rfstudio_ds.visualization._plot_cube_curve import plot_cube_curve, plot_curve


cmu_band1_task = DS_TrainTask(
    dataset=CMUPanonicRGBADataset(
        path=Path("data/sdfflow/band1/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=0.6,
        low_freq_fourier_bands=[1, 3],
        mid_freq_fourier_bands=[4, 9],
        high_freq_fourier_bands=[10,18],

        dynamic_texture=Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 96, 32, 6],
                activation='sigmoid',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[16,16,2],
            deform_desired_resolution=[4096,4096,32],
            deform_num_levels=32,
        ),

        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-2],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_cmu/band1', timestamp='test_s1'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=2,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_save=1000,
        hold_after_train=False,

        geometry_lr=3e-3, # 用于flexicube的 deform 和 weight 参数
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

        appearance_learning_rate=5e-3,
        appearance_decay=1500,

        light_learning_rate=1e-3, # 用于light的参数
        light_learning_rate_decay=800,

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
            sdf_entropy_weight_end = 0.05,
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

            # curve_coeff_tv loss
            curve_coeff_tv_weight_begin = 0.1, # 控制curve coefficient tv smooth loss的权重
            curve_coeff_tv_weight_end = 10,
            curve_coeff_tv_decay_steps = 1250, # 控制curve coefficient tv smooth loss的线性衰减步数
            curve_coeff_tv_start_step = 0, # 控制curve coefficient tv smooth loss的起始步数
            curve_coeff_tv_end_step=-1, # 控制curve coefficient tv smooth loss的终止步数

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
        full_test_after_train=False,
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)

cmu_cello1_task = DS_TrainTask(
    dataset=CMUPanonicRGBADataset(
        path=Path("data/sdfflow/cello1/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=1.2,
        low_freq_fourier_bands=[1, 3],
        mid_freq_fourier_bands=[4, 9],
        high_freq_fourier_bands=[10,18],

        dynamic_texture=Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 96, 32, 6],
                activation='sigmoid',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[16,16,2],
            deform_desired_resolution=[4096,4096,32],
            deform_num_levels=32,
        ),

        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-2],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_cmu/cello1', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=2,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_save=1000,
        hold_after_train=True,

        geometry_lr=3e-3, # 用于flexicube的 deform 和 weight 参数
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

        appearance_learning_rate=5e-3,
        appearance_decay=1500,

        light_learning_rate=1e-3, # 用于light的参数
        light_learning_rate_decay=800,

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
            sdf_entropy_weight_end = 0.05,
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

            # curve_coeff_tv loss
            curve_coeff_tv_weight_begin = 0.1, # 控制curve coefficient tv smooth loss的权重
            curve_coeff_tv_weight_end = 10,
            curve_coeff_tv_decay_steps = 1250, # 控制curve coefficient tv smooth loss的线性衰减步数
            curve_coeff_tv_start_step = 0, # 控制curve coefficient tv smooth loss的起始步数
            curve_coeff_tv_end_step=-1, # 控制curve coefficient tv smooth loss的终止步数

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
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)

cmu_hanggling_b2_task = DS_TrainTask(
    dataset=CMUPanonicRGBADataset(
        path=Path("data/sdfflow/hanggling_b2/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=1.5,
        low_freq_fourier_bands=[1, 3],
        mid_freq_fourier_bands=[4, 9],
        high_freq_fourier_bands=[10,18],

        dynamic_texture=Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 96, 32, 6],
                activation='sigmoid',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[16,16,2],
            deform_desired_resolution=[4096,4096,32],
            deform_num_levels=32,
        ),

        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-2],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_cmu/hanggling_b2', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=2,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_save=1000,
        hold_after_train=True,

        geometry_lr=3e-3, # 用于flexicube的 deform 和 weight 参数
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

        appearance_learning_rate=5e-3,
        appearance_decay=1500,

        light_learning_rate=1e-3, # 用于light的参数
        light_learning_rate_decay=800,

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
            sdf_entropy_weight_end = 0.05,
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

            # curve_coeff_tv loss
            curve_coeff_tv_weight_begin = 0.1, # 控制curve coefficient tv smooth loss的权重
            curve_coeff_tv_weight_end = 10,
            curve_coeff_tv_decay_steps = 1250, # 控制curve coefficient tv smooth loss的线性衰减步数
            curve_coeff_tv_start_step = 0, # 控制curve coefficient tv smooth loss的起始步数
            curve_coeff_tv_end_step=-1, # 控制curve coefficient tv smooth loss的终止步数

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
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)

cmu_ian3_task = DS_TrainTask(
    dataset=CMUPanonicRGBADataset(
        path=Path("data/sdfflow/ian3/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=1.2,
        low_freq_fourier_bands=[1, 3],
        mid_freq_fourier_bands=[4, 9],
        high_freq_fourier_bands=[10,18],

        dynamic_texture=Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 96, 32, 6],
                activation='sigmoid',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[16,16,2],
            deform_desired_resolution=[4096,4096,32],
            deform_num_levels=32,
        ),

        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-2],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_cmu/ian3', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=2,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_save=1000,
        hold_after_train=True,

        geometry_lr=3e-3, # 用于flexicube的 deform 和 weight 参数
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

        appearance_learning_rate=5e-3,
        appearance_decay=1500,

        light_learning_rate=1e-3, # 用于light的参数
        light_learning_rate_decay=800,

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
            sdf_entropy_weight_end = 0.05,
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

            # curve_coeff_tv loss
            curve_coeff_tv_weight_begin = 0.1, # 控制curve coefficient tv smooth loss的权重
            curve_coeff_tv_weight_end = 10,
            curve_coeff_tv_decay_steps = 1250, # 控制curve coefficient tv smooth loss的线性衰减步数
            curve_coeff_tv_start_step = 0, # 控制curve coefficient tv smooth loss的起始步数
            curve_coeff_tv_end_step=-1, # 控制curve coefficient tv smooth loss的终止步数

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
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)

cmu_pizza1_task = DS_TrainTask(
    dataset=CMUPanonicRGBADataset(
        path=Path("data/sdfflow/pizza1/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=1.2,
        low_freq_fourier_bands=[1, 3],
        mid_freq_fourier_bands=[4, 9],
        high_freq_fourier_bands=[10,18],

        dynamic_texture=Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 96, 32, 6],
                activation='sigmoid',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[16,16,2],
            deform_desired_resolution=[4096,4096,32],
            deform_num_levels=32,
        ),

        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-2],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_cmu/pizza1', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=2,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_save=1000,
        hold_after_train=True,

        geometry_lr=3e-3, # 用于flexicube的 deform 和 weight 参数
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

        appearance_learning_rate=5e-3,
        appearance_decay=1500,

        light_learning_rate=1e-3, # 用于light的参数
        light_learning_rate_decay=800,

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
            sdf_entropy_weight_end = 0.05,
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

            # curve_coeff_tv loss
            curve_coeff_tv_weight_begin = 0.1, # 控制curve coefficient tv smooth loss的权重
            curve_coeff_tv_weight_end = 10,
            curve_coeff_tv_decay_steps = 1250, # 控制curve coefficient tv smooth loss的线性衰减步数
            curve_coeff_tv_start_step = 0, # 控制curve coefficient tv smooth loss的起始步数
            curve_coeff_tv_end_step=-1, # 控制curve coefficient tv smooth loss的终止步数

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
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)

zjumocap_task = DS_TrainTask(
    dataset=CMUPanonicRGBADataset(
        path=Path("data/zju-mocap/CoreView_394_blender/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=48,
        geometry_scale=2,
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
            deform_base_resolution=[16,16,2],
            deform_desired_resolution=[4096,4096,256],
            deform_num_levels=32,
        ),

        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_zjumocap/case1', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=2,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_save=1000,
        hold_after_train=True,

        geometry_lr=3e-3, # 用于flexicube的 deform 和 weight 参数
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

        appearance_learning_rate=5e-3,
        appearance_decay=1500,

        light_learning_rate=1e-3, # 用于light的参数
        light_learning_rate_decay=800,

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
            sdf_entropy_weight_end = 0.05,
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

            # curve_coeff_tv loss
            curve_coeff_tv_weight_begin = 0.1, # 控制curve coefficient tv smooth loss的权重
            curve_coeff_tv_weight_end = 10,
            curve_coeff_tv_decay_steps = 1250, # 控制curve coefficient tv smooth loss的线性衰减步数
            curve_coeff_tv_start_step = 0, # 控制curve coefficient tv smooth loss的起始步数
            curve_coeff_tv_end_step=-1, # 控制curve coefficient tv smooth loss的终止步数

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
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)


@dataclass
class Extract_results(Task):
    step: Optional[int] = None

    load: Optional[Path] = None

    def process_loader(self, split_name, export_pred_light=False):
        full_batch_size = self.dataset.dataparser.model.num_frames
        loader = self.dataset.get_test_iter(batch_size=25, shuffle=False, infinite=False)
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )

        # 初始化存储容器
        gt_images = []
        pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
        pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs = [], [], [], []

        iter_count = 0
        for inputs, gt_outputs, indices in loader:
            _, _, visualization, pred_meshes = self.trainer.step(
                self.model,
                inputs,
                gt_outputs,
                indices=indices,
                mode='test',
                visual=True,
                val_pbr_attr=True,
                return_pred_mesh=True,
            )
            
            for i in range(len(visualization)):
                count_id = iter_count * 25 + i
                view_num = count_id // full_batch_size
                if view_num > 5:
                    break
                split_name = f'test_view{view_num}'
                frame_id = count_id % full_batch_size
                
                camera = inputs[i]
                vis = visualization[i]
                vis_hight, vis_width, _ = vis.shape
                image_hight = int(vis_hight / 4)
                image_width = int(vis_width / 3)

                gt_images.append(vis[:image_hight, :image_width, :3].cpu())
                pred_images.append(vis[image_hight:2*image_hight, :image_width, :3].cpu())
                pred_normals.append(vis[image_hight:2*image_hight, image_width:2*image_width, :3].cpu())
                pred_depths.append(vis[image_hight:2*image_hight, 2*image_width:, :3].cpu())
                pred_kds.append(vis[2*image_hight:3*image_hight, :image_width, :3].cpu())
                if export_pred_light:
                    self.experiment.dump_image('eval/extract', image=vis[2*image_hight:3*image_hight, image_width:, :3], name='pred_light')
                    export_pred_light = False
                pbr_roughnesses.append(vis[3*image_hight:, :image_width, :3].cpu())
                pbr_metallics.append(vis[3*image_hight:, image_width:2*image_width, :3].cpu())
                pbr_occs.append(vis[3*image_hight:, 2*image_width:, :3].cpu())

                pred_mesh = pred_meshes[i].clone()
                pred_pretty_meshes.append(
                    pred_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                )

                if view_num == 0:
                    pred_mesh.export(
                        path=self.experiment.dump_file_path(subfolder='eval/pred_mesh', file_name=f'frame{frame_id}.obj'),
                        only_geometry=True
                    )

                # 保存 GT 图像
                for idx, (img, sub) in enumerate([
                    (gt_images[-1], 'gt_image'), 
                ]):
                    self.experiment.dump_image(f'eval/extract/{split_name}/gt/{sub}', image=img, name=f'{frame_id}')

                # 保存 Pred 图像
                for idx, (img, sub) in enumerate([
                    (pred_images[-1], 'pred_image'), 
                    (pred_normals[-1], 'pred_normal'), 
                    (pred_depths[-1], 'pred_depth'),
                    (pred_pretty_meshes[-1], 'pred_mesh'),
                    (pred_kds[-1], 'pbr_kd'), 
                    (pbr_roughnesses[-1], 'pbr_roughness'), 
                    (pbr_metallics[-1], 'pbr_metallic'),
                    (pbr_occs[-1], 'pbr_occ')
                ]):
                    self.experiment.dump_image(f'eval/extract/{split_name}/pred/{sub}', image=img, name=f'{frame_id}')

                if frame_id == full_batch_size - 1:
                    # 导出视频
                    video_dict = {
                        'gt_image': gt_images,
                        'pred_image': pred_images,
                        'pred_mesh': pred_pretty_meshes,
                        'pred_normal': pred_normals,
                        'pred_depth': pred_depths,
                        'pbr_kd': pred_kds,
                        'pbr_roughness': pbr_roughnesses,
                        'pbr_metallic': pbr_metallics,
                        'pbr_occ': pbr_occs
                    }
                    for name, imgs in video_dict.items():
                        self.experiment.dump_images2video(
                            f'eval/extract/test_view{view_num}',
                            name=name, 
                            images=imgs, 
                            downsample=1, 
                            fps=8,
                        )
                    del gt_images, pred_images, pred_normals, pred_depths, pred_pretty_meshes, pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs
                    torch.cuda.empty_cache()
                    gc.collect()
                    gt_images = []
                    pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
                    pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs = [], [], [], []

            iter_count += 1
            torch.cuda.empty_cache()
                
        del gt_images, pred_images, pred_normals, pred_depths, pred_pretty_meshes, pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs, loader
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def run(self) -> None:

        print(f'Processing: {self.load}...')
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            self.experiment: DS_Experiment = task.experiment
            self.model: D_Joint = task.model
            self.model.background_color = 'white'
            self.dataset: CMUPanonicRGBADataset = task.dataset
            self.trainer: D_JointTrainer = task.trainer
            self.trainer.setup(model=self.model, dataset=self.dataset)
            self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
        self.model.eval()

        self.process_loader('test_view', export_pred_light=True)

        del self.model, self.dataset, self.trainer, self.experiment
        torch.cuda.empty_cache()
        gc.collect()


@dataclass
class Eval_results(Task):
    """渲染训练好的模型"""
    step: Optional[int] = None

    load: Optional[Path] = None

    def process_loader(self):
        full_batch_size = self.dataset.dataparser.model.num_frames
        loader = self.dataset.get_test_iter(batch_size=25, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='test')

        # 初始化存储容器
        chamfer_dists = []
        
        iter_count = 0
        for inputs, gt_outputs, indices in loader:
            _, _, visualization, pred_meshes = self.trainer.step(
                self.model,
                inputs,
                gt_outputs,
                indices=indices,
                mode='test',
                visual=True,
                val_pbr_attr=True,
                return_pred_mesh=True,
            )
            
            for i in range(len(visualization)):
                count_id = iter_count * 25 + i
                view_num = count_id // full_batch_size
                if view_num > 0:
                    break
                frame_id = count_id % full_batch_size

                pred_mesh = pred_meshes[i].clone()
                gt_mesh = gt_meshes[frame_id].clone()

                chamfer_dist = ChamferDistanceMetric(target_num_points=1000000)(gt_mesh, pred_mesh)
                chamfer_dists.append(chamfer_dist)
                self.experiment.log(f"Frame {frame_id}: Chamfer Distance: {chamfer_dist}",new_logfile='eval.txt')
                # pred_mesh.export(path=Path('temp_pred.obj'),only_geometry=True)
                # gt_mesh.export(path=Path('temp_gt.ply'))
            break
            iter_count += 1
            torch.cuda.empty_cache()

        mean_chamfer_dist = torch.stack(chamfer_dists).mean().item()
        self.experiment.log(f"Mean Chamfer Distance: {mean_chamfer_dist}")
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def run(self) -> None:
        print(f'Processing: {self.load}...')
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            self.experiment: DS_Experiment = task.experiment
            self.model: D_Joint = task.model
            self.dataset: CMUPanonicRGBADataset = task.dataset
            self.trainer: D_JointTrainer = task.trainer
            self.trainer.setup(model=self.model, dataset=self.dataset)
            self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
        self.model.eval()

        self.process_loader()
        self.experiment.parse_log_auto(self.experiment.log_path) # 分析日志

        del self.model, self.dataset, self.trainer, self.experiment
        torch.cuda.empty_cache()
        gc.collect()

@dataclass
class Gather_results(Task):
    step: Optional[int] = None

    load: Optional[Path] = None

    def process_loader(self, split_name, export_pred_light=False):
        full_batch_size = self.dataset.dataparser.model.num_frames
        loader = self.dataset.get_test_iter(batch_size=25, shuffle=False, infinite=False)
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )

        # 初始化存储容器
        gt_images = []
        pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
        pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs = [], [], [], []

        iter_count = 0
        from tqdm import tqdm
        with tqdm(total=full_batch_size * 6, desc="Evaluating and Extracting results", unit="frame") as pbar:
            for inputs, gt_outputs, indices in loader:
                _, _, visualization, pred_meshes = self.trainer.step(
                    self.model,
                    inputs,
                    gt_outputs,
                    indices=indices,
                    mode='test',
                    visual=True,
                    val_pbr_attr=True,
                    return_pred_mesh=True,
                )
                
                for i in range(len(visualization)):
                    count_id = iter_count * 25 + i
                    view_num = count_id // full_batch_size
                    if view_num > 5:
                        break
                    split_name = f'test_view{view_num}'
                    frame_id = count_id % full_batch_size
                    
                    camera = inputs[i]
                    vis = visualization[i]
                    vis_hight, vis_width, _ = vis.shape
                    image_hight = int(vis_hight / 4)
                    image_width = int(vis_width / 3)

                    gt_images.append(vis[:image_hight, :image_width, :3].cpu())
                    pred_images.append(vis[image_hight:2*image_hight, :image_width, :3].cpu())
                    pred_normals.append(vis[image_hight:2*image_hight, image_width:2*image_width, :3].cpu())
                    pred_depths.append(vis[image_hight:2*image_hight, 2*image_width:, :3].cpu())
                    pred_kds.append(vis[2*image_hight:3*image_hight, :image_width, :3].cpu())
                    if export_pred_light:
                        self.experiment.dump_image(f'{self.load.parent.parent}/gather_image', image=vis[2*image_hight:3*image_hight, image_width:, :3], name='pred_light')
                        export_pred_light = False
                    pbr_roughnesses.append(vis[3*image_hight:, :image_width, :3].cpu())
                    pbr_metallics.append(vis[3*image_hight:, image_width:2*image_width, :3].cpu())
                    pbr_occs.append(vis[3*image_hight:, 2*image_width:, :3].cpu())

                    pred_mesh = pred_meshes[i].clone()
                    pred_pretty_meshes.append(
                        pred_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                    )

                    # 保存 GT 图像
                    for idx, (img, sub) in enumerate([
                        (gt_images[-1], 'gt_image'), 
                    ]):
                        self.experiment.dump_image(
                            f'{self.load.parent.parent}/gather_image/test_view{view_num}/frame_{frame_id}',
                            image=img, 
                            name=f'{sub}'
                        )
                    # 保存 Pred 图像
                    for idx, (img, sub) in enumerate([
                        (pred_images[-1], 'pred_image'), 
                        (pred_normals[-1], 'pred_normal'), 
                        # (pred_depths[-1], 'pred_depth'),
                        (pred_pretty_meshes[-1], 'pred_mesh'),
                        (pred_kds[-1], 'pbr_kd'), 
                        (pbr_roughnesses[-1], 'pbr_roughness'), 
                        (pbr_metallics[-1], 'pbr_metallic'),
                        (pbr_occs[-1], 'pbr_occ')
                    ]):
                        self.experiment.dump_image(
                            f'{self.load.parent.parent}/gather_image/{split_name}/frame_{frame_id}',
                            image=img, 
                            name=f'psdf_{sub}'
                        )
                    if frame_id == full_batch_size - 1:
                        del gt_images, pred_images, pred_normals, pred_depths, pred_pretty_meshes, pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs
                        torch.cuda.empty_cache()
                        gc.collect()
                        gt_images = []
                        pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
                        pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs = [], [], [], []
                    pbar.update(1)
                iter_count += 1
                torch.cuda.empty_cache()
                
        del gt_images, pred_images, pred_normals, pred_depths, pred_pretty_meshes, pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs, loader
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def run(self) -> None:

        print(f'Processing: {self.load}...')
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            self.experiment: DS_Experiment = task.experiment
            self.model: D_Joint = task.model
            self.model.background_color = 'white'
            self.dataset: CMUPanonicRGBADataset = task.dataset
            self.trainer: D_JointTrainer = task.trainer
            self.trainer.setup(model=self.model, dataset=self.dataset)
            self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
        self.model.eval()

        self.process_loader('test_view', export_pred_light=True)

        del self.model, self.dataset, self.trainer, self.experiment
        torch.cuda.empty_cache()
        gc.collect()

@dataclass
class Gather_mesh(Task):
    step: Optional[int] = None

    load: Optional[Path] = None

    baseline_visual: Optional[Path] = None

    baseline_mesh: Optional[Path] = None
    
    gt_mesh: Optional[Path] = None
    
    method_name: Optional[str] = None

    def process_loader(self):
        full_batch_size = self.dataset.dataparser.model.num_frames
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)

        iter_count = 0
        with tqdm(total=full_batch_size * 6, desc="Evaluating and Extracting results", unit="frame") as pbar:
            for inputs, gt_outputs, _ in loader:

                for i in range(len(inputs)):
                    count_id = iter_count * 50 + i
                    view_num = count_id // full_batch_size
                    if view_num >= 1:
                        break
                    frame_id = count_id % full_batch_size

                    pred_mesh_path = self.baseline_mesh / f'frame{frame_id}.obj'
                    pred_mesh_pkl_path = pred_mesh_path.with_suffix('.pkl')
                    if pred_mesh_pkl_path.exists():
                        pred_mesh = DS_TriangleMesh.deserialize(pred_mesh_pkl_path).to(self.device)
                    else:
                        pred_mesh = DS_TriangleMesh.from_file(pred_mesh_path, read_mtl=False).to(self.device)
                        pred_mesh.serialize(pred_mesh_pkl_path)
                    inv_trans = torch.tensor([[1,0,0],[0,-1,0],[0,0,-1]]).to(pred_mesh.vertices)
                    pred_mesh.replace_(vertices=(inv_trans@(pred_mesh.vertices * 2).unsqueeze(-1)).squeeze(-1)) # * 2 是把0.5的范围映射到1
                    pred_mesh.export(
                        path=self.experiment.dump_file_path(subfolder=f'{self.load.parent.parent}/gather_mesh/frame_{frame_id}', file_name=f'{self.method_name}.obj'),
                        only_geometry=True
                    )

                    pbar.update(1)
                iter_count += 1
                torch.cuda.empty_cache()

    @torch.no_grad()
    def run(self) -> None:
        print(f'Processing: {self.load}...')
        print(f'Baseline: {self.baseline_visual}')
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step, load_checkpoint=False)
            self.experiment: DS_Experiment = task.experiment
            self.model: D_Joint = task.model
            self.model.background_color = 'white'
            self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
            self.trainer: D_JointTrainer = task.trainer
            self.trainer.setup(model=self.model, dataset=self.dataset)
            self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)

        self.process_loader()
        torch.cuda.empty_cache()



if __name__ == '__main__':
    TaskGroup(
        # train task
        band1 = cmu_band1_task,
        cello1 = cmu_cello1_task,
        hanggling2 = cmu_hanggling_b2_task,
        ian3 = cmu_ian3_task,
        pizza1 = cmu_pizza1_task,
        
        zjumocap1 = zjumocap_task,
        
        # eval task
        extract = Extract_results(cuda=0),
        eval = Eval_results(cuda=0),
        gather = Gather_results(cuda=0),
        gathermesh = Gather_mesh(cuda=0),
    ).run()
