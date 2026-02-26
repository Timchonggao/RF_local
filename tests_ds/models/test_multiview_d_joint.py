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

toy_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/ObjSel-Dyn/toy/"),
        # costume_padding_size=15
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
            deform_desired_resolution=[4096,4096,128],
            deform_num_levels=32,
        ),

        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_joint/toy', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
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
        full_fix_vis_after_train=True,
        full_orbit_vis_after_train=True,
        detect_anomaly=True,
    ),
    cuda=0,
    seed=1
)

cat_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/ObjSel-Dyn/cat/"),
        # costume_padding_size=1
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=0.5,
        # low_freq_fourier_bands=[1, 10],
        # mid_freq_fourier_bands=[11, 20],
        # high_freq_fourier_bands=[21,30],
        low_freq_fourier_bands=[1, 30],
        mid_freq_fourier_bands=[31, 60],
        high_freq_fourier_bands=[61,100],
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
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_joint/cat', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=8000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
        num_steps_per_save=1000,
        hold_after_train=False,

        use_multi_model_stage=True,
        model_start_steps=[0,0,0,2000,4000],  # static, poly, fourier_low, fourier_mid, fourier_high

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
        appearance_decay=1250,

        light_learning_rate=1e-3, # 用于light的参数
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
        full_fix_vis_after_train=True,
        full_orbit_vis_after_train=True,
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)

deer_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/ObjSel-Dyn/deer/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=0.5,
        # low_freq_fourier_bands=[1, 10],
        # mid_freq_fourier_bands=[11, 20],
        # high_freq_fourier_bands=[21,30],
        low_freq_fourier_bands=[1, 30],
        mid_freq_fourier_bands=[31, 60],
        high_freq_fourier_bands=[61,100],
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
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_joint/deer', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
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
        appearance_decay=1250,

        light_learning_rate=1e-3, # 用于light的参数
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

            psnr_weight_begin = 1., # psnr loss
            psnr_weight_end = 1.,
            psnr_weight_decay_steps = 1250,
            psnr_weight_start_step = -1,

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

footballplayer_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/ObjSel-Dyn/football_player/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=0.5,
        # low_freq_fourier_bands=[1, 10],
        # mid_freq_fourier_bands=[11, 20],
        # high_freq_fourier_bands=[21,30],
        low_freq_fourier_bands=[1, 30],
        mid_freq_fourier_bands=[31, 60],
        high_freq_fourier_bands=[61,100],
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
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 5e-3],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 5e-3],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 5e-3],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_joint/footballplayer', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
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
        appearance_decay=1250,

        light_learning_rate=1e-3, # 用于light的参数
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
        full_fix_vis_after_train=True,
        full_orbit_vis_after_train=True,
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)

lego_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/ObjSel-Dyn/lego/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=0.5,
        # low_freq_fourier_bands=[1, 10],
        # mid_freq_fourier_bands=[11, 20],
        # high_freq_fourier_bands=[21,30],
        low_freq_fourier_bands=[1, 50],
        mid_freq_fourier_bands=[51, 100],
        high_freq_fourier_bands=[101,150],
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
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_joint/lego', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=8000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
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
        appearance_decay=1250,

        light_learning_rate=1e-3, # 用于light的参数
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

            psnr_weight_begin = 1., # psnr loss
            psnr_weight_end = 1.,
            psnr_weight_decay_steps = 1250,
            psnr_weight_start_step = -1,

            # sdf entropy loss
            sdf_entropy_weight_begin = 0.2, # 控制sdf entropy loss的权重
            sdf_entropy_weight_end = 0.01,
            sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
            sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
            sdf_entropy_end_step = -1, # 控制sdf entropy loss的终止步数

            # curve derivative smooth loss
            time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
            time_tv_weight_end = 0.01,
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

rabbit_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/ObjSel-Dyn/rabbit/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=0.5,
        # low_freq_fourier_bands=[1, 10],
        # mid_freq_fourier_bands=[11, 20],
        # high_freq_fourier_bands=[21,30],
        low_freq_fourier_bands=[1, 30],
        mid_freq_fourier_bands=[31, 60],
        high_freq_fourier_bands=[61,100],
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
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_joint/rabbit', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
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
        appearance_decay=1250,

        light_learning_rate=1e-3, # 用于light的参数
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

            psnr_weight_begin = 1., # psnr loss
            psnr_weight_end = 1.,
            psnr_weight_decay_steps = 1250,
            psnr_weight_start_step = -1,

            # sdf entropy loss
            sdf_entropy_weight_begin = 0.2, # 控制sdf entropy loss的权重
            sdf_entropy_weight_end = 0.01,
            sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
            sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
            sdf_entropy_end_step = -1, # 控制sdf entropy loss的终止步数

            # curve derivative smooth loss
            time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
            time_tv_weight_end = 0.01,
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

spidermanfight_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/ObjSel-Dyn/spiderman_fight/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=0.5,
        # low_freq_fourier_bands=[1, 10],
        # mid_freq_fourier_bands=[11, 20],
        # high_freq_fourier_bands=[21,30],
        low_freq_fourier_bands=[1, 30],
        mid_freq_fourier_bands=[31, 60],
        high_freq_fourier_bands=[61,100],
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
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 7e-3],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 7e-3],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 7e-3],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_joint/spidermanfight', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
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

        appearance_learning_rate=3e-2,
        appearance_decay=1500,

        light_learning_rate=5e-3, # 用于light的参数
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
        full_fix_vis_after_train=True,
        full_orbit_vis_after_train=True,
        detect_anomaly=True
    ),
    cuda=0,
    seed=1
)

penguin_task = DS_TrainTask(
    dataset=RealDynamicMultiviewObjectRGBADataset(
        path=Path("data/diva360/penguin/"),
    ),
    model=D_Joint(
        geometry='DD_isocubes',
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
            deform_desired_resolution=[4096,4096,128],
            deform_num_levels=32,
        ),

        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='realobject_mv_d_joint/penguin', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
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

blue_car_task = DS_TrainTask(
    dataset=RealDynamicMultiviewObjectRGBADataset(
        path=Path("data/diva360/blue_car/"),
    ),
    model=D_Joint(
        geometry='DD_isocubes',
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
            deform_desired_resolution=[4096,4096,128],
            deform_num_levels=32,
        ),

        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='realobject_mv_d_joint/blue_car', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
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

dog_task = DS_TrainTask(
    dataset=RealDynamicMultiviewObjectRGBADataset(
        path=Path("data/diva360/dog/"),
    ),
    model=D_Joint(
        geometry='DD_isocubes',
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
            deform_desired_resolution=[4096,4096,128],
            deform_num_levels=32,
        ),

        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='realobject_mv_d_joint/dog', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
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

k1_double_punch_task = DS_TrainTask(
    dataset=RealDynamicMultiviewObjectRGBADataset(
        path=Path("data/diva360/k1_double_punch/"),
    ),
    model=D_Joint(
        geometry='DD_isocubes',
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
            deform_desired_resolution=[4096,4096,128],
            deform_num_levels=32,
        ),

        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='realobject_mv_d_joint/k1_double_punch', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
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

wolf_task = DS_TrainTask(
    dataset=RealDynamicMultiviewObjectRGBADataset(
        path=Path("data/diva360/wolf/"),
    ),
    model=D_Joint(
        geometry='DD_isocubes',
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
            deform_desired_resolution=[4096,4096,128],
            deform_num_levels=32,
        ),

        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='realobject_mv_d_joint/wolf', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
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

cmu_pose1_task = DS_TrainTask(
    dataset=CMUPanonicRGBADataset(
        path=Path("data/cmu_panonic/171204_pose1_sample/"),
    ),
    model=D_Joint(
        geometry='DD_isocubes',
        geometry_resolution=96,
        geometry_scale=1,
        low_freq_fourier_bands=[1, 20],
        mid_freq_fourier_bands=[21, 40],
        high_freq_fourier_bands=[41,60],

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
    experiment=DS_Experiment(name='cmupanonic_mv_d_joint/pose1', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
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

cmu_cello1_task = DS_TrainTask(
    dataset=CMUPanonicRGBADataset(
        path=Path("data/sdfflow/cello1/"),
    ),
    model=D_Joint(
        geometry='DD_isocubes',
        geometry_resolution=96,
        geometry_scale=1,
        low_freq_fourier_bands=[1, 20],
        mid_freq_fourier_bands=[21, 40],
        high_freq_fourier_bands=[41,60],

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
    experiment=DS_Experiment(name='cmupanonic_mv_d_joint/cello1', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=4,
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

cmu_band1_task = DS_TrainTask(
    dataset=CMUPanonicRGBADataset(
        path=Path("data/sdfflow/band1/"),
    ),
    model=D_Joint(
        geometry='DD_isocubes',
        geometry_resolution=96,
        geometry_scale=0.6,
        low_freq_fourier_bands=[1, 20],
        mid_freq_fourier_bands=[21, 40],
        high_freq_fourier_bands=[41,60],

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
    experiment=DS_Experiment(name='cmupanonic_mv_d_joint/band1', timestamp='test'),
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
        geometry='DD_isocubes',
        geometry_resolution=96,
        geometry_scale=1.5,
        low_freq_fourier_bands=[1, 20],
        mid_freq_fourier_bands=[21, 40],
        high_freq_fourier_bands=[41,60],

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
    experiment=DS_Experiment(name='cmupanonic_mv_d_joint/hanggling_b2', timestamp='test'),
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
        geometry='DD_isocubes',
        geometry_resolution=96,
        geometry_scale=1,
        low_freq_fourier_bands=[1, 20],
        mid_freq_fourier_bands=[21, 40],
        high_freq_fourier_bands=[41,60],

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
    experiment=DS_Experiment(name='cmupanonic_mv_d_joint/ian3', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=4,
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
        geometry='DD_isocubes',
        geometry_resolution=96,
        geometry_scale=1,
        low_freq_fourier_bands=[1, 20],
        mid_freq_fourier_bands=[21, 40],
        high_freq_fourier_bands=[41,60],

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
    experiment=DS_Experiment(name='cmupanonic_mv_d_joint/pizza1', timestamp='test'),
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


eagle_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/multiview_dynamic_blender/eagle/"),
        # costume_sample_frames=[0,239],
        # costume_padding_size=1
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=0.5,
        # low_freq_fourier_bands=[1, 10],
        # mid_freq_fourier_bands=[11, 20],
        # high_freq_fourier_bands=[21,30],
        low_freq_fourier_bands=[1, 30],
        mid_freq_fourier_bands=[31, 60],
        high_freq_fourier_bands=[61,100],
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
    experiment=DS_Experiment(name='blender_mv_d_joint/eagle', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
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
        appearance_decay=1250,

        light_learning_rate=1e-3, # 用于light的参数
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

            psnr_weight_begin = 1., # psnr loss
            psnr_weight_end = 1.,
            psnr_weight_decay_steps = 1250,
            psnr_weight_start_step = -1,

            # sdf entropy loss
            sdf_entropy_weight_begin = 0.2, # 控制sdf entropy loss的权重
            sdf_entropy_weight_end = 0.01,
            sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
            sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
            sdf_entropy_end_step = -1, # 控制sdf entropy loss的终止步数

            # curve derivative smooth loss
            time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
            time_tv_weight_end = 0.005,
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

girlwalk_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/multiview_dynamic_blender/girlwalk/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=0.5,
        # low_freq_fourier_bands=[1, 10],
        # mid_freq_fourier_bands=[11, 20],
        # high_freq_fourier_bands=[21,30],
        low_freq_fourier_bands=[1, 30],
        mid_freq_fourier_bands=[31, 60],
        high_freq_fourier_bands=[61,100],
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
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 5e-3],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 5e-3],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 5e-3],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_joint/girlwalk', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
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
        appearance_decay=1250,

        light_learning_rate=1e-3, # 用于light的参数
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

            psnr_weight_begin = 1., # psnr loss
            psnr_weight_end = 1.,
            psnr_weight_decay_steps = 1250,
            psnr_weight_start_step = -1,

            # sdf entropy loss
            sdf_entropy_weight_begin = 0.2, # 控制sdf entropy loss的权重
            sdf_entropy_weight_end = 0.01,
            sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
            sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
            sdf_entropy_end_step = -1, # 控制sdf entropy loss的终止步数

            # curve derivative smooth loss
            time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
            time_tv_weight_end = 0.01,
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

spidermanwalk_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/multiview_dynamic_blender/spidermanwalk/"),
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
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 2e-2],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 2e-2],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 2e-2],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_joint/spidermanwalk', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
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

            psnr_weight_begin = 1., # psnr loss
            psnr_weight_end = 1.,
            psnr_weight_decay_steps = 1250,
            psnr_weight_start_step = -1,

            # sdf entropy loss
            sdf_entropy_weight_begin = 0.2, # 控制sdf entropy loss的权重
            sdf_entropy_weight_end = 0.001,
            sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
            sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
            sdf_entropy_end_step = -1, # 控制sdf entropy loss的终止步数

            # curve derivative smooth loss
            time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
            time_tv_weight_end = 0.01,
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

excavator_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/ObjSel-Dyn/excavator/"),
        # costume_sample_frames=[0,70],
        # costume_padding_size=15
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
            deform_desired_resolution=[4096,4096,128],
            deform_num_levels=32,
        ),
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 9e-3],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 9e-3],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 9e-3],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_joint/excavator', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
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

        light_learning_rate=5e-3, # 用于light的参数
        light_learning_rate_decay=800,

        regularization=JointRegularizationConfig(
            ssim_weight_begin = 1., # ssim loss
            ssim_weight_end  = 10.,
            ssim_weight_decay_steps = 1250,
            ssim_weight_start_step = 0,

            psnr_weight_begin = 1., # psnr loss
            psnr_weight_end = 1.,
            psnr_weight_decay_steps = 1250,
            psnr_weight_start_step = -1,

            mask_weight_begin = 10., # mask supervision loss
            mask_weight_end = 10.,
            mask_weight_decay_steps = 1250,
            mask_weight_start_step = 0,

            # sdf entropy loss
            sdf_entropy_weight_begin = 0.2, # 控制sdf entropy loss的权重
            sdf_entropy_weight_end = 0.001,
            sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
            sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
            sdf_entropy_end_step = -1, # 控制sdf entropy loss的终止步数

            # curve derivative smooth loss
            time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
            time_tv_weight_end = 0.001,
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

            reg_light_begin=0.01,
            reg_light_end=0.05,
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

monsterroar_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/multiview_dynamic_blender/monster_roar/"),
        # costume_sample_frames=[0,70],
        # costume_padding_size=15
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=0.5,
        # low_freq_fourier_bands=[1, 10],
        # mid_freq_fourier_bands=[11, 20],
        # high_freq_fourier_bands=[21,30],
        low_freq_fourier_bands=[1, 30],
        mid_freq_fourier_bands=[31, 60],
        high_freq_fourier_bands=[61,100],
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
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 6e-3],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 6e-3],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 6e-3],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_joint/monster_roar', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
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

            psnr_weight_begin = 1., # psnr loss
            psnr_weight_end = 1.,
            psnr_weight_decay_steps = 1250,
            psnr_weight_start_step = -1,

            # sdf entropy loss
            sdf_entropy_weight_begin = 0.2, # 控制sdf entropy loss的权重
            sdf_entropy_weight_end = 0.001,
            sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
            sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
            sdf_entropy_end_step = -1, # 控制sdf entropy loss的终止步数

            # curve derivative smooth loss
            time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
            time_tv_weight_end = 0.01,
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

boywarrior_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/multiview_dynamic_blender/boy_warrior/"),
        # costume_sample_frames=[0,70],
        # costume_padding_size=15
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
    experiment=DS_Experiment(name='blender_mv_d_joint/boy_warrior', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
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

        appearance_learning_rate=3e-2,
        appearance_decay=1500,

        light_learning_rate=5e-3, # 用于light的参数
        light_learning_rate_decay=800,

        regularization=JointRegularizationConfig(
            ssim_weight_begin = 1., # ssim loss
            ssim_weight_end  = 10.,
            ssim_weight_decay_steps = 1250,
            ssim_weight_start_step = 0,

            psnr_weight_begin = 1., # psnr loss
            psnr_weight_end = 1.,
            psnr_weight_decay_steps = 1250,
            psnr_weight_start_step = -1,

            depth_weight_begin = 100., # depth supervision loss
            depth_weight_end = 1.,
            depth_weight_decay_steps = 1250,
            depth_weight_start_step = -1, # -1 代表“不启用”

            mask_weight_begin = 10., # mask supervision loss
            mask_weight_end = 10.,
            mask_weight_decay_steps = 1250,
            mask_weight_start_step = 0,

            # sdf entropy loss
            sdf_entropy_weight_begin = 0.2, # 控制sdf entropy loss的权重
            sdf_entropy_weight_end = 0.001,
            sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
            sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
            sdf_entropy_end_step = 5000, # 控制sdf entropy loss的终止步数

            # curve derivative smooth loss
            time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
            time_tv_weight_end = 0.001,
            time_tv_decay_steps = 1250, # 控制curve derivative smooth loss的线性衰减步数
            time_tv_start_step = 0, # 控制curve derivative smooth loss的起始步数
            time_tv_end_step=5000, # 控制curve derivative smooth loss的终止步数

            # sdf eikonal loss，开启后，大概需要占用1G显存
            sdf_eikonal_weight_begin = 0.01, # 控制sdf eikonal loss的权重
            sdf_eikonal_weight_end = 0.001,
            sdf_eikonal_decay_steps = 1250, # 控制sdf eikonal loss的线性衰减步数
            sdf_eikonal_start_step = 0, # 控制sdf eikonal loss的起始步数
            sdf_eikonal_end_step=5000, # 控制sdf eikonal loss的终止步数

            # curve coefficient tv smooth loss，开启后，大概需要占用5G显存
            curve_coeff_tv_weight_begin = 0.1, # 控制curve coefficient tv smooth loss的权重
            curve_coeff_tv_weight_end = 10,
            curve_coeff_tv_decay_steps = 1250, # 控制curve coefficient tv smooth loss的线性衰减步数
            curve_coeff_tv_start_step = -1, # 控制curve coefficient tv smooth loss的起始步数
            curve_coeff_tv_end_step=5000, # 控制curve coefficient tv smooth loss的终止步数

            reg_temporal_hashgrid_begin = 0.1,
            reg_temporal_hashgrid_end = 0.01,
            reg_temporal_hashgrid_decay_steps = 1250,
            reg_temporal_hashgrid_start_step = 0,
            reg_temporal_hashgrid_end_step=5000,

            reg_occ_begin=0.0,
            reg_occ_end=0.001,
            reg_occ_decay_steps=1250,
            reg_occ_start_step=0,
            reg_occ_end_step=5000,

            reg_light_begin=0.001,
            reg_light_end=0.01,
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

refrigerator_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/multiview_dynamic_blender/refrigerator/"),
        # costume_sample_frames=[0,70],
        # costume_padding_size=15
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
            deform_desired_resolution=[4096,4096,128],
            deform_num_levels=32,
        ),
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 6e-3],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 6e-3],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 6e-3],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_joint/refrigerator', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
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

        appearance_learning_rate=3e-2,
        appearance_decay=1500,

        light_learning_rate=5e-3, # 用于light的参数
        light_learning_rate_decay=800,

        regularization=JointRegularizationConfig(
            ssim_weight_begin = 1., # ssim loss
            ssim_weight_end  = 10.,
            ssim_weight_decay_steps = 1250,
            ssim_weight_start_step = 0,

            psnr_weight_begin = 1., # psnr loss
            psnr_weight_end = 1.,
            psnr_weight_decay_steps = 1250,
            psnr_weight_start_step = -1,

            depth_weight_begin = 100., # depth supervision loss
            depth_weight_end = 1.,
            depth_weight_decay_steps = 1250,
            depth_weight_start_step = -1, # -1 代表“不启用”

            mask_weight_begin = 10., # mask supervision loss
            mask_weight_end = 10.,
            mask_weight_decay_steps = 1250,
            mask_weight_start_step = 0,

            # sdf entropy loss
            sdf_entropy_weight_begin = 0.2, # 控制sdf entropy loss的权重
            sdf_entropy_weight_end = 0.001,
            sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
            sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
            sdf_entropy_end_step = 5000, # 控制sdf entropy loss的终止步数

            # curve derivative smooth loss
            time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
            time_tv_weight_end = 0.001,
            time_tv_decay_steps = 1250, # 控制curve derivative smooth loss的线性衰减步数
            time_tv_start_step = 0, # 控制curve derivative smooth loss的起始步数
            time_tv_end_step=5000, # 控制curve derivative smooth loss的终止步数

            # sdf eikonal loss，开启后，大概需要占用1G显存
            sdf_eikonal_weight_begin = 0.01, # 控制sdf eikonal loss的权重
            sdf_eikonal_weight_end = 0.001,
            sdf_eikonal_decay_steps = 1250, # 控制sdf eikonal loss的线性衰减步数
            sdf_eikonal_start_step = 0, # 控制sdf eikonal loss的起始步数
            sdf_eikonal_end_step=5000, # 控制sdf eikonal loss的终止步数

            # curve coefficient tv smooth loss，开启后，大概需要占用5G显存
            curve_coeff_tv_weight_begin = 0.1, # 控制curve coefficient tv smooth loss的权重
            curve_coeff_tv_weight_end = 10,
            curve_coeff_tv_decay_steps = 1250, # 控制curve coefficient tv smooth loss的线性衰减步数
            curve_coeff_tv_start_step = -1, # 控制curve coefficient tv smooth loss的起始步数
            curve_coeff_tv_end_step=5000, # 控制curve coefficient tv smooth loss的终止步数

            reg_temporal_hashgrid_begin = 0.1,
            reg_temporal_hashgrid_end = 0.01,
            reg_temporal_hashgrid_decay_steps = 1250,
            reg_temporal_hashgrid_start_step = 0,
            reg_temporal_hashgrid_end_step=5000,

            reg_occ_begin=0.0,
            reg_occ_end=0.001,
            reg_occ_decay_steps=1250,
            reg_occ_start_step=0,
            reg_occ_end_step=5000,

            reg_light_begin=0.001,
            reg_light_end=0.01,
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

dumptruck_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/multiview_dynamic_blender/dump_truck/"),
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
            deform_desired_resolution=[4096,4096,128],
            deform_num_levels=32,
        ),
        reg_spatial_random_perturb_range=[1e-2, 1e-2, 1e-2, 9e-3],
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 9e-3],
        reg_kd_random_perturb_range=[1e-2, 1e-2, 1e-2, 9e-3],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='blender_mv_d_joint/dumptruck', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=500,
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
        appearance_decay=1250,

        light_learning_rate=1e-3, # 用于light的参数
        light_learning_rate_decay=1250,

        regularization=JointRegularizationConfig(
            ssim_weight_begin = 1., # ssim loss
            ssim_weight_end  = 10.,
            ssim_weight_decay_steps = 2500,
            ssim_weight_start_step = 0,

            psnr_weight_begin = 1., # psnr loss
            psnr_weight_end = 1.,
            psnr_weight_decay_steps = 1250,
            psnr_weight_start_step = -1,

            depth_weight_begin = 100., # depth supervision loss
            depth_weight_end = 1.,
            depth_weight_decay_steps = 1250,
            depth_weight_start_step = -1, # -1 代表“不启用”

            mask_weight_begin = 10., # mask supervision loss
            mask_weight_end = 10.,
            mask_weight_decay_steps = 1250,
            mask_weight_start_step = 0,

            # sdf entropy loss
            sdf_entropy_weight_begin = 0.2, # 控制sdf entropy loss的权重
            sdf_entropy_weight_end = 0.01,
            sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
            sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
            sdf_entropy_end_step = 5000, # 控制sdf entropy loss的终止步数

            # curve derivative smooth loss
            time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
            time_tv_weight_end = 0.01,
            time_tv_decay_steps = 1250, # 控制curve derivative smooth loss的线性衰减步数
            time_tv_start_step = 0, # 控制curve derivative smooth loss的起始步数
            time_tv_end_step=5000, # 控制curve derivative smooth loss的终止步数

            # sdf eikonal loss，开启后，大概需要占用1G显存
            sdf_eikonal_weight_begin = 0.01, # 控制sdf eikonal loss的权重
            sdf_eikonal_weight_end = 0.001,
            sdf_eikonal_decay_steps = 1250, # 控制sdf eikonal loss的线性衰减步数
            sdf_eikonal_start_step = 0, # 控制sdf eikonal loss的起始步数
            sdf_eikonal_end_step=5000, # 控制sdf eikonal loss的终止步数

            # curve coefficient tv smooth loss，开启后，大概需要占用5G显存
            curve_coeff_tv_weight_begin = 0.1, # 控制curve coefficient tv smooth loss的权重
            curve_coeff_tv_weight_end = 10,
            curve_coeff_tv_decay_steps = 1250, # 控制curve coefficient tv smooth loss的线性衰减步数
            curve_coeff_tv_start_step = -1, # 控制curve coefficient tv smooth loss的起始步数
            curve_coeff_tv_end_step=5000, # 控制curve coefficient tv smooth loss的终止步数

            reg_temporal_hashgrid_begin = 0.1,
            reg_temporal_hashgrid_end = 0.01,
            reg_temporal_hashgrid_decay_steps = 1250,
            reg_temporal_hashgrid_start_step = 0,
            reg_temporal_hashgrid_end_step=5000,

            reg_occ_begin=0.0,
            reg_occ_end=0.001,
            reg_occ_decay_steps=1250,
            reg_occ_start_step=0,
            reg_occ_end_step=5000,

            reg_light_begin=0.001,
            reg_light_end=0.01,
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
            dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
            trainer: D_JointTrainer = task.trainer
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
class Extract_results(Task):
    """渲染训练好的模型"""
    step: Optional[int] = None

    def process_loader(self, split_name, export_pred_light=False):
        """
        处理一个数据迭代器：
        - 推理模型
        - 渲染网格
        - 保存图像和 PBR 属性
        - 导出网格（可选）
        """
        full_batch_size = self.dataset.get_size(split='fix_vis') # full time size
        if 'orbit' in split_name:
            loader = self.dataset.get_orbit_vis_iter(batch_size=50, shuffle=False, infinite=False)
        elif 'fix' in split_name:
            loader = self.dataset.get_fix_vis_iter(batch_size=50, shuffle=False, infinite=False)
        elif 'test' in split_name:
            loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='orbit_vis')
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )

        # 初始化存储容器
        gt_images, gt_normals, gt_depths, gt_pretty_meshes = [], [], [], []
        pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
        pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs = [], [], [], []
        # scene_flow_pred_next_frame_pretty_meshes = []

        iter_count = 0
        for inputs, gt_outputs, indices in loader:
            _, _, visualization, pred_meshes = self.trainer.step(
                self.model,
                inputs,
                gt_outputs,
                indices=indices,
                mode='orbit_vis',
                visual=True,
                val_pbr_attr=True,
                return_pred_mesh=True,
            )
            
            for i in range(len(visualization)):
                count_id = iter_count * 50 + i
                if 'test' in split_name:
                    view_num = count_id // full_batch_size
                    split_name = f'test_view{view_num}'
                    frame_id = count_id % full_batch_size
                else:
                    frame_id = count_id

                camera = inputs[i]
                vis = visualization[i]

                gt_mesh = gt_meshes[frame_id].clone()
                gt_pretty_meshes.append(
                    gt_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                )
                pred_mesh = pred_meshes[i].clone()
                pred_pretty_meshes.append(
                    pred_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                )

                # 切分可视化图像
                gt_images.append(vis[:800, :800, :3].cpu())
                gt_normals.append(vis[:800, 800:1600, :3].cpu())
                gt_depths.append(vis[:800, 1600:, :3].cpu())
                pred_images.append(vis[800:1600, :800, :3].cpu())
                pred_normals.append(vis[800:1600, 800:1600, :3].cpu())
                pred_depths.append(vis[800:1600, 1600:, :3].cpu())
                pred_kds.append(vis[1600:2400, :800, :3].cpu())
                if export_pred_light:
                    self.experiment.dump_image('eval/extract', image=vis[1600:2400, 800:, :3], name='pred_light')
                    export_pred_light = False
                pbr_roughnesses.append(vis[2400:3200, :800, :3].cpu())
                pbr_metallics.append(vis[2400:3200, 800:1600, :3].cpu())
                pbr_occs.append(vis[2400:3200, 1600:, :3].cpu())

                # 保存 GT 图像
                for idx, (img, sub) in enumerate([
                    (gt_images[-1], 'gt_image'), 
                    (gt_normals[-1], 'gt_normal'), 
                    (gt_depths[-1], 'gt_depth'),
                    (gt_pretty_meshes[-1], 'gt_mesh')
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
                        'gt_mesh': gt_pretty_meshes,
                        'gt_image': gt_images,
                        'gt_normal': gt_normals,
                        'gt_depth': gt_depths,
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
                        if 'test' in split_name:
                            split_name = f'test_view{view_num}'
                            self.experiment.dump_images2video(
                                f'eval/extract/{split_name}',
                                name=name, 
                                images=imgs, 
                                downsample=1, 
                                fps=48, 
                                duration=5
                            )
                        else:
                            self.experiment.dump_images2video(
                                f'eval/extract/{split_name}',
                                name=name, 
                                images=imgs, 
                                downsample=1, 
                                fps=48, 
                                duration=5
                            )
                    del gt_images, gt_normals, gt_depths, gt_pretty_meshes, pred_images, pred_normals, pred_depths, pred_pretty_meshes, pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs
                    torch.cuda.empty_cache()
                    gc.collect()
                    gt_images, gt_normals, gt_depths, gt_pretty_meshes = [], [], [], []
                    pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
                    pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs = [], [], [], []

            iter_count += 1
            torch.cuda.empty_cache()
                
        del gt_images, gt_normals, gt_depths, gt_pretty_meshes, pred_images, pred_normals, pred_depths, pred_pretty_meshes, pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs, loader, gt_meshes
        torch.cuda.empty_cache()
        gc.collect()


    @torch.no_grad()
    def run(self) -> None:
        # categories = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        categories = ['spidermanfight']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint')
        cases = []
        for category in categories:
            # case路径
            case = root / category / 'test' / 'task.py'
            cases.append(case)

        for case in cases:
            self.load = case
            print(f'Processing: {self.load}...')
            with console.status(desc='Loading Model'):
                task = DS_TrainTask.load_from_script(self.load, step=self.step)
                self.experiment: DS_Experiment = task.experiment
                self.model: D_Joint = task.model
                self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
                self.trainer: D_JointTrainer = task.trainer
                self.trainer.setup(model=self.model, dataset=self.dataset)
                self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
            self.model.eval()
            # self.model.reg_scene_flow_smoothness_able = True
            
            # self.process_loader('orbit', export_pred_light=True)
            # self.process_loader('fix', export_pred_light=False)
            self.process_loader('test_view', export_pred_light=False)

            del self.model, self.dataset, self.trainer, self.experiment
            torch.cuda.empty_cache()
            gc.collect()


@dataclass
class Eval_results(Task):
    """渲染训练好的模型"""
    step: Optional[int] = None

    def process_loader(self):
        """
        处理一个数据迭代器：
        - 推理模型
        - 渲染网格
        - 保存图像和 PBR 属性
        - 导出网格（可选）
        """
        full_batch_size = self.dataset.get_size(split='fix_vis') # full time size
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='orbit_vis')

        # 初始化存储容器
        psnrs = []
        ssims = []
        lpipss = []
        chamfer_dists = []
        normal_maes = []
        
        iter_count = 0
        for inputs, gt_outputs, indices in loader:
            outputs = self.model.render_report(
                camera_inputs=inputs, 
                trainer_mode='orbit_vis',
                return_pred_mesh=True,
                time_window_indices=None, # only train use time window data
                return_pbr_attrs=False,
            )
            (
                pred_outputs,
                reg_loss_dict,
                pred_meshes,
                pred_depths, gt_depths,
                pred_normals, gt_normals,
                gt_next_frame_depths, pred_next_frame_depths, scene_flow_pred_next_frame_depths,
                gt_next_frame_normals, pred_next_frame_normals, scene_flow_pred_next_frame_normals,
                pbr_attrs
            ) = outputs
            bg_color = self.model.get_background_color().to(self.model.device)
            if isinstance(pred_outputs, PBRAImages):
                pred_rgbs = pred_outputs.rgb2srgb().blend(bg_color)
            else:
                pred_rgbs = pred_outputs.blend(bg_color)
            gt_rgbs = gt_outputs.blend(bg_color)
            
            for i in range(len(pred_rgbs)):
                count_id = iter_count * 50 + i
                view_num = count_id // full_batch_size
                frame_id = count_id % full_batch_size

                gt_image = gt_rgbs[i].clamp(0, 1)
                pred_image = pred_rgbs[i].clamp(0, 1)
                psnr = PSNRLoss()(gt_image, pred_image)
                ssim = (1 - SSIMLoss()(gt_image, pred_image))
                lpips = LPIPSLoss()(gt_image, pred_image)

                camera = inputs[i]
                normal_bg = torch.tensor([0, 0, 1]).float().to(self.device)
                gt_mesh = gt_meshes[frame_id].clone()
                pred_mesh = pred_meshes[i].clone()
                gt_normal_ = gt_mesh.render(camera, shader=NormalShader()).item()
                gt_normal = torch.add(
                    gt_normal_[..., :3] / gt_normal_[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-6) * gt_normal_[..., 3:],
                    normal_bg * (1 - gt_normal_[..., 3:]),
                ) # [H, W, 3]
                pred_normal_ = pred_mesh.render(camera, shader=NormalShader()).item()
                pred_normal = torch.add(
                    pred_normal_[..., :3] / pred_normal_[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-6) * pred_normal_[..., 3:],
                    normal_bg * (1 - pred_normal_[..., 3:]),
                ) # [H, W, 3]
                
                ae = (pred_normal * gt_normal).sum(-1, keepdim=True).clamp(-1, 1)
                mae = ae.arccos().rad2deg().mean()
                
                self.experiment.log(f"Test view {view_num}, Frame {frame_id}: PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}, MAE: {mae}", new_logfile='eval.txt')
                psnrs.append(psnr)
                ssims.append(ssim)
                lpipss.append(lpips)
                normal_maes.append(mae)

                if view_num == 0:
                    inv_trans = torch.tensor([[1,0,0],[0,0,1],[0,-1,0]]).to(pred_mesh.vertices)
                    pred_mesh.replace_(vertices=(inv_trans @ (pred_mesh.vertices * (3/2) * 2).unsqueeze(-1)).squeeze(-1))
                    gt_mesh.replace_(vertices=(inv_trans @ (gt_mesh.vertices * (3/2) * 2).unsqueeze(-1)).squeeze(-1))
                    chamfer_dist = ChamferDistanceMetric(target_num_points=1000000)(gt_mesh, pred_mesh)
                    chamfer_dists.append(chamfer_dist)
                    pred_mesh.export(
                        path=self.experiment.dump_file_path(subfolder='eval/scale_pred_mesh', file_name=f'frame{frame_id}.obj'),
                        only_geometry=True
                    )
                    gt_mesh.export(
                        path=self.experiment.dump_file_path(subfolder='eval/scale_gt_mesh', file_name=f'frame{frame_id}.obj'),
                        only_geometry=True
                    )   
            iter_count += 1
            torch.cuda.empty_cache()
        for i in range(len(chamfer_dists)):
            self.experiment.log(f"Frame {i}: Chamfer Distance: {chamfer_dists[i]}")
        mean_chamfer_dist = torch.stack(chamfer_dists).mean().item()
        self.experiment.log(f"Mean Chamfer Distance: {mean_chamfer_dist}")
        mean_psnr = torch.stack(psnrs).mean().item()
        mean_ssim = torch.stack(ssims).mean().item()
        mean_lpips = torch.stack(lpipss).mean().item()
        mean_normal_mae = torch.stack(normal_maes).mean().item()
        self.experiment.log(f"Test view, Mean PSNR: {mean_psnr}, Mean SSIM: {mean_ssim}, Mean LPIPS: {mean_lpips}, Mean Normal MAE: {mean_normal_mae}")

        del psnrs, ssims, lpipss, chamfer_dists, normal_maes, loader, gt_meshes
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def run(self) -> None:
        # categories = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        categories = ['toy']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint')
        cases = []
        for category in categories:
            # case路径
            case = root / category / 'debug3.6' / 'task.py'
            cases.append(case)

        for case in cases:
            self.load = case
            print(f'Processing: {self.load}...')
            with console.status(desc='Loading Model'):
                task = DS_TrainTask.load_from_script(self.load, step=self.step)
                self.experiment: DS_Experiment = task.experiment
                self.model: D_Joint = task.model
                self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
                self.trainer: D_JointTrainer = task.trainer
                self.trainer.setup(model=self.model, dataset=self.dataset)
                self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
            self.model.eval()
            # self.model.reg_scene_flow_smoothness_able = True
            
            self.process_loader()
            del self.model, self.dataset, self.trainer, self.experiment
            torch.cuda.empty_cache()
            gc.collect()

            

@dataclass
class EvalSceneFlow(Task):
    """为训练好的 D_DiffDR 模型评估场景流，生成网格并计算场景流用在mesh上看预测效果。"""

    step: Optional[int] = None

    def process_loader(self, split_name):
        """
        处理一个数据迭代器：
        - 推理模型
        - 渲染网格
        - 保存图像和 PBR 属性
        - 导出网格（可选）
        """
        full_batch_size = self.dataset.get_size(split='fix_vis') # full time size
        loader = self.dataset.get_fix_vis_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='orbit_vis')
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )

        # 初始化存储容器
        gt_pretty_meshes, pred_pretty_meshes, scene_flow_pred_next_frame_pretty_meshes  = [], [], []
        interpolate_pred_pretty_meshes, interpolate_scene_flow_pred_next_frame_pretty_meshes = [], []

        iter_count = 0
        inv_trans = torch.tensor([[1,0,0],[0,0,1],[0,-1,0]]).to(gt_meshes[0].vertices)
        for inputs, _, _ in loader:
            dt = inputs.dts[0].item()
            dt_ = dt / 11
            interpolate_times = []
            
            for i in range((inputs.times.shape[0] - 1) if inputs.times.shape[0] < 50 else (inputs.times.shape[0])):
                t0 = inputs.times[i].item()
                # 生成 11-1 个等分点 (包含 t0，但不包含下一个点，因为会在下一轮拼接)
                inter_times = [t0 + k * dt_ for k in range(11)]
                interpolate_times.extend(inter_times)
            interpolate_times.append(inputs.times[-1].item())
            interpolate_times = torch.tensor(interpolate_times, dtype=inputs.times.dtype, device=inputs.times.device)
            
            pred_meshes, _ = self.model.get_geometry(times=inputs.times)
            interpolate_pred_meshes, _ = self.model.get_geometry(times=interpolate_times)
            
            for i in range(inputs.times.shape[0]):
                count_id = iter_count * 50 + i
                if 'test' in split_name:
                    view_num = count_id // full_batch_size
                    split_name = f'test_view{view_num}'
                    frame_id = count_id % full_batch_size
                else:
                    frame_id = count_id

                camera = inputs[i]

                gt_mesh = gt_meshes[frame_id].clone()
                gt_pretty_meshes.append(
                    gt_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                )
                self.experiment.dump_image(f'eval/extract/{split_name}/gt/gt_mesh', image=gt_pretty_meshes[-1], name=f'{frame_id}')
                gt_mesh.replace_(vertices=(inv_trans @ (gt_mesh.vertices * (3/2)).unsqueeze(-1)).squeeze(-1))
                gt_mesh.export(
                    path=self.experiment.dump_file_path(subfolder='eval/extract/gt_mesh', file_name=f'frame{frame_id}.obj'),
                    only_geometry=True
                )

                pred_mesh = pred_meshes[i].clone()
                pred_pretty_meshes.append(
                    pred_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                )
                self.experiment.dump_image(f'eval/extract/{split_name}/pred/pred_mesh', image=pred_pretty_meshes[-1], name=f'{frame_id}')
                if pred_mesh.vertices_scene_flow is None:
                    pred_mesh.compute_scene_flow()
                scene_flow_pred_next_frame_mesh = pred_mesh.get_next_frame_mesh(dt)
                scene_flow_pred_next_frame_pretty_meshes.append(
                    scene_flow_pred_next_frame_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1)
                )
                self.experiment.dump_image(f'eval/extract/{split_name}/pred_scene_flow/pred_scene_flow_mesh', image=scene_flow_pred_next_frame_pretty_meshes[-1], name=f'{frame_id+1}')

                pred_mesh.replace_(vertices=(inv_trans @ (pred_mesh.vertices * (3/2)).unsqueeze(-1)).squeeze(-1))
                pred_mesh.export(
                    path=self.experiment.dump_file_path(subfolder='eval/extract/pred_mesh', file_name=f'frame{frame_id}.obj'),
                    only_geometry=True
                )
                scene_flow_pred_next_frame_mesh.replace_(
                    vertices=(inv_trans @ (scene_flow_pred_next_frame_mesh.vertices * (3/2)).unsqueeze(-1)).squeeze(-1),
                )
                scene_flow_pred_next_frame_mesh.export(
                    path=self.experiment.dump_file_path(subfolder='eval/extract/pred_scene_flow_mesh', file_name=f'frame{frame_id+1}.obj'),
                    only_geometry=True
                )

                if frame_id < full_batch_size - 1:
                    inter_meshes = interpolate_pred_meshes[i * 11 : i * 11 + 11]
                    for j, inter_mesh in enumerate(inter_meshes):
                        inter_mesh = inter_mesh.clone()
                        interpolate_pred_pretty_meshes.append(
                            inter_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                        )
                        # 保存插值 Mesh 图像
                        self.experiment.dump_image(
                            f'eval/extract/{split_name}/pred_interpolated/pred_interpolated_mesh',
                            image=interpolate_pred_pretty_meshes[-1],
                            name=f'{frame_id}_interp{j}'
                        )
                        
                        if inter_mesh.vertices_scene_flow is None:
                            inter_mesh.compute_scene_flow()
                        inter_scene_flow_pred_next_frame_mesh = inter_mesh.get_next_frame_mesh(dt_)
                        interpolate_scene_flow_pred_next_frame_pretty_meshes.append(
                            inter_scene_flow_pred_next_frame_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                        )
                        self.experiment.dump_image(
                            f'eval/extract/{split_name}/pred_interpolated_scene_flow/pred_interpolated_scene_flow_mesh',
                            image=interpolate_scene_flow_pred_next_frame_pretty_meshes[-1],
                            name=f'{frame_id}_interp{j+1}'
                        )

                        inter_mesh.replace_(vertices=(inv_trans @ (inter_mesh.vertices * (3/2)).unsqueeze(-1)).squeeze(-1))
                        inter_mesh.export(
                            path=self.experiment.dump_file_path(subfolder='eval/extract/pred_interpolated_mesh', file_name=f'frame{frame_id}_interp{j}.obj'),
                            only_geometry=True
                        )
                        inter_scene_flow_pred_next_frame_mesh.replace_(
                            vertices=(inv_trans @ (inter_scene_flow_pred_next_frame_mesh.vertices * (3/2)).unsqueeze(-1)).squeeze(-1),
                        )
                        inter_scene_flow_pred_next_frame_mesh.export(
                            path=self.experiment.dump_file_path(subfolder='eval/extract/pred_interpolated_scene_flow_mesh', file_name=f'frame{frame_id}_interp{j+1}.obj'),
                            only_geometry=True
                        )

                if frame_id == full_batch_size - 1:
                    # 导出视频
                    video_dict = {
                        'gt_mesh': gt_pretty_meshes,
                        'pred_mesh': pred_pretty_meshes,
                        'pred_scene_flow_mesh': scene_flow_pred_next_frame_pretty_meshes,
                        'pred_interpolated_mesh': interpolate_pred_pretty_meshes,
                        'pred_interpolated_scene_flow_mesh': interpolate_scene_flow_pred_next_frame_pretty_meshes,
                    }
                    for name, imgs in video_dict.items():
                        self.experiment.dump_images2video(
                            f'eval/extract/{split_name}',
                            name=name, 
                            images=imgs, 
                            downsample=1, 
                            fps=48, 
                            duration=5
                        )

            iter_count += 1
            torch.cuda.empty_cache()
                
        torch.cuda.empty_cache()
        gc.collect()


    @torch.no_grad()
    def run(self) -> None:
        # categories = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        categories = ['toy']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint')
        cases = []
        for category in categories:
            # case路径
            case = root / category / 'debug3' / 'task.py'
            cases.append(case)

        for case in cases:
            self.load = case
            print(f'Processing: {self.load}...')
            with console.status(desc='Loading Model'):
                task = DS_TrainTask.load_from_script(self.load, step=self.step)
                self.experiment: DS_Experiment = task.experiment
                self.model: D_Joint = task.model
                self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
                self.trainer: D_JointTrainer = task.trainer
                self.trainer.setup(model=self.model, dataset=self.dataset)
                self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
            self.model.eval()
            self.model.reg_scene_flow_smoothness_able = True
            
            self.process_loader('fix')

            del self.model, self.dataset, self.trainer, self.experiment
            torch.cuda.empty_cache()
            gc.collect()


@dataclass
class ExtractSceneFlowMap(Task):
    """为训练好的 D_DiffDR 模型评估场景流，生成网格并计算场景流用在mesh上看预测效果。"""

    step: Optional[int] = None

    def get_scene_flow_map_by_vetrice_sceneflow(self, camera, pred_mesh):
        ctx = dr.RasterizeCudaContext(camera.device)
        
        vertices = torch.cat((
            pred_mesh.vertices,
            torch.ones_like(pred_mesh.vertices[..., :1]),
        ), dim=-1).view(-1, 4, 1)
        H, W = resolution = [camera.height.item(), camera.width.item()]
        mvp = camera.projection_matrix @ camera.view_matrix                    # [4, 4]
        projected = (mvp @ vertices).view(1, -1, 4)                            # [1, V, 4]
        indices = pred_mesh.indices.int()
        with dr.DepthPeeler(ctx, projected, indices, resolution=resolution) as peeler:
            rast, _ = peeler.rasterize_next_layer()
        alphas = (rast[..., -1:] > 0).float() # [1, H, W, 1]
        position, _ = dr.interpolate(pred_mesh.vertices[None], rast, indices)
        position_map = torch.cat((position, alphas), dim=-1) # [1， H, W, 4]
        scene_flow, _ = dr.interpolate(pred_mesh.vertices_scene_flow[None], rast, indices) # [1, H, W, 3]
        scene_flow_map = torch.cat((scene_flow, alphas), dim=-1) # [1， H, W, 4]
        image_array = self.plot_scene_flow_quiver3d(
            position_map=position_map, 
            scene_flow_map=scene_flow_map, 
            save_path=Path('temp.png')
        )
        
        return image_array
    

    def plot_scene_flow_quiver3d(self, position_map, scene_flow_map, save_path, step=25):
        """
        将三维场景流图可视化为附着在物体表面的箭头图 (quiver plot)。
        
        参数:
            position_map (torch.Tensor): [1, H, W, 4] 的张量，前3个通道是 (x, y, z) 世界坐标，第4通道是mask。
            scene_flow_map (torch.Tensor): [1, H, W, 4] 的张量，前3个通道是 (vx, vy, vz) 场景流，第4通道是mask。
            save_path (str): 图像保存路径。
            step (int): 降采样步长，每隔 `step` 个像素取一个点进行可视化。
        """
        # 1. 提取数据并转换到 CPU NumPy 数组
        pos_np = position_map[0].detach().cpu().numpy()      # [H, W, 4]
        flow_np = scene_flow_map[0].detach().cpu().numpy()   # [H, W, 4]

        H, W, _ = pos_np.shape
        
        # 2. 为了画面清晰，对图进行降采样
        yy, xx = np.mgrid[0:H:step, 0:W:step]
        
        # 获取降采样后各点的三维坐标 (x,y,z) 和可见性 mask
        x, y, z, mask = pos_np[yy, xx].T
        
        # 获取这些点对应的三维流向量 (u,v,w)
        u, v, w, _ = flow_np[yy, xx].T

        # 将数组展平以便处理
        x, y, z, mask = x.flatten(), y.flatten(), z.flatten(), mask.flatten()
        u, v, w = u.flatten(), v.flatten(), w.flatten()

        # 3. 使用 mask 过滤掉背景中不可见的点
        valid_indices = mask > 0.5
        x, y, z = x[valid_indices], y[valid_indices], z[valid_indices]
        u, v, w = u[valid_indices], v[valid_indices], w[valid_indices]
        
        if len(x) == 0:
            print("警告: 没有有效的点可用于绘制场景流箭头图。")
            return None

        # 4. 标准化箭头长度以获得更好的可视化效果
        magnitudes = np.linalg.norm(np.stack([u, v, w]), axis=0)
        max_magnitude = np.max(magnitudes) if len(magnitudes) > 0 else 1.0
        
        # 缩放向量，使得最长的箭头有一个合理的显示尺寸
        # 我们可以让它相对于可见物体的大小进行缩放
        object_size = np.max(np.ptp(np.stack([x, y, z]), axis=1)) if len(x) > 1 else 1.0
        # 让最长箭头的长度是物体尺寸的10%
        scale_factor = (object_size / max_magnitude) * 0.1 if max_magnitude > 0 else 0.0
        
        u_scaled, v_scaled, w_scaled = u * scale_factor, v * scale_factor, w * scale_factor

        # 5. 绘图
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 使用流向量的模长来为箭头着色
        colors = plt.cm.viridis(magnitudes / max_magnitude)

        ax.quiver(x, y, z, u_scaled, v_scaled, w_scaled, colors=colors, length=1.0, normalize=False)
        
        # 设置绘图范围，确保xyz轴比例尺一致，防止物体变形
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x, mid_y, mid_z = (x.max()+x.min())*0.5, (y.max()+y.min())*0.5, (z.max()+z.min())*0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_xlabel('X 轴')
        ax.set_ylabel('Y 轴')
        ax.set_zlabel('Z 轴')
        ax.set_title('三维场景流可视化')
        
        # 保存图像
        plt.savefig(save_path)

    
    def process_loader(self, split_name):
        """
        处理一个数据迭代器：
        - 推理模型
        - 渲染网格
        - 保存图像和 PBR 属性
        - 导出网格（可选）
        """
        full_batch_size = self.dataset.get_size(split='fix_vis') # full time size
        loader = self.dataset.get_fix_vis_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='orbit_vis')
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )

        # 初始化存储容器
        gt_pretty_meshes, pred_pretty_meshes, scene_flow_pred_next_frame_pretty_meshes  = [], [], []
        interpolate_pred_pretty_meshes, interpolate_scene_flow_pred_next_frame_pretty_meshes = [], []

        iter_count = 0
        inv_trans = torch.tensor([[1,0,0],[0,0,1],[0,-1,0]]).to(gt_meshes[0].vertices)
        for inputs, _, _ in loader: 
            pred_meshes, _ = self.model.get_geometry(times=inputs.times)

            for i in range(inputs.times.shape[0]):
                count_id = iter_count * 50 + i
                if 'test' in split_name:
                    view_num = count_id // full_batch_size
                    split_name = f'test_view{view_num}'
                    frame_id = count_id % full_batch_size
                else:
                    frame_id = count_id

                camera = inputs[i]

                pred_mesh = pred_meshes[i].clone()
                scene_flow_map = self.get_scene_flow_map_by_vetrice_sceneflow(camera, pred_mesh)
     
            iter_count += 1
            torch.cuda.empty_cache()
                
        torch.cuda.empty_cache()
        gc.collect()


    @torch.no_grad()
    def run(self) -> None:
        # categories = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        categories = ['toy']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint')
        cases = []
        for category in categories:
            # case路径
            case = root / category / 'debug3' / 'task.py'
            cases.append(case)

        for case in cases:
            self.load = case
            print(f'Processing: {self.load}...')
            with console.status(desc='Loading Model'):
                task = DS_TrainTask.load_from_script(self.load, step=self.step)
                self.experiment: DS_Experiment = task.experiment
                self.model: D_Joint = task.model
                self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
                self.trainer: D_JointTrainer = task.trainer
                self.trainer.setup(model=self.model, dataset=self.dataset)
                self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
            self.model.eval()
            self.model.reg_scene_flow_smoothness_able = True
            
            self.process_loader('fix')

            del self.model, self.dataset, self.trainer, self.experiment
            torch.cuda.empty_cache()
            gc.collect()



@dataclass
class EvalCubeCurve(Task):

    load: Path = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/toy/debug2.1/task.py')

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
        trainer: D_JointTrainer,
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

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            experiment: DS_Experiment = task.experiment
            model: D_Joint = task.model
            dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
            trainer: D_JointTrainer = task.trainer
            sampler = NearestGridSampler(
                resolution=model.geometry_resolution,
                scale=model.geometry_scale,    
            )

            trainer.setup(model=model, dataset=dataset)
            trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)
            sampler.__setup__()
            sampler.to(self.device)
            sampler.reset()
            
            model.eval()
            if model.geometric_repr.device != self.device:
                model.geometric_repr.swap_(model.geometric_repr.to(self.device))
            if model.geometry == 'DD_isocubes':
                geometric_repr = model.geometric_repr.replace(
                    static_sdf_values = model.static_sdf_params,
                    sdf_curve_poly_coefficient = model.sdf_curve_poly_coefficient,
                    sdf_curve_low_freq_fourier_coefficient = model.sdf_curve_low_freq_fourier_coefficient,
                    sdf_curve_mid_freq_fourier_coefficient = model.sdf_curve_mid_freq_fourier_coefficient,
                    sdf_curve_high_freq_fourier_coefficient = model.sdf_curve_high_freq_fourier_coefficient,
                )
            elif model.geometry == 'DD_flexicubes':
                geometric_repr = model.geometric_repr.replace(
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
            pred_meshes, _, = model.get_geometry(times=times)
            pred_sdf, pred_sdf_flow = geometric_repr.query_sdf_at_times(t=times,model_stage=model.dynamic_model_stage,compute_sdf_flow=True)

        max_error_fame = 15
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
        max_error_cube_curve_info = geometric_repr.get_cube_curve_info(indices=max_error_cube_indices)
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
        experiment.log(f"Max Error Cube indices: {max_error_cube_indices.tolist()}") 

        fix_cube_indices = torch.tensor(
            [
                [56, 69, 96],
                [97, 79, 81],

                [67, 33, 64],
                [20, 67, 64],

                [20, 67, 65],

                [70, 64, 16]
            ]
        )
        fix_cube_curve_info = geometric_repr.get_cube_curve_info(indices=fix_cube_indices)
        self.analyze_cubes_curve_info(
            trainer=trainer,
            experiment=experiment,
            save_subfolder='eval_fix_cubes_curve',
            cubes_info=fix_cube_curve_info,
            times=times,
            pred_data=pred_sdf.squeeze(-1),
            highlight_indices=[max_error_fame],
        )
        
        
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
            [0.03149604797363281, -0.3779527544975281, -0.12598425149917603],
        ], device=self.device)
        fix_points_cube_info = geometric_repr.get_cube_curve_info(positions=fix_positions)
        self.analyze_cubes_curve_info(
            trainer=trainer,
            experiment=experiment,
            save_subfolder='eval_fix_points_curve',
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


@dataclass
class Extract_Realobject_results(Task):
    """渲染训练好的模型"""
    step: Optional[int] = None

    def process_loader(self, split_name, export_pred_light=False):
        """
        处理一个数据迭代器：
        - 推理模型
        - 渲染网格
        - 保存图像和 PBR 属性
        - 导出网格（可选）
        """
        full_batch_size = self.dataset.dataparser.model.num_frames
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)
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
                count_id = iter_count * 50 + i
                view_num = count_id // full_batch_size
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
                        if 'test' in split_name:
                            split_name = f'test_view{view_num}'
                            self.experiment.dump_images2video(
                                f'eval/extract/{split_name}',
                                name=name, 
                                images=imgs, 
                                downsample=1, 
                                fps=48, 
                                duration=5
                            )
                        else:
                            self.experiment.dump_images2video(
                                f'eval/extract/{split_name}',
                                name=name, 
                                images=imgs, 
                                downsample=1, 
                                fps=48, 
                                duration=5
                            )
                    del gt_images, pred_images, pred_normals, pred_depths, pred_pretty_meshes, pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs
                    torch.cuda.empty_cache()
                    gc.collect()
                    gt_images, gt_normals, gt_depths, gt_pretty_meshes = [], [], [], []
                    pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
                    pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs = [], [], [], []

            iter_count += 1
            torch.cuda.empty_cache()
                
        del gt_images, gt_normals, gt_depths, gt_pretty_meshes, pred_images, pred_normals, pred_depths, pred_pretty_meshes, pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs, loader, gt_meshes
        torch.cuda.empty_cache()
        gc.collect()


    @torch.no_grad()
    def run(self) -> None:
        # categories = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        categories = ['penguin']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/realobject_mv_d_joint')
        cases = []
        for category in categories:
            # case路径
            case = root / category / 'test' / 'task.py'
            cases.append(case)

        for case in cases:
            self.load = case
            print(f'Processing: {self.load}...')
            with console.status(desc='Loading Model'):
                task = DS_TrainTask.load_from_script(self.load, step=self.step)
                self.experiment: DS_Experiment = task.experiment
                self.model: D_Joint = task.model
                self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
                self.trainer: D_JointTrainer = task.trainer
                self.trainer.setup(model=self.model, dataset=self.dataset)
                self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
            self.model.eval()
            # self.model.reg_scene_flow_smoothness_able = True
            
            # self.process_loader('orbit', export_pred_light=True)
            # self.process_loader('fix', export_pred_light=False)
            self.process_loader('test_view', export_pred_light=False)

            del self.model, self.dataset, self.trainer, self.experiment
            torch.cuda.empty_cache()
            gc.collect()


@dataclass
class Extract_CMUPanonic_results(Task):
    """渲染训练好的模型"""
    step: Optional[int] = None

    def process_loader(self, split_name, export_pred_light=False):
        """
        处理一个数据迭代器：
        - 推理模型
        - 渲染网格
        - 保存图像和 PBR 属性
        - 导出网格（可选）
        """
        full_batch_size = self.dataset.dataparser.model.num_frames
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)
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
                count_id = iter_count * 50 + i
                view_num = count_id // full_batch_size
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
                        if 'test' in split_name:
                            split_name = f'test_view{view_num}'
                            self.experiment.dump_images2video(
                                f'eval/extract/{split_name}',
                                name=name, 
                                images=imgs, 
                                downsample=1, 
                                fps=48, 
                                duration=5
                            )
                        else:
                            self.experiment.dump_images2video(
                                f'eval/extract/{split_name}',
                                name=name, 
                                images=imgs, 
                                downsample=1, 
                                fps=48, 
                                duration=5
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
        # categories = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        categories = ['pose1']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/cmupanonic_mv_d_joint')
        cases = []
        for category in categories:
            # case路径
            case = root / category / 'test2.1' / 'task.py'
            cases.append(case)

        for case in cases:
            self.load = case
            print(f'Processing: {self.load}...')
            with console.status(desc='Loading Model'):
                task = DS_TrainTask.load_from_script(self.load, step=self.step)
                self.experiment: DS_Experiment = task.experiment
                self.model: D_Joint = task.model
                self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
                self.trainer: D_JointTrainer = task.trainer
                self.trainer.setup(model=self.model, dataset=self.dataset)
                self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
            self.model.eval()
            # self.model.reg_scene_flow_smoothness_able = True
            
            # self.process_loader('orbit', export_pred_light=True)
            # self.process_loader('fix', export_pred_light=False)
            self.process_loader('test_view', export_pred_light=False)

            del self.model, self.dataset, self.trainer, self.experiment
            torch.cuda.empty_cache()
            gc.collect()





if __name__ == '__main__':
    TaskGroup(
        # train task
        toy = toy_task,
        cat = cat_task,
        deer = deer_task,
        footballplayer = footballplayer_task,
        lego = lego_task,
        rabbit = rabbit_task,
        spidermanfight = spidermanfight_task,

        penguin = penguin_task,
        bluecar = blue_car_task,
        dog = dog_task,
        punch = k1_double_punch_task,
        wolf = wolf_task,

        pose1 = cmu_pose1_task,
        cello1 = cmu_cello1_task,
        band1 = cmu_band1_task,
        hanggling2 = cmu_hanggling_b2_task,
        ian3 = cmu_ian3_task,
        pizza1 = cmu_pizza1_task,

        spidermanwalk = spidermanwalk_task,
        excavator = excavator_task,
        eagle = eagle_task,
        girlwalk = girlwalk_task,
        monsterroar = monsterroar_task,
        
        boywarrior = boywarrior_task,
        refrigerator = refrigerator_task,
        dumptruck = dumptruck_task,
        
        # eval task
        rendermodel = RenderAfterTrain(cuda=0), 
        extract = Extract_results(cuda=0), 
        eval = Eval_results(cuda=0), 
        evalsceneflow = EvalSceneFlow(cuda=0),
        extractsceneflowmap = ExtractSceneFlowMap(cuda=0),

        evalcubecurve = EvalCubeCurve(cuda=0),

        extractreal = Extract_Realobject_results(cuda=0),
        extractcmu = Extract_CMUPanonic_results(cuda=0),
    ).run()
