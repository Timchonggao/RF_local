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
import cv2

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
from rfstudio_ds.data import SyntheticDynamicMultiViewBlenderRGBADataset, SyntheticTemporalDynamicMultiviewBlenderRGBADataset
from rfstudio_ds.model import D_Joint # rgb image loss optimization model
from rfstudio_ds.trainer import D_JointTrainer, JointRegularizationConfig # rgb image loss optimization trainer, dynamic nvdiffrec
from rfstudio_ds.graphics.spatial_sampler import NearestGridSampler
from rfstudio_ds.visualization import Visualizer
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh
from rfstudio_ds.model.density_field.components.encoding_4d import Grid4d_HashEncoding, KplaneEncoding
from rfstudio_ds.nn import MLPDecoderNetwork, Grid4DDecoderNetwork
from rfstudio.io import load_float32_image, open_video_renderer, dump_float32_image
import natsort
from rfstudio_ds.visualization._plot_cube_curve import plot_cube_curve, plot_curve

################# Train Task
toy_task = DS_TrainTask(
    dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
        path=Path("data/ObjSel-Dyn/toy/"),
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
            deform_base_resolution=[16,16,16],
            deform_desired_resolution=[4096,4096,128],
            deform_num_levels=32,
        ),

        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_blender/toy', timestamp='test1_s1'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=1000,
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
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=0.5,
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
        
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_blender/cat', timestamp='test_s1'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=1000,
        num_steps_per_save=1000,
        hold_after_train=False,

        use_multi_model_stage=True,
        model_start_steps=[0,0,0,1000,2500],  # static, poly, fourier_low, fourier_mid, fourier_high

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
        full_test_after_train=False,
        full_fix_vis_after_train=True,
        full_orbit_vis_after_train=True,
        detect_anomaly=True,
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
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_blender/deer', timestamp='test_s1'),
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
        full_test_after_train=False,
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
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 5e-3],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_blender/footballplayer', timestamp='test1_s1'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=1000,
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
        full_fix_vis_after_train=True,
        full_orbit_vis_after_train=True,
        detect_anomaly=True,
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
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-3],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_blender/lego', timestamp='test_s1'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=1000,
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
        #  dynamic_texture=KplaneEncoding(
        #     decoder=MLPDecoderNetwork(
        #         layers=[-1, 96, 32, 6],
        #         activation='sigmoid',
        #         bias=False,
        #         initialization='kaiming-uniform',
        #     ),
        # ),
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_blender/rabbit', timestamp='test'),
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
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 7e-3],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_blender/spidermanfight', timestamp='test'),
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

################### Temporal Train Task
temporal_toy_task = DS_TrainTask(
    dataset=SyntheticTemporalDynamicMultiviewBlenderRGBADataset(
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

        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        background_color="black",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_blender/temporal_toy', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=None,
        num_steps_per_fix_vis=None,
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
            sdf_eikonal_weight_end = 0.01,
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
        full_fix_vis_after_train=False,
        full_orbit_vis_after_train=False,
        detect_anomaly=True,
    ),
    cuda=0,
    seed=1
)

temporal_spidermanfight_task = DS_TrainTask(
    dataset=SyntheticTemporalDynamicMultiviewBlenderRGBADataset(
        path=Path("data/ObjSel-Dyn/spiderman_fight/"),
    ),
    model=D_Joint(
        geometry='DD_flexicubes',
        geometry_resolution=96,
        geometry_scale=0.5,
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
        reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 7e-3],
        background_color="white",
        shader_type="split_sum_pbr",
        min_roughness=0.1,
    ),
    experiment=DS_Experiment(name='psdf_blender/temporal_spidermanfight', timestamp='test'),
    trainer=D_JointTrainer(
        num_steps=5000,
        batch_size=8,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=None,
        num_steps_per_fix_vis=None,
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
        full_fix_vis_after_train=False,
        full_orbit_vis_after_train=False,
        detect_anomaly=True,
    ),
    cuda=0,
    seed=1
)

################# Export Task
@dataclass
class ExportModel(Task):

    load: Path = ...

    @torch.no_grad()
    def run(self) -> None:
        train_task = DS_TrainTask.load_from_script(self.load)
        model = train_task.model
        dataset = train_task.dataset
        trainer = train_task.trainer
        trainer.setup(model=model, dataset=dataset)
        trainer.before_update(model=model, optimizers=None, curr_step=trainer.num_steps)

        assert isinstance(model, D_Joint)
        output = self.load.parent / 'dump' / 'model.pth'
        model_attributes=model.export_model()
        dataset_attributes=dataset.export_dataset_attributes()
        attributes = {**model_attributes, **dataset_attributes}
        torch.save(attributes, output)
        print(f"Model and dataset attributes {attributes} saved to {output}")


################# Eval Task
@dataclass
class Extract_results(Task):
    step: Optional[int] = None

    load: Optional[Path] = None

    def process_loader(self, split_name, export_pred_light=False):
        full_batch_size = self.dataset.get_size(split='fix_vis') # full time size
        if 'orbit' in split_name:
            loader = self.dataset.get_orbit_vis_iter(batch_size=50, shuffle=False, infinite=False)
        elif 'fix' in split_name:
            loader = self.dataset.get_fix_vis_iter(batch_size=50, shuffle=False, infinite=False)
        elif 'test' in split_name:
            loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='fix_vis')
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )

        # 初始化存储容器
        gt_images, gt_normals, gt_depths, gt_pretty_meshes = [], [], [], []
        pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
        pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs = [], [], [], []

        iter_count = 0
        for inputs, gt_outputs, indices in loader:
            _, _, visualization, pred_meshes = self.trainer.step(
                self.model,
                inputs,
                gt_outputs,
                indices=indices,
                mode='orbit_vis',
                visual=True,
                val_pbr_attr=False if 'nopbr' in str(self.load) else True,
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
                if 'nopbr' in str(self.load):
                    gt_images.append(vis[:800, :800, :3].cpu())
                    gt_normals.append(vis[:800, 800:1600, :3].cpu())
                    gt_depths.append(vis[:800, 1600:, :3].cpu())
                    pred_images.append(vis[800:1600, :800, :3].cpu())
                    pred_normals.append(vis[800:1600, 800:1600, :3].cpu())
                    pred_depths.append(vis[800:1600, 1600:, :3].cpu())
                    
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
                        (pred_pretty_meshes[-1], 'pred_mesh')
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
                            'pred_depth': pred_depths
                        }
                        for name, imgs in video_dict.items():
                            if 'test' in split_name:
                                video_dir = f'eval/extract/test_view{view_num}'
                            else:
                                video_dir = f'eval/extract/{split_name}'
                            self.experiment.dump_images2video(
                                video_dir,
                                name=name, 
                                images=imgs, 
                                downsample=1, 
                                fps=48, 
                                duration=5
                            )
                        del gt_images, gt_normals, gt_depths, gt_pretty_meshes, pred_images, pred_normals, pred_depths, pred_pretty_meshes
                        torch.cuda.empty_cache()
                        gc.collect()
                        gt_images, gt_normals, gt_depths, gt_pretty_meshes = [], [], [], []
                        pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
                else:
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
                                video_dir = f'eval/extract/test_view{view_num}'
                            else:
                                video_dir = f'eval/extract/{split_name}'
                            self.experiment.dump_images2video(
                                video_dir,
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
        # self.load = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/psdf_blender/cat/debug1.1/task.py')

        print(f'Processing: {self.load}...')
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            self.experiment: DS_Experiment = task.experiment
            self.model: D_Joint = task.model
            self.model.background_color = 'white'
            self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
            self.trainer: D_JointTrainer = task.trainer
            self.trainer.setup(model=self.model, dataset=self.dataset)
            self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
        self.model.eval()

        # self.process_loader('orbit', export_pred_light=False)
        # self.process_loader('fix', export_pred_light=False)
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
        full_batch_size = self.dataset.get_size(split='fix_vis') # full time size
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='fix_vis')

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
                trainer_mode='fix_vis',
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
                normal_bg = torch.tensor([0, 0, 1]).float().to(self.device) # 单位向量作为normal的背景
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
                    pred_mesh.replace_(vertices=(inv_trans @ (pred_mesh.vertices * (3/2) * 2).unsqueeze(-1)).squeeze(-1)) # * (3/2) 是把超参数scale还原回去（还原为gt的0.5范围）， * 2 是把0.5的范围映射到1
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
        # self.load = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/psdf_blender/cat/debug1.1/task.py')
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

        self.process_loader()
        del self.model, self.dataset, self.trainer, self.experiment
        torch.cuda.empty_cache()
        gc.collect()

@dataclass
class Gather_results(Task):
    step: Optional[int] = None

    load: Optional[Path] = None

    def process_loader(self, split_name, export_pred_light=False):
        full_batch_size = self.dataset.get_size(split='fix_vis') # full time size
        if 'orbit' in split_name:
            loader = self.dataset.get_orbit_vis_iter(batch_size=50, shuffle=False, infinite=False)
        elif 'fix' in split_name:
            loader = self.dataset.get_fix_vis_iter(batch_size=50, shuffle=False, infinite=False)
        elif 'test' in split_name:
            loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='fix_vis')
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )
        normal_shader = NormalShader(
            antialias=True, 
            normal_type='vertex'
        )

        # 初始化存储容器
        gt_images, gt_normals, gt_depths, gt_pretty_meshes = [], [], [], []
        pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
        pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs = [], [], [], []

        iter_count = 0
        with tqdm(total=full_batch_size * 6, desc="Evaluating and Extracting results", unit="frame") as pbar:
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
                        self.experiment.dump_image('{self.load.parent.parent}/gather_image', image=vis[1600:2400, 800:, :3], name='pred_light')
                        export_pred_light = False
                    pbr_roughnesses.append(vis[2400:3200, :800, :3].cpu())
                    pbr_metallics.append(vis[2400:3200, 800:1600, :3].cpu())
                    pbr_occs.append(vis[2400:3200, 1600:, :3].cpu())

                    # 保存 GT 图像
                    for idx, (img, sub) in enumerate([
                        (gt_images[-1], 'gt_image'), 
                        (gt_normals[-1], 'gt_normal'), 
                        # (gt_depths[-1], 'gt_depth'),
                        (gt_pretty_meshes[-1], 'gt_mesh')
                    ]):
                        self.experiment.dump_image(
                            f'{self.load.parent.parent}/gather_image/{split_name}/frame_{frame_id}',
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
                        del gt_images, gt_normals, gt_depths, gt_pretty_meshes, pred_images, pred_normals, pred_depths, pred_pretty_meshes, pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs
                        torch.cuda.empty_cache()
                        gc.collect()
                        gt_images, gt_normals, gt_depths, gt_pretty_meshes = [], [], [], []
                        pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
                        pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs = [], [], [], []
                    pbar.update(1)
                iter_count += 1
                torch.cuda.empty_cache()
                
        del gt_images, gt_normals, gt_depths, gt_pretty_meshes, pred_images, pred_normals, pred_depths, pred_pretty_meshes, pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs, loader, gt_meshes
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
            self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
            self.trainer: D_JointTrainer = task.trainer
            self.trainer.setup(model=self.model, dataset=self.dataset)
            self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
        self.model.eval()

        self.process_loader('test_view', export_pred_light=True)
        # self.process_loader('orbit', export_pred_light=False)
        # self.process_loader('fix', export_pred_light=False)
        
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
        full_batch_size = self.dataset.get_size(split='fix_vis') # full time size
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
                        
                    gt_mesh_path = self.gt_mesh / f'frame{frame_id}.obj'
                    gt_mesh_pkl_path = gt_mesh_path.with_suffix('.pkl')
                    if gt_mesh_pkl_path.exists():
                        gt_mesh = DS_TriangleMesh.deserialize(gt_mesh_pkl_path).to(self.device)
                    else:
                        gt_mesh = DS_TriangleMesh.from_file(gt_mesh_path, read_mtl=False).to(self.device)
                        gt_mesh.serialize(pred_mesh_pkl_path)
                    
                    pred_mesh.export(
                        path=self.experiment.dump_file_path(subfolder=f'{self.load.parent.parent}/gather_mesh/frame_{frame_id}', file_name=f'{self.method_name}.obj'),
                        only_geometry=True
                    )
                    gt_mesh.export(
                        path=self.experiment.dump_file_path(subfolder=f'{self.load.parent.parent}/gather_mesh/frame_{frame_id}', file_name=f'gt.obj'),
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
class EvalCubeCurve(Task):

    load: Path = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/psdf_blender/toy/debug2.1/task.py')

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
                # def get_next_frame_mesh(self, dt: float, compute_scene_flow: bool = False) -> DS_TriangleMesh:
                    # assert not (self.vertices_scene_flow is None and not compute_scene_flow), "Scene flow must be computed before next frame mesh can be generated."
                    # if self.vertices_scene_flow is None:
                    #     self.compute_scene_flow()
                    # if compute_scene_flow:
                    #     self.compute_scene_flow()
                    # vertices = self.vertices + self.vertices_scene_flow * dt  # Update vertices by scene flow
                    
                    # return self.replace(vertices=vertices)
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
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/psdf_blender')
        cases = []
        for category in categories:
            # case路径
            case = root / category / 'testek' / 'task.py'
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

def compute_optical_flow_from_scene_flow(camera, pred_mesh,dt):
        # ===================== [修改] =====================
        #  我们将完全使用MVP矩阵和nvdiffrast的坐标系来计算一切
        # ================================================
        
        ctx = dr.RasterizeCudaContext(camera.device)
        H, W = camera.height.item(), camera.width.item()

        # --- 步骤 1: 将当前帧和下一帧的顶点都转换到裁剪空间 ---
        # 准备当前帧的顶点
        verts_t0 = pred_mesh.vertices
        verts_t0_hom = torch.cat([verts_t0, torch.ones_like(verts_t0[..., :1])], dim=-1) # 转齐次坐标 (V, 4)

        # 准备下一帧的顶点 (基于3D场景流)
        verts_t1 = verts_t0 + pred_mesh.vertices_scene_flow * dt
        verts_t1_hom = torch.cat([verts_t1, torch.ones_like(verts_t1[..., :1])], dim=-1) # 转齐次坐标 (V, 4)
        
        # 获取MVP矩阵
        mvp = camera.projection_matrix @ camera.view_matrix
        
        # 将两组顶点都投影到裁剪空间 (Clip Space)
        clip_t0 = (mvp @ verts_t0_hom.T).T.contiguous() # (V, 4)
        clip_t1 = (mvp @ verts_t1_hom.T).T.contiguous() # (V, 4)

        # --- 步骤 2: 从裁剪空间计算像素坐标和光流 ---
        # 透视除法，得到NDC坐标 (-1, 1)
        ndc_t0 = clip_t0[..., :2] / clip_t0[..., 3:4]
        ndc_t1 = clip_t1[..., :2] / clip_t1[..., 3:4]
        
        # 从NDC坐标转换到屏幕像素坐标 (0, W) 和 (0, H)
        # 公式: screen = (ndc + 1) * 0.5 * image_size
        # nvdiffrast的Y轴是向下的, 和常规图像坐标系一致，通常不需要反转
        viewport = torch.tensor([W, H], dtype=torch.float32, device=camera.device)
        uv0 = (ndc_t0 + 1) * 0.5 * viewport
        uv1 = (ndc_t1 + 1) * 0.5 * viewport
        
        # 计算每个顶点的2D光流 (单位：像素)
        flow = uv1 - uv0 # (V, 2)

        # --- 步骤 3: 光栅化和插值 (这部分和你原来的一样) ---
        indices = pred_mesh.indices.int()
        
        # 注意：光栅化器需要 (B, V, 4) 的输入，B=1
        with dr.DepthPeeler(ctx, clip_t0[None, ...], indices, resolution=(H, W)) as peeler:
            rast, _ = peeler.rasterize_next_layer()
        
        alphas = (rast[..., -1:] > 0).float()

        # 插值我们刚刚正确计算出的光流
        flow_attr = flow[None, ...]  # (1, V, 2)
        flow_image, _ = dr.interpolate(flow_attr, rast, indices)
        flow_image = flow_image[0]

        return flow_image, alphas[0]

def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_uv = flow_uv.detach().cpu().numpy() if torch.is_tensor(flow_uv) else flow_uv
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel

def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return torch.from_numpy(flow_image).float() / 255.0

@dataclass
class ExtractSceneFlowMap(Task):
    """为训练好的 D_DiffDR 模型评估场景流，生成网格并计算场景流用在mesh上看预测效果。"""

    step: Optional[int] = None

    
    def process_loader(self, split_name):

        full_batch_size = self.dataset.get_size(split='fix_vis') # full time size
        loader = self.dataset.get_fix_vis_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='orbit_vis')
        
        iter_count = 0
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

                if frame_id < full_batch_size-1:
                    gt_mesh = gt_meshes[frame_id].clone()
                    next_gt_mesh = gt_meshes[frame_id+1].clone()
                    scene_flow = next_gt_mesh.vertices - gt_mesh.vertices
                    scene_flow = scene_flow / camera.dts.item()  
                    if gt_mesh.vertices_scene_flow is None:
                        gt_mesh.annotate_(vertices_scene_flow=scene_flow)
                    else:
                        gt_mesh.replace_(vertices_scene_flow=scene_flow)
                    gt_optical_flow_map,_= compute_optical_flow_from_scene_flow(camera, gt_mesh, dt=camera.dts.item())
                    gt_optical_flow_flow_vis = flow_to_image(gt_optical_flow_map.cpu())
                    self.experiment.dump_image(
                        f'eval/extract/{split_name}/gt_optical_flow_flow_vis/',
                        image=gt_optical_flow_flow_vis,
                        name=f'{frame_id}'
                    )

                pred_mesh = pred_meshes[i].clone()
                optical_flow_map,_ = compute_optical_flow_from_scene_flow(camera, pred_mesh, dt=camera.dts.item())
                optical_flow_flow_vis = flow_to_image(optical_flow_map.cpu())
                self.experiment.dump_image(
                    f'eval/extract/{split_name}/optical_flow_flow_vis/',
                    image=optical_flow_flow_vis,
                    name=f'{frame_id}'
                )

            iter_count += 1
            torch.cuda.empty_cache()
                
        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def run(self) -> None:
        categories = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        # categories = ['toy']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/psdf_blender')
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
            self.model.reg_scene_flow_smoothness_able = True
            
            self.process_loader('fix')

            del self.model, self.dataset, self.trainer, self.experiment
            torch.cuda.empty_cache()
            gc.collect()

@dataclass
class ExtractArbitraryTimeResolution(Task):
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
                        inter_mesh_optical_flow_map,_ = compute_optical_flow_from_scene_flow(camera, inter_mesh, dt=dt_)
                        inter_optical_flow_flow_vis = flow_to_image(inter_mesh_optical_flow_map.cpu())
                        self.experiment.dump_image(
                            f'eval/extract/{split_name}/pred_interpolated_optical_flow_flow_vis',
                            image=inter_optical_flow_flow_vis,
                            name=f'{frame_id}_interp{j}'
                        )
                        
                        inter_scene_flow_pred_next_frame_mesh = inter_mesh.get_next_frame_mesh(dt_)
                        interpolate_scene_flow_pred_next_frame_pretty_meshes.append(
                            inter_scene_flow_pred_next_frame_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                        )
                        self.experiment.dump_image(
                            f'eval/extract/{split_name}/pred_interpolated_scene_flow/pred_interpolated_scene_flow_mesh',
                            image=interpolate_scene_flow_pred_next_frame_pretty_meshes[-1],
                            name=f'{frame_id}_interp{j+1}'
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
        categories = ['spidermanfight']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/psdf_blender')
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
            self.model.reg_scene_flow_smoothness_able = True
            
            self.process_loader('fix')

            del self.model, self.dataset, self.trainer, self.experiment
            torch.cuda.empty_cache()
            gc.collect()

@dataclass
class Relighter(Task):
    step: Optional[int] = None

    load: Optional[Path] = None

    def process_loader(self, split_name, export_pred_light=False):
        full_batch_size = self.dataset.get_size(split='fix_vis') # full time size
        if 'orbit' in split_name:
            loader = self.dataset.get_orbit_vis_iter(batch_size=50, shuffle=False, infinite=False)
        elif 'fix' in split_name:
            loader = self.dataset.get_fix_vis_iter(batch_size=50, shuffle=False, infinite=False)
        elif 'test' in split_name:
            loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='fix_vis')
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )

        # 初始化存储容器
        gt_images, gt_normals, gt_depths, gt_pretty_meshes = [], [], [], []
        pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
        pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs = [], [], [], []

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

                # pred_mesh = pred_meshes[i].clone()
                # pred_pretty_meshes.append(
                #     pred_mesh.render(camera, shader=pretty_shader).rgb2srgb().blend((1, 1, 1)).item().clamp(0, 1).cpu()
                # )

                # 切分可视化图像
                pred_images.append(vis[800:1600, :800, :3].cpu())
                if export_pred_light:
                    self.experiment.dump_image('eval/extract', image=vis[1600:2400, 800:, :3], name='relight')
                    export_pred_light = False

                # 保存 Pred 图像
                for idx, (img, sub) in enumerate([
                    (pred_images[-1], 'pred_image'), 
                ]):
                    self.experiment.dump_image(f'eval/extract/bridge_relight_{split_name}/pred/{sub}', image=img, name=f'{frame_id}')

                if frame_id == full_batch_size - 1:
                    # 导出视频
                    video_dict = {

                        'pred_image': pred_images,
                    }
                    for name, imgs in video_dict.items():
                        if 'test' in split_name:
                            video_dir = f'eval/extract/bridge_relight_test_view{view_num}'
                        else:
                            video_dir = f'eval/extract/bridge_relight_{split_name}'
                        self.experiment.dump_images2video(
                            video_dir,
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
        self.relight_envmap = Path('/data3/gaochong/project/RadianceFieldStudio/data/tensoir/bridge.hdr')
        print(f'Processing: {self.load}...  Rlighting with {self.relight_envmap}')
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            self.experiment: DS_Experiment = task.experiment
            self.model: D_Joint = task.model
            self.model.background_color = 'white'
            self.model.set_relight_envmap(self.relight_envmap)
            self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
            self.trainer: D_JointTrainer = task.trainer
            self.trainer.setup(model=self.model, dataset=self.dataset)
            self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
        self.model.eval()

        # self.process_loader('orbit', export_pred_light=False)
        self.process_loader('fix', export_pred_light=False)
        # self.process_loader('test_view', export_pred_light=True)

        del self.model, self.dataset, self.trainer, self.experiment
        torch.cuda.empty_cache()
        gc.collect()


@dataclass
class Extract_temporal_results(Task):
    step: Optional[int] = None

    load: Optional[Path] = None

    def process_test_loader(self):
        full_batch_size = self.dataset.get_size(eval_split='test', split='fix_vis') # full time size
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='test')
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )

        # 初始化存储容器
        gt_images, gt_normals, gt_depths, gt_pretty_meshes = [], [], [], []
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
                val_pbr_attr=False if 'nopbr' in str(self.load) else True,
                return_pred_mesh=True,
            )
            
            for i in range(len(visualization)):
                count_id = iter_count * 50 + i

                view_num = count_id // full_batch_size
                split_name = f'view{view_num}'
                frame_id = count_id % full_batch_size

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
                if 'nopbr' in str(self.load):
                    gt_images.append(vis[:800, :800, :3].cpu())
                    gt_normals.append(vis[:800, 800:1600, :3].cpu())
                    gt_depths.append(vis[:800, 1600:, :3].cpu())
                    pred_images.append(vis[800:1600, :800, :3].cpu())
                    pred_normals.append(vis[800:1600, 800:1600, :3].cpu())
                    pred_depths.append(vis[800:1600, 1600:, :3].cpu())
                    
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
                        (pred_pretty_meshes[-1], 'pred_mesh')
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
                            'pred_depth': pred_depths
                        }
                        for name, imgs in video_dict.items():
                            if 'test' in split_name:
                                video_dir = f'eval/extract/test_view{view_num}'
                            else:
                                video_dir = f'eval/extract/{split_name}'
                            self.experiment.dump_images2video(
                                video_dir,
                                name=name, 
                                images=imgs, 
                                downsample=1, 
                                fps=48, 
                                duration=5
                            )
                        del gt_images, gt_normals, gt_depths, gt_pretty_meshes, pred_images, pred_normals, pred_depths, pred_pretty_meshes
                        torch.cuda.empty_cache()
                        gc.collect()
                        gt_images, gt_normals, gt_depths, gt_pretty_meshes = [], [], [], []
                        pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
                else:
                    gt_images.append(vis[:800, :800, :3].cpu())
                    gt_normals.append(vis[:800, 800:1600, :3].cpu())
                    gt_depths.append(vis[:800, 1600:, :3].cpu())
                    pred_images.append(vis[800:1600, :800, :3].cpu())
                    pred_normals.append(vis[800:1600, 800:1600, :3].cpu())
                    pred_depths.append(vis[800:1600, 1600:, :3].cpu())
                    pred_kds.append(vis[1600:2400, :800, :3].cpu())
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
                        self.experiment.dump_image(f'eval/test_extract/{split_name}/gt/{sub}', image=img, name=f'{frame_id}')

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
                        self.experiment.dump_image(f'eval/test_extract/{split_name}/pred/{sub}', image=img, name=f'{frame_id}')

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
                                video_dir = f'eval/test_extract/{split_name}'
                            else:
                                video_dir = f'eval/test_extract/{split_name}'
                            self.experiment.dump_images2video(
                                video_dir,
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

    def process_train_loader(self):
        full_batch_size = self.dataset.get_size(eval_split='train', split='fix_vis') # full time size
        loader = self.dataset.get_train_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='train')
        pretty_shader = PrettyShader(
            z_up=True,
            antialias=True,
            culling=True,
        )
        padding_size = self.dataset.dataparser.padding_size
        true_full_batch_size = full_batch_size - 2 * padding_size

        # 初始化存储容器
        gt_images, gt_normals, gt_depths, gt_pretty_meshes = [], [], [], []
        pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
        pred_kds, pbr_roughnesses, pbr_metallics, pbr_occs = [], [], [], []

        iter_count = 0
        for inputs, gt_outputs, indices in loader:
            _, _, visualization, pred_meshes = self.trainer.step(
                self.model,
                inputs,
                gt_outputs,
                indices=indices,
                mode='train',
                visual=True,
                val_pbr_attr=False if 'nopbr' in str(self.load) else True,
                return_pred_mesh=True,
            )
            
            for i in range(len(visualization)):
                count_id = iter_count * 50 + i

                view_num = count_id // full_batch_size
                split_name = f'view{view_num}'
                frame_id = count_id % full_batch_size
                if frame_id < padding_size or frame_id >= (true_full_batch_size + padding_size):
                    continue
                true_frame_id = frame_id - padding_size
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
                if 'nopbr' in str(self.load):
                    gt_images.append(vis[:800, :800, :3].cpu())
                    gt_normals.append(vis[:800, 800:1600, :3].cpu())
                    gt_depths.append(vis[:800, 1600:, :3].cpu())
                    pred_images.append(vis[800:1600, :800, :3].cpu())
                    pred_normals.append(vis[800:1600, 800:1600, :3].cpu())
                    pred_depths.append(vis[800:1600, 1600:, :3].cpu())
                    
                    # 保存 GT 图像
                    for idx, (img, sub) in enumerate([
                        (gt_images[-1], 'gt_image'), 
                        (gt_normals[-1], 'gt_normal'), 
                        (gt_depths[-1], 'gt_depth'),
                        (gt_pretty_meshes[-1], 'gt_mesh')
                    ]):
                        self.experiment.dump_image(f'eval/train_extract/{split_name}/gt/{sub}', image=img, name=f'{frame_id}')

                    # 保存 Pred 图像
                    for idx, (img, sub) in enumerate([
                        (pred_images[-1], 'pred_image'), 
                        (pred_normals[-1], 'pred_normal'), 
                        (pred_depths[-1], 'pred_depth'),
                        (pred_pretty_meshes[-1], 'pred_mesh')
                    ]):
                        self.experiment.dump_image(f'eval/train_extract/{split_name}/pred/{sub}', image=img, name=f'{frame_id}')

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
                            'pred_depth': pred_depths
                        }
                        for name, imgs in video_dict.items():
                            if 'test' in split_name:
                                video_dir = f'eval/extract/test_view{view_num}'
                            else:
                                video_dir = f'eval/extract/{split_name}'
                            self.experiment.dump_images2video(
                                video_dir,
                                name=name, 
                                images=imgs, 
                                downsample=1, 
                                fps=48, 
                                duration=5
                            )
                        del gt_images, gt_normals, gt_depths, gt_pretty_meshes, pred_images, pred_normals, pred_depths, pred_pretty_meshes
                        torch.cuda.empty_cache()
                        gc.collect()
                        gt_images, gt_normals, gt_depths, gt_pretty_meshes = [], [], [], []
                        pred_images, pred_normals, pred_depths, pred_pretty_meshes = [], [], [], []
                else:
                    gt_images.append(vis[:800, :800, :3].cpu())
                    gt_normals.append(vis[:800, 800:1600, :3].cpu())
                    gt_depths.append(vis[:800, 1600:, :3].cpu())
                    pred_images.append(vis[800:1600, :800, :3].cpu())
                    pred_normals.append(vis[800:1600, 800:1600, :3].cpu())
                    pred_depths.append(vis[800:1600, 1600:, :3].cpu())
                    pred_kds.append(vis[1600:2400, :800, :3].cpu())
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
                        self.experiment.dump_image(f'eval/train_extract/{split_name}/gt/{sub}', image=img, name=f'{true_frame_id}')

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
                        self.experiment.dump_image(f'eval/train_extract/{split_name}/pred/{sub}', image=img, name=f'{true_frame_id}')

                    if true_frame_id == true_full_batch_size - 1:
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
                                video_dir = f'eval/train_extract/{split_name}'
                            else:
                                video_dir = f'eval/train_extract/{split_name}'
                            self.experiment.dump_images2video(
                                video_dir,
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
        # self.load = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/psdf_blender/cat/debug1.1/task.py')

        print(f'Processing: {self.load}...')
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            self.experiment: DS_Experiment = task.experiment
            self.model: D_Joint = task.model
            self.model.background_color = 'white'
            self.dataset: SyntheticTemporalDynamicMultiviewBlenderRGBADataset = task.dataset
            self.trainer: D_JointTrainer = task.trainer
            self.trainer.setup(model=self.model, dataset=self.dataset)
            self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
        self.model.eval()

        self.process_test_loader()
        self.process_train_loader()

        del self.model, self.dataset, self.trainer, self.experiment
        torch.cuda.empty_cache()
        gc.collect()


@dataclass
class Eval_Temporal_results(Task):
    """渲染训练好的模型"""
    step: Optional[int] = None

    load: Optional[Path] = None

    def process_test_loader(self):
        full_batch_size = self.dataset.get_size(eval_split='test', split='fix_vis') # full time size
        loader = self.dataset.get_test_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='test')

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
                trainer_mode='test',
                return_pred_mesh=True,
                return_pbr_attrs=False,
            )
            (
                pred_outputs,
                reg_loss_dict,
                pred_meshes,
                pred_depths, gt_depths,
                pred_normals, gt_normals,
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
                normal_bg = torch.tensor([0, 0, 1]).float().to(self.device) # 单位向量作为normal的背景
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
                
                self.experiment.log(f"Test view {view_num}, Frame {frame_id}: PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}, MAE: {mae}", new_logfile='eval_test.txt')
                psnrs.append(psnr)
                ssims.append(ssim)
                lpipss.append(lpips)
                normal_maes.append(mae)

                if view_num == 0:
                    inv_trans = torch.tensor([[1,0,0],[0,0,1],[0,-1,0]]).to(pred_mesh.vertices)
                    pred_mesh.replace_(vertices=(inv_trans @ (pred_mesh.vertices * (3/2) * 2).unsqueeze(-1)).squeeze(-1)) # * (3/2) 是把超参数scale还原回去（还原为gt的0.5范围）， * 2 是把0.5的范围映射到1
                    gt_mesh.replace_(vertices=(inv_trans @ (gt_mesh.vertices * (3/2) * 2).unsqueeze(-1)).squeeze(-1))
                    chamfer_dist = ChamferDistanceMetric(target_num_points=1000000)(gt_mesh, pred_mesh)
                    chamfer_dists.append(chamfer_dist)
                    pred_mesh.export(
                        path=self.experiment.dump_file_path(subfolder='eval/test_scale_pred_mesh', file_name=f'frame{frame_id}.obj'),
                        only_geometry=True
                    )
                    gt_mesh.export(
                        path=self.experiment.dump_file_path(subfolder='eval/test_scale_gt_mesh', file_name=f'frame{frame_id}.obj'),
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

    def process_train_loader(self):
        full_batch_size = self.dataset.get_size(eval_split='train', split='fix_vis') # full time size
        loader = self.dataset.get_train_iter(batch_size=50, shuffle=False, infinite=False)
        gt_meshes = self.dataset.get_meshes(split='train')
        padding_size = self.dataset.dataparser.padding_size
        true_full_batch_size = full_batch_size - 2 * padding_size

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
                trainer_mode='train',
                return_pred_mesh=True,
                return_pbr_attrs=False,
            )
            (
                pred_outputs,
                reg_loss_dict,
                pred_meshes,
                pred_depths, gt_depths,
                pred_normals, gt_normals,
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
                if frame_id < padding_size or frame_id >= (true_full_batch_size + padding_size):
                    continue
                true_frame_id = frame_id - padding_size

                gt_image = gt_rgbs[i].clamp(0, 1)
                pred_image = pred_rgbs[i].clamp(0, 1)
                psnr = PSNRLoss()(gt_image, pred_image)
                ssim = (1 - SSIMLoss()(gt_image, pred_image))
                lpips = LPIPSLoss()(gt_image, pred_image)

                camera = inputs[i]
                normal_bg = torch.tensor([0, 0, 1]).float().to(self.device) # 单位向量作为normal的背景
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
                
                self.experiment.log(f"Test view {view_num}, Frame {true_frame_id}: PSNR: {psnr}, SSIM: {ssim}, LPIPS: {lpips}, MAE: {mae}", new_logfile='eval_train.txt')
                psnrs.append(psnr)
                ssims.append(ssim)
                lpipss.append(lpips)
                normal_maes.append(mae)

                if view_num == 0:
                    inv_trans = torch.tensor([[1,0,0],[0,0,1],[0,-1,0]]).to(pred_mesh.vertices)
                    pred_mesh.replace_(vertices=(inv_trans @ (pred_mesh.vertices * (3/2) * 2).unsqueeze(-1)).squeeze(-1)) # * (3/2) 是把超参数scale还原回去（还原为gt的0.5范围）， * 2 是把0.5的范围映射到1
                    gt_mesh.replace_(vertices=(inv_trans @ (gt_mesh.vertices * (3/2) * 2).unsqueeze(-1)).squeeze(-1))
                    chamfer_dist = ChamferDistanceMetric(target_num_points=1000000)(gt_mesh, pred_mesh)
                    chamfer_dists.append(chamfer_dist)
                    pred_mesh.export(
                        path=self.experiment.dump_file_path(subfolder='eval/train_scale_pred_mesh', file_name=f'frame{true_frame_id}.obj'),
                        only_geometry=True
                    )
                    gt_mesh.export(
                        path=self.experiment.dump_file_path(subfolder='eval/train_scale_gt_mesh', file_name=f'frame{true_frame_id}.obj'),
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
        # self.load = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/psdf_blender/cat/debug1.1/task.py')
        print(f'Processing: {self.load}...')
        with console.status(desc='Loading Model'):
            task = DS_TrainTask.load_from_script(self.load, step=self.step)
            self.experiment: DS_Experiment = task.experiment
            self.model: D_Joint = task.model
            self.dataset: SyntheticTemporalDynamicMultiviewBlenderRGBADataset = task.dataset
            self.trainer: D_JointTrainer = task.trainer
            # self.dataset.__setup__()
            self.trainer.setup(model=self.model, dataset=self.dataset)
            self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
        self.model.eval()

        self.process_test_loader()
        self.process_train_loader()
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
        
        # temporal train task
        temp_toy = temporal_toy_task,
        temp_spidermanfight = temporal_spidermanfight_task,
        
        # export task
        exportmodel = ExportModel(cuda=0),
        
        # eval task
        extract = Extract_results(cuda=0), 
        eval = Eval_results(cuda=0), 
        evaltemporal = Eval_Temporal_results(cuda=0), 
        extracttemporal = Extract_temporal_results(cuda=0), 
        gather = Gather_results(cuda=0),
        gathermesh = Gather_mesh(cuda=0),

        rendermodel = RenderAfterTrain(cuda=0), 
        evalcubecurve = EvalCubeCurve(cuda=0),
        evalsceneflow = EvalSceneFlow(cuda=0),
        extractsceneflowmap = ExtractSceneFlowMap(cuda=0),
        extractarbitrarytime = ExtractArbitraryTimeResolution(cuda=0),
        relight = Relighter(cuda=0),
    ).run()
