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
from rfstudio_ds.model import D_Joint_S2 # rgb image loss optimization model
from rfstudio_ds.trainer import D_Joint_S2Trainer, JointRegularizationConfig_S2 # rgb image loss optimization trainer, dynamic nvdiffrec
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
    model=D_Joint_S2(
        load='/data3/gaochong/project/RadianceFieldStudio/outputs/psdf_cmu/band1/test_s1_v1/dump/model.pth',
        # geometry_residual = Grid4d_HashEncoding(
        #     decoder=MLPDecoderNetwork(
        #         layers=[-1, 96, 64, 25],
        #         activation='sigmoid',
        #         bias=False,
        #         initialization='kaiming-uniform',
        #     ),
        #     grad_scaling=16.0,
        #     backend='grid4d',
        #     deform_base_resolution=[32,32,4],
        #     deform_desired_resolution=[2048, 2048, 32],
        #     deform_num_levels=32,
        # ),
        geometry_sdf_residual = Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 32, 32, 1],
                activation='tanh',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[32,32,4],
            deform_desired_resolution=[2048, 2048, 32],
            deform_num_levels=32,
        ),
        geometry_deform_residual = Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 32, 32, 3],
                activation='none',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[32,32,4],
            deform_desired_resolution=[2048, 2048, 32],
            deform_num_levels=32,
        ),
        geometry_weight_residual = Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 96, 32, 21],
                activation='none',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[32,32,4],
            deform_desired_resolution=[2048, 2048, 32],
            deform_num_levels=32,
        ),
        targe_high_flexicube_res=160,
        geometry_residual_enabled=True,
        reg_geometry_residual_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 4e-2],

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
    experiment=DS_Experiment(name='psdf_cmu/band1', timestamp='test_s2'),
    trainer=D_Joint_S2Trainer(
        num_steps=5000,
        batch_size=2,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_save=1000,
        hold_after_train=False,

        geometry_residual_lr=5e-4, # 用于flexicube的 residual 参数
        geometry_residual_lr_decay=1500,
        
        static_sdf_params_learning_rate=1e-3,
        static_sdf_params_decay=800, # 控制学习率的半衰期，在trainer中默认是800
        static_sdf_params_max_norm=None, # 控制系数优化的最大step，在trainer中默认没有开启
        
        poly_coeffs_learning_rate=5e-4,
        poly_coeffs_decay=800,
        poly_coeffs_max_norm=None,

        fourier_low_learning_rate = 5e-4,
        fourier_low_decay=800,
        fourier_low_max_norm=0.005,

        fourier_mid_learning_rate = 1e-4,
        fourier_mid_decay=1200,
        fourier_mid_max_norm=0.005,

        fourier_high_learning_rate = 1e-4, 
        fourier_high_decay=1600,
        fourier_high_max_norm=0.005,

        appearance_learning_rate=5e-4,
        appearance_decay=1500,

        light_learning_rate=1e-3, # 用于light的参数
        light_learning_rate_decay=800,
        
        use_multi_model_stage=True,
        # model_start_steps=[200,200,200,200,200],  # static, poly, fourier_low, fourier_mid, fourier_high

        regularization=JointRegularizationConfig_S2(
            ssim_weight_begin = 10., # ssim loss
            ssim_weight_end  = 10.,
            ssim_weight_decay_steps = 2500,
            ssim_weight_start_step = 0,

            mask_weight_begin = 1., # mask supervision loss
            mask_weight_end = 1.,
            mask_weight_decay_steps = 1250,
            mask_weight_start_step = 0,
            
            # geometry regularization loss
            # geometry residual temporal hashgrid loss
            reg_geometry_residual_temporal_hashgrid_begin = 0.1,
            reg_geometry_residual_temporal_hashgrid_end = 0.01,
            reg_geometry_residual_temporal_hashgrid_decay_steps = 1250,
            reg_geometry_residual_temporal_hashgrid_start_step = -1,
            reg_geometry_residual_temporal_hashgrid_end_step=-1,
            
            geometry_residual_weight_begin=0.05,
            geometry_residual_weight_end=1,
            geometry_residual_decay_steps=500,
            geometry_residual_start_step=0,
            geometry_residual_end_step=-1,

            # sdf entropy loss
            sdf_entropy_weight_begin = 0.01, # 控制sdf entropy loss的权重
            sdf_entropy_weight_end = 0.005,
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
            sdf_eikonal_weight_begin = 0.001, # 控制sdf eikonal loss的权重
            sdf_eikonal_weight_end = 0.001,
            sdf_eikonal_decay_steps = 1250, # 控制sdf eikonal loss的线性衰减步数
            sdf_eikonal_start_step = 0, # 控制sdf eikonal loss的起始步数
            sdf_eikonal_end_step=-1, # 控制sdf eikonal loss的终止步数

            # curve_coeff_tv loss
            curve_coeff_tv_weight_begin = 0.1, # 控制curve coefficient tv smooth loss的权重
            curve_coeff_tv_weight_end = 1,
            curve_coeff_tv_decay_steps = 1250, # 控制curve coefficient tv smooth loss的线性衰减步数
            curve_coeff_tv_start_step = 0, # 控制curve coefficient tv smooth loss的起始步数
            curve_coeff_tv_end_step=-1, # 控制curve coefficient tv smooth loss的终止步数

            # appearance regularization loss
            # appearance temporal hashgrid loss
            reg_appearance_temporal_hashgrid_begin = 0.01,
            reg_appearance_temporal_hashgrid_end = 0.01,
            reg_appearance_temporal_hashgrid_decay_steps = 1250,
            reg_appearance_temporal_hashgrid_start_step = -1,
            reg_appearance_temporal_hashgrid_end_step=-1,

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




if __name__ == '__main__':
    TaskGroup(
        # train task
        band1 = cmu_band1_task,
        test = cmu_band1_task,
    ).run()
