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
from rfstudio_ds.model import D_Joint_S2 # rgb image loss optimization model
from rfstudio_ds.trainer import D_Joint_S2Trainer, JointRegularizationConfig_S2 # rgb image loss optimization trainer, dynamic nvdiffrec
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
    model=D_Joint_S2(
        load='/data3/gaochong/project/RadianceFieldStudio/outputs/psdf_blender/toy/test_s1/dump/model.pth',
        geometry_sdf_residual = Grid4d_HashEncoding(
            decoder=MLPDecoderNetwork(
                layers=[-1, 32, 32, 1],
                activation='tanh',
                bias=False,
                initialization='kaiming-uniform',
            ),
            grad_scaling=16.0,
            backend='grid4d',
            deform_base_resolution=[32,32,32],
            deform_desired_resolution=[2048, 2048, 128],
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
            deform_base_resolution=[32,32,32],
            deform_desired_resolution=[2048, 2048, 128],
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
            deform_base_resolution=[32,32,32],
            deform_desired_resolution=[2048, 2048, 128],
            deform_num_levels=32,
        ),
        geometry_residual_enabled=True,
        reg_geometry_residual_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 1e-2],
        
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
    experiment=DS_Experiment(name='psdf_blender/toy', timestamp='test_s2'),
    trainer=D_Joint_S2Trainer(
        num_steps=2500,
        batch_size=4,
        num_steps_per_val=50,
        num_steps_per_val_pbr_attr=250,
        num_steps_per_fix_vis=1000,
        num_steps_per_save=1000,
        hold_after_train=False,

        geometry_residual_lr=1e-3, # 用于flexicube的 residual 参数
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
            # main supervision loss
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
        full_fix_vis_after_train=True,
        full_orbit_vis_after_train=True,
        detect_anomaly=True,
    ),
    cuda=0,
    seed=1
)

# cat_task = DS_TrainTask(
#     dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
#         path=Path("data/ObjSel-Dyn/cat/"),
#     ),
#     model=D_Joint_S2(
#         load='/data3/gaochong/project/RadianceFieldStudio/outputs/psdf_blender/cat/test_s1/dump/model.pth',
#         geometry_residual = Grid4d_HashEncoding(
#             decoder=MLPDecoderNetwork(
#                 layers=[-1, 96, 64, 25],
#                 activation='sigmoid',
#                 bias=False,
#                 initialization='kaiming-uniform',
#             ),
#             grad_scaling=16.0,
#             backend='grid4d',
#             deform_base_resolution=[32,32,8],
#             deform_desired_resolution=[2048, 2048, 256],
#             deform_num_levels=32,
#         ),
#         geometry_residual_enabled=True,
#         reg_geometry_residual_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
        
#         dynamic_texture=Grid4d_HashEncoding(
#             decoder=MLPDecoderNetwork(
#                 layers=[-1, 96, 32, 6],
#                 activation='sigmoid',
#                 bias=False,
#                 initialization='kaiming-uniform',
#             ),
#             grad_scaling=16.0,
#             backend='grid4d',
#             deform_base_resolution=[16,16,8],
#             deform_desired_resolution=[4096,4096,256],
#             deform_num_levels=32,
#         ),
#         reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 3e-3],
#         background_color="white",
#         shader_type="split_sum_pbr",
#         min_roughness=0.1,
#     ),
#     experiment=DS_Experiment(name='psdf_blender/cat', timestamp='test_s2'),
#     trainer=D_Joint_S2Trainer(
#         num_steps=2500,
#         batch_size=4,
#         num_steps_per_val=50,
#         num_steps_per_val_pbr_attr=250,
#         num_steps_per_fix_vis=1000,
#         num_steps_per_save=1000,
#         hold_after_train=False,

#         geometry_residual_lr=1e-3, # 用于flexicube的 residual 参数
#         geometry_residual_lr_decay=1500,

#         static_sdf_params_learning_rate=1e-3,
#         static_sdf_params_decay=800, # 控制学习率的半衰期，在trainer中默认是800
#         static_sdf_params_max_norm=None, # 控制系数优化的最大step，在trainer中默认没有开启
        
#         poly_coeffs_learning_rate=5e-4,
#         poly_coeffs_decay=800,
#         poly_coeffs_max_norm=None,

#         fourier_low_learning_rate = 5e-4,
#         fourier_low_decay=800,
#         fourier_low_max_norm=0.005,

#         fourier_mid_learning_rate = 1e-4,
#         fourier_mid_decay=1200,
#         fourier_mid_max_norm=0.005,

#         fourier_high_learning_rate = 1e-4, 
#         fourier_high_decay=1600,
#         fourier_high_max_norm=0.005,

#         appearance_learning_rate=5e-4,
#         appearance_decay=1500,

#         light_learning_rate=1e-3, # 用于light的参数
#         light_learning_rate_decay=800,
        
#         use_multi_model_stage=True,
#         # model_start_steps=[200,200,200,200,200],  # static, poly, fourier_low, fourier_mid, fourier_high

#         regularization=JointRegularizationConfig_S2(
#             # main supervision loss
#             ssim_weight_begin = 10., # ssim loss
#             ssim_weight_end  = 10.,
#             ssim_weight_decay_steps = 2500,
#             ssim_weight_start_step = 0,

#             mask_weight_begin = 1., # mask supervision loss
#             mask_weight_end = 1.,
#             mask_weight_decay_steps = 1250,
#             mask_weight_start_step = 0,

#             # geometry regularization loss
#             # geometry residual temporal hashgrid loss
#             reg_geometry_residual_temporal_hashgrid_begin = 0.1,
#             reg_geometry_residual_temporal_hashgrid_end = 0.01,
#             reg_geometry_residual_temporal_hashgrid_decay_steps = 1250,
#             reg_geometry_residual_temporal_hashgrid_start_step = 0,
#             reg_geometry_residual_temporal_hashgrid_end_step=-1,
            
#             geometry_residual_weight_begin=0.05,
#             geometry_residual_weight_end=1,
#             geometry_residual_decay_steps=500,
#             geometry_residual_start_step=0,
#             geometry_residual_end_step=-1,
            
#             # sdf entropy loss
#             sdf_entropy_weight_begin = 0.01, # 控制sdf entropy loss的权重
#             sdf_entropy_weight_end = 0.005,
#             sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
#             sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
#             sdf_entropy_end_step = -1, # 控制sdf entropy loss的终止步数

#             # curve derivative smooth loss
#             time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
#             time_tv_weight_end = 0.05,
#             time_tv_decay_steps = 1250, # 控制curve derivative smooth loss的线性衰减步数
#             time_tv_start_step = 0, # 控制curve derivative smooth loss的起始步数
#             time_tv_end_step=-1, # 控制curve derivative smooth loss的终止步数

#             # sdf eikonal loss，开启后，大概需要占用1G显存
#             sdf_eikonal_weight_begin = 0.001, # 控制sdf eikonal loss的权重
#             sdf_eikonal_weight_end = 0.001,
#             sdf_eikonal_decay_steps = 1250, # 控制sdf eikonal loss的线性衰减步数
#             sdf_eikonal_start_step = 0, # 控制sdf eikonal loss的起始步数
#             sdf_eikonal_end_step=-1, # 控制sdf eikonal loss的终止步数

#             # curve_coeff_tv loss
#             curve_coeff_tv_weight_begin = 0.1, # 控制curve coefficient tv smooth loss的权重
#             curve_coeff_tv_weight_end = 1,
#             curve_coeff_tv_decay_steps = 1250, # 控制curve coefficient tv smooth loss的线性衰减步数
#             curve_coeff_tv_start_step = 0, # 控制curve coefficient tv smooth loss的起始步数
#             curve_coeff_tv_end_step=-1, # 控制curve coefficient tv smooth loss的终止步数

#             # appearance regularization loss
#             # appearance temporal hashgrid loss
#             reg_appearance_temporal_hashgrid_begin = 0.01,
#             reg_appearance_temporal_hashgrid_end = 0.01,
#             reg_appearance_temporal_hashgrid_decay_steps = 1250,
#             reg_appearance_temporal_hashgrid_start_step = 0,
#             reg_appearance_temporal_hashgrid_end_step=-1,

#             reg_occ_begin=0.0,
#             reg_occ_end=0.001,
#             reg_occ_decay_steps=1250,
#             reg_occ_start_step=0,
#             reg_occ_end_step=-1,

#             reg_light_begin=0.001,
#             reg_light_end=0.01,
#             reg_light_decay_steps=1250,
#             reg_light_start_step=0,
#             reg_light_end_step=-1,
#         ),

#         mixed_precision=False,
#         full_test_after_train=False,
#         full_fix_vis_after_train=True,
#         full_orbit_vis_after_train=True,
#         detect_anomaly=True,
#     ),
#     cuda=0,
#     seed=1
# )

# footballplayer_task = DS_TrainTask(
#     dataset=SyntheticDynamicMultiViewBlenderRGBADataset(
#         path=Path("data/ObjSel-Dyn/football_player/"),
#     ),
#     model=D_Joint_S2(
#         load='/data3/gaochong/project/RadianceFieldStudio/outputs/psdf_blender/footballplayer/test_s1/dump/model.pth',
#         geometry_residual = Grid4d_HashEncoding(
#             decoder=MLPDecoderNetwork(
#                 layers=[-1, 96, 64, 25],
#                 activation='sigmoid',
#                 bias=False,
#                 initialization='kaiming-uniform',
#             ),
#             grad_scaling=16.0,
#             backend='grid4d',
#             deform_base_resolution=[32,32,8],
#             deform_desired_resolution=[2048, 2048, 192],
#             deform_num_levels=32,
#         ),
#         geometry_residual_enabled=True,
#         reg_geometry_residual_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 5e-3],
        
#         dynamic_texture=Grid4d_HashEncoding(
#             decoder=MLPDecoderNetwork(
#                 layers=[-1, 96, 32, 6],
#                 activation='sigmoid',
#                 bias=False,
#                 initialization='kaiming-uniform',
#             ),
#             grad_scaling=16.0,
#             backend='grid4d',
#             deform_base_resolution=[16,16,8],
#             deform_desired_resolution=[4096,4096,192],
#             deform_num_levels=32,
#         ),
#         reg_temporal_random_perturb_range=[1e-2, 1e-2, 1e-2, 5e-3],
#         background_color="black",
#         shader_type="split_sum_pbr",
#         min_roughness=0.1,
#     ),
#     experiment=DS_Experiment(name='psdf_blender/footballplayer', timestamp='test_s2'),
#     trainer=D_Joint_S2Trainer(
#         num_steps=2500,
#         batch_size=4,
#         num_steps_per_val=50,
#         num_steps_per_val_pbr_attr=250,
#         num_steps_per_fix_vis=1000,
#         num_steps_per_save=1000,
#         hold_after_train=False,

#         geometry_residual_lr=1e-3, # 用于flexicube的 residual 参数
#         geometry_residual_lr_decay=1500,

#         static_sdf_params_learning_rate=1e-3,
#         static_sdf_params_decay=800, # 控制学习率的半衰期，在trainer中默认是800
#         static_sdf_params_max_norm=None, # 控制系数优化的最大step，在trainer中默认没有开启
        
#         poly_coeffs_learning_rate=5e-4,
#         poly_coeffs_decay=800,
#         poly_coeffs_max_norm=None,

#         fourier_low_learning_rate = 5e-4,
#         fourier_low_decay=800,
#         fourier_low_max_norm=0.005,

#         fourier_mid_learning_rate = 1e-4,
#         fourier_mid_decay=1200,
#         fourier_mid_max_norm=0.005,

#         fourier_high_learning_rate = 1e-4, 
#         fourier_high_decay=1600,
#         fourier_high_max_norm=0.005,

#         appearance_learning_rate=5e-4,
#         appearance_decay=1500,

#         light_learning_rate=1e-3, # 用于light的参数
#         light_learning_rate_decay=800,
        
#         use_multi_model_stage=True,
#         # model_start_steps=[200,200,200,200,200],  # static, poly, fourier_low, fourier_mid, fourier_high

#         regularization=JointRegularizationConfig_S2(
#             # main supervision loss
#             ssim_weight_begin = 10., # ssim loss
#             ssim_weight_end  = 10.,
#             ssim_weight_decay_steps = 2500,
#             ssim_weight_start_step = 0,

#             mask_weight_begin = 1., # mask supervision loss
#             mask_weight_end = 1.,
#             mask_weight_decay_steps = 1250,
#             mask_weight_start_step = 0,

#             # geometry regularization loss
#             # geometry residual temporal hashgrid loss
#             reg_geometry_residual_temporal_hashgrid_begin = 0.1,
#             reg_geometry_residual_temporal_hashgrid_end = 0.01,
#             reg_geometry_residual_temporal_hashgrid_decay_steps = 1250,
#             reg_geometry_residual_temporal_hashgrid_start_step = 0,
#             reg_geometry_residual_temporal_hashgrid_end_step=-1,
            
#             geometry_residual_weight_begin=0.05,
#             geometry_residual_weight_end=1,
#             geometry_residual_decay_steps=500,
#             geometry_residual_start_step=0,
#             geometry_residual_end_step=-1,
            
#             # sdf entropy loss
#             sdf_entropy_weight_begin = 0.01, # 控制sdf entropy loss的权重
#             sdf_entropy_weight_end = 0.005,
#             sdf_entropy_decay_steps = 1250, # 控制sdf entropy loss的线性衰减步数
#             sdf_entropy_start_step = 0, # 控制sdf entropy loss的起始步数
#             sdf_entropy_end_step = -1, # 控制sdf entropy loss的终止步数

#             # curve derivative smooth loss
#             time_tv_weight_begin = 0.1, # 控制curve derivative smooth loss的权重
#             time_tv_weight_end = 0.05,
#             time_tv_decay_steps = 1250, # 控制curve derivative smooth loss的线性衰减步数
#             time_tv_start_step = 0, # 控制curve derivative smooth loss的起始步数
#             time_tv_end_step=-1, # 控制curve derivative smooth loss的终止步数

#             # sdf eikonal loss，开启后，大概需要占用1G显存
#             sdf_eikonal_weight_begin = 0.001, # 控制sdf eikonal loss的权重
#             sdf_eikonal_weight_end = 0.001,
#             sdf_eikonal_decay_steps = 1250, # 控制sdf eikonal loss的线性衰减步数
#             sdf_eikonal_start_step = 0, # 控制sdf eikonal loss的起始步数
#             sdf_eikonal_end_step=-1, # 控制sdf eikonal loss的终止步数

#             # curve_coeff_tv loss
#             curve_coeff_tv_weight_begin = 0.1, # 控制curve coefficient tv smooth loss的权重
#             curve_coeff_tv_weight_end = 1,
#             curve_coeff_tv_decay_steps = 1250, # 控制curve coefficient tv smooth loss的线性衰减步数
#             curve_coeff_tv_start_step = 0, # 控制curve coefficient tv smooth loss的起始步数
#             curve_coeff_tv_end_step=-1, # 控制curve coefficient tv smooth loss的终止步数

#             # appearance regularization loss
#             # appearance temporal hashgrid loss
#             reg_appearance_temporal_hashgrid_begin = 0.01,
#             reg_appearance_temporal_hashgrid_end = 0.01,
#             reg_appearance_temporal_hashgrid_decay_steps = 1250,
#             reg_appearance_temporal_hashgrid_start_step = 0,
#             reg_appearance_temporal_hashgrid_end_step=-1,

#             reg_occ_begin=0.0,
#             reg_occ_end=0.001,
#             reg_occ_decay_steps=1250,
#             reg_occ_start_step=0,
#             reg_occ_end_step=-1,

#             reg_light_begin=0.001,
#             reg_light_end=0.01,
#             reg_light_decay_steps=1250,
#             reg_light_start_step=0,
#             reg_light_end_step=-1,
#         ),

#         mixed_precision=False,
#         full_test_after_train=False,
#         full_fix_vis_after_train=True,
#         full_orbit_vis_after_train=True,
#         detect_anomaly=True,
#     ),
#     cuda=0,
#     seed=1
# )


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
            self.model: D_Joint_S2 = task.model
            self.model.background_color = 'white'
            self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
            self.trainer: D_Joint_S2Trainer = task.trainer
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
            self.model: D_Joint_S2 = task.model
            self.dataset: SyntheticDynamicMultiViewBlenderRGBADataset = task.dataset
            self.trainer: D_Joint_S2Trainer = task.trainer
            self.trainer.setup(model=self.model, dataset=self.dataset)
            self.trainer.before_update(model=self.model, optimizers=None, curr_step=self.trainer.num_steps)
        self.model.eval()

        self.process_loader()
        del self.model, self.dataset, self.trainer, self.experiment
        torch.cuda.empty_cache()
        gc.collect()



if __name__ == '__main__':
    TaskGroup(
        test = toy_task,
        # train task
        toy = toy_task,
        # cat = cat_task,
        # footballplayer = footballplayer_task,
        
        # eval task
        extract = Extract_results(cuda=0), 
        eval = Eval_results(cuda=0), 

    ).run()
