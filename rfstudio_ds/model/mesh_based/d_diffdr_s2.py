from __future__ import annotations

# import modules
from dataclasses import dataclass, field
from typing import Literal, Any, Tuple, List, Dict, Optional, Union
from jaxtyping import Float32
from pathlib import Path

import torch
from torch import Tensor

# import rfstudio modules
from rfstudio.graphics import (
    DepthImages,
    VectorImages,
)
from rfstudio.graphics.shaders import DepthShader, NormalShader
from rfstudio.nn import Module
from rfstudio.utils.decorator import chains
from rfstudio.io import open_video_renderer, dump_float32_image

# import rfstudio_ds modules
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh, DualDomain4DIsoCubes, DualDomain4DFlexiCubes
from rfstudio_ds.model.density_field.components.encoding_4d import Grid4d_HashEncoding
from rfstudio_ds.nn import MLPDecoderNetwork, Grid4DDecoderNetwork

@dataclass
class D_DiffDR_S2(Module):

    # geometry setting
    load: Path = ...

    # rendering setting
    antialias: bool = True

    sdf_residual_enc: Grid4d_HashEncoding = Grid4d_HashEncoding(
        decoder=MLPDecoderNetwork(
            layers=[-1, 32, 1],
            activation='sigmoid',
            bias=False,
            initialization='kaiming-uniform',
        ),
        grad_scaling=16.0,
        backend='grid4d',
        deform_base_resolution=[8,8,8],
        deform_desired_resolution=[128,128,256],
        deform_num_levels=16,
    )

    wavelet_name: str = 'db1'
    wavelet_level: int = 3
    wavelet_min_step: int = 2

    def __setup__(self) -> None:
        state_dict = torch.load(self.load, map_location='cpu')

        self.geometry = state_dict['geometry']
        self.geometry_resolution = state_dict['geometry_resolution']
        self.geometry_scale = state_dict['geometry_scale']
        self.poly_degree = state_dict['poly_degree']
        self.low_freq_fourier_bands = state_dict['low_freq_fourier_bands']
        self.mid_freq_fourier_bands = state_dict['mid_freq_fourier_bands']
        self.high_freq_fourier_bands = state_dict['high_freq_fourier_bands']
        self.time_resolution = state_dict['time_resolution']

        self.dynamic_model_stage = None

        self.reg_wavelet_sparse_able = False
        self.reg_wavelet_sparse_weight = 0.

        self.reg_sdf_entropy_able = False
        self.reg_sdf_entropy_weight = 0.0

        self.reg_time_tv_able = False
        self.reg_time_tv_weight = 0.0
        
        self.use_sdf_residual_enc = False
        self.sdf_residual_enc_weight = 0.001

        if self.geometry == 'gt':
            self.geometric_repr = None
        elif self.geometry == 'DD_isocubes':
            self.geometric_repr = DualDomain4DIsoCubes.from_resolution(
                self.geometry_resolution,
                scale=self.geometry_scale,
                poly_degree=self.poly_degree,
                low_freq_fourier_bands=self.low_freq_fourier_bands,
                mid_freq_fourier_bands=self.mid_freq_fourier_bands,
                high_freq_fourier_bands=self.high_freq_fourier_bands,
                time_frames=torch.linspace(0, 1, self.time_resolution),
                wavelet_name=self.wavelet_name,
                wavelet_level=self.wavelet_level,
                wavelet_min_step=self.wavelet_min_step,
            )
            self.static_sdf_params = torch.nn.Parameter(state_dict['static_sdf_params'], requires_grad=False)
            self.sdf_curve_poly_coefficient = torch.nn.Parameter(state_dict['sdf_curve_poly_coefficient'], requires_grad=False)
            self.sdf_curve_low_freq_fourier_coefficient = torch.nn.Parameter(state_dict['sdf_curve_low_freq_fourier_coefficient'], requires_grad=False)
            self.sdf_curve_mid_freq_fourier_coefficient = torch.nn.Parameter(state_dict['sdf_curve_mid_freq_fourier_coefficient'], requires_grad=False)
            self.sdf_curve_high_freq_fourier_coefficient = torch.nn.Parameter(state_dict['sdf_curve_high_freq_fourier_coefficient'], requires_grad=False)
            self.deform_params = torch.nn.Parameter(state_dict['deform_params'], requires_grad=False)
            self.weight_params = torch.nn.Parameter(state_dict['weight_params'], requires_grad=False)

            self.sdf_curve_wavelet_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_wavelet_coefficient.clone())
        elif self.geometry == 'DD_flexicubes':
            self.geometric_repr = DualDomain4DFlexiCubes.from_resolution(
                self.geometry_resolution,
                scale=self.geometry_scale,
                poly_degree=self.poly_degree,
                low_freq_fourier_bands=self.low_freq_fourier_bands,
                mid_freq_fourier_bands=self.mid_freq_fourier_bands,
                high_freq_fourier_bands=self.high_freq_fourier_bands,
                time_frames=torch.linspace(0, 1, self.time_resolution),
                wavelet_name=self.wavelet_name,
                wavelet_level=self.wavelet_level,
                wavelet_min_step=self.wavelet_min_step,
            )
            self.static_sdf_params = torch.nn.Parameter(state_dict['static_sdf_params'], requires_grad=False)
            self.sdf_curve_poly_coefficient = torch.nn.Parameter(state_dict['sdf_curve_poly_coefficient'], requires_grad=False)
            self.sdf_curve_low_freq_fourier_coefficient = torch.nn.Parameter(state_dict['sdf_curve_low_freq_fourier_coefficient'], requires_grad=False)
            self.sdf_curve_mid_freq_fourier_coefficient = torch.nn.Parameter(state_dict['sdf_curve_mid_freq_fourier_coefficient'], requires_grad=False)
            self.sdf_curve_high_freq_fourier_coefficient = torch.nn.Parameter(state_dict['sdf_curve_high_freq_fourier_coefficient'], requires_grad=False)
            self.deform_params = torch.nn.Parameter(state_dict['deform_params'], requires_grad=False)
            self.weight_params = torch.nn.Parameter(state_dict['weight_params'], requires_grad=False)

            self.sdf_curve_wavelet_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_wavelet_coefficient.clone())
        else:
            raise ValueError(self.geometry)

    def set_batch_gt_geometry(self, batch_mesh: List[DS_TriangleMesh]) -> None:
        self.batch_gt_geometry = batch_mesh
    
    def get_gt_geometry(self, indice: int) -> DS_TriangleMesh:
        assert self.batch_gt_geometry is not None # need to set batch_gt_geometry before call this function, ussually called in trainer.step
        return self.batch_gt_geometry[indice].to(self.device)

    def get_geometry(self, indice: int = None, times: Tensor = None) -> Tuple[Union[DS_TriangleMesh, List[DS_TriangleMesh]], Dict[str, Tensor]]:
        if self.geometry == 'gt':
            assert indice is not None
            reg_losses_dict = {
                'sdf_entropy_loss': 0.0,
                'wavelet_sparse_loss': 0.0
            }
            return self.get_gt_geometry(indice), reg_losses_dict
        elif self.geometry in ['DD_isocubes', 'DD_flexicubes']:
            assert times is not None
            with torch.no_grad():
                if self.geometric_repr.device != self.device:
                    self.geometric_repr.swap_(self.geometric_repr.to(self.device))
            
            if self.geometry == 'DD_isocubes':
                geometric_repr = self.geometric_repr.replace(
                    static_sdf_values = self.static_sdf_params,
                    sdf_curve_poly_coefficient = self.sdf_curve_poly_coefficient,
                    sdf_curve_low_freq_fourier_coefficient = self.sdf_curve_low_freq_fourier_coefficient,
                    sdf_curve_mid_freq_fourier_coefficient = self.sdf_curve_mid_freq_fourier_coefficient,
                    sdf_curve_high_freq_fourier_coefficient = self.sdf_curve_high_freq_fourier_coefficient,
                    sdf_curve_wavelet_coefficient = self.sdf_curve_wavelet_coefficient,
                )
                sdfs,sdfs_flow = geometric_repr.query_sdf_at_times(t=times,model_stage=self.dynamic_model_stage,compute_sdf_flow=self.reg_time_tv_able)
                if self.use_sdf_residual_enc:
                    xyz = geometric_repr.vertices
                    xyz_normal = xyz / self.geometry_scale # normalize to [-1, 1]. [ V, 3]
                    times = times * 2 - 1 # normalize to [-1, 1]. [Batch, 1]
                    sdfs_residual = []
                    for i in range(len(times)):
                        t = times[i].unsqueeze(0).expand(xyz.shape[0],-1) # [V, 1]
                        xyzt = torch.cat((xyz_normal, t), dim=-1) # [V, 4]
                        sdf_residual, spatial_h, temporal_h = self.sdf_residual_enc(xyzt)
                        sdfs_residual.append(sdf_residual)
                    sdfs_residual = torch.stack(sdfs_residual, dim=0)
                    sdfs = sdfs + sdfs_residual * self.sdf_residual_enc_weight

                pred_meshes, loss_dict = geometric_repr.marching_cubes_at_times_precomputesdfs(
                    dynamic_sdf_values=sdfs,
                    dynamic_sdf_flow_values=sdfs_flow,
                    compute_sdf_entropy=self.reg_sdf_entropy_able,
                    compute_time_tv_loss=self.reg_time_tv_able,
                    compute_wavelet_sparse_loss=self.reg_wavelet_sparse_able,
                    sdf_eps=None,
                )
                reg_losses_dict = {
                    'sdf_entropy_loss': loss_dict['sdf_entropy_loss'] * self.reg_sdf_entropy_weight,
                    'time_tv_loss': loss_dict['time_tv_loss'] * self.reg_time_tv_weight,
                    'wavelet_sparse_loss': loss_dict['wavelet_sparse_loss'] * self.reg_wavelet_sparse_weight,
                }
                return pred_meshes, reg_losses_dict
            elif self.geometry == 'DD_flexicubes':
                deform_vertices = self.geometric_repr.vertices + self.deform_params.tanh() * (0.5 * self.geometry_scale / self.geometry_resolution)
                geometric_repr = self.geometric_repr.replace(
                    vertices=deform_vertices,
                    alpha=self.weight_params[:, :8],
                    beta=self.weight_params[:, 8:20],
                    gamma=self.weight_params[:, 20:],
                    static_sdf_values = self.static_sdf_params,
                    sdf_curve_poly_coefficient = self.sdf_curve_poly_coefficient,
                    sdf_curve_low_freq_fourier_coefficient = self.sdf_curve_low_freq_fourier_coefficient,
                    sdf_curve_mid_freq_fourier_coefficient = self.sdf_curve_mid_freq_fourier_coefficient,
                    sdf_curve_high_freq_fourier_coefficient = self.sdf_curve_high_freq_fourier_coefficient,
                    sdf_curve_wavelet_coefficient = self.sdf_curve_wavelet_coefficient,
                )
                sdfs,sdfs_flow = geometric_repr.query_sdf_at_times(t=times,model_stage=self.dynamic_model_stage,compute_sdf_flow=self.reg_time_tv_able)
                pred_meshes, loss_dict = geometric_repr.dual_marching_cubes_at_times_precomputesdfs(
                    dynamic_sdf_values=sdfs,
                    dynamic_sdf_flow_values=sdfs_flow,
                    compute_sdf_entropy=self.reg_sdf_entropy_able,
                    compute_time_tv_loss=self.reg_time_tv_able,
                    compute_wavelet_sparse_loss=self.reg_wavelet_sparse_able,
                    sdf_eps=None,
                )
                reg_losses_dict = {
                    'L_dev': loss_dict['L_dev'] * 0.25,
                    'sdf_entropy_loss': loss_dict['sdf_entropy_loss'] * self.reg_sdf_entropy_weight,
                    'time_tv_loss': loss_dict['time_tv_loss'] * self.reg_time_tv_weight,
                    'wavelet_sparse_loss': loss_dict['wavelet_sparse_loss'] * self.reg_wavelet_sparse_weight,
                }
                return pred_meshes, reg_losses_dict
        else:
            raise ValueError(self.geometry)

    def render_report(
        self,
        camera_inputs: DS_Cameras,
        return_pred_mesh: bool = False,
    ) -> Tuple[
        Optional[List[DS_TriangleMesh]],
        DepthImages, DepthImages,
        VectorImages, VectorImages,
        Dict[str, Tensor],
    ]:
        '''
        input multi view camera, time_batch
        output time_batch mesh, multi view depth, normal, geometry_reg
        '''
        batch = len(camera_inputs)

        report = ([], [], [], [], []) # pred_mesh, depth, gt_depth, normal, gt_normal
        times = camera_inputs.times
        pred_meshes, reg_loss_dict = self.get_geometry(times=times)
        
        depth_shader = DepthShader(antialias=self.antialias, culling=False)
        normal_shader = NormalShader(antialias=self.antialias, normal_type='flat')

        def render_views(mesh: DS_TriangleMesh, gt_mesh: DS_TriangleMesh, cameras: DS_Cameras):
            depth_images, gt_depth_images = [], []
            normal_images, gt_normal_images = [], []
            for cam in cameras:
                depth_images.append(mesh.render(cam, shader=depth_shader).item())
                gt_depth_images.append(gt_mesh.render(cam, shader=depth_shader).item())
                normal_images.append(mesh.render(cam, shader=normal_shader).item())
                gt_normal_images.append(gt_mesh.render(cam, shader=normal_shader).item())
            return depth_images, gt_depth_images, normal_images, gt_normal_images


        for i in range(batch):
            mesh, gt_mesh = pred_meshes[i], self.get_gt_geometry(i)
            if return_pred_mesh:
                report[0].append(mesh)
            
            cams = [camera_inputs[i]]
            d, gd, n, gn = render_views(mesh, gt_mesh, cams)
            report[1].extend(d); report[2].extend(gd)
            report[3].extend(n); report[4].extend(gn)
            

        return (
            report[0] if return_pred_mesh else None, 
            DepthImages(report[1]) if report[1] != [] else None, 
            DepthImages(report[2]) if report[2] != [] else None, 
            VectorImages(report[3]) if report[3] != [] else None, 
            VectorImages(report[4]) if report[4] != [] else None, 
            reg_loss_dict,    
            )

    def get_background_color(self) -> Float32[Tensor, "3 "]:
        if self.background_color == 'black':
            return torch.zeros(3)
        if self.background_color == 'white':
            return torch.ones(3)
        if self.training:
            return torch.rand(3)
        return torch.tensor([0.1490, 0.1647, 0.2157])

    @chains
    def as_module(self, *, field_name: Literal['poly_coeffs', 'fourier_low_coeffs', 'fourier_mid_coeffs', 'fourier_high_coeffs']) -> Any:
        def parameters(self) -> Any:
            params = {
                'dddm_wavelet_coeffs': self.sdf_curve_wavelet_coefficient,
            }[field_name]
            return [params]
        return parameters
