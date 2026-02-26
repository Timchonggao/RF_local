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


@dataclass
class D_DiffDR(Module):

    # geometry setting
    geometry: Literal['gt', 'DD_isocubes', 'DD_flexicubes'] = 'DD_flexicubes' # method to generate geometry, gt: ground truth, DD_isocubes: Dual-Domain 4D isosurface cubes; 
    geometry_resolution: int = 128
    geometry_scale: float = 1.0
    # dynamic geometry setting
    poly_degree: int = 3
    low_freq_fourier_bands: List[int] = field(default_factory=lambda: [1, 3])
    mid_freq_fourier_bands: List[int] = field(default_factory=lambda: [4, 9])
    high_freq_fourier_bands: List[int] = field(default_factory=lambda: [10, 18])
    # geometry regularization setting
    reg_sdf_entropy_able: bool = False
    reg_time_tv_able: bool = False

    reg_sdf_eikonal_able: bool = False
    reg_coeff_tv_able: bool = False
    
    reg_scene_flow_smoothness_able: bool = False
    reg_scene_flow_consistency_able: bool = False

    # rendering setting
    antialias: bool = True

    def __setup__(self) -> None:
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
            )
            self.static_sdf_params = torch.nn.Parameter(self.geometric_repr.static_sdf_values.clone())
            self.sdf_curve_poly_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_poly_coefficient.clone())
            self.sdf_curve_low_freq_fourier_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_low_freq_fourier_coefficient.clone())
            self.sdf_curve_mid_freq_fourier_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_mid_freq_fourier_coefficient.clone())
            self.sdf_curve_high_freq_fourier_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_high_freq_fourier_coefficient.clone())
            # todo 支持omega learnable
            self.deform_params = torch.nn.Parameter(torch.empty(0))
            self.weight_params = torch.nn.Parameter(torch.empty(0))
        elif self.geometry == 'DD_flexicubes':
            self.geometric_repr = DualDomain4DFlexiCubes.from_resolution(
                self.geometry_resolution,
                scale=self.geometry_scale,
                poly_degree=self.poly_degree,
                low_freq_fourier_bands=self.low_freq_fourier_bands,
                mid_freq_fourier_bands=self.mid_freq_fourier_bands,
                high_freq_fourier_bands=self.high_freq_fourier_bands,
            )
            self.static_sdf_params = torch.nn.Parameter(self.geometric_repr.static_sdf_values.clone())
            self.sdf_curve_poly_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_poly_coefficient.clone())
            self.sdf_curve_low_freq_fourier_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_low_freq_fourier_coefficient.clone())
            self.sdf_curve_mid_freq_fourier_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_mid_freq_fourier_coefficient.clone())
            self.sdf_curve_high_freq_fourier_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_high_freq_fourier_coefficient.clone())
            # todo 支持omega learnable
            self.deform_params = torch.nn.Parameter(torch.zeros_like(self.geometric_repr.vertices))
            self.weight_params = torch.nn.Parameter(torch.ones(self.geometric_repr.indices.shape[0], 21))
        else:
            raise ValueError(self.geometry)

        self.dynamic_model_stage: dict = field(default_factory=dict)

        self.batch_gt_geometry = None
        self.batch_next_frame_gt_geometry = None # use for scene flow smoothness regularization, consistency loss
        
        self.reg_sdf_entropy_weight = 0.0 
        self.reg_time_tv_weight = 0.0
        self.reg_sdf_eikonal_weight = 0.0
        self.reg_coeff_tv_weight = 0.0
        self.reg_scene_flow_smoothness_weight = 0.0

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
                'time_tv_loss': 0.0,
                'sdf_eikonal_loss': 0.0,
                'coeff_tv_loss': 0.0,
                'scene_flow_smoothness_loss': 0.0,
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
                )
                pred_meshes, loss_dict = geometric_repr.marching_cubes_at_times(
                    t=times, 
                    scale=self.geometry_scale,
                    model_stage=self.dynamic_model_stage,
                    compute_sdf_flow=self.reg_scene_flow_consistency_able or self.reg_scene_flow_smoothness_able or self.reg_time_tv_able, # 三者都需要开启计算sdf_flow
                    compute_sdf_entropy=self.reg_sdf_entropy_able,
                    compute_sdf_eikonal=self.reg_sdf_eikonal_able,
                    compute_scene_flow_smoothness=self.reg_scene_flow_smoothness_able,
                    compute_coeff_tv_loss=self.reg_coeff_tv_able,
                    compute_time_tv_loss=self.reg_time_tv_able,
                    sdf_eps=None,
                )
                reg_losses_dict = {
                    'sdf_entropy_loss': loss_dict['sdf_entropy_loss'] * self.reg_sdf_entropy_weight,
                    'time_tv_loss': loss_dict['time_tv_loss'] * self.reg_time_tv_weight,
                    'coeff_tv_loss': loss_dict['coeff_tv_loss'] * self.reg_coeff_tv_weight,
                    'sdf_eikonal_loss': loss_dict['sdf_eikonal_loss'] * self.reg_sdf_eikonal_weight,
                    'scene_flow_smoothness_loss': loss_dict['scene_flow_smoothness_loss'] * self.reg_scene_flow_smoothness_weight,
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
                )
                pred_meshes, loss_dict = geometric_repr.dual_marching_cubes_at_times(
                    t=times, 
                    scale=self.geometry_scale,
                    model_stage=self.dynamic_model_stage,
                    compute_sdf_flow=self.reg_scene_flow_consistency_able or self.reg_scene_flow_smoothness_able or self.reg_time_tv_able, # 三者都需要开启计算sdf_flow
                    compute_sdf_entropy=self.reg_sdf_entropy_able,
                    compute_sdf_eikonal=self.reg_sdf_eikonal_able,
                    compute_scene_flow_smoothness=self.reg_scene_flow_smoothness_able,
                    compute_coeff_tv_loss=self.reg_coeff_tv_able,
                    compute_time_tv_loss=self.reg_time_tv_able,
                    sdf_eps=None,
                )
                reg_losses_dict = {
                    'L_dev': loss_dict['L_dev'] * 0.25,
                    'sdf_entropy_loss': loss_dict['sdf_entropy_loss'] * self.reg_sdf_entropy_weight,
                    'time_tv_loss': loss_dict['time_tv_loss'] * self.reg_time_tv_weight,
                    'coeff_tv_loss': loss_dict['coeff_tv_loss'] * self.reg_coeff_tv_weight,
                    'sdf_eikonal_loss': loss_dict['sdf_eikonal_loss'] * self.reg_sdf_eikonal_weight,
                    'scene_flow_smoothness_loss': loss_dict['scene_flow_smoothness_loss'] * self.reg_scene_flow_smoothness_weight,
                }
                return pred_meshes, reg_losses_dict
        else:
            raise ValueError(self.geometry)

    def render_report(
        self,
        camera_inputs: DS_Cameras,
        return_pred_mesh: bool = False,
        frame_batch: Optional[int] = None,
        camera_batch: Optional[int] = None,
        time_window_indices: Optional[Tensor] = None,
    ) -> Tuple[
        Optional[List[DS_TriangleMesh]],
        DepthImages, DepthImages,
        VectorImages, VectorImages,
        Dict[str, Tensor],
        Optional[DepthImages], Optional[VectorImages],
        Optional[DepthImages], Optional[VectorImages],
        Optional[DepthImages], Optional[VectorImages],
    ]:
        '''
        input multi view camera, time_batch
        output time_batch mesh, multi view depth, normal, geometry_reg
        '''
        batch = len(camera_inputs)

        report = ([], [], [], [], []) # pred_mesh, depth, gt_depth, normal, gt_normal

        scene_flow_pred_next_frame_depths, scene_flow_pred_next_frame_normals = [], []
        gt_next_frame_depths, gt_next_frame_normals = [], []
        pred_next_frame_depths, pred_next_frame_normals = [], []

        times = camera_inputs.times[::camera_batch] if (frame_batch and camera_batch) else camera_inputs.times
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

        if frame_batch and camera_batch:
            for i in range(frame_batch):
                mesh, gt_mesh = pred_meshes[i], self.get_gt_geometry(i)
                if return_pred_mesh:
                    report[0].append(mesh)
                cams = camera_inputs[i * camera_batch : (i + 1) * camera_batch]
                d, gd, n, gn = render_views(mesh, gt_mesh, cams)
                report[1].extend(d); report[2].extend(gd)
                report[3].extend(n); report[4].extend(gn)
        else:
            if self.reg_scene_flow_consistency_able and time_window_indices is not None:
                valid_time_indices_set = set(time_window_indices.tolist())
            for i in range(batch):
                mesh, gt_mesh = pred_meshes[i], self.get_gt_geometry(i)
                if return_pred_mesh:
                    report[0].append(mesh)
                
                if self.reg_scene_flow_consistency_able and time_window_indices is not None:
                    next_frame_index = time_window_indices[i].item() + 1
                    if next_frame_index in valid_time_indices_set:
                        next_mesh = mesh.get_next_frame_mesh(dt=camera_inputs[i].dts)
                        scene_flow_pred_next_frame_depths.append(next_mesh.render(camera_inputs[i+1], shader=depth_shader).item())
                        scene_flow_pred_next_frame_normals.append(next_mesh.render(camera_inputs[i+1], shader=normal_shader).item()) # dump_float32_image(Path('temp_scene_flow1.png'),next_mesh.render(camera_inputs[i+1], shader=normal_shader).visualize((1, 1, 1)).item())
                
                cams = [camera_inputs[i]]
                d, gd, n, gn = render_views(mesh, gt_mesh, cams)
                report[1].extend(d); report[2].extend(gd)
                report[3].extend(n); report[4].extend(gn)
                
                if self.reg_scene_flow_consistency_able and time_window_indices is not None:
                    last_frame_index = time_window_indices[i].item() - 1
                    if last_frame_index in valid_time_indices_set:
                        gt_next_frame_depths.append(gd) # dump_float32_image(Path('temp_gt1.png'),gt_mesh.render(camera_inputs[i], shader=normal_shader).visualize((1, 1, 1)).item())
                        gt_next_frame_normals.append(gn)
                        pred_next_frame_depths.append(d)
                        pred_next_frame_normals.append(n) # dump_float32_image(Path('temp_pred1.png'),mesh.render(camera_inputs[i], shader=normal_shader).visualize((1, 1, 1)).item())

        return (
            report[0] if return_pred_mesh else None, 
            DepthImages(report[1]) if report[1] != [] else None, 
            DepthImages(report[2]) if report[2] != [] else None, 
            VectorImages(report[3]) if report[3] != [] else None, 
            VectorImages(report[4]) if report[4] != [] else None, 
            reg_loss_dict,    
            DepthImages(scene_flow_pred_next_frame_depths) if scene_flow_pred_next_frame_depths != [] else None,
            VectorImages(scene_flow_pred_next_frame_normals) if scene_flow_pred_next_frame_normals != [] else None,
            DepthImages(gt_next_frame_depths) if gt_next_frame_depths != [] else None,
            VectorImages(gt_next_frame_normals) if gt_next_frame_normals != [] else None,
            DepthImages(pred_next_frame_depths) if pred_next_frame_depths != [] else None,
            VectorImages(pred_next_frame_normals) if pred_next_frame_normals != [] else None,
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
                'static_sdf_params': self.static_sdf_params,
                'dddm_poly_coeffs': self.sdf_curve_poly_coefficient,
                'dddm_fourier_low_coeffs': self.sdf_curve_low_freq_fourier_coefficient,
                'dddm_fourier_mid_coeffs': self.sdf_curve_mid_freq_fourier_coefficient,
                'dddm_fourier_high_coeffs': self.sdf_curve_high_freq_fourier_coefficient,
                'geometry_repr_vertices_deforms': self.deform_params,
                'geometry_repr_weights': self.weight_params,
            }[field_name]
            return [params]
        return parameters

    @torch.no_grad()
    def export_model(self) -> None:
        attributes = {
            'geometry': self.geometry,
            'geometry_resolution': self.geometry_resolution,
            'geometry_scale': self.geometry_scale,
            'poly_degree': self.poly_degree,
            'low_freq_fourier_bands': self.low_freq_fourier_bands,
            'mid_freq_fourier_bands': self.mid_freq_fourier_bands,
            'high_freq_fourier_bands': self.high_freq_fourier_bands,

            'static_sdf_params': self.static_sdf_params,
            'sdf_curve_poly_coefficient': self.sdf_curve_poly_coefficient,
            'sdf_curve_low_freq_fourier_coefficient': self.sdf_curve_low_freq_fourier_coefficient,
            'sdf_curve_mid_freq_fourier_coefficient': self.sdf_curve_mid_freq_fourier_coefficient,
            'sdf_curve_high_freq_fourier_coefficient': self.sdf_curve_high_freq_fourier_coefficient,
            
            'deform_params': self.deform_params,
            'weight_params': self.weight_params,
        }
        return attributes
