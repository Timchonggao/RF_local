from __future__ import annotations

# import modules
from dataclasses import dataclass, field
from typing import Literal, Any, Tuple, List, Optional, List, Union, Dict
from jaxtyping import Float32
from pathlib import Path

import torch
from torch import Tensor

import nvdiffrast.torch as dr

# import rfstudio modules
from rfstudio.io import dump_float32_image
from rfstudio.graphics import (
    DepthImages, VectorImages,
    RGBAImages, PBRAImages, RGBImages,
)
from rfstudio.graphics.shaders import (
    DepthShader, NormalShader,
    _get_fg_lut
)
from rfstudio.graphics.math import safe_normalize
from rfstudio.nn import Module, MLP
from rfstudio.utils.decorator import chains

# import rfstudio_ds modules
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh, DualDomain4DIsoCubes
from rfstudio_ds.model.density_field.components.encoding_4d import Grid4d_HashEncoding
from rfstudio_ds.nn import MLPDecoderNetwork, Grid4DDecoderNetwork

@dataclass
class D_NVDiffRec(Module):

    # geometry setting
    geometry: Literal['gt', 'DD_isocubes'] = 'DD_isocubes' # method to generate geometry, gt: ground truth, DD_isocubes: Dual-Domain 4D isosurface cubes; 
    geometry_resolution: int = 128
    geometry_scale: float = 1.0
    # dynamic geometry setting
    poly_degree: int = 3
    low_freq_fourier_bands: List[int] = field(default_factory=lambda: [1, 3])
    mid_freq_fourier_bands: List[int] = field(default_factory=lambda: [4, 9])
    high_freq_fourier_bands: List[int] = field(default_factory=lambda: [10, 18])
    # geometry regularization setting
    reg_time_tv_able: bool = False
    reg_coeff_tv_able: bool = False
    reg_sdf_entropy_able: bool = False
    reg_sdf_eikonal_able: bool = False
    reg_scene_flow_smoothness_able: bool = False

    # dynamic texture setting
    texture_able: bool = True
    dynamic_texture: Grid4d_HashEncoding = Grid4d_HashEncoding(
        decoder=MLPDecoderNetwork(
            layers=[-1, 96, 32, 3],
            activation='sigmoid',
            bias=False,
            initialization='kaiming-uniform',
        ),
        grad_scaling=16.0,
        backend='grid4d',
    )
    reg_type: Literal['random', 'flow'] = 'random'
    reg_spatial_hashgrid_able: bool = False
    reg_spatial_downsample_ratio: float = 0.5
    reg_spatial_random_perturb_range: Union[float, List[float]] = field(default_factory=lambda:[1e-2, 1e-2, 1e-2, 1e-3])
    reg_temporal_hashgrid_able: bool = False
    reg_temporal_downsample_ratio: float = 0.5
    reg_temporal_random_perturb_range: Union[float, List[float]] = field(default_factory=lambda:[1e-2, 1e-2, 1e-2, 1e-3])
    reg_kd_enc_able: bool = False
    reg_kd_random_perturb_range: Union[float, List[float]] = field(default_factory=lambda:[1e-2, 1e-2, 1e-2, 1e-3])

    # rendering setting
    antialias: bool = True
    background_color: Literal["random", "black", "white"] = "random"
    z_up: bool = False
    
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
            self.deform_params = torch.nn.Parameter(torch.empty(0)) # disable in isocubes
            self.weight_params = torch.nn.Parameter(torch.empty(0)) # disable in isocubes
        else:
            raise ValueError(self.geometry)

        self.batch_gt_geometry = None
        self.batch_next_frame_gt_geometry = None # use for scene flow smoothness regularization, consistency loss
        
        self.reg_coeff_tv_weight = 0.0
        self.reg_time_tv_weight = 0.0
        self.reg_sdf_entropy_weight = 0.0 
        self.reg_sdf_eikonal_weight = 0.0
        self.reg_scene_flow_smoothness_weight = 0.0
        
        self.reg_spatial_hashgrid_weight = 0.0
        self.reg_temporal_hashgrid_weight = 0.0
        self.reg_kd_enc_weight = 0.0

    def set_batch_gt_geometry(self, batch_mesh: List[DS_TriangleMesh]) -> None:
        self.batch_gt_geometry = batch_mesh
    
    def get_gt_geometry(self, indice: int) -> DS_TriangleMesh:
        assert self.batch_gt_geometry is not None # need to set batch_gt_geometry before call this function, ussually called in trainer.step
        return self.batch_gt_geometry[indice].to(self.device)

    def get_geometry(self, indice: int = None, times: Tensor = None) -> Tuple[DS_TriangleMesh, Tensor, Tensor, Tensor, Tensor]:
        if self.geometry == 'gt':
            assert indice is not None
            return self.get_gt_geometry(indice), 0, 0, 0, 0
        elif self.geometry in ['DD_isocubes']:
            assert times is not None
            with torch.no_grad():
                if self.geometric_repr.device != self.device:
                    self.geometric_repr.swap_(self.geometric_repr.to(self.device))
            if self.geometry == 'DD_isocubes':
                # deform_vertices = self.geometric_repr.vertices + self.deform_params.tanh() * (0.5 * self.geometry_scale / self.geometry_resolution)
                geometric_repr = self.geometric_repr.replace(
                    # vertices=deform_vertices,
                    static_sdf_values = self.static_sdf_params,
                    sdf_curve_poly_coefficient = self.sdf_curve_poly_coefficient,
                    sdf_curve_low_freq_fourier_coefficient = self.sdf_curve_low_freq_fourier_coefficient,
                    sdf_curve_mid_freq_fourier_coefficient = self.sdf_curve_mid_freq_fourier_coefficient,
                    sdf_curve_high_freq_fourier_coefficient = self.sdf_curve_high_freq_fourier_coefficient,
                )
            else:
                raise ValueError(self.geometry)
            pred_meshes, sdf_entropy_loss, sdf_eikonal_loss, scene_flow_smoothness_loss, coefficent_tv_loss, time_tv_loss = geometric_repr.marching_cubes_at_times(
                t=times, 
                scale=self.geometry_scale,
                model_stage=-1,
                compute_sdf_flow=self.reg_time_tv_able,
                compute_sdf_entropy=self.reg_sdf_entropy_able,
                compute_sdf_eikonal=self.reg_sdf_eikonal_able,
                compute_scene_flow_smoothness=self.reg_scene_flow_smoothness_able,
                compute_coeff_tv_loss=self.reg_coeff_tv_able,
                compute_time_tv_loss=self.reg_time_tv_able,
                sdf_eps=None,
            )
            reg_sdf_entropy_loss = torch.stack(sdf_entropy_loss, dim=0).mean(dim=0) * self.reg_sdf_entropy_weight if sdf_entropy_loss else 0.0
            reg_sdf_eikonal_loss = torch.stack(sdf_eikonal_loss, dim=0).mean(dim=0) * self.reg_sdf_eikonal_weight if sdf_eikonal_loss else 0.0
            reg_scene_flow_smoothness_loss = torch.stack(scene_flow_smoothness_loss, dim=0).mean(dim=0) * self.reg_scene_flow_smoothness_weight if scene_flow_smoothness_loss else 0.0
            reg_coeff_tv_loss = coefficent_tv_loss * self.reg_coeff_tv_weight if coefficent_tv_loss else 0.0
            reg_time_tv_loss = time_tv_loss * self.reg_time_tv_weight if time_tv_loss else 0.0
            return (
                pred_meshes, 
                reg_sdf_entropy_loss, reg_sdf_eikonal_loss, 
                reg_scene_flow_smoothness_loss, reg_coeff_tv_loss, reg_time_tv_loss
            )
        else:
            raise ValueError(self.geometry)

    def render_report(
            self, 
            camera_inputs: DS_Cameras,
            frame_batch: Optional[int] = None,
            camera_batch: Optional[int] = None,
            trainer_mode: Literal['train', 'val', 'test', 'orbit_vis', 'fix_vis'] = 'train',
        ) -> Tuple[
            Union[RGBAImages, None], 
            Union[DepthImages, None], Union[DepthImages, None],
            Union[VectorImages, None], Union[VectorImages, None],
            Dict[str, Tensor],
        ]:
        
        batch = len(camera_inputs)
        report = ([], [], [], [], []) # srgb, depth, gt_depth, normal, gt_normal
        reg_loss_dict = {}

        if self.geometry != 'gt':
            times = camera_inputs.times[::camera_batch] if (frame_batch and camera_batch) else camera_inputs.times
            times = times
            pred_meshes, *reg_losses = self.get_geometry(times=times)
            reg_loss_dict.update(dict(zip(['sdf_entropy_loss', 'sdf_eikonal_loss', 'scene_flow_smoothness_loss', 'coeff_tv_loss', 'time_tv_loss'], reg_losses)))

        ctx = dr.RasterizeCudaContext(camera_inputs.device)
        depth_shader = DepthShader(antialias=self.antialias, culling=False)
        normal_shader = NormalShader(antialias=self.antialias, normal_type='flat')

        def render_views(mesh: DS_TriangleMesh, gt_mesh: DS_TriangleMesh, cameras: DS_Cameras): # todo support parallel rendering after support multiview dataset
            depth_images, gt_depth_images = [], []
            normal_images, gt_normal_images = [], []
            for cam in cameras:
                depth_images.append(mesh.render(cam, shader=depth_shader).item())
                gt_depth_images.append(gt_mesh.render(cam, shader=depth_shader).item())
                normal_images.append(mesh.render(cam, shader=normal_shader).item())
                gt_normal_images.append(gt_mesh.render(cam, shader=normal_shader).item())
            return depth_images, gt_depth_images, normal_images, gt_normal_images

        if frame_batch and camera_batch: # todo support multiview
            for i in range(frame_batch):
                mesh, gt_mesh = pred_meshes[i], self.get_gt_geometry(i)
                cams = camera_inputs[i * camera_batch : (i + 1) * camera_batch] # multi-view cameras
                d, gd, n, gn = render_views(mesh, gt_mesh, cams)
                report[1].extend(d); report[2].extend(gd)
                report[3].extend(n); report[4].extend(gn)
        else:
            for i in range(batch):
                if self.geometry != 'gt':
                    mesh, gt_mesh = pred_meshes[i], self.get_gt_geometry(i)

                    cams = [camera_inputs[i]] # single camera
                    d, gd, n, gn = render_views(mesh, gt_mesh, cams)
                    report[1].extend(d); report[2].extend(gd)
                    report[3].extend(n); report[4].extend(gn)
                else:
                    mesh, gt_mesh = self.get_gt_geometry(i), self.get_gt_geometry(i)

                if self.texture_able:
                    camera = camera_inputs[i]
                    vertices = torch.cat((
                        mesh.vertices,
                        torch.ones_like(mesh.vertices[..., :1]),
                    ), dim=-1).view(-1, 4, 1) # todo check 范围
                    H, W = resolution = [camera.height.item(), camera.width.item()]
                    mvp = camera.projection_matrix @ camera.view_matrix                    # [4, 4]
                    projected = (mvp @ vertices).view(1, -1, 4)                            # [1, V, 4]
                    indices = mesh.indices.int()
                    with dr.DepthPeeler(ctx, projected, indices, resolution=resolution) as peeler:
                        rast, _ = peeler.rasterize_next_layer()
                    alphas = (rast[..., -1:] > 0).float() # [1, H, W, 1]
                    frag_pos, _ = dr.interpolate(mesh.vertices[None], rast, indices) # [1, H, W, 3]
                    
                    alpha_mask = alphas.view(-1) > 0.5  # [H*W]
                    valid_indices = torch.where(alpha_mask)[0]  # [V], 有效像素的扁平化索引
                    xyz = frag_pos.view(-1, 3)[alpha_mask] # [V, 3] get partial vertices position
                    xyz_normal = xyz / self.geometry_scale # 归一化到[-1, 1]
                    t = (camera.times * 2 - 1).unsqueeze(0).expand(xyz_normal.shape[0], -1) # [-1, 1]
                    
                    xyzt = torch.cat((xyz_normal, t), dim=-1) # [V, 4]
                    kd, spatial_h, temporal_h = self.dynamic_texture(xyzt) # [V, 3], [V, D], [V, D']
                    kd_colormap_flattened = torch.zeros(H*W, 3, device=kd.device, dtype=kd.dtype)  # [H*W, 3]
                    kd_colormap_flattened[valid_indices] = kd  # 将 kd 的值赋给有效索引
                    colors = kd_colormap_flattened.view(1, H, W, 3) # [1, H, W, 3] 
                    srgba = torch.cat((colors, alphas), dim=-1)
                    if self.antialias:
                        srgba = dr.antialias(srgba, rast, projected, indices)
                    report[0].append(srgba.squeeze(0))

                    reg_spatial_hashgrid_loss = 0.0
                    reg_temporal_hashgrid_loss = 0.0
                    reg_kd_enc_loss = 0.0
                    if trainer_mode in ['train', 'val', 'test']:
                        if self.reg_spatial_hashgrid_able:
                            if self.reg_type == 'random':
                                if type(self.reg_spatial_random_perturb_range) is float:
                                    reg_spatial_random_perturb_range_list = [self.reg_spatial_random_perturb_range for _ in range(3)] # 空间正则化扰动：若启用，默认范围为1e-3，沿x、y、z三个方向。
                                    self.reg_spatial_random_perturb_range = torch.tensor(reg_spatial_random_perturb_range_list, device="cuda", dtype=torch.float32)
                                else:
                                    assert len(self.reg_spatial_random_perturb_range) == 3, "reg_spatial_random_perturb_range should be a float or a list of 3 floats"
                                    self.reg_spatial_random_perturb_range = torch.tensor(self.reg_spatial_random_perturb_range, device="cuda", dtype=torch.float32)

                                if self.reg_spatial_downsample_ratio < 1.0:
                                    choice = torch.randperm(xyz.shape[0], device=xyz.device)[: int(max(1, xyz.shape[0] * self.reg_spatial_downsample_ratio))]
                                    xyz_normal_choice = xyz_normal[choice]
                                    xyz_perturb = xyz_normal_choice + (torch.rand_like(xyz_normal_choice) * 2.0 - 1.0) * self.reg_spatial_random_perturb_range[None, ...] 
                                    spatial_h = spatial_h[choice]
                                else:
                                    xyz_perturb = xyz_normal + (torch.rand_like(xyz_normal) * 2.0 - 1.0) * self.reg_spatial_random_perturb_range[None, ...]
                                
                                spatial_h_perturb = self.dynamic_texture.encode_spatial(xyz_perturb)
                                reg_spatial = torch.sum(torch.abs(spatial_h_perturb - spatial_h) ** 2, dim=-1)
                                reg_spatial_hashgrid_loss += self.reg_spatial_hashgrid_weight * torch.mean(reg_spatial)
                            elif self.reg_type == 'flow':
                                pass
                        
                        if self.reg_temporal_hashgrid_able:
                            if self.reg_type == 'random':
                                if type(self.reg_temporal_random_perturb_range) is float:
                                    reg_temporal_random_perturb_range_list = [self.reg_temporal_random_perturb_range for _ in range(4)] # 时间正则化扰动：若启用，默认范围为1e-2，沿x、y、z、t四个维度。
                                    self.reg_temporal_random_perturb_range = torch.tensor(reg_temporal_random_perturb_range_list, device="cuda", dtype=torch.float32)
                                else:
                                    assert len(self.reg_temporal_random_perturb_range) == 4, "reg_temporal_random_perturb_range should be a float or a list of 4 floats"
                                    self.reg_temporal_random_perturb_range = torch.tensor(self.reg_temporal_random_perturb_range, device="cuda", dtype=torch.float32)

                                if self.reg_temporal_downsample_ratio < 1.0:
                                    choice = torch.randperm(xyzt.shape[0], device=xyzt.device)[: int(max(1, xyzt.shape[0] * self.reg_temporal_downsample_ratio))]
                                    xyzt_choice = xyzt[choice]
                                    # xyzt_perturb = xyzt_choice + (torch.rand_like(xyzt_choice) * 2.0 - 1.0) * self.temporal_perturb_range[None, ...]
                                    xyz_perturb = (torch.rand_like(xyzt_choice[:, :3]) * 2.0 - 1.0) * self.reg_temporal_random_perturb_range[:3]
                                    t_perturb = (torch.rand(1, device=xyzt.device) * 2.0 - 1.0).expand(xyzt_choice.shape[0], 1) * self.reg_temporal_random_perturb_range[3]
                                    # xyzt_perturb = torch.cat([xyzt_choice[:, :3], xyzt_choice[:, 3:4] + t_perturb], dim=-1) # 仅扰动t
                                    # xyzt_perturb = torch.cat([xyzt_choice[:, :3] + xyz_perturb, xyzt_choice[:, 3:4]], dim=-1) # 仅扰动xyz
                                    xyzt_perturb = torch.cat([xyzt_choice[:, :3] + xyz_perturb, xyzt_choice[:, 3:4] + t_perturb], dim=-1) # 扰动xyz和t
                                    temporal_h = temporal_h[choice]
                                else:
                                    # xyzt_perturb = xyzt + (torch.rand_like(xyzt) * 2.0 - 1.0) * self.temporal_perturb_range[None, ...]
                                    xyz_perturb = (torch.rand_like(xyzt[:, :3]) * 2.0 - 1.0) * self.temporal_perturb_range[:3]
                                    t_perturb = (torch.rand(1, device=xyzt.device) * 2.0 - 1.0).expand(xyzt.shape[0], 1) * self.temporal_perturb_range[3]
                                    # xyzt_perturb = torch.cat([xyzt[:, :3], xyzt[:, 3:4] + t_perturb], dim=-1) # 仅扰动t
                                    # xyzt_perturb = torch.cat([xyzt[:, :3] + xyz_perturb, xyzt[:, 3:4]], dim=-1) # 仅扰动xyz
                                    xyzt_perturb = torch.cat([xyzt[:, :3] + xyz_perturb, xyzt[:, 3:4] + t_perturb], dim=-1) # 扰动xyz和t
                                
                                temporal_h_perturb = self.dynamic_texture.encode_temporal(xyzt_perturb)
                                reg_temporal = torch.sum(torch.abs(temporal_h_perturb - temporal_h) ** 2, dim=-1)
                                reg_temporal_hashgrid_loss += self.reg_temporal_hashgrid_weight * torch.mean(reg_temporal)
                            elif self.reg_type == 'flow':
                                pass
                        
                        if self.reg_kd_enc_able:
                            if self.reg_type == 'random':
                                if type(self.reg_kd_random_perturb_range) is float:
                                    reg_kd_random_perturb_range_list = [self.reg_kd_random_perturb_range for _ in range(4)] # 编码器正则化扰动：若启用，默认范围为1e-2，沿x、y、z、t四个维度。
                                    self.reg_kd_random_perturb_range = torch.tensor(reg_kd_random_perturb_range_list, device="cuda", dtype=torch.float32)
                                else:
                                    assert len(self.reg_kd_random_perturb_range) == 4, "reg_kd_random_perturb_range should be a float or a list of 4 floats"
                                    self.reg_kd_random_perturb_range = torch.tensor(self.reg_kd_random_perturb_range, device="cuda", dtype=torch.float32)

                                # xyzt_perturb = xyzt + (torch.rand_like(xyzt) * 2.0 - 1.0) * self.temporal_perturb_range[None, ...]
                                xyz_perturb = (torch.rand_like(xyzt[:, :3]) * 2.0 - 1.0) * self.reg_kd_random_perturb_range[:3]
                                t_perturb = (torch.rand(1, device=xyzt.device) * 2.0 - 1.0).expand(xyzt.shape[0], 1) * self.reg_kd_random_perturb_range[3]
                                # xyzt_perturb = torch.cat([xyzt[:, :3], xyzt[:, 3:4] + t_perturb], dim=-1) # 仅扰动t
                                # xyzt_perturb = torch.cat([xyzt[:, :3] + xyz_perturb, xyzt[:, 3:4]], dim=-1) # 仅扰动xyz
                                xyzt_perturb = torch.cat([xyzt[:, :3] + xyz_perturb, xyzt[:, 3:4] + t_perturb], dim=-1) # 扰动xyz和t

                                kd_perturb, _, _ = self.dynamic_texture(xyzt_perturb)
                                reg_kd_enc_loss += torch.mean(torch.abs(kd_perturb - kd))
                            elif self.reg_type == 'flow':
                                pass
            
            if self.texture_able:
                reg_loss_dict['reg_spatial_hashgrid'] = (reg_spatial_hashgrid_loss / batch) * self.reg_spatial_hashgrid_weight
                reg_loss_dict['reg_temporal_hashgrid'] = (reg_temporal_hashgrid_loss / batch) * self.reg_temporal_hashgrid_weight
                reg_loss_dict['reg_kd_enc'] = (reg_kd_enc_loss / batch) * self.reg_kd_enc_weight


        return (
            RGBAImages(report[0]) if report[0] != [] else None, 
            DepthImages(report[1]) if report[1] != [] else None, 
            DepthImages(report[2]) if report[2] != [] else None, 
            VectorImages(report[3]) if report[3] != [] else None, 
            VectorImages(report[4]) if report[4] != [] else None, 
            reg_loss_dict
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

