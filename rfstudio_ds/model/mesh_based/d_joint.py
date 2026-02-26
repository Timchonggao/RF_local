from __future__ import annotations

# import modules
from dataclasses import dataclass, field
from typing import Literal, Any, Tuple, List, Optional, List, Union, Dict
from jaxtyping import Float32
from pathlib import Path

import torch
from torch import Tensor

from rfstudio.utils.lazy_module import dr


# import rfstudio modules
from rfstudio.io import dump_float32_image
from rfstudio.graphics import (
    DepthImages, VectorImages,
    RGBAImages, PBRAImages, RGBImages,
    TextureCubeMap, TextureSplitSum,
    TextureLatLng,
    TriangleMesh,
)
from rfstudio.graphics.shaders import (
    DepthShader, NormalShader,
    _get_fg_lut
)
from rfstudio.graphics.math import safe_normalize
from rfstudio.nn import Module, MLP
from rfstudio.utils.decorator import chains

# import rfstudio_ds modules
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh, DualDomain4DIsoCubes, DualDomain4DFlexiCubes
from rfstudio_ds.model.density_field.components.encoding_4d import Grid4d_HashEncoding, KplaneEncoding
from rfstudio_ds.nn import MLPDecoderNetwork, Grid4DDecoderNetwork

@dataclass
class D_Joint(Module):

    # dynamic geometry setting
    geometry: Literal['gt', 'DD_isocubes', 'DD_flexicubes'] = 'DD_flexicubes' # method to generate geometry, gt: ground truth, DD_isocubes: Dual-Domain 4D isosurface cubes; 
    geometry_resolution: int = 64
    geometry_scale: float = 1.0
    poly_degree: int = 3
    low_freq_fourier_bands: List[int] = field(default_factory=lambda: [1, 10])
    mid_freq_fourier_bands: List[int] = field(default_factory=lambda: [11, 20])
    high_freq_fourier_bands: List[int] = field(default_factory=lambda: [21, 30])
    # geometry regularization setting
    reg_sdf_entropy_able: bool = False
    reg_time_tv_able: bool = False
    reg_sdf_eikonal_able: bool = False
    reg_coeff_tv_able: bool = False

    # dynamic texture setting
    min_roughness: float = 0.1
    gt_envmap: Optional[Path] = None
    dynamic_texture: Grid4d_HashEncoding = Grid4d_HashEncoding(
        decoder=MLPDecoderNetwork(
            layers=[-1, 96, 32, 6],
            activation='sigmoid',
            bias=False,
            initialization='kaiming-uniform',
        ),
        grad_scaling=16.0,
        backend='grid4d',
        deform_base_resolution=[32,32,32],
        deform_desired_resolution=[4096,4096,256],
        deform_num_levels=32,
    )
    # texture regularization setting
    reg_type: Literal['random', 'flow'] = 'random'
    reg_temporal_hashgrid_able: bool = False
    reg_temporal_downsample_ratio: float = 0.5
    reg_temporal_random_perturb_range: Union[float, List[float]] = field(default_factory=lambda:[1e-2, 1e-2, 1e-2, 1e-3])
    reg_occ_able: bool = False
    reg_light_able: bool = False

    # rendering setting
    shader_type: Literal["split_sum_pbr", "direct"] = "split_sum_pbr"
    antialias: bool = True
    background_color: Literal["random", "black", "white"] = "white"
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
            self.deform_params = torch.nn.Parameter(torch.zeros_like(self.geometric_repr.vertices))
            self.weight_params = torch.nn.Parameter(torch.ones(self.geometric_repr.indices.shape[0], 21))
        else:
            raise ValueError(self.geometry)
        
        self.dynamic_model_stage: dict = field(default_factory=dict)
        self.batch_gt_geometry = None
        self.reg_sdf_entropy_weight = 0.0 
        self.reg_time_tv_weight = 0.0
        self.reg_sdf_eikonal_weight = 0.0
        self.reg_sdf_eikonal_type = "L2"
        self.reg_coeff_tv_weight = 0.0
        self.reg_reg_coeff_tv_type = "L1"
        self.reg_temporal_hashgrid_weight = 0.0
        self.reg_occ_weight = 0.0
        self.reg_light_weight = 0.0
        self.envmap = None
        if self.gt_envmap is None:
            self.cubemap = torch.nn.Parameter(torch.empty(6, 512, 512, 3).fill_(0.5))
        else:
            self.cubemap = torch.nn.Parameter(torch.empty(0))

    def get_envmap(self) -> TextureSplitSum:
        if self.gt_envmap is None:
            return TextureCubeMap(data=self.cubemap, transform=None).as_splitsum()
        if self.envmap is None:
            self.envmap = TextureCubeMap.from_image_file(self.gt_envmap, device=self.device).as_splitsum()
        return self.envmap
    
    def set_relight_envmap(self, envmap_path: Path) -> None:
        self.gt_envmap = envmap_path
        self.envmap = TextureCubeMap.from_image_file(envmap_path, device=self.device).as_splitsum()
        self.envmap = self.envmap.y_up_to_z_up_()

    def get_light_regularization(self) -> Tensor:
        if self.gt_envmap is None:
            white = self.cubemap.mean(-1, keepdim=True) # [6, R, R, 1]
            return (self.cubemap - white).abs().mean()
        return torch.zeros(1, device=self.device)
    
    def set_batch_gt_geometry(self, batch_mesh: List[DS_TriangleMesh]) -> None:
        self.batch_gt_geometry = batch_mesh
    
    def get_gt_geometry(self, indice: int) -> DS_TriangleMesh:
        if self.batch_gt_geometry is None or not isinstance(self.batch_gt_geometry[-1], DS_TriangleMesh):
            return None
        else:
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
                    compute_coeff_tv_type=self.reg_reg_coeff_tv_type,
                    compute_sdf_eikonal_type=self.reg_sdf_eikonal_type,
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
                pred_meshes, loss_dict, _ = geometric_repr.dual_marching_cubes_at_times(
                    t=times, 
                    scale=self.geometry_scale,
                    model_stage=self.dynamic_model_stage,
                    compute_sdf_flow=self.reg_time_tv_able,
                    compute_time_tv_loss=self.reg_time_tv_able,
                    compute_sdf_entropy=self.reg_sdf_entropy_able,
                    compute_sdf_eikonal=self.reg_sdf_eikonal_able,
                    compute_sdf_eikonal_type=self.reg_sdf_eikonal_type,
                    compute_coeff_tv_loss=self.reg_coeff_tv_able,
                    compute_coeff_tv_type=self.reg_reg_coeff_tv_type,
                    sdf_eps=None,
                )
                reg_losses_dict = {
                    'L_dev': loss_dict['L_dev'] * 0.25,
                    'sdf_entropy_loss': loss_dict['sdf_entropy_loss'] * self.reg_sdf_entropy_weight,
                    'time_tv_loss': loss_dict['time_tv_loss'] * self.reg_time_tv_weight,
                    'coeff_tv_loss': loss_dict['coeff_tv_loss'] * self.reg_coeff_tv_weight,
                    'sdf_eikonal_loss': loss_dict['sdf_eikonal_loss'] * self.reg_sdf_eikonal_weight,
                }
                return pred_meshes, reg_losses_dict
        else:
            raise ValueError(self.geometry)

    def render_report(
        self, 
        camera_inputs: DS_Cameras,
        trainer_mode: Literal['train', 'val', 'test', 'orbit_vis', 'fix_vis'] = 'train',
        return_pred_mesh: bool = False,
        return_pbr_attrs: bool = False,
    ) -> Tuple[
        Union[RGBAImages, PBRAImages, None],
        Dict[str, Tensor],
        Optional[List[DS_TriangleMesh]],
        Optional[DepthImages], Optional[DepthImages],
        Optional[VectorImages], Optional[VectorImages],
        Optional[Dict[str, RGBImages]]
    ]:
        if return_pbr_attrs:
            assert self.shader_type == "split_sum_pbr", "PBR attributes can only be returned when using split_sum_pbr shader"

        batch = len(camera_inputs)

        pred_images, kd_vis, roughness_vis, metallic_vis, occ_vis = [], [], [], [], []
        pred_meshes, pred_depths, gt_depths, pred_normals, gt_normals = [], [], [], [], []
        reg_temporal_hashgrid_loss = 0.0
        reg_occ_loss = 0.0

        pred_meshes, reg_loss_dict = self.get_geometry(times=camera_inputs.times)
        envmap = self.get_envmap()
        ctx = dr.RasterizeCudaContext(camera_inputs.device)
        depth_shader = DepthShader(antialias=self.antialias, culling=False)
        normal_shader = NormalShader(antialias=self.antialias, normal_type='vertex')

        for i in range(batch):
            camera = camera_inputs[i]
            pred_mesh, gt_mesh = pred_meshes[i], self.get_gt_geometry(i)
            if pred_mesh.normals is None:
                pred_mesh = pred_mesh.compute_vertex_normals(fix=True)

            vertices = torch.cat((
                pred_mesh.vertices,
                torch.ones_like(pred_mesh.vertices[..., :1]),
            ), dim=-1).view(-1, 4, 1)
            camera_pos = camera.c2w[:, 3]
            H, W = resolution = [camera.height.item(), camera.width.item()]
            mvp = camera.projection_matrix @ camera.view_matrix                    # [4, 4]
            projected = (mvp @ vertices).view(1, -1, 4)                            # [1, V, 4]
            indices = pred_mesh.indices.int()
            with dr.DepthPeeler(ctx, projected, indices, resolution=resolution) as peeler:
                rast, _ = peeler.rasterize_next_layer()
            alphas = (rast[..., -1:] > 0).float() # [1, H, W, 1]
            alpha_mask = alphas.view(-1) > 0.5  # [H*W]
            valid_indices = torch.where(alpha_mask)[0]  # [V], 有效像素的索引
            frag_pos, _ = dr.interpolate(pred_mesh.vertices[None], rast, indices) # [1, H, W, 3] 
            xyz = frag_pos.view(-1, 3)[alpha_mask] # [V, 3] get partial vertices position
            xyz_normal = (xyz / self.geometry_scale).clamp(-1, 1) # 归一化到[-1, 1]
            t_normal = (camera.times * 2 - 1).unsqueeze(0).expand(xyz_normal.shape[0], -1).clamp(-1, 1) # [-1, 1]
            xyzt_normal = torch.cat((xyz_normal, t_normal), dim=-1) # [V, 4]
            
            # shader module
            if isinstance(self.dynamic_texture, Grid4d_HashEncoding):
                kdks, temporal_h = self.dynamic_texture(xyzt_normal) # [V, 6], [V, D']
            elif isinstance(self.dynamic_texture, KplaneEncoding):
                kdks = self.dynamic_texture(xyzt_normal) # [V, 6], [V, D], [V, D']
            if self.shader_type == "direct":
                kd_colormap_flattened = torch.ones(H*W, 3, device=kdks.device, dtype=kdks.dtype)  # [H*W, 3]
                kd_colormap_flattened[valid_indices] = kdks[..., :3]  # 将 kd 的值赋给有效索引
                kd = kd_colormap_flattened.view(1, H, W, 3) # [1, H, W, 3] 
                
                colors = kd # [1, H, W, 3] 
            elif self.shader_type == "split_sum_pbr":
                kd_colormap_flattened = torch.ones(H*W, 3, device=kdks.device, dtype=kdks.dtype)  # [H*W, 3]
                kd_colormap_flattened[valid_indices] = kdks[..., :3]  # 将 kd 的值赋给有效索引
                kd = kd_colormap_flattened.view(1, H, W, 3) # [1, H, W, 3] 
                ks_colormap_flattened = torch.ones(H*W, 3, device=kdks.device, dtype=kdks.dtype)  # [H*W, 3]
                ks_colormap_flattened[valid_indices] = kdks[..., 3:]  # 将 ks 的值赋给有效索引
                ks = ks_colormap_flattened.view(1, H, W, 3) # [1, H, W, 3] 

                frag_n, _ = dr.interpolate(pred_mesh.normals[None], rast, indices) # [1, H, W, 3]
                frag_n = safe_normalize(frag_n) # [1, H, W, 3]

                # todo 尝试保持kdks的维度，不用扩展到map维度，这样可以节省内存、避免背景色干扰，或者将背景色设置为0
                roughness = ks[..., 1:2] * (1 - self.min_roughness) + self.min_roughness # [1, H, W, 1]
                metallic  = ks[..., 2:3] # [1, H, W, 1]
                specular  = (1.0 - metallic) * 0.04 + kd * metallic # [1, H, W, 3]
                diffuse  = kd * (1.0 - metallic) # [1, H, W, 3]

                frag_wo = safe_normalize(camera_pos - frag_pos) # [1, H, W, 3]
                n_dot_v = (frag_n * frag_wo).sum(-1, keepdim=True).clamp(min=1e-4) # [1, H, W, 1]
                fg_uv = torch.cat((n_dot_v, roughness), dim=-1) # [1, H, W, 2]
                fg_lookup = dr.texture(
                    _get_fg_lut(resolution=256, device=fg_uv.device),
                    fg_uv,
                    filter_mode='linear',
                    boundary_mode='clamp',
                ) # [1, H, W, 2]

                # Compute aggregate lighting
                frag_inv_wi = 2 * (frag_wo * frag_n).sum(-1, keepdim=True) * frag_n - frag_wo # [1, H, W, 3]
                l_diff, l_spec = envmap.sample(
                    normals=frag_n,
                    directions=frag_inv_wi,
                    roughness=roughness,
                ) # [1, H, W, 3]
                reflectance = specular * fg_lookup[..., 0:1] + fg_lookup[..., 1:2] # [1, H, W, 3]
                colors = (l_diff * diffuse + l_spec * reflectance) * (1.0 - ks[..., 0:1]) # [1, H, W, 3]

                if return_pbr_attrs:
                    kd_vis.append(
                        torch.where(
                        kd <= 0.0031308,
                        kd * 12.92,
                        torch.clamp(kd, min=0.0031308).pow(1.0 / 2.4) * 1.055 - 0.055,
                    ).squeeze(0))
                    roughness_vis.append(roughness.squeeze(0).repeat(1, 1, 3))
                    metallic_vis.append(metallic.squeeze(0).repeat(1, 1, 3))
                    occ_vis.append(ks[..., 0:1].squeeze(0).repeat(1, 1, 3))

            image = torch.cat((colors, alphas), dim=-1)
            if self.antialias:
                image = dr.antialias(image, rast, projected, indices)
            pred_images.append(image.squeeze(0))
            
            pred_depth = pred_mesh.render(camera, shader=depth_shader).item()
            pred_depths.append(pred_depth)
            gt_depth = gt_mesh.render(camera, shader=depth_shader).item() if gt_mesh is not None else pred_depth
            gt_depths.append(gt_depth); 
            
            pred_normal = pred_mesh.render(camera, shader=normal_shader).item()
            pred_normals.append(pred_normal)
            gt_normal = gt_mesh.render(camera, shader=normal_shader).item() if gt_mesh is not None else pred_normal
            gt_normals.append(gt_normal)

            # reg loss
            if trainer_mode in ['train', 'val', 'test']:
                if self.reg_temporal_hashgrid_able:
                    if self.reg_type == 'random':
                        if type(self.reg_temporal_random_perturb_range) is float:
                            reg_temporal_random_perturb_range_list = [self.reg_temporal_random_perturb_range for _ in range(4)] # 时间正则化扰动：若启用，默认范围为1e-2，沿x、y、z、t四个维度。
                            self.reg_temporal_random_perturb_range = torch.tensor(reg_temporal_random_perturb_range_list, device="cuda", dtype=torch.float32)
                        else:
                            assert len(self.reg_temporal_random_perturb_range) == 4, "reg_temporal_random_perturb_range should be a float or a list of 4 floats"
                            self.reg_temporal_random_perturb_range = torch.tensor(self.reg_temporal_random_perturb_range, device="cuda", dtype=torch.float32)

                        if self.reg_temporal_downsample_ratio < 1.0:
                            choice = torch.randperm(xyzt_normal.shape[0], device=xyzt_normal.device)[: int(max(1, xyzt_normal.shape[0] * self.reg_temporal_downsample_ratio))]
                            temporal_h = temporal_h[choice]
                            xyzt_choice = xyzt_normal[choice]
                            
                            # xyzt_perturb = xyzt_choice + (torch.rand_like(xyzt_choice) * 2.0 - 1.0) * self.temporal_perturb_range[None, ...]
                            
                            xyz_perturb = (torch.rand_like(xyzt_choice[:, :3]) * 2.0 - 1.0) * self.reg_temporal_random_perturb_range[:3]
                            t_perturb = (torch.rand(1, device=xyzt_normal.device) * 2.0 - 1.0).expand(xyzt_choice.shape[0], 1) * self.reg_temporal_random_perturb_range[3]
                            # xyzt_perturb = torch.cat([xyzt_choice[:, :3], xyzt_choice[:, 3:4] + t_perturb], dim=-1) # 仅扰动t
                            # xyzt_perturb = torch.cat([xyzt_choice[:, :3] + xyz_perturb, xyzt_choice[:, 3:4]], dim=-1) # 仅扰动xyz
                            xyzt_perturb = torch.cat([xyzt_choice[:, :3] + xyz_perturb, xyzt_choice[:, 3:4] + t_perturb], dim=-1) # 扰动xyz和t
                            
                        else:
                            # xyzt_perturb = xyzt + (torch.rand_like(xyzt) * 2.0 - 1.0) * self.temporal_perturb_range[None, ...]
                            
                            xyz_perturb = (torch.rand_like(xyzt_normal[:, :3]) * 2.0 - 1.0) * self.reg_temporal_random_perturb_range[:3]
                            t_perturb = (torch.rand(1, device=xyzt_normal.device) * 2.0 - 1.0).expand(xyzt_normal.shape[0], 1) * self.temporal_perturb_range[3]
                            # xyzt_perturb = torch.cat([xyzt[:, :3], xyzt[:, 3:4] + t_perturb], dim=-1) # 仅扰动t
                            # xyzt_perturb = torch.cat([xyzt[:, :3] + xyz_perturb, xyzt[:, 3:4]], dim=-1) # 仅扰动xyz
                            xyzt_perturb = torch.cat([xyzt_normal[:, :3] + xyz_perturb, xyzt_normal[:, 3:4] + t_perturb], dim=-1) # 扰动xyz和t
                        
                        temporal_h_perturb = self.dynamic_texture.encode_temporal(xyzt_perturb)
                        reg_temporal = torch.sum(torch.abs(temporal_h_perturb - temporal_h) ** 2, dim=-1)
                        reg_temporal_hashgrid_loss += torch.mean(reg_temporal)
                    elif self.reg_type == 'flow':
                        pass
                 
                if self.reg_occ_able and self.shader_type == "split_sum_pbr":
                    reg_occ_loss += ks[..., 0:1].mean()

      
        reg_loss_dict['reg_temporal_hashgrid'] = (reg_temporal_hashgrid_loss / batch) * self.reg_temporal_hashgrid_weight
        reg_loss_dict['reg_occ'] = (reg_occ_loss / batch) * self.reg_occ_weight
        if self.reg_light_able:
            reg_loss_dict['reg_light'] = self.get_light_regularization() * self.reg_light_weight
        else:
            reg_loss_dict['reg_light'] = 0.0
        
        if self.shader_type == "direct":
            pred_images = RGBAImages(pred_images) if pred_images else None
        elif self.shader_type == "split_sum_pbr":
            pred_images = PBRAImages(pred_images) if pred_images else None
        
        if return_pbr_attrs:
            cubemap = TextureCubeMap(data=self.cubemap)
            light_vis = cubemap.visualize(
                width=W * 2,
                height=H,
            )
            pbr_attrs = {
                'kd': RGBImages(kd_vis),
                'roughness': RGBImages(roughness_vis),
                'metallic': RGBImages(metallic_vis),
                'occ': RGBImages(occ_vis),
                'light': light_vis,
            }
        else:
            pbr_attrs = None
            
        return (
            pred_images,
            reg_loss_dict,
            pred_meshes if return_pred_mesh else None,
            DepthImages(pred_depths) if pred_depths else None, DepthImages(gt_depths) if gt_depths else None,
            VectorImages(pred_normals) if pred_normals else None, VectorImages(gt_normals) if gt_normals else None,
            pbr_attrs
        )
    
    def render_rgb(self, inputs: DS_Cameras) -> RGBImages:
        if self.shader_type == "direct":
            return self.render_report(inputs)[0].blend(self.get_background_color())
        elif self.shader_type == "split_sum_pbr":
            return self.render_report(inputs)[0].rgb2srgb().blend(self.get_background_color())

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
                'light': self.cubemap,
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
            
            # 'deform_params': self.deform_params,
            # 'weight_params': self.weight_params,

            'light': self.cubemap,
            
            'dynamic_texture': self.dynamic_texture.state_dict(),
            'z_up': self.z_up,
            'background_color': self.background_color,

        }
        return attributes
