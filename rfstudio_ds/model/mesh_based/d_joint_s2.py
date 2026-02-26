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
    FlexiCubes
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
class D_Joint_S2(Module):

    load : Path = ...
    
    # dynamic geometry setting
    geometry_sdf_residual: Grid4d_HashEncoding = Grid4d_HashEncoding(
        decoder=MLPDecoderNetwork(
            layers=[-1, 32, 32, 1],
            activation='tanh',
            bias=False,
            initialization='kaiming-uniform',
        ),
        grad_scaling=16.0,
        backend='grid4d',
        deform_base_resolution=[16,16,32],
        deform_desired_resolution=[1024, 1024, 256],
        deform_num_levels=32,
    )
    geometry_deform_residual: Grid4d_HashEncoding = Grid4d_HashEncoding(
        decoder=MLPDecoderNetwork(
            layers=[-1, 32, 32, 3],
            activation='none',
            bias=False,
            initialization='kaiming-uniform',
        ),
        grad_scaling=16.0,
        backend='grid4d',
        deform_base_resolution=[16,16,32],
        deform_desired_resolution=[1024, 1024, 256],
        deform_num_levels=32,
    )
    geometry_weight_residual: Grid4d_HashEncoding = Grid4d_HashEncoding(
        decoder=MLPDecoderNetwork(
            layers=[-1, 96, 32, 21],
            activation='none',
            bias=False,
            initialization='kaiming-uniform',
        ),
        grad_scaling=16.0,
        backend='grid4d',
        deform_base_resolution=[16,16,32],
        deform_desired_resolution=[1024, 1024, 256],
        deform_num_levels=32,
    )
    targe_high_flexicube_res: int = 128
    geometry_residual_enabled: bool = True
    reg_geometry_residual_type: Literal['random', 'flow'] = 'random'
    reg_geometry_residual_temporal_hashgrid_able: bool = False
    reg_geometry_residual_temporal_downsample_ratio: float = 0.5
    reg_geometry_residual_temporal_random_perturb_range: Union[float, List[float]] = field(default_factory=lambda:[1e-2, 1e-2, 1e-2, 1e-2])
    # geometry regularization setting
    reg_sdf_entropy_able: bool = False
    reg_time_tv_able: bool = False
    reg_sdf_eikonal_able: bool = False
    reg_coeff_tv_able: bool = False
    
    # dynamic texture setting
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
    min_roughness: float = 0.1
    gt_envmap: Optional[Path] = None
    # texture regularization setting
    reg_type: Literal['random', 'flow'] = 'random'
    reg_appearance_temporal_hashgrid_able: bool = False
    reg_temporal_downsample_ratio: float = 0.5
    reg_temporal_random_perturb_range: Union[float, List[float]] = field(default_factory=lambda:[1e-2, 1e-2, 1e-2, 1e-2])
    reg_occ_able: bool = False
    reg_light_able: bool = False

    # rendering setting
    shader_type: Literal["split_sum_pbr", "direct"] = "split_sum_pbr"
    antialias: bool = True
    background_color: Literal["random", "black", "white"] = "white"
    z_up: bool = False
    
    def __setup__(self) -> None:
        state_dict = torch.load(self.load, map_location='cpu')
        
        self.geometry_resolution = state_dict['geometry_resolution']
        self.geometry_scale = state_dict['geometry_scale']
        self.poly_degree = state_dict['poly_degree']
        self.low_freq_fourier_bands = state_dict['low_freq_fourier_bands']
        self.mid_freq_fourier_bands = state_dict['mid_freq_fourier_bands']
        self.high_freq_fourier_bands = state_dict['high_freq_fourier_bands']
        
        self.geometric_repr = DualDomain4DFlexiCubes.from_resolution(
            self.geometry_resolution,
            scale=self.geometry_scale,
            poly_degree=self.poly_degree,
            low_freq_fourier_bands=self.low_freq_fourier_bands,
            mid_freq_fourier_bands=self.mid_freq_fourier_bands,
            high_freq_fourier_bands=self.high_freq_fourier_bands,
        )
        self.static_sdf_params = torch.nn.Parameter(state_dict['static_sdf_params'], requires_grad=True)
        self.sdf_curve_poly_coefficient = torch.nn.Parameter(state_dict['sdf_curve_poly_coefficient'], requires_grad=True)
        self.sdf_curve_low_freq_fourier_coefficient = torch.nn.Parameter(state_dict['sdf_curve_low_freq_fourier_coefficient'], requires_grad=True)
        self.sdf_curve_mid_freq_fourier_coefficient = torch.nn.Parameter(state_dict['sdf_curve_mid_freq_fourier_coefficient'], requires_grad=True)
        self.sdf_curve_high_freq_fourier_coefficient = torch.nn.Parameter(state_dict['sdf_curve_high_freq_fourier_coefficient'], requires_grad=True)
        self.dynamic_texture.load_state_dict(state_dict['dynamic_texture'])
        
        self.batch_gt_geometry = None
        self.geometry_residual_weight = 0.0
        self.dynamic_model_stage: dict = field(default_factory=dict)
        self.reg_sdf_entropy_weight = 0.0 
        self.reg_time_tv_weight = 0.0
        self.reg_sdf_eikonal_weight = 0.0
        self.reg_sdf_eikonal_type = "L2"
        self.reg_coeff_tv_weight = 0.0
        self.reg_reg_coeff_tv_type = "L1"
        self.reg_geometry_residual_temporal_hashgrid_weight = 0.0
        self.reg_appearance_temporal_hashgrid_weight = 0.0
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

    def get_corse_geometry(self, times: Tensor = None, return_sdf_values: bool = False, return_meshes: bool = False):

        assert times is not None
        with torch.no_grad():
            if self.geometric_repr.device != self.device:
                self.geometric_repr.swap_(self.geometric_repr.to(self.device))
        geometric_repr = self.geometric_repr.replace(
            static_sdf_values = self.static_sdf_params,
            sdf_curve_poly_coefficient = self.sdf_curve_poly_coefficient,
            sdf_curve_low_freq_fourier_coefficient = self.sdf_curve_low_freq_fourier_coefficient,
            sdf_curve_mid_freq_fourier_coefficient = self.sdf_curve_mid_freq_fourier_coefficient,
            sdf_curve_high_freq_fourier_coefficient = self.sdf_curve_high_freq_fourier_coefficient,
        )
        pred_meshes, loss_dict, dynamic_sdf_values = geometric_repr.dual_marching_cubes_at_times(
            t=times, 
            scale=self.geometry_scale,
            model_stage=self.dynamic_model_stage,
            compute_sdf_flow=self.reg_time_tv_able,
            compute_time_tv_loss=self.reg_time_tv_able,
            compute_coeff_tv_loss=self.reg_coeff_tv_able,
            compute_coeff_tv_type=self.reg_reg_coeff_tv_type,
            sdf_eps=None,
            return_sdf_values=return_sdf_values,
            return_meshes=return_meshes,
        )

        return pred_meshes, loss_dict, dynamic_sdf_values

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
        reg_appearance_temporal_hashgrid_loss = 0.0
        reg_geometry_residual_temporal_hashgrid_loss = 0.0
        reg_occ_loss = 0.0
        detail_L_dev = 0.0
        detail_sdf_entropy_loss = 0.0
        detail_sdf_eikonal_loss = 0.0

        corse_pred_meshes, reg_loss_dict, dynamic_sdf_values = self.get_corse_geometry(times=camera_inputs.times, return_sdf_values=True, return_meshes=self.geometry_residual_enabled)
        envmap = self.get_envmap()
        ctx = dr.RasterizeCudaContext(camera_inputs.device)
        depth_shader = DepthShader(antialias=self.antialias, culling=False)
        normal_shader = NormalShader(antialias=self.antialias, normal_type='vertex')

        for i in range(batch):
            camera = camera_inputs[i]
            gt_mesh = self.get_gt_geometry(i)
            corse_sdf_value = dynamic_sdf_values[i]
            if self.geometry_residual_enabled:
                corse_pred_mesh = corse_pred_meshes[i]
            
            fc_low = FlexiCubes.from_resolution(self.geometry_resolution, scale=self.geometry_scale, device=self.device)
            fc_low = fc_low.replace(sdf_values=corse_sdf_value)
            
            fc_high = fc_low.upsample(new_resolution=self.targe_high_flexicube_res, scale=self.geometry_scale)
            if self.geometry_residual_enabled:
                unique_indices = fc_high.query_cubes( # fc_high.vertices.max(0) values=tensor([0.500, 0.500, 0.500], fc_high.vertices.min(0) values=tensor([-0.500, -0.500, -0.500]
                    corse_pred_mesh.vertices, #  corse_pred_mesh.vertices.max(0) values=tensor([0.111, 0.137, 0.335], corse_pred_mesh.vertices.min(0) values=tensor([-0.148, -0.137, -0.334]
                    scale=self.geometry_scale,
                    dilation=1
                ) # unique_indices是有效的cube indice，用于fc_high.indices第一维度的索引
                activated_cube_vertex_indices = fc_high.indices[unique_indices].view(-1) # [M * 8] 纯vertice索引
                activated_unique_cube_vertex_indices = torch.unique(activated_cube_vertex_indices, sorted=True) # [N_unique]
                activated_unique_cube_vertex_positions = fc_high.vertices[activated_unique_cube_vertex_indices] # [N_unique, 3] activated_unique_cube_vertex_positions.max(0) values=tensor([0.070, 0.078, 0.180], activated_unique_cube_vertex_positions.min(0) values=tensor([-0.086, -0.078, -0.180],
                activated_unique_cube_vertex_positions_normalize = (activated_unique_cube_vertex_positions / self.geometry_scale).clamp(-1, 1) # 归一化到[-1, 1]
                t_normalize = (camera.times * 2 - 1).unsqueeze(0).expand(activated_unique_cube_vertex_positions_normalize.shape[0], -1).clamp(-1, 1) # [-1, 1]
                activated_vertex_positions_t_normalize = torch.cat((activated_unique_cube_vertex_positions_normalize, t_normalize), dim=-1) # [N_unique, 4]
                # activated_unique_cube_vertex_residual, activated_unique_cube_vertex_residual_temporal_h = self.geometry_residual(activated_vertex_positions_t_normalize)
                
                detail_sdf_value = fc_high.sdf_values
                # sdf_residual = activated_unique_cube_vertex_residual[:,0].unsqueeze(-1)
                sdf_residual, sdf_residual_temporal_h = self.geometry_sdf_residual(activated_vertex_positions_t_normalize)
                detail_sdf_value[activated_unique_cube_vertex_indices] += sdf_residual * self.geometry_residual_weight
                
                detail_flexicube_vertives = fc_high.vertices
                # deform_residual = activated_unique_cube_vertex_residual[:,1:4]
                deform_residual, deform_residual_temporal_h = self.geometry_deform_residual(activated_vertex_positions_t_normalize)
                detail_flexicube_vertives[activated_unique_cube_vertex_indices] += deform_residual.tanh() * (0.5 * self.geometry_scale / self.targe_high_flexicube_res)
                
                detail_flexicube_weight_params = torch.nn.Parameter(torch.ones(fc_high.indices.shape[0], 21)).to(self.device)
                # weight_residual = activated_unique_cube_vertex_residual[:,4:]
                weight_residual, weight_residual_temporal_h = self.geometry_weight_residual(activated_vertex_positions_t_normalize)
                vertex_residual_full = torch.zeros(fc_high.vertices.shape[0], 21, device=self.device)
                vertex_residual_full[activated_unique_cube_vertex_indices] = weight_residual
                cube_vertex_residuals = vertex_residual_full[fc_high.indices[unique_indices]]  
                cube_weight_residual = cube_vertex_residuals.mean(dim=1) 
                detail_flexicube_weight_params[unique_indices] += cube_weight_residual

                fc_high.replace_(
                    vertices=detail_flexicube_vertives,
                    sdf_values = detail_sdf_value,
                    alpha=detail_flexicube_weight_params[:, :8],
                    beta=detail_flexicube_weight_params[:, 8:20],
                    gamma=detail_flexicube_weight_params[:, 20:],
                )
            
            pred_mesh, detail_L_dev_ = fc_high.dual_marching_cubes()
            detail_sdf_entropy_loss_ = fc_high.compute_entropy() * self.reg_sdf_entropy_weight if self.reg_sdf_entropy_able else 0.0
            detail_sdf_eikonal_loss_ = fc_high.compute_sdf_eikonal_loss(scale=self.geometry_scale) * self.reg_sdf_eikonal_weight if self.reg_sdf_eikonal_able else 0.0
            detail_L_dev += detail_L_dev_.mean() * 0.25
            detail_sdf_entropy_loss += detail_sdf_entropy_loss_
            detail_sdf_eikonal_loss += detail_sdf_eikonal_loss_
            # test vis debug code below
            # pred_normal = VectorImages(pred_mesh.render(camera, shader=normal_shader).item())
            # dump_float32_image(Path(f"./low_res_pred_normal.png"), pred_normal[0].visualize((1,1,1)).item())
            # dump_float32_image(Path(f"./high_res_pred_normal.png"), pred_normal[0].visualize((1,1,1)).item())
            
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
            
            if return_pred_mesh:
                pred_meshes.append(pred_mesh)

            # reg loss
            if trainer_mode in ['train', 'val', 'test']:
                if self.reg_appearance_temporal_hashgrid_able:
                    if self.reg_type == 'random':
                        if type(self.reg_temporal_random_perturb_range) is float:
                            reg_temporal_random_perturb_range_list = [self.reg_temporal_random_perturb_range for _ in range(4)] # 时间正则化扰动：若启用，默认范围为1e-2，沿x、y、z、t四个维度。
                            self.reg_temporal_random_perturb_range = torch.tensor(reg_temporal_random_perturb_range_list, device="cuda", dtype=torch.float32)
                        else:
                            assert len(self.reg_temporal_random_perturb_range) == 4, "reg_temporal_random_perturb_range should be a float or a list of 4 floats"
                            self.reg_temporal_random_perturb_range = torch.tensor(self.reg_temporal_random_perturb_range, device="cuda", dtype=torch.float32)

                        if self.reg_temporal_downsample_ratio < 1.0:
                            choice = torch.randperm(xyzt_normal.shape[0], device=xyzt_normal.device)[: int(max(1, xyzt_normal.shape[0] * self.reg_temporal_downsample_ratio))]
                            temporal_h_choice = temporal_h[choice]
                            xyzt_choice = xyzt_normal[choice]
                            
                            # xyzt_perturb = xyzt_choice + (torch.rand_like(xyzt_choice) * 2.0 - 1.0) * self.temporal_perturb_range[None, ...]
                            
                            xyz_perturb = (torch.rand_like(xyzt_choice[:, :3]) * 2.0 - 1.0) * self.reg_temporal_random_perturb_range[:3]
                            t_perturb = (torch.rand(1, device=xyzt_normal.device) * 2.0 - 1.0).expand(xyzt_choice.shape[0], 1) * self.reg_temporal_random_perturb_range[3]
                            # xyzt_perturb = torch.cat([xyzt_choice[:, :3], xyzt_choice[:, 3:4] + t_perturb], dim=-1) # 仅扰动t
                            # xyzt_perturb = torch.cat([xyzt_choice[:, :3] + xyz_perturb, xyzt_choice[:, 3:4]], dim=-1) # 仅扰动xyz
                            xyzt_perturb = torch.cat([xyzt_choice[:, :3] + xyz_perturb, xyzt_choice[:, 3:4] + t_perturb], dim=-1) # 扰动xyz和t
                            
                        else:
                            # xyzt_perturb = xyzt + (torch.rand_like(xyzt) * 2.0 - 1.0) * self.temporal_perturb_range[None, ...]
                            temporal_h_choice = temporal_h
                            xyz_perturb = (torch.rand_like(xyzt_normal[:, :3]) * 2.0 - 1.0) * self.reg_temporal_random_perturb_range[:3]
                            t_perturb = (torch.rand(1, device=xyzt_normal.device) * 2.0 - 1.0).expand(xyzt_normal.shape[0], 1) * self.temporal_perturb_range[3]
                            # xyzt_perturb = torch.cat([xyzt[:, :3], xyzt[:, 3:4] + t_perturb], dim=-1) # 仅扰动t
                            # xyzt_perturb = torch.cat([xyzt[:, :3] + xyz_perturb, xyzt[:, 3:4]], dim=-1) # 仅扰动xyz
                            xyzt_perturb = torch.cat([xyzt_normal[:, :3] + xyz_perturb, xyzt_normal[:, 3:4] + t_perturb], dim=-1) # 扰动xyz和t
                        
                        temporal_h_perturb = self.dynamic_texture.encode_temporal(xyzt_perturb)
                        reg_temporal = torch.sum(torch.abs(temporal_h_perturb - temporal_h_choice) ** 2, dim=-1)
                        reg_appearance_temporal_hashgrid_loss += torch.mean(reg_temporal) * self.reg_appearance_temporal_hashgrid_weight
                    elif self.reg_type == 'flow':
                        pass
                
                if self.reg_geometry_residual_temporal_hashgrid_able and self.geometry_residual_enabled:
                    if self.reg_geometry_residual_type == 'random':
                        if type(self.reg_geometry_residual_temporal_random_perturb_range) is float:
                            reg_geometry_residual_temporal_random_perturb_range_list = [self.reg_geometry_residual_temporal_random_perturb_range for _ in range(4)] # 时间正则化扰动：若启用，默认范围为1e-2，沿x、y、z、t四个维度。
                            self.reg_geometry_residual_temporal_random_perturb_range = torch.tensor(reg_geometry_residual_temporal_random_perturb_range_list, device="cuda", dtype=torch.float32)
                        else:
                            assert len(self.reg_geometry_residual_temporal_random_perturb_range) == 4, "reg_geometry_residual_temporal_random_perturb_range should be a float or a list of 4 floats"
                            self.reg_geometry_residual_temporal_random_perturb_range = torch.tensor(self.reg_geometry_residual_temporal_random_perturb_range, device="cuda", dtype=torch.float32)

                        if self.reg_geometry_residual_temporal_downsample_ratio < 1.0:
                            choice = torch.randperm(activated_vertex_positions_t_normalize.shape[0], device=activated_vertex_positions_t_normalize.device)[: int(max(1, activated_vertex_positions_t_normalize.shape[0] * self.reg_geometry_residual_temporal_downsample_ratio))]
                            # temporal_h = activated_unique_cube_vertex_residual_temporal_h[choice]
                            sdf_residual_temporal_h_choice = sdf_residual_temporal_h[choice]
                            deform_residual_temporal_h_choice = deform_residual_temporal_h[choice]
                            weight_residual_temporal_h_choice = weight_residual_temporal_h[choice]
                            xyzt_choice = activated_vertex_positions_t_normalize[choice]

                            xyz_perturb = (torch.rand_like(xyzt_choice[:, :3]) * 2.0 - 1.0) * self.reg_geometry_residual_temporal_random_perturb_range[:3]
                            t_perturb = (torch.rand(1, device=activated_vertex_positions_t_normalize.device) * 2.0 - 1.0).expand(xyzt_choice.shape[0], 1) * self.reg_geometry_residual_temporal_random_perturb_range[3]
                            xyzt_perturb = torch.cat([xyzt_choice[:, :3] + xyz_perturb, xyzt_choice[:, 3:4] + t_perturb], dim=-1) # 扰动xyz和t
                        else:
                            pass
                        # temporal_h_perturb = self.geometry_residual.encode_temporal(xyzt_perturb)
                        # reg_temporal = torch.sum(torch.abs(temporal_h_perturb - temporal_h) ** 2, dim=-1)
                        sdf_residual_temporal_h_perturb = self.geometry_sdf_residual.encode_temporal(xyzt_perturb)
                        reg_sdf_residual_temporal = torch.sum(torch.abs(sdf_residual_temporal_h_perturb - sdf_residual_temporal_h_choice) ** 2, dim=-1)
                        deform_residual_temporal_h_perturb = self.geometry_deform_residual.encode_temporal(xyzt_perturb)
                        reg_deform_residual_temporal = torch.sum(torch.abs(deform_residual_temporal_h_perturb - deform_residual_temporal_h_choice) ** 2, dim=-1)
                        weight_residual_temporal_h_perturb = self.geometry_weight_residual.encode_temporal(xyzt_perturb)
                        reg_weight_residual_temporal = torch.sum(torch.abs(weight_residual_temporal_h_perturb - weight_residual_temporal_h_choice) ** 2, dim=-1)
                        reg_temport = (reg_sdf_residual_temporal + reg_deform_residual_temporal + reg_weight_residual_temporal) / 3.0
                        reg_geometry_residual_temporal_hashgrid_loss += torch.mean(reg_temporal) * self.reg_geometry_residual_temporal_hashgrid_weight
                
                if self.reg_occ_able and self.shader_type == "split_sum_pbr":
                    reg_occ_loss += ks[..., 0:1].mean() * self.reg_occ_weight
      
        reg_loss_dict['detail_L_dev'] = (detail_L_dev / batch)
        reg_loss_dict['detail_sdf_entropy_loss'] = (detail_sdf_entropy_loss / batch)
        reg_loss_dict['detail_sdf_eikonal_loss'] = (detail_sdf_eikonal_loss / batch)
        reg_loss_dict['reg_appearance_temporal_hashgrid'] = (reg_appearance_temporal_hashgrid_loss / batch) 
        reg_loss_dict['reg_geometry_residual_temporal_hashgrid'] = (reg_geometry_residual_temporal_hashgrid_loss / batch) 
        reg_loss_dict['reg_occ'] = (reg_occ_loss / batch)
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
                'light': self.cubemap,
            }[field_name]
            return [params]
        return parameters


