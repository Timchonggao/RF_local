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
    TextureCubeMap, TextureSplitSum,
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
class D_Texture(Module):

    # geometry setting
    geometry = 'gt' 
    geometry_scale: float = 0.5

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
    reg_type: Literal['random', 'flow'] = 'random'
    reg_spatial_hashgrid_able: bool = False
    reg_spatial_downsample_ratio: float = 0.5
    reg_spatial_random_perturb_range: Union[float, List[float]] = field(default_factory=lambda:[1e-2, 1e-2, 1e-2, 1e-3])
    reg_temporal_hashgrid_able: bool = False
    reg_temporal_downsample_ratio: float = 0.5
    reg_temporal_random_perturb_range: Union[float, List[float]] = field(default_factory=lambda:[1e-2, 1e-2, 1e-2, 1e-3])
    reg_kd_enc_able: bool = False
    reg_kd_random_perturb_range: Union[float, List[float]] = field(default_factory=lambda:[1e-2, 1e-2, 1e-2, 1e-3])
    reg_occ_able: bool = False
    reg_light_able: bool = False

    min_roughness: float = 0.25

    gt_envmap: Optional[Path] = None

    # rendering setting
    shader_type: Literal["split_sum_pbr", "direct"] = "direct"
    antialias: bool = True
    background_color: Literal["random", "black", "white"] = "random"
    z_up: bool = False
    
    def __setup__(self) -> None:
        if self.geometry == 'gt':
            self.geometric_repr = None
        else:
            raise ValueError(self.geometry)

        self.batch_gt_geometry = None

        self.reg_spatial_hashgrid_weight = 0.0
        self.reg_temporal_hashgrid_weight = 0.0
        self.reg_kd_enc_weight = 0.0
        self.reg_occ_weight = 0.0
        self.reg_light_weight = 0.0
        self.envmap = None
        if self.gt_envmap is None:
            self.cubemap = torch.nn.Parameter(torch.empty(6, 512, 512, 3).fill_(0.5))
        else:
            self.cubemap = torch.nn.Parameter(torch.empty(0))

    def set_batch_gt_geometry(self, batch_mesh: List[DS_TriangleMesh]) -> None:
        self.batch_gt_geometry = batch_mesh
    
    def get_gt_geometry(self, indice: int) -> DS_TriangleMesh:
        assert self.batch_gt_geometry is not None # need to set batch_gt_geometry before call this function, ussually called in trainer.step
        return self.batch_gt_geometry[indice].to(self.device)

    def get_envmap(self) -> TextureSplitSum:
        if self.gt_envmap is None:
            return TextureCubeMap(data=self.cubemap, transform=None).as_splitsum()
        if self.envmap is None:
            self.envmap = TextureCubeMap.from_image_file(self.gt_envmap, device=self.device).as_splitsum()
        return self.envmap
    
    def get_light_regularization(self) -> Tensor:
        if self.gt_envmap is None:
            white = self.cubemap.mean(-1, keepdim=True) # [6, R, R, 1]
            return (self.cubemap - white).abs().mean()
        return torch.zeros(1, device=self.device)
    
    def render_report(
            self, 
            camera_inputs: DS_Cameras,
            trainer_mode: Literal['train', 'val', 'test', 'orbit_vis', 'fix_vis'] = 'train',
            return_pbr_attrs: bool = False,
        ) -> Tuple[
            Union[RGBAImages, PBRAImages], 
            Dict[str, Tensor],
            Optional[Dict[str, RGBImages]]
        ]:

        if return_pbr_attrs:
            assert trainer_mode in ['val', 'test', 'orbit_vis', 'fix_vis'] and self.shader_type == "split_sum_pbr"
        
        batch = len(camera_inputs)
        pred_images = [] # color
        kd_vis = []
        roughness_vis = []
        metallic_vis = []
        occ_vis = []
        reg_loss_dict = {}
        reg_spatial_hashgrid_loss = 0.0
        reg_temporal_hashgrid_loss = 0.0
        reg_kd_enc_loss = 0.0
        reg_occ_loss = 0.0
        
        envmap = self.get_envmap()
        ctx = dr.RasterizeCudaContext(camera_inputs.device)
        for i in range(batch):
            gt_mesh = self.get_gt_geometry(i)
            gt_mesh = gt_mesh.compute_vertex_normals(fix=True)
            camera = camera_inputs[i]
            vertices = torch.cat((
                gt_mesh.vertices,
                torch.ones_like(gt_mesh.vertices[..., :1]),
            ), dim=-1).view(-1, 4, 1)
            camera_pos = camera.c2w[:, 3]
            H, W = resolution = [camera.height.item(), camera.width.item()]
            mvp = camera.projection_matrix @ camera.view_matrix                    # [4, 4]
            projected = (mvp @ vertices).view(1, -1, 4)                            # [1, V, 4]
            indices = gt_mesh.indices.int()
            with dr.DepthPeeler(ctx, projected, indices, resolution=resolution) as peeler:
                rast, _ = peeler.rasterize_next_layer()
            alphas = (rast[..., -1:] > 0).float() # [1, H, W, 1]
            frag_pos, _ = dr.interpolate(gt_mesh.vertices[None], rast, indices) # [1, H, W, 3]
            
            alpha_mask = alphas.view(-1) > 0.5  # [H*W]
            valid_indices = torch.where(alpha_mask)[0]  # [V], 有效像素的扁平化索引
            xyz = frag_pos.view(-1, 3)[alpha_mask] # [V, 3] get partial vertices position
            xyz_normal = (xyz / self.geometry_scale).clamp(-1, 1) # 归一化到[-1, 1]
            t_normal = (camera.times * 2 - 1).unsqueeze(0).expand(xyz_normal.shape[0], -1).clamp(-1, 1) # [-1, 1]
            xyzt_normal = torch.cat((xyz_normal, t_normal), dim=-1) # [V, 4]
            kdks, spatial_h, temporal_h = self.dynamic_texture(xyzt_normal) # [V, 6], [V, D], [V, D']
            
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

                frag_n, _ = dr.interpolate(gt_mesh.normals[None], rast, indices) # [1, H, W, 3]
                frag_n = safe_normalize(frag_n) # [1, H, W, 3]

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
                            choice = torch.randperm(xyz_normal.shape[0], device=xyz_normal.device)[: int(max(1, xyz_normal.shape[0] * self.reg_spatial_downsample_ratio))]
                            spatial_h = spatial_h[choice]
                            xyz_normal_choice = xyz_normal[choice]
                            xyz_perturb = xyz_normal_choice + (torch.rand_like(xyz_normal_choice) * 2.0 - 1.0) * self.reg_spatial_random_perturb_range[None, ...]  
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
                        
                        xyz_perturb = (torch.rand_like(xyzt_normal[:, :3]) * 2.0 - 1.0) * self.reg_kd_random_perturb_range[:3]
                        t_perturb = (torch.rand(1, device=xyzt_normal.device) * 2.0 - 1.0).expand(xyzt_normal.shape[0], 1) * self.reg_kd_random_perturb_range[3]
                        # xyzt_perturb = torch.cat([xyzt[:, :3], xyzt[:, 3:4] + t_perturb], dim=-1) # 仅扰动t
                        # xyzt_perturb = torch.cat([xyzt[:, :3] + xyz_perturb, xyzt[:, 3:4]], dim=-1) # 仅扰动xyz
                        xyzt_perturb = torch.cat([xyzt_normal[:, :3] + xyz_perturb, xyzt_normal[:, 3:4] + t_perturb], dim=-1) # 扰动xyz和t

                        kdks_perturb, _, _ = self.dynamic_texture(xyzt_perturb)
                        reg_kd_enc_loss += torch.mean(torch.abs(kdks_perturb[..., :3] - kdks[..., :3]))
                    elif self.reg_type == 'flow':
                        pass
                
                if self.reg_occ_able and self.shader_type == "split_sum_pbr":
                    reg_occ_loss += ks[..., 0:1].mean()

        reg_loss_dict['reg_spatial_hashgrid'] = (reg_spatial_hashgrid_loss / batch) * self.reg_spatial_hashgrid_weight
        reg_loss_dict['reg_temporal_hashgrid'] = (reg_temporal_hashgrid_loss / batch) * self.reg_temporal_hashgrid_weight
        reg_loss_dict['reg_kd_enc'] = (reg_kd_enc_loss / batch) * self.reg_kd_enc_weight
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
            pbr_attrs,
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
                'light': self.cubemap,
            }[field_name]
            return [params]
        return parameters
