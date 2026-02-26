from __future__ import annotations

import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Tuple, Union

import torch
from gsplat import rasterization
from jaxtyping import Float32
from kornia.filters import spatial_gradient
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from rfstudio.graphics import (
    Cameras,
    DMTet,
    FlexiCubes,
    PBRAImages,
    Points,
    RGBAImages,
    RGBImages,
    Splats,
    TextureCubeMap,
    TextureLatLng,
    TextureSplitSum,
    TriangleMesh,
    VectorImages,
)
from rfstudio.graphics._mesh._optix import OptiXContext, bilateral_denoiser, optix_build_bvh, optix_env_shade
from rfstudio.graphics.math import (
    get_rotation_from_relative_vectors,
    rot2quat,
    safe_normalize,
)
from rfstudio.graphics.shaders import NormalShader, PrettyShader, _get_fg_lut
from rfstudio.model.density_field.components import SDFField
from rfstudio.model.density_field.components.encoding import HashEncoding
from rfstudio.nn import MLP, Module
from rfstudio.utils.decorator import chains
from rfstudio.utils.lazy_module import dr
from rfstudio.utils.tensor_dataclass import Float, TensorDataclass

from .gsplat import GSplatter


@dataclass
class RenderableAttrs(TensorDataclass):

    kd: Tensor = Float[..., 3]
    ks: Tensor = Float[..., 2]
    occ: Optional[Tensor] = Float[..., 6]
    normals: Tensor = Float[..., 3]
    kd_jitter: Optional[Tensor] = Float[..., 3]
    ks_jitter: Optional[Tensor] = Float[..., 2]

    def splat(
        self,
        gsplat: GSplatter,
        cameras: Cameras,
        *,
        exposure: Float32[Tensor, "1"],
        envmap: TextureSplitSum,
        occ_type: Literal['texture', 'none'],
        min_roughness: float,
        max_metallic: float,
        mode: Literal['pbr', 'diffuse', 'specular'] = 'pbr',
        tone_type: Literal['aces', 'naive', 'none'] = 'naive',
        factor: float = 1.0,
        culling: bool = False,
    ) -> Float32[Tensor, "H W 4"]:
        assert cameras.shape == (1, )
        camera = cameras[0]
        camera_pos = camera.c2w[:, 3] # [3]

        if culling:
            with torch.no_grad():
                wo = safe_normalize(camera_pos - gsplat.gaussians.means) # [N, 3]
                mask = (self.normals * wo).sum(-1) > 0.0 # [N]

            if not mask.any():
                raise ValueError("No valid splat found.")
        else:
            mask = ...

        origin_gaussians = gsplat.gaussians
        gsplat.gaussians = origin_gaussians[mask]

        filtered_normals = self.normals[mask]
        filtered_kd = self.kd[mask] # [N_, 3]
        filtered_roughness = self.ks[mask, 0:1] * (1 - min_roughness) + min_roughness # [N_, 1]
        filtered_metallic = self.ks[mask, 1:2] * max_metallic # [N_, 1]
        filtered_specular = (1.0 - filtered_metallic) * 0.04 + filtered_kd * filtered_metallic # [N_, 3]
        filtered_diffuse = filtered_kd * (1.0 - filtered_metallic) # [N_, 3]

        filtered_wo = safe_normalize(camera_pos - gsplat.gaussians.means) # [N_, 3]
        filtered_n_dot_v = (filtered_normals * filtered_wo).sum(-1, keepdim=True).clamp(min=1e-6) # [N_, 1]
        filtered_fg_uv = torch.cat((filtered_n_dot_v, filtered_roughness), dim=-1)[None, None] # [1, 1, N_, 2]
        filtered_fg_lookup: Tensor = dr.texture(
            _get_fg_lut(resolution=256, device=camera.device),
            filtered_fg_uv,
            filter_mode='linear',
            boundary_mode='clamp',
        ).view(-1, 2)

        # Compute aggregate lighting
        filtered_inv_wi = torch.sub(
            2 * (filtered_wo * filtered_normals).sum(-1, keepdim=True) * filtered_normals,
            filtered_wo
        ) # [N_, 3]
        filtered_l_diff, filtered_l_spec = envmap.sample(
            normals=filtered_normals[None, None],
            directions=filtered_inv_wi[None, None],
            roughness=filtered_roughness[None, None],
        ) # [1, 1, N_, 3]
        filtered_reflectance = filtered_specular * filtered_fg_lookup[..., 0:1] + filtered_fg_lookup[..., 1:2] # [N_, 3]
        # if occ_type != 'none':
        #     filtered_diffuse = filtered_diffuse * (self.occ[mask].sigmoid() * 0.4 + 0.8)
        if mode == 'pbr':
            filtered_colors = torch.add(
                # filtered_l_diff.view(-1, 3) * filtered_diffuse,
                filtered_diffuse,
                filtered_l_spec.view(-1, 3) * filtered_reflectance,
            ) # [N_, 3]
        elif mode == 'diffuse':
            filtered_colors = filtered_l_diff.view(-1, 3) * filtered_diffuse
        elif mode == 'specular':
            filtered_colors = filtered_l_spec.view(-1, 3) * filtered_reflectance
        else:
            raise ValueError(mode)
        gsplat.gaussians.replace_(colors=filtered_colors)
        if tone_type == 'none':
            pbr_image = gsplat.render_rgba(cameras).item() * exposure
        elif tone_type == 'aces':
            pbr_image = _tone_mapping_aces(gsplat.render_rgba(cameras).item(), exposure)
        elif tone_type == 'naive':
            pbr_image = _tone_mapping_naive(gsplat.render_rgba(cameras).item(), exposure)
        else:
            raise ValueError(tone_type)
        gsplat.gaussians = origin_gaussians
        return pbr_image

    def splat_mc(
        self,
        gsplat: GSplatter,
        cameras: Cameras,
        *,
        gt_image: Optional[Tensor],
        context: OptiXContext,
        exposure: Float32[Tensor, "1"],
        positions: Tensor,
        envmap: TextureLatLng,
        min_roughness: float,
        max_metallic: float,
        mode: Literal['pbr', 'diffuse', 'specular'] = 'pbr',
        tone_type: Literal['aces', 'naive', 'none'] = 'naive',
        num_samples_per_ray: int = 32,
        shadow_scale: float = 1.0,
        denoise: bool = False,
        residual: bool = False,
    ) -> Float32[Tensor, "H W 4"]:
        assert cameras.shape == (1, )
        camera = cameras[0]
        camera_pos = camera.c2w[:, 3] # [3]
        camera_lookat = -camera.c2w[:, 2] # [3]

        frag_pos = positions.view(1, 1, -1, 3)
        frag_n = self.normals.view(1, 1, -1, 3)
        bend = ((frag_n.detach() * camera_lookat).sum(-1, keepdim=True) > 1e-3)
        frag_n = torch.where(bend, -frag_n, frag_n)
        frag_depth = ((frag_pos - camera_pos) * camera_lookat).sum(-1, keepdim=True)
        assert envmap.transform is None
        kd = self.kd.view(1, 1, -1, 3)
        roughness = self.ks[..., :1] * (1 - min_roughness) + min_roughness
        metallic = self.ks[..., 1:] * max_metallic
        ks = torch.cat((torch.zeros_like(roughness), roughness, metallic), dim=-1).view(1, 1, -1, 3)
        diffuse_accum, specular_accum, residual_accum = optix_env_shade(
            context,
            torch.ones_like(frag_n[..., :1]),
            frag_pos + self.normals.detach().view(1, 1, -1, 3) * 1e-5,
            frag_pos,
            frag_n,
            camera_pos.contiguous().view(1, 1, 1, 3),
            kd,
            ks,
            envmap.data,
            envmap.pdf[..., 0],
            envmap.pdf[:, 0, 1],
            envmap.pdf[..., 2],
            BSDF='pbr',
            n_samples_x=num_samples_per_ray,
            rnd_seed=None,
            shadow_scale=shadow_scale,
        )

        diffuse_accum = diffuse_accum.clamp_min(1e-4)
        specular_accum = specular_accum.clamp_min(1e-4)
        residual_accum = residual_accum.clamp(0, 1)
        kd_factor = kd * (1 - metallic)

        if denoise:
            sigma = max(shadow_scale * 2, 0.0001)
            diffuse_accum  = bilateral_denoiser(diffuse_accum, frag_n, frag_depth, sigma)
            specular_accum = bilateral_denoiser(specular_accum, frag_n, frag_depth, sigma)

        if residual:
            if denoise:
                sigma = max(shadow_scale * 2, 0.0001)
                residual_accum = bilateral_denoiser(
                    torch.cat((torch.zeros_like(residual_accum[..., :1]), residual_accum), dim=-1),
                    frag_n,
                    frag_depth,
                    sigma,
                )[..., 1:]
            residual_light = (self.occ - 3).sigmoid()
            residual_diff_accum = residual_accum[..., 0:1] * residual_light[..., :3]
            residual_spec_accum = residual_accum[..., 1:2] * residual_light[..., 3:]
            diffuse_accum = diffuse_accum + residual_diff_accum
            specular_accum = specular_accum + residual_spec_accum

        if mode == 'pbr':
            filtered_colors = diffuse_accum * kd_factor + specular_accum
        elif mode == 'diffuse':
            filtered_colors = diffuse_accum * kd_factor
        elif mode == 'specular':
            filtered_colors = specular_accum
        else:
            raise ValueError(mode)

        reg = 0
        if gt_image is not None and 0:
            gsplat.gaussians.replace_(colors=diffuse_accum.view(-1, 3))
            diffuse_map = gsplat.render_rgb(cameras).item()
            diffuse_luma = diffuse_map.mean(-1, keepdim=True)
            gsplat.gaussians.replace_(colors=specular_accum.view(-1, 3))
            specular_luma = gsplat.render_rgb(cameras).item().mean(-1, keepdim=True)
            ref_luma = gt_image[..., :3].max(-1, keepdim=True).values
            img_luma = 1 - torch.nn.Softplus(beta=100)(1 - (diffuse_luma + specular_luma) * exposure)
            error = ((img_luma - ref_luma) * gt_image[..., 3:]).abs() * diffuse_luma / (diffuse_luma + specular_luma).clamp_min(1e-3)
            reg = error.mean() * 0.15 + specular_luma.mean() / diffuse_luma.mean().clamp_min(1e-3) * 0.0025
            # reg = error.mean() * 0.05 + (diffuse_map - diffuse_luma).abs().mean() * 0.01 + specular_luma.mean() / diffuse_luma.mean().clamp_min(1e-3) * 0.005

        original_colors = gsplat.gaussians.colors
        gsplat.gaussians.replace_(colors=filtered_colors.view(-1, 3))
        if tone_type == 'none':
            pbr_image = gsplat.render_rgba(cameras).item() * exposure
        elif tone_type == 'aces':
            pbr_image = _tone_mapping_aces(gsplat.render_rgba(cameras).item(), exposure)
        elif tone_type == 'naive':
            pbr_image = _tone_mapping_naive(gsplat.render_rgba(cameras).item(), exposure)
        else:
            raise ValueError(tone_type)
        gsplat.gaussians.replace_(colors=original_colors)
        return pbr_image.squeeze(0), reg


    def splat_mc_deferred(
        self,
        gsplat: GSplatter,
        cameras: Cameras,
        *,
        gt_image: Optional[Tensor],
        context: OptiXContext,
        exposure: Float32[Tensor, "1"],
        positions: Tensor,
        mask: Tensor,
        envmap: TextureLatLng,
        min_roughness: float,
        max_metallic: float,
        mode: Literal['pbr', 'diffuse', 'specular'] = 'pbr',
        tone_type: Literal['aces', 'naive', 'none'] = 'naive',
        num_samples_per_ray: int = 32,
        shadow_scale: float = 1.0,
        denoise: bool = False,
        residual: bool = False,
    ) -> Float32[Tensor, "H W 4"]:
        assert cameras.shape == (1, )
        camera = cameras[0]
        camera_pos = camera.c2w[:, 3] # [3]
        camera_lookat = -camera.c2w[:, 2] # [3]

        bend = ((self.normals.detach() * camera_lookat).sum(-1, keepdim=True) > 0)
        normals = torch.where(bend, -self.normals, self.normals)
        opacities = torch.where(bend, -2, gsplat.gaussians.opacities)

        render, alpha, _ = rasterization(
            means=gsplat.gaussians.means,
            quats=gsplat.gaussians.quats,
            scales=gsplat.gaussians.scales.exp(),
            opacities=opacities.sigmoid().squeeze(-1),
            colors=torch.cat((normals, self.kd, self.ks, self.occ), dim=-1),
            viewmats=camera.view_matrix.view(1, 4, 4),  # [1, 4, 4]
            Ks=camera.intrinsic_matrix.view(1, 3, 3),  # [1, 3, 3]
            width=camera.width.item(),
            height=camera.height.item(),
            tile_size=gsplat.block_width,
            packed=True,
            near_plane=0.01,
            far_plane=1e10,
            render_mode='RGB',
            sh_degree=None,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode='antialiased',
        )
        render = render / alpha.detach().clamp_min(1e-6)
        frag_n = safe_normalize(render[..., 0:3])
        frag_kd = render[..., 3:6]
        frag_roughness = render[..., 6:7] * (1 - min_roughness) + min_roughness
        frag_metallic = render[..., 7:8] * max_metallic
        frag_occ = render[..., 8:14]
        frag_ks = torch.cat((torch.zeros_like(frag_roughness), frag_roughness, frag_metallic), dim=-1)
        frag_depth = ((positions - camera_pos) * camera_lookat).sum(-1, keepdim=True)
        assert envmap.transform is None
        diffuse_accum, specular_accum, residual_accum = optix_env_shade(
            context,
            mask,
            frag_n.detach() * 1e-3 + positions,
            positions,
            frag_n,
            camera_pos.contiguous().view(1, 1, 1, 3),
            frag_kd,
            frag_ks,
            envmap.data,
            envmap.pdf[..., 0],
            envmap.pdf[:, 0, 1],
            envmap.pdf[..., 2],
            BSDF='pbr',
            n_samples_x=num_samples_per_ray,
            rnd_seed=None,
            shadow_scale=shadow_scale,
        )

        diffuse_accum = diffuse_accum.clamp_min(1e-4)
        specular_accum = specular_accum.clamp_min(1e-4)
        residual_accum = residual_accum.clamp(0, 1)
        kd_factor = frag_kd * (1 - frag_metallic)

        if denoise:
            sigma = max(shadow_scale * 2, 0.0001)
            diffuse_accum  = bilateral_denoiser(diffuse_accum, frag_n, frag_depth, sigma)
            specular_accum = bilateral_denoiser(specular_accum, frag_n, frag_depth, sigma)

        if residual:
            if denoise:
                sigma = max(shadow_scale * 2, 0.0001)
                residual_accum = bilateral_denoiser(
                    torch.cat((torch.zeros_like(residual_accum[..., :1]), residual_accum), dim=-1),
                    frag_n,
                    frag_depth,
                    sigma,
                )[..., 1:]
            residual_light = (frag_occ - 3).sigmoid()
            residual_diff_accum = residual_accum[..., 0:1] * residual_light[..., :3]
            residual_spec_accum = residual_accum[..., 1:2] * residual_light[..., 3:]
            diffuse_accum = diffuse_accum + residual_diff_accum
            specular_accum = specular_accum + residual_spec_accum

        if mode == 'pbr':
            filtered_colors = diffuse_accum * kd_factor + specular_accum
        elif mode == 'diffuse':
            filtered_colors = diffuse_accum * kd_factor
        elif mode == 'specular':
            filtered_colors = specular_accum
        else:
            raise ValueError(mode)
        filtered_colors = torch.cat((filtered_colors * alpha.detach(), alpha), dim=-1)

        reg = 0
        if gt_image is not None and 0:
            diffuse_luma = diffuse_accum.mean(-1, keepdim=True) * mask
            specular_luma = specular_accum.mean(-1, keepdim=True) * mask
            ref_luma = gt_image[..., :3].max(-1, keepdim=True).values
            img_luma = 1 - torch.nn.Softplus(beta=100)(1 - (diffuse_luma + specular_luma) * exposure)
            error = ((img_luma - ref_luma) * gt_image[..., 3:]).abs() * diffuse_luma / (diffuse_luma + specular_luma).clamp_min(1e-3)
            reg = error.mean() * 0.15 + specular_luma.mean() / diffuse_luma.mean().clamp_min(1e-3) * 0.0025
            # reg = error.mean() * 0.05 + (diffuse_map - diffuse_luma).abs().mean() * 0.01 + specular_luma.mean() / diffuse_luma.mean().clamp_min(1e-3) * 0.005

        if tone_type == 'none':
            pbr_image = filtered_colors * exposure
        elif tone_type == 'aces':
            pbr_image = _tone_mapping_aces(filtered_colors, exposure)
        elif tone_type == 'naive':
            pbr_image = _tone_mapping_naive(filtered_colors, exposure)
        else:
            raise ValueError(tone_type)
        return pbr_image.squeeze(0), reg

    def splat_deferred(
        self,
        gsplat: GSplatter,
        cameras: Cameras,
        *,
        exposure: Float32[Tensor, "1"],
        envmap: TextureSplitSum,
        occ_type: Literal['texture', 'none'],
        min_roughness: float,
        max_metallic: float,
        mode: Literal['pbr', 'diffuse', 'specular'] = 'pbr',
        tone_type: Literal['aces', 'naive', 'none'] = 'naive',
    ) -> Float32[Tensor, "H W 4"]:
        assert cameras.shape == (1, )
        camera = cameras[0]
        camera_pos = camera.c2w[:, 3] # [3]

        with torch.no_grad():
            wo = safe_normalize(camera_pos - gsplat.gaussians.means) # [N, 3]
            mask = (self.normals * wo).sum(-1) > 0.0 # [N]

        if not mask.any():
            raise ValueError("No valid splat found.")

        render, alpha, _ = rasterization(
            means=gsplat.gaussians.means[mask],
            quats=gsplat.gaussians.quats[mask],
            scales=gsplat.gaussians.scales[mask].exp(),
            opacities=torch.sigmoid(gsplat.gaussians.opacities[mask]).squeeze(-1),
            colors=torch.cat((self.normals, self.kd, self.ks, self.occ, gsplat.gaussians.means), dim=-1)[mask],
            viewmats=camera.view_matrix.view(1, 4, 4),  # [1, 4, 4]
            Ks=camera.intrinsic_matrix.view(1, 3, 3),  # [1, 3, 3]
            width=camera.width.item(),
            height=camera.height.item(),
            tile_size=gsplat.block_width,
            packed=True,
            near_plane=0.01,
            far_plane=1e10,
            render_mode='RGB',
            sh_degree=None,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode='antialiased',
        )
        render = render / alpha.detach().clamp_min(1e-6)
        filtered_normals = safe_normalize(render[..., 0:3])
        filtered_kd = render[..., 3:6]
        filtered_roughness = render[..., 6:7] * (1 - min_roughness) + min_roughness
        filtered_metallic = render[..., 7:8] * max_metallic
        filtered_pos = render[..., 9:12]

        filtered_specular = (1.0 - filtered_metallic) * 0.04 + filtered_kd * filtered_metallic
        filtered_diffuse = filtered_kd * (1.0 - filtered_metallic)
        filtered_wo = safe_normalize(camera_pos - filtered_pos)
        filtered_n_dot_v = (filtered_normals * filtered_wo).sum(-1, keepdim=True).clamp(min=1e-6)
        filtered_fg_uv = torch.cat((filtered_n_dot_v, filtered_roughness), dim=-1)
        filtered_fg_lookup: Tensor = dr.texture(
            _get_fg_lut(resolution=256, device=camera.device),
            filtered_fg_uv,
            filter_mode='linear',
            boundary_mode='clamp',
        )

        # Compute aggregate lighting
        filtered_inv_wi = torch.sub(
            2 * (filtered_wo * filtered_normals).sum(-1, keepdim=True) * filtered_normals,
            filtered_wo
        )
        filtered_l_diff, filtered_l_spec = envmap.sample(
            normals=filtered_normals.contiguous(),
            directions=filtered_inv_wi,
            roughness=filtered_roughness,
        )
        filtered_l_diff = filtered_l_diff * 0.2 + 0.8 * filtered_l_diff.detach()
        filtered_reflectance = filtered_specular * filtered_fg_lookup[..., 0:1] + filtered_fg_lookup[..., 1:2]
        if mode == 'pbr':
            filtered_colors = torch.add(
                filtered_l_diff * filtered_diffuse,
                filtered_l_spec * filtered_reflectance,
            )
        elif mode == 'diffuse':
            filtered_colors = filtered_l_diff * filtered_diffuse
        elif mode == 'specular':
            filtered_colors = filtered_l_spec * filtered_reflectance
        else:
            raise ValueError(mode)
        if occ_type != 'none':
            filtered_colors = (1 - render[..., 8:9]) * filtered_colors
        if tone_type == 'none':
            pbr_image = torch.cat((filtered_colors * alpha.detach() * exposure, alpha), dim=-1)
        elif tone_type == 'aces':
            pbr_image = _tone_mapping_aces(torch.cat((filtered_colors * alpha.detach(), alpha), dim=-1), exposure)
        elif tone_type == 'naive':
            pbr_image = _tone_mapping_naive(torch.cat((filtered_colors * alpha.detach(), alpha), dim=-1), exposure)
        else:
            raise ValueError(tone_type)
        return pbr_image.squeeze(0)


@dataclass
class MGAdapter:

    scale_ratio1: float = 0.5
    scale_ratio2: float = 1.3
    g_scale_ratio: float = 1.6
    l_scale_ratio1: float = 1 / 3
    l_scale_ratio2: float = 3

    bias1: float = -1 / 24
    bias2: float = 0.0

    def bary2gs(
        self,
        p0: Float32[Tensor, "N 3"],
        p1: Float32[Tensor, "N 3"],
        area: Float32[Tensor, "N 1"],
        normals: Float32[Tensor, "N 3"],
        *,
        max_scale_ratio: float,
    ) -> Splats:
        means = (p0 + p1) / 2 # [N, 3]
        max_rots = p1 - means # [N, 3]
        max_scales: Tensor = (p1 - means).norm(dim=-1, keepdim=True).clamp(min=1e-10) # [N, 1]
        min_scales = area / 4 / max_scales # [N, 1]
        max_rots = max_rots / max_scales # [N, 3]
        scales = torch.cat((
            (self.g_scale_ratio * max_scale_ratio * max_scales).log(),
            (self.g_scale_ratio / max_scale_ratio * min_scales).log(),
            torch.empty_like(max_scales).fill_(-10),
        ), dim=-1)
        min_rots = normals.cross(max_rots, dim=-1) # [N, 3]
        quats = rot2quat(
            torch.stack((
                max_rots,
                min_rots,
                normals,
            ), dim=-1) # [N, 3, 3]
        ) # [N, 4]
        return Splats(
            means=means,
            scales=scales,
            quats=quats,
            colors=normals,
            opacities=torch.empty_like(means[:, :1]).fill_(0.99).logit(),
            shs=means.new_empty(means.shape[0], 0, 3),
        )

    def make(
        self,
        mesh: TriangleMesh,
        *,
        normal_interpolation: bool = True,
    ) -> Tuple[Splats, Float32[Tensor, "N 3"]]:

        splats = []

        p0 = mesh.vertices[mesh.indices[..., 0], :] # [F, 3]
        p1 = mesh.vertices[mesh.indices[..., 1], :] # [F, 3]
        p2 = mesh.vertices[mesh.indices[..., 2], :] # [F, 3]
        if normal_interpolation:
            vn0 = mesh.normals[mesh.indices[..., 0], :] # [F, 3]
            vn1 = mesh.normals[mesh.indices[..., 1], :] # [F, 3]
            vn2 = mesh.normals[mesh.indices[..., 2], :] # [F, 3]
        normals = (p1 - p0).cross(p2 - p0) # [F, 3]
        area: Tensor = normals.norm(dim=-1, keepdim=True).clamp(min=1e-10) / 2 # [F, 1]
        normals = safe_normalize(normals)
        offsets = normals.detach() * area.detach().sqrt()
        for u_coeff, a_coeff, s_ratio in zip(
            [1 / 9 + self.bias1, 2 / 9 + self.bias2],
            [1 / 4 * self.l_scale_ratio1, 1 / 12 * self.l_scale_ratio2],
            [self.scale_ratio1, self.scale_ratio2]
        ):
            u0 = p0 * (1 - 2 * u_coeff) + (p1 + p2) * u_coeff # [F, 3]
            u1 = p1 * (1 - 2 * u_coeff) + (p2 + p0) * u_coeff # [F, 3]
            u2 = p2 * (1 - 2 * u_coeff) + (p0 + p1) * u_coeff # [F, 3]
            if normal_interpolation:
                n0 = vn0 * (1 - 2 * u_coeff) + (vn1 + vn2) * u_coeff # [F, 3]
                n1 = vn1 * (1 - 2 * u_coeff) + (vn2 + vn0) * u_coeff # [F, 3]
                n2 = vn2 * (1 - 2 * u_coeff) + (vn0 + vn1) * u_coeff # [F, 3]
            a = area * a_coeff
            splats += [
                self.bary2gs(u0, u1, a, normals, max_scale_ratio=s_ratio),
                self.bary2gs(u1, u2, a, normals, max_scale_ratio=s_ratio),
                self.bary2gs(u2, u0, a, normals, max_scale_ratio=s_ratio),
            ]
            if normal_interpolation:
                splats[-3].replace_(colors=safe_normalize((n0 + n1) / 2))
                splats[-2].replace_(colors=safe_normalize((n1 + n2) / 2))
                splats[-1].replace_(colors=safe_normalize((n2 + n0) / 2))

        return (
            Splats.cat(splats, dim=0),
            torch.cat([offsets] * len(splats), dim=0),
        )

def _tone_mapping_naive(rgba: Float32[Tensor, "*bs 4"], exposure: Float32[Tensor, "1"]) -> Float32[Tensor, "*bs 4"]:
    rgb = rgba[..., :3] * exposure
    return torch.cat((1 - torch.nn.Softplus(beta=100)(1 - rgb), rgba[..., 3:]), dim=-1)

def _tone_mapping_aces(rgba: Float32[Tensor, "*bs 4"], exposure: Float32[Tensor, "1"]) -> Float32[Tensor, "*bs 4"]:
    rgb = rgba[..., :3] * exposure
    return torch.cat(((rgb * (2.51 * rgb + 0.03)) / (rgb * (2.43 * rgb + 0.59) + 0.14), rgba[..., 3:]), dim=-1)

@dataclass
class GaussianField(Module):

    kd_enc: HashEncoding = HashEncoding(
        mlp=MLP(
            layers=[-1, 32, 32, 3],
            activation='sigmoid',
            bias=False,
            initialization='kaiming-uniform',
        ),
        grad_scaling=16.0,
        max_res=4096,
        log2_hashmap_size=18,
    )
    occ_enc: HashEncoding = None
    ks_enc: HashEncoding = HashEncoding(
        mlp=MLP(
            layers=[-1, 32, 2],
            activation='none',
            bias=False,
            initialization='kaiming-uniform',
        ),
        grad_scaling=16.0,
        max_res=4096,
        log2_hashmap_size=18,
    )
    z_enc: HashEncoding = HashEncoding(
        mlp=MLP(
            layers=[-1, 32, 1],
            activation='none',
            bias=False,
            initialization='kaiming-uniform',
        ),
        grad_scaling=16.0,
        max_res=4096,
        log2_hashmap_size=18,
    )

    def get_patches(
        self,
        mesh: TriangleMesh
    ) -> Tuple[Points, Float32[Tensor, "V 1"]]:
        assert mesh.normals is None
        assert mesh.shape == ()
        F = mesh.num_faces
        expanded_inds = mesh.indices.view(F, 3, 1).expand(F, 3, 3)     # [F, 3, 3]
        vertices = mesh.vertices.gather(
            dim=-2,
            index=expanded_inds.flatten(-3, -2)                        # [3F, 3]
        ).view(F, 3, 3)                                                # [F, 3, 3]
        weighted_face_normals = torch.cross(
            vertices[:, 1, :] - vertices[:, 0, :],
            vertices[:, 2, :] - vertices[:, 0, :],
            dim=-1,
        ).unsqueeze(-2).expand(expanded_inds.shape)                    # [F, 3, 3]
        normals = torch.zeros_like(mesh.vertices)                      # [V, 3]
        normals.scatter_add_(
            dim=-2,
            index=expanded_inds.flatten(-3, -2),                       # [3F, 3]
            src=safe_normalize(weighted_face_normals).flatten(-3, -2), # [3F, 3]
        )
        normals = safe_normalize(normals)

        face_vertex_normals = normals.gather(
            dim=-2,
            index=expanded_inds.flatten(-3, -2)                          # [3F, 3]
        ).view(F, 3, 3)                                                  # [F, 3, 3]
        products = (weighted_face_normals * face_vertex_normals).sum(-1) # [F, 3]
        vertex_areas = torch.zeros_like(normals[:, :1])                  # [V, 1]
        vertex_areas.scatter_add_(
            dim=-2,
            index=mesh.indices.view(-1, 1),                              # [3F, 1]
            src=products.view(-1, 1),                                    # [3F, 1]
        )
        return Points(mesh.vertices, normals=normals), vertex_areas.clamp_min(1e-10) / 6

    def get_gaussians_from_vertex(
        self,
        kd_perturb_std: float,
        ks_perturb_std: float,
        scale: float,
        mesh: TriangleMesh,
        initial_guess: Float32[Tensor, "2"],
    ) -> Tuple[Splats, RenderableAttrs]:
        points, areas = self.get_patches(mesh)
        log_sqrt_areas = (areas * (1 / 2.5)).log() * 0.5 # [S, 1]
        encoder_inputs = (points.positions / scale).clamp(-1, 1)                     # [S, 3]

        kd_jitter = None
        ks_jitter = None
        # TODO 实现随机选择10%的点进行jitter，只需要参考下面的代码保存indice即可
        # choice = torch.randperm(xyzt.shape[0], device=xyzt.device)[: int(max(1, xyzt.shape[0] * self.temporal_downsample_ratio))]
        if kd_perturb_std > 0:
            perturb_kd = torch.normal(
                mean=0,
                std=kd_perturb_std,
                size=points.positions.shape,
                device=points.positions.device,
            )                                                                # [S, 3]
            kd_jitter = self.kd_enc((encoder_inputs + perturb_kd).clamp(-1, 1)) # [S, 3]
        if ks_perturb_std > 0:
            perturb_ks = torch.normal(
                mean=0,
                std=ks_perturb_std,
                size=points.positions.shape,
                device=points.positions.device,
            )                                                                # [S, 3]
            ks_jitter = (self.ks_enc((encoder_inputs + perturb_ks).clamp(-1, 1)) + initial_guess).sigmoid() # [S, 2]

        attrs = RenderableAttrs(
            kd=self.kd_enc(encoder_inputs),
            ks=(self.ks_enc(encoder_inputs) + initial_guess).sigmoid(),
            normals=points.normals,
            occ=None if self.occ_enc is None else self.occ_enc(encoder_inputs),
            kd_jitter=kd_jitter,
            ks_jitter=ks_jitter,
        )

        zs = self.z_enc(encoder_inputs.detach()).sigmoid()                                             # [S, 1]

        z_axis = torch.tensor([0, 0, 1]).to(points.normals)
        base_rot = get_rotation_from_relative_vectors(z_axis, points.normals.detach())  # [N, 3, 3]
        scales = torch.cat((
            log_sqrt_areas,
            log_sqrt_areas,
            torch.empty_like(log_sqrt_areas).fill_(1e-10).log(),
        ), dim=-1)                                                                      # [S, 3]

        z_offsets = log_sqrt_areas.detach().exp() * zs # [S, 1]
        positions = points.positions - points.normals * z_offsets # [S, 3]

        V = positions.shape[0]

        return Splats(
            means=positions,
            scales=scales,
            quats=rot2quat(base_rot),
            opacities=torch.logit(0.99 * torch.ones((V, 1), device=self.device)),
            colors=torch.empty_like(points.normals),
            shs=torch.zeros((V, 0, 3), device=self.device)
        ), attrs

    def get_gaussians_from_face(
        self,
        mesh: TriangleMesh,
        kd_perturb_std: float,
        ks_perturb_std: float,
        *,
        scale: float,
        initial_guess: Float32[Tensor, "2"],
        use_checkpoint: bool = False,
    ) -> Tuple[Splats, RenderableAttrs, Float32[Tensor, "V 3"]]:
        splats, offsets = MGAdapter().make(mesh.compute_vertex_normals(fix=True))
        means = (splats.means / scale).clamp(-1, 1)                                # [S, 3]

        if use_checkpoint:
            dummy = torch.ones(1, dtype=torch.float32, requires_grad=True)
            apply_ = lambda enc, inputs, _: enc(inputs) # noqa: E731
            apply: Callable[[HashEncoding, Tensor], Tensor] = ( # noqa: E731
                lambda enc, inputs: checkpoint(apply_, enc, inputs, dummy, use_reentrant=False)
            )
        else:
            apply: Callable[[HashEncoding, Tensor], Tensor] = lambda enc, inputs: enc(inputs) # noqa: E731

        offsets = offsets * apply(self.z_enc, means.detach()).sigmoid()                  # [S, 3]
        shifted_means = splats.means - offsets
        kd_jitter = None
        ks_jitter = None
        if kd_perturb_std > 0:
            perturb_kd = torch.normal(
                mean=0,
                std=kd_perturb_std,
                size=means.shape,
                device=means.device,
            )                                                                # [S, 3]
            kd_jitter = apply(self.kd_enc, (means + perturb_kd).clamp(-1, 1)) # [S, 3]
        if ks_perturb_std > 0:
            perturb_ks = torch.normal(
                mean=0,
                std=ks_perturb_std,
                size=means.shape,
                device=means.device,
            )                                                                # [S, 3]
            ks_jitter = (apply(self.ks_enc, (means + perturb_ks).clamp(-1, 1)) + initial_guess).sigmoid() # [S, 2]

        attrs = RenderableAttrs(
            kd=apply(self.kd_enc, means),
            ks=(apply(self.ks_enc, means) + initial_guess).sigmoid(),
            normals=splats.colors,
            occ=None if self.occ_enc is None else apply(self.occ_enc, means),
            kd_jitter=kd_jitter,
            ks_jitter=ks_jitter,
        )

        return splats.replace(means=shifted_means), attrs, offsets


@dataclass
class GeoSplatter(Module):

    background_color: Literal["random", "black", "white"] = "random"

    use_2dgs: bool = False
    geometry: Literal['dmtet', 'flexicubes', 'gt'] = 'flexicubes'

    sdf_representation: Literal['mlp', 'grid'] = 'grid'

    resolution: int = 32
    light_resolution: int = 512

    field: GaussianField = GaussianField()

    gaussian_limits_hard: int = 1500000
    gaussian_limits_soft: int = 1000000

    scale: float = 1.05
    min_roughness: float = 0.1
    max_metallic: float = 1.0

    learn_per_view_exposure: bool = False
    exp_lighting: bool = False

    gt_envmap: Optional[Path] = None
    relight_envmap: Optional[Path] = None

    occ_type: Literal['texture', 'none'] = 'texture'
    smooth_type: Literal['jitter', 'grad', 'tv'] = 'jitter'
    initial_guess: Literal['specular', 'diffuse', 'hybrid', 'glossy', 'outdoor'] = 'hybrid'

    def setup_train_data_size(self, size: int) -> None:
        if self.learn_per_view_exposure:
            self.exposure_params.materialize((size, ), device=self.device)
        else:
            self.exposure_params.materialize((1, ), device=self.device)
        self.exposure_params.data.zero_()

    def setup_gt_mesh(self, gt_mesh: TriangleMesh) -> None:
        assert self.geometry == 'gt'
        self.gt_mesh = gt_mesh

    @property
    def minimal_memory(self) -> bool:
        return self.last_num_gaussians > self.gaussian_limits_hard and self.training

    @property
    def save_memory(self) -> bool:
        return self.last_num_gaussians > self.gaussian_limits_soft and self.training

    def __setup__(self) -> None:
        self.shadow_scale = 1.0
        self.last_num_gaussians = 0
        self.gt_mesh = None
        self.max_sh_degree = 0
        self.exposure_params =  torch.nn.UninitializedParameter()
        self.optix_ctx = None
        if self.geometry == 'dmtet':
            self.geometric_repr = DMTet.from_predefined(
                resolution=self.resolution,
                scale=self.scale,
            )
            self.deform_params = torch.nn.Parameter(torch.zeros_like(self.geometric_repr.vertices))
            if self.sdf_representation == 'mlp':
                self.sdf_field = SDFField()
                self.sdf_params = torch.nn.Parameter(torch.empty(0))
            elif self.sdf_representation == 'grid':
                self.sdf_params = torch.nn.Parameter(self.geometric_repr.sdf_values.clone())
            else:
                raise ValueError(self.sdf_representation)
            self.weight_params = torch.nn.Parameter(torch.empty(0))
        elif self.geometry == 'flexicubes':
            self.geometric_repr = FlexiCubes.from_resolution(
                self.resolution,
                scale=self.scale,
            )
            self.deform_params = torch.nn.Parameter(torch.zeros_like(self.geometric_repr.vertices))
            if self.sdf_representation == 'mlp':
                self.sdf_field = SDFField()
                self.sdf_params = torch.nn.Parameter(torch.empty(0))
            elif self.sdf_representation == 'grid':
                self.sdf_params = torch.nn.Parameter(self.geometric_repr.sdf_values.clone())
            else:
                raise ValueError(self.sdf_representation)
            self.weight_params = torch.nn.Parameter(torch.zeros(self.geometric_repr.indices.shape[0], 21))
        elif self.geometry == 'gt':
            self.deform_params = torch.nn.Parameter(torch.empty(0))
            self.sdf_params = torch.nn.Parameter(torch.empty(0))
            self.weight_params = torch.nn.Parameter(torch.empty(0))
        else:
            raise ValueError(self.geometry)
        self.envmap = None
        self.sdf_weight = 0.0
        self.occ_weight = 0.0
        self.light_weight = 0.0
        self.kd_factor = 1.0
        self.kd_grad_weight = 0.0
        self.kd_regualr_perturb_std = 0.0
        self.ks_grad_weight = 0.0
        self.ks_regualr_perturb_std = 0.0
        self.normal_grad_weight = 0.0
        self.sample_method: Literal['face', 'vertex'] = 'face'
        self.render_method: Literal['split-sum', 'mc'] = 'split-sum'
        if self.initial_guess == 'outdoor':
            self.initial_guess_bias = torch.nn.Parameter(torch.tensor([0, 0]).float(), requires_grad=False)
        elif self.initial_guess == 'diffuse':
            self.initial_guess_bias = torch.nn.Parameter(torch.tensor([0, -3]).float(), requires_grad=False)
        elif self.initial_guess == 'hybrid':
            self.initial_guess_bias = torch.nn.Parameter(torch.tensor([-3, -3]).float(), requires_grad=False)
        elif self.initial_guess == 'specular':
            self.initial_guess_bias = torch.nn.Parameter(torch.tensor([-3, 0]).float(), requires_grad=False)
        elif self.initial_guess == 'glossy':
            self.initial_guess_bias = torch.nn.Parameter(torch.tensor([-3, 0]).float(), requires_grad=False)
        else:
            raise ValueError(self.initial_guess)
        if self.gt_envmap is None:
            self.cubemap = torch.nn.Parameter(
                torch.empty(
                    6,
                    self.light_resolution,
                    self.light_resolution,
                    3,
                ).fill_(0.5)
            )
            self.latlng = torch.nn.Parameter(torch.empty(256, 512, 3))
        else:
            self.cubemap = torch.nn.Parameter(torch.empty(0))
            self.latlng = torch.nn.Parameter(torch.empty(0))

    def get_geometry(self) -> Tuple[TriangleMesh, Tensor]:
        if self.geometry == 'gt':
            return self.gt_mesh, self.gt_mesh.vertices.new_zeros(1)
        # TODO: fix the device
        with torch.no_grad():
            if self.geometric_repr.device != self.device:
                self.geometric_repr.swap_(self.geometric_repr.to(self.device))
        if self.geometry == 'dmtet':
            vertices = self.geometric_repr.vertices + self.deform_params.tanh() * (self.scale / self.resolution)
            dmtet = self.geometric_repr.replace(
                vertices=vertices,
                sdf_values=self.sdf_field(vertices) if self.sdf_representation == 'mlp' else self.sdf_params,
            )
            mesh = dmtet.marching_tets()
            return mesh, dmtet.compute_entropy() * self.sdf_weight
        if self.geometry == 'flexicubes':
            vertices = self.geometric_repr.vertices + self.deform_params.tanh() * (0.5 * self.scale / self.resolution)
            flexicubes = self.geometric_repr.replace(
                vertices=vertices,
                sdf_values=self.sdf_field(self.geometric_repr.vertices) if self.sdf_representation == 'mlp' else self.sdf_params,
                alpha=self.weight_params[:, :8],
                beta=self.weight_params[:, 8:20],
                gamma=self.weight_params[:, 20:],
            )
            mesh, L_dev = flexicubes.dual_marching_cubes()
            reg_loss = torch.add(
                L_dev.mean() * 0.5 + self.weight_params[:, :20].abs().mean() * 0.1,
                flexicubes.compute_entropy() * self.sdf_weight
            )
            return mesh, reg_loss
        raise ValueError(self.geometry)

    def get_background_color(self) -> Float32[Tensor, "3 "]:
        if self.background_color == 'black':
            return torch.zeros(3)
        if self.background_color == 'white':
            return torch.ones(3)
        if self.training:
            return torch.rand(3)
        return torch.tensor([0.1490, 0.1647, 0.2157])

    def get_envmap(self) -> Union[Tuple[TextureSplitSum, Tensor], Tuple[TextureLatLng, Tensor]]:
        if self.gt_envmap is not None:
            raise NotImplementedError
        if self.render_method == 'split-sum':
            cubemap = self.cubemap
            if self.exp_lighting:
                cubemap = cubemap.exp()
            white = cubemap.mean(-1, keepdim=True) # [6, R, R, 1]
            white_balance_reg = (cubemap - white).abs().mean()
            texture = TextureCubeMap(data=cubemap, transform=None).as_splitsum()
            return texture, white_balance_reg
        return TextureLatLng(data=self.latlng), self.latlng.new_zeros(1)

    def set_relight_envmap(self, envmap_path: Path) -> None:
        self.gt_envmap = envmap_path
        self.envmap = TextureCubeMap.from_image_file(envmap_path, device=self.device).as_splitsum()

    def get_gsplat(
        self,
        sampling: Literal['face', 'vertex'],
    ) -> Tuple[TriangleMesh, GSplatter, RenderableAttrs, Tensor, Tensor]:
        mesh, reg = self.get_geometry()
        gsplat = GSplatter(
            num_random=4,
            sh_degree=0,
            background_color=self.background_color,
            rasterize_mode='antialiased' if not self.use_2dgs else '2dgs',
        ).to(self.device)
        gsplat.__setup__()
        gsplat.set_max_sh_degree(self.max_sh_degree)
        kd_regualr_perturb_std = self.kd_regualr_perturb_std if self.smooth_type == 'jitter' else 0
        ks_regualr_perturb_std = self.ks_regualr_perturb_std if self.smooth_type == 'jitter' else 0
        if sampling == 'face':
            self.last_num_gaussians = mesh.num_faces * 6
            splats, attrs, offsets = self.field.get_gaussians_from_face(
                mesh=mesh,
                scale=self.scale,
                kd_perturb_std=kd_regualr_perturb_std,
                ks_perturb_std=ks_regualr_perturb_std,
                use_checkpoint=self.minimal_memory,
                initial_guess=self.initial_guess_bias,
            )
        elif sampling == 'vertex':
            self.last_num_gaussians = mesh.num_vertices
            splats, attrs = self.field.get_gaussians_from_vertex(
                mesh=mesh,
                scale=self.scale,
                kd_perturb_std=kd_regualr_perturb_std,
                ks_perturb_std=ks_regualr_perturb_std,
                initial_guess=self.initial_guess_bias,
            )
            offsets = None
        else:
            raise ValueError(sampling)
        gsplat.gaussians = splats
        if kd_regualr_perturb_std > 0 and self.kd_grad_weight > 0:
            # TODO 在计算perturb loss的时候，使用随机的10%，使用之前保存的indices即可
            reg = reg + self.kd_grad_weight * (attrs.kd_jitter - attrs.kd).abs().mean()
        if ks_regualr_perturb_std > 0 and self.ks_grad_weight > 0:
            reg = reg + self.ks_grad_weight * (attrs.ks_jitter - attrs.ks).abs().mean()
        if self.occ_weight > 0:
            reg = reg + self.occ_weight * attrs.occ.abs().mean()
        return mesh, gsplat, attrs.replace(kd_jitter=None, ks_jitter=None), reg, offsets

    @torch.no_grad()
    def export_point_cloud(self, path: pathlib.Path) -> None:
        _, gsplat, attrs, _, _ = self.get_gsplat('face')
        gsplat.gaussians.replace_(colors=attrs.kd.clamp(0, 1))
        gsplat.export_point_cloud(path)

    def get_splats(self) -> Splats:
        _, gsplat, attrs, _, _ = self.get_gsplat('face')
        return gsplat.gaussians.replace(colors=attrs.kd.clamp(0, 1))

    @torch.no_grad()
    def export_mesh_with_splats(self, path: pathlib.Path) -> None:
        assert self.occ_type != 'none'

        mesh, _ = self.get_geometry()
        _, attrs, offsets = self.field.get_gaussians_from_face(
            mesh=mesh,
            scale=self.scale,
            kd_perturb_std=0,
            ks_perturb_std=0,
        )
        attributes = {
            'vertices': mesh.vertices, # [V, 3]
            'indices': mesh.indices, # [F, 3]
            'kdks': torch.cat((attrs.kd, torch.zeros_like(attrs.ks[..., :1]), attrs.ks), dim=-1), # [6F, 6]
            'offsets': offsets, # [6F, 3]
            'min_roughness': self.min_roughness, # [1]
            'max_metallic': self.max_metallic, # [1]
            'exposure': self.exposure_params.mean() if self.learn_per_view_exposure else self.exposure_params, # [1]
        }
        if self.gt_envmap is None:
            attributes['cubemap'] = self.cubemap # [6, R, R, 3]
        torch.save(attributes, path)

    @torch.no_grad()
    def export_splats(self, path: Path) -> None:
        attributes = {
            'geom_scale': self.scale,
            'resolution': self.resolution,
            'min_roughness': self.min_roughness, # [1]
            'max_metallic': self.max_metallic, # [1]
            'exposure': self.exposure_params, # [1]
            'cubemap': self.cubemap,
            'deforms': self.deform_params,
            'weights': self.weight_params,
            'sdfs': self.sdf_params,
            'ks_enc': self.field.ks_enc.state_dict(),
            'initial_guess': self.initial_guess_bias,
        }
        torch.save(attributes, path)

    def render_report(self, inputs: Cameras, *, indices: Optional[Tensor], gt_outputs: RGBAImages) -> Tuple[
        PBRAImages,
        RGBImages,
        VectorImages,
        int,
        Tensor,
    ]:
        images = []
        mesh, gsplat, attrs, regularization, offsets = self.get_gsplat(self.sample_method)
        envmap, light_reg = self.get_envmap()
        batch_size = len(inputs)
        if self.learn_per_view_exposure and indices is not None:
            exposures = self.exposure_params[indices].exp()
        else:
            exposures = self.exposure_params.mean().exp().expand(batch_size)
        if self.save_memory:
            torch.cuda.empty_cache()

        if self.render_method == 'mc':
            with torch.no_grad():
                if self.optix_ctx is None:
                    self.optix_ctx = OptiXContext()
                optix_build_bvh(self.optix_ctx, mesh.vertices.contiguous(), mesh.indices.int(), rebuild=1)
                mc_positions = gsplat.gaussians.means + offsets
                gt_pbra = gt_outputs.srgb2rgb()
            envmap.compute_pdf_()
        for i in range(batch_size):
            if self.minimal_memory:
                torch.cuda.empty_cache()
            if self.render_method == 'split-sum':
                images.append(attrs.splat(
                    gsplat,
                    inputs[i:i+1],
                    envmap=envmap,
                    exposure=exposures[i],
                    occ_type=self.occ_type,
                    min_roughness=self.min_roughness,
                    max_metallic=self.max_metallic,
                    factor=self.kd_factor,
                ))
            else:
                img, reg = attrs.splat_mc(
                    gsplat,
                    inputs[i:i+1],
                    envmap=envmap,
                    gt_image=gt_pbra[i].item(),
                    context=self.optix_ctx,
                    positions=mc_positions,
                    exposure=exposures[i],
                    num_samples_per_ray=8,
                    denoise=True,
                    residual=True,
                    min_roughness=self.min_roughness,
                    max_metallic=self.max_metallic,
                    shadow_scale=self.shadow_scale,
                )
                images.append(img)
                regularization = regularization + reg / batch_size

        if self.smooth_type == 'grad':
            for camera, gt_rgb in zip(inputs.view(-1, 1), gt_outputs.blend(self.get_background_color())):
                if self.kd_grad_weight > 0:
                    gsplat.gaussians.replace_(colors=attrs.kd)
                    first_order_edge_aware_loss = torch.mul(
                        spatial_gradient(gsplat.render_rgb(camera).item()[None].permute(0, 3, 1, 2), order=1)[0].abs(),
                        (-spatial_gradient(gt_rgb[None].permute(0, 3, 1, 2), order=1)[0].abs()).exp()
                    ).sum(1).mean()
                    regularization = regularization + first_order_edge_aware_loss * self.kd_grad_weight / batch_size
                if self.ks_grad_weight > 0:
                    gsplat.gaussians.replace_(colors=torch.cat((torch.zeros_like(attrs.ks[..., :1]), attrs.ks), dim=-1))
                    first_order_edge_aware_loss = torch.mul(
                        spatial_gradient(gsplat.render_rgb(camera).item()[None].permute(0, 3, 1, 2), order=1)[0].abs(),
                        (-spatial_gradient(gt_rgb[None].permute(0, 3, 1, 2), order=1)[0].abs()).exp()
                    ).sum(1).mean()
                    regularization = regularization + first_order_edge_aware_loss * self.ks_grad_weight / batch_size
        if self.normal_grad_weight > 0:
            for camera, gt_rgb in zip(inputs.view(-1, 1), gt_outputs.blend(self.get_background_color())):
                gsplat.gaussians.replace_(colors=attrs.normals * 0.5 + 0.5)
                first_order_edge_aware_loss = torch.mul(
                    spatial_gradient(gsplat.render_rgb(camera).item()[None].permute(0, 3, 1, 2), order=1)[0].abs(),
                    (-spatial_gradient(gt_rgb[None].permute(0, 3, 1, 2), order=1)[0].abs()).exp()
                ).sum(1).mean()
                regularization = regularization + first_order_edge_aware_loss * self.normal_grad_weight / batch_size
        if self.smooth_type == 'tv':
            for camera in inputs.view(-1, 1):
                if self.kd_grad_weight > 0:
                    gsplat.gaussians.replace_(colors=attrs.kd)
                    rendered = gsplat.render_rgb(camera).item()
                    tv_loss = torch.add(
                        (rendered[1:, :] - rendered[:-1, :]).square().mean(),
                        (rendered[:, 1:] - rendered[:, :-1]).square().mean(),
                    )
                    regularization = regularization + tv_loss * self.kd_grad_weight / batch_size
                if self.ks_grad_weight > 0:
                    gsplat.gaussians.replace_(colors=torch.cat((torch.zeros_like(attrs.ks[..., :1]), attrs.ks), dim=-1))
                    rendered = gsplat.render_rgb(camera).item()
                    tv_loss = torch.add(
                        (rendered[1:, :] - rendered[:-1, :]).square().mean(),
                        (rendered[:, 1:] - rendered[:, :-1]).square().mean(),
                    )
                    regularization = regularization + tv_loss * self.ks_grad_weight / batch_size
        if self.minimal_memory:
            return (
                PBRAImages(images),
                None,
                None,
                gsplat.gaussians.shape[0],
                regularization + light_reg * self.light_weight,
            )
        with torch.no_grad():
            camera = inputs.view(-1, 1)[0]
            gsplat.gaussians.replace_(colors=attrs.kd)
            kd = gsplat.render_rgb(camera).item()
            kd = torch.where(
                kd <= 0.0031308,
                kd * 12.92,
                torch.clamp(kd, min=0.0031308).pow(1.0 / 2.4) * 1.055 - 0.055,
            )
            gsplat.gaussians.replace_(colors=torch.cat((
                torch.zeros_like(attrs.ks[..., :1]),
                attrs.ks[..., :1] * (1 - self.min_roughness) + self.min_roughness,
                attrs.ks[..., 1:] * self.max_metallic,
            ), dim=-1))
            ks = gsplat.render_rgb(camera).item()
            gsplat.gaussians.replace_(colors=attrs.normals * 0.5 + 0.5)
            normal_map = gsplat.render_rgb(camera).item()
            if self.gt_envmap is None:
                cubemap = (
                    TextureCubeMap(data=self.cubemap)
                    if self.render_method == 'split-sum'
                    else TextureLatLng(data=self.latlng).as_cubemap(resolution=self.light_resolution)
                )
                light = cubemap.visualize(
                    width=kd.shape[1] * 2,
                    height=kd.shape[0],
                ).item()
            else:
                light = torch.zeros_like(normal_map)
            mesh_normal = mesh.render(camera, shader=NormalShader())
            pretty_mesh = mesh.render(
                camera,
                shader=PrettyShader(occlusion_type='none'),
            ).rgb2srgb().blend(self.get_background_color()).item()
        return (
            PBRAImages(images),
            RGBImages([kd, ks, normal_map, light, pretty_mesh]),
            mesh_normal,
            gsplat.gaussians.shape[0],
            regularization + light_reg * self.light_weight
        )

    @chains
    def as_module(self, *, field_name: Literal['deforms', 'sdfs', 'weights', 'light', 'exposure']) -> Any:

        def parameters(self) -> Any:
            params = {
                'deforms': self.deform_params,
                'sdfs': self.sdf_params,
                'weights': self.weight_params,
                'light': self.cubemap if self.render_method == 'split-sum' else self.latlng,
                'exposure': self.exposure_params,
            }[field_name]
            return [params]

        return parameters
