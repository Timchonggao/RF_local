from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.graphics import (
    Cameras,
    DMTet,
    FlexiCubes,
    PBRAImages,
    RGBAImages,
    RGBImages,
    TextureCubeMap,
    TextureSplitSum,
    TriangleMesh,
)
from rfstudio.graphics.math import safe_normalize
from rfstudio.graphics.shaders import _get_fg_lut
from rfstudio.model.density_field.components.encoding import HashEncoding
from rfstudio.nn import MLP, Module
from rfstudio.utils.decorator import chains
from rfstudio.utils.lazy_module import dr


@dataclass
class NVDiffRec(Module):

    geometry: Literal['dmtet', 'flexicubes', 'gt'] = 'dmtet'

    background_color: Literal["random", "black", "white"] = "random"

    resolution: int = 64

    min_roughness: float = 0.25

    scale: float = 1.05

    antialias: bool = True

    mlp_texture: HashEncoding = HashEncoding(
        mlp=MLP(
            layers=[-1, 32, 32, 6],
            activation='sigmoid',
            bias=False,
            initialization='kaiming-uniform',
        ),
        max_res=4096,
        grad_scaling=16.0,
    )

    gt_envmap: Optional[Path] = None

    def __setup__(self) -> None:
        if self.geometry == 'dmtet':
            self.geometric_repr = DMTet.from_predefined(
                resolution=self.resolution,
                scale=self.scale,
            )
            self.deform_params = torch.nn.Parameter(torch.zeros_like(self.geometric_repr.vertices))
            self.sdf_params = torch.nn.Parameter(self.geometric_repr.sdf_values.clone())
            self.weight_params = torch.nn.Parameter(torch.empty(0))
        elif self.geometry == 'flexicubes':
            self.geometric_repr = FlexiCubes.from_resolution(
                self.resolution,
                scale=self.scale,
            )
            self.deform_params = torch.nn.Parameter(torch.zeros_like(self.geometric_repr.vertices))
            self.sdf_params = torch.nn.Parameter(self.geometric_repr.sdf_values.clone())
            self.weight_params = torch.nn.Parameter(torch.ones(self.geometric_repr.indices.shape[0], 21))
        elif self.geometry == 'gt':
            self.deform_params = torch.nn.Parameter(torch.empty(0))
            self.sdf_params = torch.nn.Parameter(torch.empty(0))
            self.weight_params = torch.nn.Parameter(torch.empty(0))
        else:
            raise ValueError(self.geometry)
        self.gt_geometry = None
        self.sdf_weight = 0.0
        self.occ_weight = 0.0
        self.kd_grad_weight = 0.0
        self.light_weight = 0.0
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

    def get_light_regularization(self) -> Tensor:
        if self.gt_envmap is None:
            white = self.cubemap.mean(-1, keepdim=True) # [6, R, R, 1]
            return (self.cubemap - white).abs().mean()
        return torch.zeros(1, device=self.device)

    def set_gt_geometry(self, mesh: TriangleMesh) -> None:
        assert self.geometry == 'gt'
        self.gt_geometry = mesh.to(self.device)

    def get_geometry(self) -> Tuple[TriangleMesh, Tensor]:
        if self.geometry == 'gt':
            assert self.gt_geometry is not None
            return self.gt_geometry, torch.zeros(1).to(self.device)
        # TODO: fix the device
        with torch.no_grad():
            if self.geometric_repr.device != self.device:
                self.geometric_repr.swap_(self.geometric_repr.to(self.device))
        if self.geometry == 'dmtet':
            vertices = self.geometric_repr.vertices + self.deform_params.tanh() * (self.scale / self.resolution)
            dmtet = self.geometric_repr.replace(vertices=vertices, sdf_values=self.sdf_params)
            if self.sdf_weight > 0:
                geom_reg = dmtet.compute_entropy() * self.sdf_weight
            else:
                geom_reg = 0.0
            return dmtet.marching_tets(), geom_reg
        if self.geometry == 'flexicubes':
            vertices = self.geometric_repr.vertices + self.deform_params.tanh() * (0.5 * self.scale / self.resolution)
            flexicubes = self.geometric_repr.replace(
                vertices=vertices,
                sdf_values=self.sdf_params,
                alpha=self.weight_params[:, :8],
                beta=self.weight_params[:, 8:20],
                gamma=self.weight_params[:, 20:],
            )
            mesh, L_dev = flexicubes.dual_marching_cubes()
            if self.sdf_weight > 0:
                geom_reg = flexicubes.compute_entropy() * self.sdf_weight
            else:
                geom_reg = 0.0
            return mesh, geom_reg + L_dev.mean() * 0.25
        raise ValueError(self.geometry)

    def sample_jitter_regularization(
            self,
            frag_pos: Tensor,                  # [1, H, W, 3]
            kdks: Tensor,                      # [1, H, W, 6]
            scale: float,
            mlp_texture: Module,
            ratio: float = 0.1,
            strategy: Literal["uniform", "feature_aware", "gumbel"] = "uniform",
        ) -> Tuple[Tensor, Tensor]:
        """
        仅对选取的像素点进行 jitter 并计算 kd/ks 的正则项。

        Args:
            frag_pos: 像素对应的世界空间位置
            kdks: 来自 MLP 的反射率与材质输出
            scale: 空间归一化因子
            mlp_texture: 网络
            ratio: 采样比例
            strategy: 采样策略 - 'uniform', 'feature_aware', 'gumbel'

        Returns:
            kd_grad, ks_grad: 平均正则项
        """
        assert frag_pos.shape[0] == 1 and kdks.shape[0] == 1, "Only supports batch size 1"
        H, W = frag_pos.shape[1], frag_pos.shape[2]
        N = H * W
        N_sample = max(1, int(N * ratio))

        frag_pos_flat = frag_pos.view(-1, 3)  # [H*W, 3]
        kdks_flat = kdks.view(-1, 6)          # [H*W, 6]

        # ----------------------------
        # Sampling choice indices
        # ----------------------------
        if strategy == "uniform":
            choice = torch.randperm(N, device=frag_pos.device)[:N_sample]

        elif strategy == "feature_aware":
            # 基于 kd（反射率）的变化程度采样
            kd = kdks[0, ..., :3]  # [H, W, 3]
            dx = kd[1:, :, :] - kd[:-1, :, :]
            dy = kd[:, 1:, :] - kd[:, :-1, :]
            grad_mag = (dx[:-1, :, :]**2 + dy[:, :-1, :]**2).sum(-1).sqrt()  # [H-1, W-1]
            grad_mag = torch.nn.functional.pad(grad_mag, (0, 1, 0, 1))       # [H, W]
            flat_grad = grad_mag.view(-1)
            _, choice = torch.topk(flat_grad, k=N_sample)

        elif strategy == "gumbel":
            # importance: kd的颜色强度
            kd_intensity = kdks_flat[:, :3].norm(dim=-1)  # [H*W]
            gumbel_noise = -torch.empty_like(kd_intensity).exponential_().log()
            scores = kd_intensity + gumbel_noise
            _, choice = torch.topk(scores, k=N_sample)

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        # ----------------------------
        # Jittered query
        # ----------------------------
        frag_pos_sampled = frag_pos_flat[choice]
        kdks_sampled = kdks_flat[choice]
        d_frag_pos = torch.normal(mean=0, std=0.01, size=frag_pos_sampled.shape, device=frag_pos.device)
        frag_pos_jittered = frag_pos_sampled + d_frag_pos
        kdks_jittered = mlp_texture((frag_pos_jittered / scale).clamp(-1, 1))

        # ----------------------------
        # Compute Regularization
        # ----------------------------
        kd_grad = (kdks_jittered[..., :3] - kdks_sampled[..., :3]).abs().mean()
        ks_grad = (kdks_jittered[..., 3:-1] - kdks_sampled[..., 3:-1]).abs().mean()

        return kd_grad, ks_grad

    def render_report(self, inputs: Cameras) -> Tuple[PBRAImages, RGBAImages, TriangleMesh, Tensor]:
        mesh, geom_reg = self.get_geometry()
        mesh = mesh.compute_vertex_normals(fix=True)
        assert inputs.is_cuda
        inputs = inputs.view(-1)

        ctx = dr.RasterizeCudaContext(inputs.device)

        vertices = torch.cat((
            mesh.vertices,
            torch.ones_like(mesh.vertices[..., :1]),
        ), dim=-1).view(-1, 4, 1)                          # [V, 4, 1]

        envmap = self.get_envmap()

        images = []
        kdks_vis = []
        occ_reg = 0
        kd_grad_reg = 0
        for camera in inputs:
            camera_pos = camera.c2w[:, 3]
            resolution = [camera.height.item(), camera.width.item()]
            mvp = camera.projection_matrix @ camera.view_matrix                    # [4, 4]
            projected = (mvp @ vertices).view(1, -1, 4)                            # [1, V, 4]
            indices = mesh.indices.int()
            with dr.DepthPeeler(ctx, projected, indices, resolution=resolution) as peeler:
                rast, _ = peeler.rasterize_next_layer()
            frag_pos, _ = dr.interpolate(mesh.vertices[None], rast, indices) # [1, H, W, 3]
            kdks = self.mlp_texture((frag_pos / self.scale).clamp(-1, 1)) # [1, H, W, 6]
            frag_n, _ = dr.interpolate(mesh.normals[None], rast, indices) # [1, H, W, 3]
            frag_n = safe_normalize(frag_n) # [1, H, W, 3]
            kd = kdks[..., :3] # [1, H, W, 3]
            # breakpoint()
            roughness = kdks[..., 4:5] * (1 - self.min_roughness) + self.min_roughness # [1, H, W, 1]
            metallic  = kdks[..., 5:6] # [1, H, W, 1]
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
            colors = (l_diff * diffuse + l_spec * reflectance) * (1.0 - kdks[..., 3:4]) # [1, H, W, 3]
            alphas = (rast[..., -1:] > 0).float() # [1, H, W, 1]
            rgba = torch.cat((colors, alphas), dim=-1)
            if self.antialias:
                rgba = dr.antialias(rgba, rast, projected, indices)
            images.append(rgba.squeeze(0))
            if not kdks_vis:
                with torch.no_grad():
                    kd = torch.where(
                        kd <= 0.0031308,
                        kd * 12.92,
                        torch.clamp(kd, min=0.0031308).pow(1.0 / 2.4) * 1.055 - 0.055,
                    )
                    kdks_vis.append(torch.cat((kd, alphas), dim=-1).squeeze(0))
                    kdks_vis.append(torch.cat((kdks[..., 3:4], roughness, metallic, alphas), dim=-1).squeeze(0))
                    kdks_vis.append(torch.cat((frag_n * 0.5 + 0.5, alphas), dim=-1).squeeze(0))
            if self.occ_weight > 0:
                occ_reg = occ_reg + kdks[..., 3:4].mean()
            if self.kd_grad_weight > 0:
                # d_frag_pos = torch.normal(mean=0, std=0.01, size=frag_pos.shape, device=frag_pos.device) # [1, H, W, 3]
                # kdks_jitter = self.mlp_texture(((frag_pos + d_frag_pos) / self.scale).clamp(-1, 1)) # [1, H, W, 6]
                # kd_grad = (kdks_jitter[..., :3] - kdks[..., :3]).abs().mean() # [1]
                kd_grad, _ = self.sample_jitter_regularization(frag_pos, kdks, self.scale, self.mlp_texture)
                kd_grad_reg = kd_grad_reg + kd_grad
        if self.light_weight > 0:
            light_reg = self.light_weight * self.get_light_regularization()
        else:
            light_reg = 0.0
        occ_reg = (self.occ_weight / inputs.shape[0]) * occ_reg
        kd_grad_reg = (self.kd_grad_weight / inputs.shape[0]) * kd_grad_reg
        regularization = geom_reg + light_reg + occ_reg + kd_grad_reg
        return (
            PBRAImages(images),
            RGBAImages(kdks_vis),
            mesh,
            regularization,
        )

    def render_rgb(self, inputs: Cameras) -> RGBImages:
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
    def as_module(self, *, field_name: Literal['deforms', 'sdfs', 'weights', 'light']) -> Any:

        def parameters(self) -> Any:
            params = {
                'deforms': self.deform_params,
                'sdfs': self.sdf_params,
                'weights': self.weight_params,
                'light': self.cubemap,
            }[field_name]
            return [params]

        return parameters
