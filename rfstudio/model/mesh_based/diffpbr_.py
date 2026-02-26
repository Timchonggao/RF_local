from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.graphics import (
    Cameras,
    RGBAImages,
    TextureSG,
    TriangleMesh,
)
from rfstudio.graphics.math import safe_normalize
from rfstudio.model.density_field.components.encoding import HashEncoding
from rfstudio.nn import MLP, Module
from rfstudio.utils.decorator import chains
from rfstudio.utils.lazy_module import dr


def tone_mapping(colors: Tensor, gamma: Tensor) -> Tensor:
    return gamma.sigmoid() ** -0.2 * ((2.51 * colors + 0.03) * colors / ((2.43 * colors + 0.59) * colors + 0.14))

@dataclass
class DiffPBR(Module):

    background_color: Literal["random", "black", "white"] = "white"

    min_roughness: float = 0.1

    antialias: bool = True

    kd_texture: HashEncoding = HashEncoding(
        mlp=MLP(
            layers=[-1, 32, 32, 3],
            activation='sigmoid',
            bias=False,
            initialization='kaiming-uniform',
        ),
        max_res=4096,
        grad_scaling=16.0,
    )

    ks_texture: HashEncoding = HashEncoding(
        mlp=MLP(
            layers=[-1, 32, 32, 2],
            activation='none',
            bias=False,
            initialization='kaiming-uniform',
        ),
        max_res=4096,
        grad_scaling=16.0,
    )

    gt_mesh: Optional[Path] = None
    z_up: bool = False

    kd_grad_weight: float = 0.1
    ks_grad_weight: float = 0.05
    light_weight: float = 0.0

    def __setup__(self) -> None:
        self.gt_geometry = None
        self.sdf_weight = 0.0
        self.occ_weight = 0.0
        self.scale = None
        self.envmap = None
        self.optix_ctx = None
        random = TextureSG.from_random(24)
        self.cubemap = torch.nn.Parameter(torch.cat((
            random.axis,
            random.sharpness,
            random.amplitude,
        ), dim=-1))
        self.gamma = torch.nn.Parameter(torch.zeros(1))

    def get_envmap(self) -> TextureSG:
        return TextureSG(
            axis=self.cubemap[..., :3],
            sharpness=self.cubemap[..., 3:4],
            amplitude=self.cubemap[..., 4:],
        )

    def get_light_regularization(self) -> Tensor:
        return torch.zeros(1, device=self.device)

    def set_gt_geometry(self, mesh: TriangleMesh) -> None:
        self.gt_geometry = mesh.to(self.device).compute_vertex_normals_(fix=True)
        self.scale = self.gt_geometry.vertices.abs().max() + 1e-4

    def render_report(self, inputs: Cameras) -> Tuple[RGBAImages, RGBAImages, TriangleMesh, Tensor]:
        mesh = self.gt_geometry
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
        kd_grad_reg = 0
        ks_grad_reg = 0
        for camera in inputs:
            camera_pos = camera.c2w[:, 3]
            resolution = [camera.height.item(), camera.width.item()]
            mvp = camera.projection_matrix @ camera.view_matrix                    # [4, 4]
            projected = (mvp @ vertices).view(1, -1, 4)                            # [1, V, 4]
            indices = mesh.indices.int()
            with dr.DepthPeeler(ctx, projected, indices, resolution=resolution) as peeler:
                rast, _ = peeler.rasterize_next_layer()
            frag_pos, _ = dr.interpolate(mesh.vertices[None], rast, indices) # [1, H, W, 3]
            kd = self.kd_texture((frag_pos / self.scale).clamp(-1, 1)) # [1, H, W, 3]
            ks = (self.ks_texture((frag_pos / self.scale).clamp(-1, 1)) - 3).sigmoid() # [1, H, W, 2]
            d_frag_pos = torch.normal(mean=0, std=0.01, size=frag_pos.shape, device=frag_pos.device) # [1, H, W, 3]
            kd_jitter = self.kd_texture(((frag_pos + d_frag_pos) / self.scale).clamp(-1, 1)) # [1, H, W, 3]
            ks_jitter = self.ks_texture(((frag_pos + d_frag_pos) / self.scale).clamp(-1, 1)) # [1, H, W, 2]
            frag_n, _ = dr.interpolate(mesh.normals[None], rast, indices) # [1, H, W, 3]
            frag_n = safe_normalize(frag_n) # [1, H, W, 3]
            roughness = ks[..., 0:1] * (1 - self.min_roughness) + self.min_roughness # [1, H, W, 1]
            metallic  = ks[..., 1:2] # [1, H, W, 1]
            ks = torch.cat((torch.zeros_like(roughness), roughness, metallic), dim=-1)

            diff, spec = envmap.integral(
                frag_n,
                frag_pos - camera_pos,
                albedo=kd,
                roughness=roughness,
                metallic=metallic,
            )
            colors = diff + spec
            alphas = (rast[..., -1:] > 0).float() # [1, H, W, 1]
            rgba = torch.cat((tone_mapping(colors, self.gamma), alphas), dim=-1)
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
                    kdks_vis.append(torch.cat((ks, alphas), dim=-1).squeeze(0))
                    kdks_vis.append(torch.cat((frag_n * 0.5 + 0.5, alphas), dim=-1).squeeze(0))
            if self.kd_grad_weight > 0:
                kd_grad = (kd_jitter - kd).abs().mean() # [1]
                kd_grad_reg = kd_grad_reg + kd_grad
            if self.ks_grad_weight > 0:
                ks_grad = (ks_jitter - ks[..., 1:]).abs().mean() # [1]
                ks_grad_reg = ks_grad_reg + ks_grad
        if self.light_weight > 0:
            light_reg = self.light_weight * self.get_light_regularization()
        else:
            light_reg = 0.0
        kd_grad_reg = (self.kd_grad_weight / inputs.shape[0]) * kd_grad_reg
        ks_grad_reg = (self.ks_grad_weight / inputs.shape[0]) * ks_grad_reg
        regularization = light_reg + kd_grad_reg + ks_grad_reg
        return (
            RGBAImages(images),
            RGBAImages(kdks_vis),
            mesh,
            regularization,
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
    def as_module(self, *, field_name: Literal['light', 'gamma']) -> Any:

        def parameters(self) -> Any:
            params = {
                'light': self.cubemap,
                'gamma': self.gamma,
            }[field_name]
            return [params]

        return parameters
