from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.graphics import (
    Cameras,
    RGBAImages,
    TextureLatLng,
    TriangleMesh,
)
from rfstudio.graphics._mesh._optix import OptiXContext, bilateral_denoiser, optix_build_bvh, optix_env_shade
from rfstudio.model.density_field.components.encoding import HashEncoding
from rfstudio.nn import MLP, Module
from rfstudio.utils.decorator import chains
from rfstudio.utils.lazy_module import dr


def tone_mapping(colors: Tensor, gamma: Tensor) -> Tensor:
    return gamma.sigmoid() ** -0.2 * ((2.51 * colors + 0.03) * colors / ((2.43 * colors + 0.59) * colors + 0.14))

@dataclass
class DLMesh(Module):

    background_color: Literal["random", "black", "white"] = "white"

    min_roughness: float = 0.1

    antialias: bool = True
    denoise: bool = True
    num_samples_per_ray: int = 8

    mlp_texture: HashEncoding = HashEncoding(
        mlp=MLP(
            layers=[32, 32, 32, 5],
            activation='sigmoid',
            bias=False,
            initialization='kaiming-uniform',
        ),
        max_res=4096,
        grad_scaling=16.0,
    )

    gt_mesh: Optional[Path] = None
    gt_envmap: Optional[Path] = None
    gt_albedo: Optional[Path] = None
    z_up: bool = False

    kd_grad_weight: float = 0.2
    ks_grad_weight: float = 0.1
    light_weight: float = 0.0

    def __setup__(self) -> None:
        self.gt_geometry = None
        self.sdf_weight = 0.0
        self.shadow_scale = 1.0
        self.envmap = None
        self.albedo = None
        self.optix_ctx = None
        if self.gt_envmap is None:
            self.cubemap = torch.nn.Parameter(torch.empty(1024, 512, 3).fill_(0.5))
        else:
            self.cubemap = torch.nn.Parameter(torch.empty(0))
        if self.gt_albedo is not None:
            self.albedo = torch.nn.Parameter(torch.load(self.gt_albedo, map_location='cpu'), requires_grad=False)
        self.gamma = torch.nn.Parameter(torch.zeros(1))

    def get_envmap(self) -> TextureLatLng:
        if self.gt_envmap is None:
            assert not self.z_up
            return TextureLatLng(data=self.cubemap, transform=None)
        if self.envmap is None:
            self.envmap = TextureLatLng.from_image_file(self.gt_envmap, device=self.device)
            if self.z_up:
                self.envmap.z_up_to_y_up_()
                self.envmap = self.envmap.as_cubemap(resolution=512).as_latlng(apply_transform=True)
        return self.envmap

    def get_light_regularization(self) -> Tensor:
        if self.gt_envmap is None:
            white = self.cubemap.mean(-1, keepdim=True) # [6, R, R, 1]
            return (self.cubemap - white).abs().mean()
        return torch.zeros(1, device=self.device)

    def set_gt_geometry(self, mesh: TriangleMesh) -> None:
        self.gt_geometry = mesh.to(self.device)
        self.offsets = torch.nn.Parameter(torch.zeros_like(self.gt_geometry.vertices))
        with torch.no_grad():
            self.optix_ctx = OptiXContext()

    def render_report(self, inputs: Cameras) -> Tuple[List[Tensor], RGBAImages, TriangleMesh, Tensor]:
        assert inputs.is_cuda
        inputs = inputs.view(-1)
        mesh = self.gt_geometry.replace(vertices=self.gt_geometry.vertices + self.offsets)
        mesh.compute_face_normals_(fix=True)

        ctx = dr.RasterizeCudaContext(inputs.device)

        envmap = self.get_envmap()
        with torch.no_grad():
            mesh_vertices = mesh.vertices
            mesh_normals = mesh.face_normals[:, 0, :].contiguous()
            if envmap.transform is not None:
                mesh_vertices = (envmap.transform @ mesh_vertices.unsqueeze(-1)).squeeze(-1)
                mesh_normals = (envmap.transform @ mesh_normals.unsqueeze(-1)).squeeze(-1)
            optix_build_bvh(self.optix_ctx, mesh_vertices.contiguous(), mesh.indices.int(), rebuild=1)

        vertices = torch.cat((
            mesh_vertices,
            torch.ones_like(mesh_vertices[..., :1]),
        ), dim=-1).view(-1, 4, 1)                          # [V, 4, 1]
        scale = mesh_vertices.abs().max(0).values

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
            fidx = (rast.detach()[..., -1:].int() - 1).clamp_min(0).flatten()
            frag_pos, _ = dr.interpolate(mesh_vertices[None], rast, indices) # [1, H, W, 3]
            kdks = self.mlp_texture((frag_pos / scale).clamp(-1, 1)) # [1, H, W, 6]
            if self.albedo is None:
                kd = kdks[..., :3].contiguous()
            else:
                kd = self.albedo[fidx, :].view_as(frag_pos).contiguous()
            d_frag_pos = torch.normal(mean=0, std=0.01, size=frag_pos.shape, device=frag_pos.device) # [1, H, W, 3]
            kdks_jitter = self.mlp_texture(((frag_pos + d_frag_pos) / scale).clamp(-1, 1))
            # frag_n, _ = dr.interpolate(mesh.normals[None], rast, indices) # [1, H, W, 3]
            # frag_n = safe_normalize(frag_n) # [1, H, W, 3]
            frag_n = mesh_normals[fidx, :].reshape_as(frag_pos)
            roughness = kdks[..., 3:4] * (1 - self.min_roughness) + self.min_roughness # [1, H, W, 1]
            metallic  = kdks[..., 4:5] # [1, H, W, 1]
            new_ks = torch.cat((torch.zeros_like(roughness), roughness, metallic), dim=-1)

            if envmap.transform is not None:
                camera_pos = (envmap.transform @ camera_pos.unsqueeze(-1)).squeeze(-1)

            envmap.compute_pdf_()
            diffuse_accum, specular_accum, _ = optix_env_shade(
                self.optix_ctx,
                rast[..., -1],
                frag_pos + frag_n * 1e-3,
                frag_pos,
                frag_n,
                camera_pos.contiguous().view(1, 1, 1, 3),
                kd,
                new_ks,
                envmap.data,
                envmap.pdf[..., 0],
                envmap.pdf[:, 0, 1],
                envmap.pdf[..., 2],
                BSDF='pbr',
                n_samples_x=self.num_samples_per_ray,
                rnd_seed=None,
                shadow_scale=self.shadow_scale,
            )
            if self.denoise:
                frag_depth = ((camera_pos - frag_pos) * camera.c2w[:, 2]).sum(-1, keepdim=True)
                sigma = max(self.shadow_scale * 2, 0.0001)
                diffuse_accum  = bilateral_denoiser(diffuse_accum, frag_n, frag_depth, sigma)
                specular_accum = bilateral_denoiser(specular_accum, frag_n, frag_depth, sigma)
            colors = torch.cat((diffuse_accum, kd * (1 - metallic), specular_accum), dim=-1) # [1, H, W, 3]
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
                    kdks_vis.append(torch.cat((new_ks, alphas), dim=-1).squeeze(0))
                    kdks_vis.append(torch.cat((frag_n * 0.5 + 0.5, alphas), dim=-1).squeeze(0))
            if self.albedo is None and self.kd_grad_weight > 0:
                kd_grad = (kdks_jitter[..., :3] - kdks[..., :3]).abs().mean() # [1]
                kd_grad_reg = kd_grad_reg + kd_grad
            if self.ks_grad_weight > 0:
                ks_grad = (kdks_jitter[..., 3:] - kdks[..., 3:]).abs().mean() # [1]
                ks_grad_reg = ks_grad_reg + ks_grad
        if self.light_weight > 0:
            light_reg = self.light_weight * self.get_light_regularization()
        else:
            light_reg = 0.0
        kd_grad_reg = (self.kd_grad_weight / inputs.shape[0]) * kd_grad_reg
        ks_grad_reg = (self.ks_grad_weight / inputs.shape[0]) * ks_grad_reg
        regularization = light_reg + kd_grad_reg + ks_grad_reg
        return (
            images,
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
    def as_module(self, *, field_name: Literal['light', 'gamma', 'offsets']) -> Any:

        def parameters(self) -> Any:
            params = {
                'light': self.cubemap,
                'gamma': self.gamma,
                'offsets': self.offsets,
            }[field_name]
            return [params]

        return parameters
