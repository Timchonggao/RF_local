from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import torch
from torch import Tensor

from rfstudio.graphics import (
    Cameras,
    DepthImages,
    DMTet,
    FlexiCubes,
    RGBAImages,
    RGBImages,
    Splats,
    TextureCubeMap,
    TriangleMesh,
)
from rfstudio.graphics.math import get_rotation_from_relative_vectors, rot2quat, safe_normalize
from rfstudio.graphics.shaders import DepthShader, NormalShader, PrettyShader
from rfstudio.model import GSplatter
from rfstudio.model.density_field.components.encoding import HashEncoding
from rfstudio.nn import MLP, Module
from rfstudio.utils.decorator import chains

from ..density_primitives.geosplat import RenderableAttrs


@dataclass
class SplattableMesh(Module):

    load: Optional[Path] = None

    mode: Literal['rgb', 'sh', 'splitsum', 'mc'] = ...

    geometry: Literal['dmtet', 'flexicubes'] = 'flexicubes'

    resolution: int = 128

    scale: float = 1.05

    max_num_gaussians: int = 5e5

    z_up: bool = False

    super_sampling: bool = True

    opacity_reset: float = 0.5

    mlp_texture: HashEncoding = HashEncoding(
        mlp=MLP(
            layers=[-1, 32, 5],
            activation='sigmoid',
            bias=False,
            initialization='kaiming-uniform',
        ),
        grad_scaling=16.0,
    )

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
        else:
            raise ValueError(self.geometry)

        # self.envmap = torch.nn.Parameter(torch.ones(6, 256, 256, 3) * 0.5)
        self.envmap = torch.nn.Parameter(TextureCubeMap.from_image_file(Path('data') / 'irrmaps' / 'aerodynamics_workshop_2k.hdr', resolution=256).data)
        self.envmap.register_hook(lambda g: g * 64)

        self.sdf_weight = 0.0
        self.sh_degree = 0

        self.mean_params = None
        self.scale_params = None
        self.quat_params = None
        self.normal_params = None
        self.opacity_params = None
        self.color_params = None
        self.shs_params = None
        self.exposure_params = torch.nn.Parameter(torch.zeros(1))

        if self.load is not None:
            state_dict = torch.load(self.load, map_location='cpu')
            self.deform_params.data.copy_(state_dict['deform_params'])
            self.sdf_params.data.copy_(state_dict['sdf_params'])
            self.weight_params.data.copy_(state_dict['weight_params'])

    def get_geometry(self) -> Tuple[TriangleMesh, Tensor]:
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
            return mesh, geom_reg + L_dev.mean() * 0.5 + self.weight_params[:, :20].abs().mean() * 0.1
        raise ValueError(self.geometry)

    def update_splats(self) -> None:

        with torch.no_grad():
            mesh, _ = self.get_geometry()
            F = mesh.num_faces

            expanded_inds = mesh.indices.view(F, 3, 1).expand(F, 3, 3)     # [F, 3, 3]
            vertices = mesh.vertices.gather(
                dim=-2,
                index=expanded_inds.view(-1, 3),                           # [3F, 3]
            ).view(F, 3, 3)                                                # [F, 3, 3]

            weighted_face_normals = torch.cross(
                vertices[:, 1, :] - vertices[:, 0, :],
                vertices[:, 2, :] - vertices[:, 0, :],
                dim=-1,
            ).unsqueeze(-2).expand(expanded_inds.shape)                    # [F, 3, 3]
            normals = torch.zeros_like(mesh.vertices)                      # [V, 3]
            normals.scatter_add_(
                dim=-2,
                index=expanded_inds.view(-1, 3),                           # [3F, 3]
                src=safe_normalize(weighted_face_normals).view(-1, 3),     # [3F, 3]
            )
            normals = safe_normalize(normals)

            face_vertex_normals = normals.gather(
                dim=-2,
                index=expanded_inds.view(-1, 3),                             # [3F, 3]
            ).view(F, 3, 3)                                                  # [F, 3, 3]
            products = (weighted_face_normals * face_vertex_normals).sum(-1) # [F, 3]
            vertex_areas = torch.zeros_like(normals[:, :1])                  # [V, 1]
            vertex_areas.scatter_add_(
                dim=-2,
                index=mesh.indices.view(-1, 1),                              # [3F, 1]
                src=products.view(-1, 1),                                    # [3F, 1]
            )
            areas = vertex_areas.clamp_min(1e-10) / 6
            log_sqrt_areas = (areas / 2).log() * 0.5                  # [S, 1]

            z_axis = torch.tensor([0, 0, 1]).to(normals)
            uniform_laplace = z_axis.expand_as(normals).clone()            # [V, 3]
            uniform_laplace.scatter_reduce_(
                dim=-2,
                index=expanded_inds.view(-1, 3),                           # [3F, 3]
                src=(vertices.roll(1, dims=1) - vertices).view(-1, 3),     # [3F, 3]
                reduce='mean',
                include_self=False,
            )

            if mesh.num_vertices > self.max_num_gaussians:
                mask = torch.randperm(mesh.num_vertices, device=normals.device) < self.max_num_gaussians
                log_sqrt_areas += torch.tensor(mesh.num_vertices / self.max_num_gaussians, device=self.device).log() / 2
            else:
                mask = ...

            quats = rot2quat(get_rotation_from_relative_vectors(z_axis, normals)) # [N, 3, 3]
            scales = torch.cat((
                log_sqrt_areas,
                log_sqrt_areas,
                torch.empty_like(log_sqrt_areas).fill_(1e-10).log(),
            ), dim=-1)                                                            # [S, 3]

            self.mean_params = (mesh.vertices - uniform_laplace * 0.5)[mask]
            self.scale_params = scales[mask]
            self.quat_params = quats[mask]
            self.normal_params = normals[mask]
            self.opacity_params = torch.empty_like(normals[mask, :1]).fill_(self.opacity_reset).logit()
            self.color_params = 0.5 * torch.ones((self.mean_params.shape[0], 5), device=self.device)

            self.mean_params.requires_grad_()
            self.scale_params.requires_grad_()
            self.quat_params.requires_grad_()
            self.normal_params.requires_grad_()
            self.opacity_params.requires_grad_()
            self.color_params.requires_grad_()

            if self.mode == 'sh':
                self.shs_params = torch.zeros(
                    (self.mean_params.shape[0], (self.sh_degree + 1) ** 2 - 1, 3),
                    device=self.device,
                )
                self.shs_params.requires_grad_()
            else:
                self.shs_params = torch.empty((self.mean_params.shape[0], 0, 3), device=self.device)

    def get_splats(self) -> Tuple[Splats, RenderableAttrs]:
        queries = (self.mean_params / self.scale).clamp(-1, 1)
        apps = (self.mlp_texture(queries) + self.color_params.clamp(0, 1)) / 2
        if self.mode == 'splitsum':
            apps_jitter = (self.mlp_texture(queries + torch.randn_like(queries) * 0.01) + self.color_params.clamp(0, 1)) / 2
            attrs = RenderableAttrs(
                kd=apps[..., :3],
                ks=apps[..., 3:],
                occ=torch.zeros_like(apps[..., :1]),
                kd_jitter=apps_jitter[..., :3],
                ks_jitter=apps_jitter[..., 3:],
                normals=safe_normalize(self.normal_params),
            )
            apps = apps[..., :3]
        elif self.mode == 'mc':
            raise NotImplementedError
        elif self.mode in ['rgb', 'sh']:
            with torch.no_grad():
                attrs = RenderableAttrs(
                    kd=apps[..., :3],
                    ks=apps[..., 3:],
                    occ=torch.zeros_like(apps[..., :1]),
                    normals=safe_normalize(self.normal_params),
                )
            apps = apps[..., :3]
        else:
            raise ValueError(self.mode)
        return Splats(
            means=self.mean_params,
            scales=self.scale_params,
            quats=self.quat_params,
            opacities=self.opacity_params,
            colors=apps,
            shs=self.shs_params,
        ), attrs

    def render_report(
        self,
        inputs: Cameras,
        *,
        vis: bool = False
    ) -> Tuple[
        DepthImages,
        RGBImages,
        DepthImages,
        TriangleMesh,
        Optional[RGBImages],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        mesh, geom_reg = self.get_geometry()
        gsplat = GSplatter(
            num_random=4,
            sh_degree=0,
            background_color='white',
            rasterize_mode='antialiased',
        ).to(self.device)
        gsplat.__setup__()
        envmap = TextureCubeMap(data=self.envmap)
        if self.mode in ['rgb', 'sh']:
            gsplat.gaussians, attrs = self.get_splats()
            gsplat_rgb = RGBImages([gsplat.render_rgb(camera).item() for camera in inputs.view(-1, 1)])
        else:
            splats, attrs = self.get_splats()
            gsplat.gaussians = splats
            gsplat_rgb = RGBAImages([
                attrs.splat(
                    gsplat,
                    camera,
                    exposure=self.exposure_params.sigmoid(),
                    envmap=envmap.as_splitsum(),
                    occ_type='none',
                    min_roughness=0.1,
                    max_metallic=1.0,
                    tone_type='aces',
                    culling=False,
                )
                for camera in inputs.view(-1, 1)
            ]).blend((1, 1, 1))
        ss_cameras = inputs.resize(2) if self.super_sampling else inputs
        mesh_normals = mesh.render(inputs, shader=NormalShader(antialias=True, normal_type='flat'))
        with torch.no_grad():
            gsplat_depth = DepthImages([gsplat.render_depth(camera).item() for camera in ss_cameras.view(-1, 1)])
        gsplat_pnormal = [gsplat.render_depth(camera).compute_pseudo_normals(camera[0]).item() for camera in inputs.view(-1, 1)]
        gsplat.gaussians.replace_(colors=attrs.normals)
        gsplat_normal = [gsplat.render_rgba(camera).item() for camera in inputs.view(-1, 1)]
        normal_reg = torch.zeros_like(geom_reg)
        for gnormal, pnormal in zip(gsplat_normal, gsplat_pnormal):
            normal_reg = normal_reg + torch.mul(
                1 - (pnormal[..., :3] * safe_normalize(gnormal[..., :3])).sum(-1, keepdim=True),
                gnormal[..., 3:],
            ).square().mean()
        kd_jitter_reg = torch.nn.functional.l1_loss(attrs.kd, attrs.kd_jitter)
        ks_jitter_reg = torch.nn.functional.l1_loss(attrs.ks, attrs.ks_jitter)
        if vis:
            with torch.no_grad():
                vis_cameras = inputs[0].resize(0.5)
                splat_rgb = gsplat_rgb[0].item()[::2, ::2, :]
                gsplat.gaussians.replace_(colors=attrs.kd)
                kd = gsplat.render_rgb(vis_cameras[None]).item()
                gsplat.gaussians.replace_(colors=torch.cat((
                    attrs.occ,
                    attrs.ks[..., :1] * 0.9 + 0.1,
                    attrs.ks[..., 1:],
                ), dim=-1))
                ks = gsplat.render_rgb(vis_cameras[None]).item()
                gsplat.gaussians.replace_(colors=attrs.normals * 0.5 + 0.5)
                splat_normal = gsplat.render_rgb(vis_cameras[None]).item()
                visualization = RGBImages([
                    mesh.render(
                        vis_cameras,
                        shader=PrettyShader(culling=False, z_up=self.z_up),
                    ).blend((1, 1, 1)).item(),
                    mesh.render(
                        vis_cameras,
                        shader=NormalShader(culling=False, normal_type='flat'),
                    ).visualize((1, 1, 1)).item(),
                    splat_rgb,
                    envmap.visualize(width=splat_rgb.shape[1], height=splat_rgb.shape[0]).item(),
                    kd,
                    ks,
                    splat_normal,
                ])
        else:
            visualization = None
        return (
            mesh.render(ss_cameras, shader=DepthShader(antialias=True)),
            gsplat_rgb,
            gsplat_depth,
            mesh,
            visualization,
            kd_jitter_reg,
            ks_jitter_reg,
            normal_reg / len(mesh_normals),
            geom_reg,
        )

    @chains
    def as_module(
        self,
        *,
        field_name: Literal[
            'deforms',
            'sdfs',
            'weights',
            'means',
            'scales',
            'normals',
            'quats',
            'opacities',
            'colors',
            'shs',
            'envmap',
            'exposure',
        ]
    ) -> Any:

        def parameters(self) -> Any:
            params = {
                'deforms': self.deform_params,
                'sdfs': self.sdf_params,
                'weights': self.weight_params,
                'means': self.mean_params,
                'scales': self.scale_params,
                'normals': self.normal_params,
                'quats': self.quat_params,
                'opacities': self.opacity_params,
                'colors': self.color_params,
                'shs': self.shs_params,
                'envmap': self.envmap,
                'exposure': self.exposure_params,
            }[field_name]
            return [params]

        return parameters
