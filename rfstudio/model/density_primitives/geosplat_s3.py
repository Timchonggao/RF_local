from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import torch
from torch import Tensor

from rfstudio.graphics import (
    Cameras,
    DepthImages,
    FlexiCubes,
    RGBAImages,
    RGBImages,
    Splats,
    TextureLatLng,
    TriangleMesh,
)
from rfstudio.graphics._mesh._optix import OptiXContext, optix_build_bvh
from rfstudio.graphics.shaders import DepthShader, NormalShader, PrettyShader
from rfstudio.model.density_field.components.encoding import HashEncoding
from rfstudio.nn import MLP, Module
from rfstudio.utils.decorator import chains

from .geosplat import MGAdapter, RenderableAttrs
from .gsplat import GSplatter


@dataclass
class GeoSplatterS3(Module):

    load: Path = ...

    resolution: int = 128

    z_up: bool = False

    super_sampling: bool = False

    background_color: Literal['white', 'black'] = 'white'

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

    occ_enc: HashEncoding = HashEncoding(
        mlp=MLP(
            layers=[-1, 32, 32, 3],
            activation='none',
            bias=False,
            initialization='kaiming-uniform',
        ),
        grad_scaling=16.0,
        max_res=4096,
        log2_hashmap_size=18,
    )

    def __setup__(self) -> None:

        state_dict = torch.load(self.load, map_location='cpu')
        self.mean_params = torch.nn.Parameter(state_dict['means'])
        self.scale_params = torch.nn.Parameter(state_dict['scales'])
        self.quat_params = torch.nn.Parameter(state_dict['quats'])
        self.opacity_params = torch.nn.Parameter(state_dict['opacities'])
        self.kd_params = torch.nn.Parameter(state_dict['kd'], requires_grad=False)
        self.ks_params = torch.nn.Parameter(state_dict['ks'], requires_grad=False)
        self.occ_params = torch.nn.Parameter(state_dict['occ'], requires_grad=False)
        self.normal_params = torch.nn.Parameter(state_dict['normals'], requires_grad=False)
        self.exposure_params = torch.nn.Parameter(state_dict['exposure'], requires_grad=False)
        self.latlng = TextureLatLng(data=state_dict['latlng'])
        self.mc_positions = torch.nn.Parameter(state_dict['mc_positions'], requires_grad=False)
        self.mc_vertices = torch.nn.Parameter(state_dict['mc_vertices'], requires_grad=False)
        self.mc_indices = torch.nn.Parameter(state_dict['mc_indices'], requires_grad=False)
        self.ks_enc.load_state_dict(state_dict['ks_enc'])
        self.occ_enc.load_state_dict(state_dict['occ_enc'])
        self.max_metallic = state_dict['max_metallic']
        self.min_roughness = state_dict['min_roughness']
        self.scale = state_dict['geom_scale']

        self.geometric_repr = FlexiCubes.from_resolution(
            self.resolution,
            scale=self.scale,
        )
        self.deform_params = torch.nn.Parameter(torch.zeros_like(self.geometric_repr.vertices))
        self.sdf_params = torch.nn.Parameter(self.geometric_repr.sdf_values.clone())
        self.weight_params = torch.nn.Parameter(torch.ones(self.geometric_repr.indices.shape[0], 21))

        self.sdf_weight = 0.0
        self.sh_degree = 0
        self.optix_ctx = None

    def get_geometry(self) -> Tuple[TriangleMesh, Tensor]:
        # TODO: fix the device
        with torch.no_grad():
            if self.geometric_repr.device != self.device:
                self.geometric_repr.swap_(self.geometric_repr.to(self.device))
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

    def get_splats(self) -> Tuple[Splats, RenderableAttrs]:
        return Splats(
            means=self.mean_params,
            scales=self.scale_params,
            quats=self.quat_params,
            opacities=self.opacity_params,
            colors=self.mean_params,
            shs=torch.zeros(self.mean_params.shape[0], 0, 3, device=self.device),
        ), RenderableAttrs(
            kd=self.kd_params,
            ks=self.ks_params,
            occ=self.occ_params,
            normals=self.normal_params,
        )

    def render_report(
        self,
        inputs: Cameras,
        *,
        vis: bool = False
    ) -> Tuple[
        DepthImages,
        RGBImages,
        RGBImages,
        DepthImages,
        TriangleMesh,
        Optional[RGBImages],
        Tensor,
    ]:
        if self.optix_ctx is None:
            with torch.no_grad():
                self.latlng = self.latlng.to(self.device).compute_pdf_()
                self.optix_ctx = OptiXContext()
                optix_build_bvh(self.optix_ctx, self.mc_vertices, self.mc_indices, rebuild=1)

        mesh, geom_reg = self.get_geometry()
        gsplat = GSplatter(
            num_random=4,
            sh_degree=0,
            background_color=self.background_color,
            rasterize_mode='antialiased',
        ).to(self.device)
        gsplat.__setup__()
        gsplat.gaussians, attrs = self.get_splats()
        gsplat_rgba = []

        for camera in inputs.view(-1, 1):
            gsplat_rgba.append(attrs.splat_mc(
                gsplat,
                camera,
                context=self.optix_ctx,
                gt_image=None,
                positions=self.mc_positions,
                envmap=self.latlng,
                exposure=self.exposure_params.exp(),
                min_roughness=self.min_roughness,
                max_metallic=self.max_metallic,
                tone_type='aces',
                num_samples_per_ray=8,
                denoise=True,
                residual=True,
            )[0])
        gsplat_rgb = RGBAImages(gsplat_rgba).blend(gsplat.get_background_color())
        ss_cameras = inputs.resize(2) if self.super_sampling else inputs
        with torch.no_grad():
            gsplat_depth = DepthImages([gsplat.render_depth(camera).item() for camera in ss_cameras.view(-1, 1)])
        gsplat.gaussians.replace_(colors=attrs.normals * 0.5 + 0.5)
        gsplat_normal = RGBImages([gsplat.render_rgb(camera).item() for camera in inputs.view(-1, 1)])
        if vis:
            with torch.no_grad():
                vis_cameras = inputs[0]
                splat_rgb = gsplat_rgb[0].item()
                splat_normal = (
                    gsplat.render_depth(vis_cameras[None])
                        .compute_pseudo_normals(vis_cameras)
                        .visualize(gsplat.get_background_color())
                        .item()
                )
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
                    splat_normal,
                    self.latlng.as_cubemap(resolution=512).visualize(
                        width=splat_rgb.shape[1],
                        height=splat_rgb.shape[0],
                    ).item(),
                ])
        else:
            visualization = None
        return (
            mesh.render(ss_cameras, shader=DepthShader(antialias=True)),
            gsplat_rgb,
            gsplat_normal,
            gsplat_depth,
            mesh,
            visualization,
            geom_reg,
        )

    @torch.no_grad()
    def export(self, path: Path) -> None:

        mesh, _ = self.get_geometry()
        splats, _ = MGAdapter().make(mesh.compute_vertex_normals(fix=True))
        attributes = {
            'geom_scale': self.scale,
            'min_roughness': self.min_roughness, # [1]
            'max_metallic': self.max_metallic, # [1]
            'exposure': self.exposure_params, # [1]
            'latlng': self.latlng.data, # [V, 3]
            'vertices': mesh.vertices, # [V, 3]
            'indices': mesh.indices, # [F, 3]
            'means': splats.means,
            'scales': splats.scales,
            'quats': splats.quats,
            'opacities': splats.opacities,
            'normals': splats.colors,
            'occ': self.occ_enc((splats.means / self.scale).clamp(-1, 1)),
            'ks_enc': self.ks_enc.state_dict(),
        }
        torch.save(attributes, path)

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
            'quats',
            'opacities',
        ]
    ) -> Any:

        def parameters(self) -> Any:
            params = {
                'deforms': self.deform_params,
                'sdfs': self.sdf_params,
                'weights': self.weight_params,
                'means': self.mean_params,
                'scales': self.scale_params,
                'quats': self.quat_params,
                'opacities': self.opacity_params,
            }[field_name]
            return [params]

        return parameters
