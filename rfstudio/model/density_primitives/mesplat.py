from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import torch
from gsplat import rasterization
from jaxtyping import Float32
from torch import Tensor, nn

from rfstudio.graphics import Cameras, Points, RGBImages, Splats, TriangleMesh
from rfstudio.graphics.math import (
    get_rotation_from_relative_vectors,
    rgb2sh,
    rot2quat,
    safe_normalize,
    sh_deg2dim,
)
from rfstudio.nn import Module
from rfstudio.utils.decorator import chains


@dataclass
class MeshSplatter(Module):

    num_samples: int = 10000

    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""

    block_width: int = 16

    background_color: Literal["random", "black", "white"] = "random"

    max_scale: float = 0.16

    max_movement: float = 0.05

    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel.
    This approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured,
    which results "aliasing-like" artifacts.
    The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers
    that were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """

    def __setup__(self) -> None:
        self.gaussians = None
        self.mean_params = nn.Parameter(torch.empty(self.num_samples, 3))
        self.scale_params = nn.Parameter(torch.empty(self.num_samples, 2))
        self.quat_params = nn.Parameter(torch.empty(self.num_samples, 4))
        self.colors = nn.Parameter(torch.empty(self.num_samples, 3))
        self.normals = nn.Parameter(torch.empty(self.num_samples, 3))
        self.shs = nn.Parameter(torch.empty(self.num_samples, sh_deg2dim(self.sh_degree) - 1, 3))
        self.max_sh_degree = None
        self.points = None

    def reset_from_mesh(self, mesh: TriangleMesh) -> None:
        points = mesh.uniformly_sample(
            self.num_samples,
            samples_per_face='uniform',
            samples_in_face='r2',
        ).to(self.device)
        self.points = points.replace(positions=points.positions + 0.001 * points.normals)
        dists = points.detach().k_nearest(k=3)[0].mean(-1, keepdim=True) # [N, 1]

        S = self.num_samples
        self.mean_params.data.zero_()
        self.scale_params.data.copy_(dists.repeat(1, 2).clamp(min=1e-10).log())
        z_axis = torch.tensor([0, 0, 1]).to(points.normals)
        self.quat_params.data.copy_(rot2quat(get_rotation_from_relative_vectors(z_axis, points.normals)))
        self.colors.data.copy_(torch.ones((S, 3), device=self.device) * 0.5)
        self.shs.data.zero_()
        self.normals.data.zero_()

        self.gaussians = Splats(
            means=torch.empty((S, 3), device=self.device),
            scales=torch.empty((S, 3), device=self.device),
            quats=torch.empty((S, 4), device=self.device),
            opacities=torch.logit(0.98 * torch.ones((S, 1), device=self.device)),
            colors=self.colors,
            shs=self.shs
        )
        self.update_cov3d()

    def export_oriented_point_cloud(self) -> Points:
        return Points(
            positions=self.gaussians.means,
            colors=self.gaussians.colors,
            normals=self.normals,
        )

    def state_dict(self) -> Dict[str, Tensor]:
        return {
            'means': self.gaussians.means,
            'colors': self.gaussians.colors,
            'shs': self.gaussians.shs,
            'opacities': self.gaussians.opacities,
            'scales': self.gaussians.scales,
            'quats': torch.where(self.gaussians.quats[..., 3:] > 0, self.gaussians.quats, -self.gaussians.quats),
            'normals': safe_normalize(self.normals),
            'state': Module.state_dict(self)
        }

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        self.gaussians = Splats(
            means=state_dict['means'],
            colors=state_dict['colors'],
            shs=state_dict['shs'],
            opacities=state_dict['opacities'],
            scales=state_dict['scales'],
            quats=state_dict['quats']
        ).to(self.device).requires_grad_(True)
        Module.load_state_dict(self, state_dict['state'])

    @torch.no_grad()
    def export_point_cloud(self, path: pathlib.Path) -> None:
        self.gaussians.export(path)

    def get_background_color(self) -> Float32[Tensor, "3 "]:
        if self.background_color == 'black':
            return torch.zeros(3)
        if self.background_color == 'white':
            return torch.ones(3)
        if self.training:
            return torch.rand(3)
        return torch.tensor([0.1490, 0.1647, 0.2157])

    def set_max_sh_degree(self, value: Optional[int] = None) -> None:
        self.max_sh_degree = value

    def update_cov3d(self) -> None:
        scales = self.scale_params.clamp(-7, torch.tensor(self.max_scale).log().item())
        self.gaussians.replace_(
            means=self.points.positions + self.mean_params.clamp(-self.max_movement, self.max_movement),
            scales=torch.cat((scales, torch.empty_like(scales[..., 0:1]).fill_(-7)), dim=-1),
            quats=self.quat_params,
        )

    @chains
    def as_module(self, *, field_name: Literal['means', 'scales', 'colors', 'shs', 'quats']) -> Any:

        def parameters(self) -> Any:
            params = {
                'means': self.mean_params,
                'scales': self.scale_params,
                'quats': self.quat_params,
                'colors': self.colors,
                'normals': self.normals,
                'shs': self.shs,
            }[field_name]
            return [params]

        return parameters

    def render_rgb(self, inputs: Cameras) -> RGBImages:

        assert inputs.shape == (1, )
        camera = inputs[0]
        W, H = int(camera.width.item()), int(camera.height.item())
        background_color = self.get_background_color().to(camera.device)
        sh_degree = (
            self.gaussians.sh_degree
            if self.max_sh_degree is None
            else min(self.max_sh_degree, self.gaussians.sh_degree)
        )

        if self.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError(f"Unknown rasterize_mode: {self.rasterize_mode}")

        if sh_degree == 0:
            colors = self.gaussians.colors
            sh_degree = None
        else:
            colors = torch.cat((rgb2sh(self.gaussians.colors[..., None, :]), self.gaussians.shs), dim=-2)

        render, alpha, info = rasterization(
            means=self.gaussians.means,
            quats=self.gaussians.quats,
            scales=self.gaussians.scales.exp(),
            opacities=torch.ones_like(self.gaussians.opacities.squeeze(-1)),
            colors=colors,
            viewmats=camera.view_matrix.view(1, 4, 4),  # [1, 4, 4]
            Ks=camera.intrinsic_matrix.view(1, 3, 3),  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=self.block_width,
            packed=True,
            near_plane=0.01,
            far_plane=1e10,
            render_mode='RGB',
            sh_degree=sh_degree,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode=self.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        rgb = render[..., :3] + (1 - alpha) * background_color
        return RGBImages([rgb.squeeze(0).clamp(0.0, 1.0)])

    def render_normals(self, inputs: Cameras) -> RGBImages:

        assert inputs.shape == (1, )
        camera = inputs[0]
        W, H = int(camera.width.item()), int(camera.height.item())
        background_color = self.get_background_color().to(camera.device)

        if self.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError(f"Unknown rasterize_mode: {self.rasterize_mode}")

        colors = self.normals * 0.5 + 0.5

        render, alpha, info = rasterization(
            means=self.gaussians.means,
            quats=self.gaussians.quats,
            scales=self.gaussians.scales.exp(),
            opacities=torch.ones_like(self.gaussians.opacities.squeeze(-1)),
            colors=colors,
            viewmats=camera.view_matrix.view(1, 4, 4),  # [1, 4, 4]
            Ks=camera.intrinsic_matrix.view(1, 3, 3),  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=self.block_width,
            packed=True,
            near_plane=0.01,
            far_plane=1e10,
            render_mode='RGB',
            sh_degree=None,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode=self.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        rgb = render[..., :3] + (1 - alpha) * background_color
        return RGBImages([rgb.squeeze(0).clamp(0.0, 1.0)])

    @torch.no_grad()
    def reset_rotations(self) -> None:
        z_axis = torch.tensor([0, 0, 1]).to(self.normals)
        self.quat_params.data.copy_(rot2quat(get_rotation_from_relative_vectors(z_axis, self.normals)))
