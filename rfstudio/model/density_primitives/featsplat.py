from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import torch
from gsplat import rasterization, rasterization_2dgs
from torch import Tensor

from rfstudio.graphics import Cameras, DepthImages, FeatureImages, FeatureSplats, RGBAImages
from rfstudio.graphics.math import rgb2sh
from rfstudio.nn import Module

from .gsplat import UpdateInfo


@dataclass
class FeatureSplatter(Module):

    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 1.0
    "Size of the cube to initialize random gaussians within"
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""

    feature_dim: int = 8

    block_width: int = 16

    prepare_densification: bool = True

    rasterize_mode: Literal["classic", "antialiased", "2dgs"] = "2dgs"
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
        self.gaussians = FeatureSplats.random(
            size=self.num_random,
            sh_degree=self.sh_degree,
            feature_dim=self.feature_dim,
            random_scale=self.random_scale,
            device=self.device,
            requires_grad=True,
        )
        self.update_info = None
        self.max_sh_degree = None

    def state_dict(self) -> Dict[str, Tensor]:
        return {
            'means': self.gaussians.means,
            'colors': self.gaussians.colors,
            'shs': self.gaussians.shs,
            'opacities': self.gaussians.opacities,
            'scales': self.gaussians.scales,
            'quats': self.gaussians.quats,
            'features': self.gaussians.features,
        }

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        self.gaussians = FeatureSplats(
            means=state_dict['means'],
            colors=state_dict['colors'],
            shs=state_dict['shs'],
            opacities=state_dict['opacities'],
            scales=state_dict['scales'],
            quats=state_dict['quats'],
            features=state_dict['features'],
        ).to(self.device).requires_grad_(self.gaussians.requires_grad)

    @torch.no_grad()
    def export_point_cloud(self, path: Path) -> None:
        self.gaussians.export(path)

    def set_max_sh_degree(self, value: Optional[int] = None) -> None:
        self.max_sh_degree = value

    def render_depth(self, inputs: Cameras) -> DepthImages:

        # TODO: fix the device
        with torch.no_grad():
            if self.gaussians.device != self.device:
                requires_grad = self.gaussians.requires_grad
                self.gaussians.swap_(self.gaussians.to(self.device))
                self.gaussians.requires_grad_(requires_grad)

        assert inputs.shape == (1, )
        camera = inputs[0]
        W, H = int(camera.width.item()), int(camera.height.item())

        if self.rasterize_mode not in ["antialiased", "classic", "2dgs"]:
            raise ValueError(f"Unknown rasterize_mode: {self.rasterize_mode}")

        if self.rasterize_mode == "2dgs":
            render, alpha, _, _, _, _, info = rasterization_2dgs(
                means=self.gaussians.means,
                quats=self.gaussians.quats,
                scales=self.gaussians.scales.exp(),
                opacities=torch.sigmoid(self.gaussians.opacities).squeeze(-1),
                colors=self.gaussians.colors.detach(),
                viewmats=camera.view_matrix.view(1, 4, 4),  # [1, 4, 4]
                Ks=camera.intrinsic_matrix.view(1, 3, 3),  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=self.block_width,
                packed=True,
                near_plane=0.01,
                far_plane=1e10,
                render_mode='ED',
                sh_degree=None,
                sparse_grad=False,
                absgrad=False,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )
        else:
            render, alpha, info = rasterization(
                means=self.gaussians.means,
                quats=self.gaussians.quats,
                scales=self.gaussians.scales.exp(),
                opacities=torch.sigmoid(self.gaussians.opacities).squeeze(-1),
                colors=self.gaussians.colors.detach(),
                viewmats=camera.view_matrix.view(1, 4, 4),  # [1, 4, 4]
                Ks=camera.intrinsic_matrix.view(1, 3, 3),  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=self.block_width,
                packed=True,
                near_plane=0.01,
                far_plane=1e10,
                render_mode='ED',
                sh_degree=None,
                sparse_grad=False,
                absgrad=False,
                rasterize_mode=self.rasterize_mode,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )

        key = "gradient_2dgs" if self.rasterize_mode == '2dgs' else "means2d"
        if self.training and self.prepare_densification and info[key].requires_grad:
            info[key].retain_grad()
            self.update_info = UpdateInfo(
                xys=info[key],
                radii=info["radii"],
                indices=info["gaussian_ids"],
                last_width=camera.width[None],
                last_height=camera.height[None]
            )

        depth = torch.cat((render, alpha), dim=-1) # [1, H, W, 2]
        return DepthImages(depth)

    def render_feature(self, inputs: Cameras) -> FeatureImages:
        # TODO: fix the device
        with torch.no_grad():
            if self.gaussians.device != self.device:
                requires_grad = self.gaussians.requires_grad
                self.gaussians.swap_(self.gaussians.to(self.device))
                self.gaussians.requires_grad_(requires_grad)

        assert inputs.shape == (1, )
        camera = inputs[0]
        W, H = int(camera.width.item()), int(camera.height.item())

        if self.rasterize_mode not in ["antialiased", "classic", "2dgs"]:
            raise ValueError(f"Unknown rasterize_mode: {self.rasterize_mode}")

        if self.rasterize_mode == "2dgs":
            render, alpha, normal, pseudo_normal, distort, _, info = rasterization_2dgs(
                means=self.gaussians.means.detach(),
                quats=self.gaussians.quats.detach(),
                scales=self.gaussians.scales.detach().exp(),
                opacities=torch.sigmoid(self.gaussians.opacities.detach()).squeeze(-1),
                colors=self.gaussians.features,
                viewmats=camera.view_matrix.view(1, 4, 4),  # [1, 4, 4]
                Ks=camera.intrinsic_matrix.view(1, 3, 3),  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=self.block_width,
                packed=True,
                near_plane=0.01,
                far_plane=1e10,
                render_mode='RGB+ED',
                sh_degree=None,
                distloss=self.training,
                sparse_grad=False,
                absgrad=False,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )
        else:
            render, alpha, info = rasterization(
                means=self.gaussians.means.detach(),
                quats=self.gaussians.quats.detach(),
                scales=self.gaussians.scales.detach().exp(),
                opacities=torch.sigmoid(self.gaussians.opacities.detach()).squeeze(-1),
                colors=self.gaussians.features,
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

        return FeatureImages(render)


    def render_rgba(self, inputs: Cameras) -> RGBAImages:

        # TODO: fix the device
        with torch.no_grad():
            if self.gaussians.device != self.device:
                requires_grad = self.gaussians.requires_grad
                self.gaussians.swap_(self.gaussians.to(self.device))
                self.gaussians.requires_grad_(requires_grad)

        assert inputs.shape == (1, )
        camera = inputs[0]
        W, H = int(camera.width.item()), int(camera.height.item())
        sh_degree = (
            self.gaussians.sh_degree
            if self.max_sh_degree is None
            else min(self.max_sh_degree, self.gaussians.sh_degree)
        )

        if self.rasterize_mode not in ["antialiased", "classic", "2dgs"]:
            raise ValueError(f"Unknown rasterize_mode: {self.rasterize_mode}")

        if sh_degree == 0:
            colors = self.gaussians.colors
            sh_degree = None
        else:
            colors = torch.cat((rgb2sh(self.gaussians.colors[..., None, :]), self.gaussians.shs), dim=-2)

        if self.rasterize_mode == "2dgs":
            render, alpha, normal, pseudo_normal, distort, _, info = rasterization_2dgs(
                means=self.gaussians.means,
                quats=self.gaussians.quats,
                scales=self.gaussians.scales.exp(),
                opacities=torch.sigmoid(self.gaussians.opacities).squeeze(-1),
                colors=colors,
                viewmats=camera.view_matrix.view(1, 4, 4),  # [1, 4, 4]
                Ks=camera.intrinsic_matrix.view(1, 3, 3),  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=self.block_width,
                packed=True,
                near_plane=0.01,
                far_plane=1e10,
                render_mode='RGB+ED',
                sh_degree=sh_degree,
                distloss=self.training,
                sparse_grad=False,
                absgrad=False,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )
        else:
            render, alpha, info = rasterization(
                means=self.gaussians.means,
                quats=self.gaussians.quats,
                scales=self.gaussians.scales.exp(),
                opacities=torch.sigmoid(self.gaussians.opacities).squeeze(-1),
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

        key = "gradient_2dgs" if self.rasterize_mode == '2dgs' else "means2d"
        if self.training and self.prepare_densification and info[key].requires_grad:
            normal_loss = None
            distort_loss = None
            if self.rasterize_mode == '2dgs':
                normal_loss = (1 - (normal * (pseudo_normal * alpha.detach())).sum(-1)).mean()[None]
                distort_loss = distort.mean()[None]
            info[key].retain_grad()
            self.update_info = UpdateInfo(
                xys=info[key],
                radii=info["radii"],
                indices=info["gaussian_ids"],
                last_width=camera.width[None],
                last_height=camera.height[None],
                distort_loss=distort_loss,
                normal_loss=normal_loss,
            )

        return RGBAImages(torch.cat((render[..., :3], alpha), dim=-1))

    @torch.no_grad()
    def update_grad_norm(self) -> None:
        assert self.update_info is not None
        # keep track of a moving average of grad norms
        grads = self.update_info.xys.grad.norm(dim=-1)                 # [V]
        if self.gaussians.xys_grad_norm is None:
            self.gaussians.annotate_(
                xys_grad_norm=grads.new_zeros(self.gaussians.shape),
                vis_counts=grads.new_ones(self.gaussians.shape),
                last_width=self.update_info.last_width,
                last_height=self.update_info.last_height
            )
        assert self.gaussians.vis_counts is not None
        self.gaussians.vis_counts[self.update_info.indices] += 1
        self.gaussians.xys_grad_norm[self.update_info.indices] += grads
