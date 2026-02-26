from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from gsplat import rasterization, rasterization_2dgs
from torch import nn

from rfstudio.graphics import Cameras, FeatureImages
from rfstudio.nn import Module
from rfstudio.utils.decorator import chains


@dataclass
class FeatureSplatterS2(Module):

    load: Path = ...

    block_width: int = 16

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
        data = torch.load(self.load)
        self.means = nn.Parameter(data['means'], requires_grad=False)
        self.colors = nn.Parameter(data['colors'], requires_grad=False)
        self.shs = nn.Parameter(data['shs'], requires_grad=False)
        self.opacities = nn.Parameter(data['opacities'], requires_grad=False)
        self.scales = nn.Parameter(data['scales'], requires_grad=False)
        self.quats = nn.Parameter(data['quats'], requires_grad=False)

        self.features = nn.Parameter(data['features']) # [N, F]
        self.codebook = nn.Parameter(data['cluster_centers']) # [C, F]

    def render_feature(self, inputs: Cameras) -> FeatureImages:

        assert inputs.shape == (1, )
        camera = inputs[0]
        W, H = int(camera.width.item()), int(camera.height.item())

        if self.rasterize_mode not in ["antialiased", "classic", "2dgs"]:
            raise ValueError(f"Unknown rasterize_mode: {self.rasterize_mode}")

        similarity = F.cosine_similarity(self.features.detach().unsqueeze(1), self.codebook.detach(), dim=-1) # [N, C]
        indices = similarity.argmax(dim=-1) # [N]
        features = self.codebook[indices] + (self.features - self.features.detach()) # [N, F]

        if self.rasterize_mode == "2dgs":
            render, alpha, normal, pseudo_normal, distort, _, info = rasterization_2dgs(
                means=self.means,
                quats=self.quats,
                scales=self.scales.exp(),
                opacities=torch.sigmoid(self.opacities).squeeze(-1),
                colors=features,
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
            )
        else:
            render, alpha, info = rasterization(
                means=self.means,
                quats=self.quats,
                scales=self.scales.exp(),
                opacities=torch.sigmoid(self.opacities).squeeze(-1),
                colors=features,
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
            )

        return FeatureImages(render)

    @chains
    def as_module(self, *, field_name: str) -> Any:

        def parameters(self) -> Any:
            return [getattr(self, field_name)]

        return parameters
