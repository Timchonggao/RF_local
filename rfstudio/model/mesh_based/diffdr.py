from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Tuple

import torch
from torch import Tensor

from rfstudio.graphics import (
    Cameras,
    DepthImages,
    DMTet,
    FlexiCubes,
    TriangleMesh,
)
from rfstudio.graphics.shaders import DepthShader
from rfstudio.nn import Module
from rfstudio.utils.decorator import chains


@dataclass
class DiffDR(Module):

    geometry: Literal['dmtet', 'flexicubes'] = 'flexicubes'

    resolution: int = 96

    scale: float = 1.05

    antialias: bool = True

    z_up: bool = False

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
        self.sdf_weight = 0.0

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

    def render_report(self, inputs: Cameras) -> Tuple[DepthImages, TriangleMesh, Tensor]:
        mesh, geom_reg = self.get_geometry()
        return (
            mesh.render(inputs, shader=DepthShader(antialias=self.antialias)),
            mesh,
            geom_reg,
        )

    @chains
    def as_module(self, *, field_name: Literal['deforms', 'sdfs', 'weights']) -> Any:

        def parameters(self) -> Any:
            params = {
                'deforms': self.deform_params,
                'sdfs': self.sdf_params,
                'weights': self.weight_params,
            }[field_name]
            return [params]

        return parameters
