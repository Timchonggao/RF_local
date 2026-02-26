from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import torch
from jaxtyping import Float32
from torch import Tensor, nn

from rfstudio.graphics import (
    Cameras,
    DepthImages,
    DMTet,
    Points,
    TriangleMesh,
    VectorImages,
)
from rfstudio.graphics.shaders import DepthShader, NormalShader
from rfstudio.graphics.spatial_sampler import NearestGridSampler
from rfstudio.nn import Module
from rfstudio.utils.decorator import chains


@dataclass
class TetWeave(Module):

    num_init_points: int = 8000
    num_target_points: int = 32000
    resampler: NearestGridSampler = NearestGridSampler(scale=1.1)

    def __setup__(self) -> None:
        dmtet = DMTet.from_ball_sampling_delaunay(self.num_init_points, scale=self.resampler.scale * 1.732)
        self.vertices = nn.Parameter(dmtet.vertices)
        self.sdf_values = nn.Parameter(dmtet.sdf_values)
        self.indices = nn.Parameter(dmtet.indices, requires_grad=False)
        self.sdf_weight = 0.0
        self.energy_weight = 0.0
        self.fairness_weight = 0.0

    @torch.no_grad()
    def load_state_dict(self, state_dict) -> None:
        self.vertices = nn.Parameter(state_dict['vertices'].to(self.device))
        self.sdf_values = nn.Parameter(state_dict['sdf_values'].to(self.device))
        self.indices = nn.Parameter(state_dict['indices'].to(self.device), requires_grad=False)

    @torch.no_grad()
    def update(self) -> None:
        dmtet = DMTet.from_delaunay(Points(positions=self.vertices), random_sdf=False)
        self.vertices.data.copy_(dmtet.vertices)
        self.indices = nn.Parameter(dmtet.indices, requires_grad=False)

    def render(self, cameras: Cameras) -> Tuple[DepthImages, VectorImages, TriangleMesh, Tensor]:
        dmtet = DMTet(vertices=self.vertices, sdf_values=self.sdf_values, indices=self.indices)
        mesh = dmtet.marching_tets()
        fairness = (mesh.compute_angles() - (torch.pi / 3)).square().mean()
        reg = (
            dmtet.compute_entropy() * self.sdf_weight +
            fairness * self.fairness_weight +
            dmtet.compute_delaunay_energy() * self.energy_weight
        )
        depths = mesh.render(cameras, shader=DepthShader(force_alpha_antialias=True))
        normals = mesh.render(cameras, shader=NormalShader(force_alpha_antialias=True, normal_type='flat'))
        return depths, normals, mesh, reg

    @torch.no_grad()
    def resample(
        self,
        num_new_points: int,
        *,
        aggregated_positions: Float32[Tensor, "N 3"],
        aggregated_errors: Float32[Tensor, "N 3"],
    ) -> None:
        dmtet = DMTet(vertices=self.vertices, sdf_values=self.sdf_values, indices=self.indices)
        neighbor_active_inds = dmtet.get_neighbor_active_vertex_indices()
        num_pruned_points = dmtet.num_vertices - neighbor_active_inds.shape[0]

        self.resampler.reset()
        self.resampler.aggregate(aggregated_positions, importances=aggregated_errors)
        resamples = self.resampler.sample(num_new_points + num_pruned_points) # [S, 3]
        interped_sdf_values = dmtet.interp_sdf_values(resamples) # [S, 1]
        self.vertices = nn.Parameter(torch.cat((dmtet.vertices[neighbor_active_inds], resamples)))
        self.sdf_values = nn.Parameter(torch.cat((dmtet.sdf_values[neighbor_active_inds], interped_sdf_values)))

    @chains
    def as_module(self, *, field_name: str) -> Any:

        def parameters(self) -> Any:
            return [getattr(self, field_name)]

        return parameters

