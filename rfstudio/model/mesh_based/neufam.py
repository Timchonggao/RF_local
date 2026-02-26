from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Tuple

import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.graphics import (
    Cameras,
    FlexiCubes,
    RGBAImages,
    TriangleMesh,
)
from rfstudio.nn import MLP, Module
from rfstudio.utils.decorator import chains
from rfstudio.utils.lazy_module import dr


@dataclass
class NeuFAM(Module):

    background_color: Literal["random", "black", "white"] = "random"

    resolution: int = 64

    feature_dim: int = 256

    scale: float = 1.05

    init_std: float = 1e-3

    antialias: bool = True

    geom_decoder: MLP = MLP(
        layers=[-1, 64, 64, 64, 1],
        activation='tanh',
        bias=False,
        initialization='kaiming-uniform',
    )

    app_decoder: MLP = MLP(
        layers=[-1, 64, 64, 64, 3],
        activation='sigmoid',
        initialization='kaiming-uniform',
    )

    def __setup__(self) -> None:
        self.geometric_repr = FlexiCubes.from_resolution(self.resolution, scale=self.scale)
        assert self.geometric_repr.num_vertices == (self.resolution + 1) ** 3
        self.feats = torch.nn.Parameter(torch.randn(self.geometric_repr.num_vertices, self.feature_dim) * self.init_std)
        centers = self.geometric_repr.vertices.view(*([self.resolution + 1] * 3), -1)
        self.grid_centers = torch.nn.Parameter(
            (centers[1:, 1:, 1:, :] + centers[:-1, :-1, :-1, :]) * 0.5,
            requires_grad=False,
        )
        self.grid_indices = torch.nn.Parameter(
            self.geometric_repr.indices.view(self.resolution, self.resolution, self.resolution, 8),
            requires_grad=False,
        )
        self.sampled_indices = torch.nn.Parameter(
            FlexiCubes.from_resolution(self.resolution - 1, random_sdf=False).indices,
            requires_grad=False,
        )
        self.rand_sdf_values = torch.nn.Parameter(
            torch.rand_like(centers[1:, 1:, 1:, :1]).view(-1, 1) * 0.5 - 0.05,
            requires_grad=False,
        )
        self.grid_size = self.scale * 2 / self.resolution
        self.sdf_weight = 0.0

    def trilinear_query(self, positions: Float32[Tensor, "... 3"]) -> Float32[Tensor, "... C"]:
        x = ((positions / (2 * self.scale) + 0.5) * self.resolution).clamp(0, self.resolution) # [..., 3]
        nearests = x.floor().int().clamp(0, self.resolution - 1) # [..., 3]
        indices = self.grid_indices[nearests[..., 0], nearests[..., 1], nearests[..., 2], :] # [..., 8]
        vertices = self.geometric_repr.vertices[indices.flatten(), :].reshape(*indices.shape, -1) # [..., 8, 3]
        nearest_feats = self.feats[indices.flatten(), :].reshape(*indices.shape, -1) # [..., 8, C]
        weights = ((positions[..., None, :] - vertices.flip(-2)) / self.grid_size).abs().prod(dim=-1, keepdim=True)
        return (nearest_feats * weights).sum(-2) # [..., C]

    def get_geometry(self) -> Tuple[TriangleMesh, Tensor]:
        # TODO: fix the device
        with torch.no_grad():
            if self.geometric_repr.device != self.device:
                self.geometric_repr.swap_(self.geometric_repr.to(self.device))
        queries = torch.add(
            self.grid_centers,
            (torch.rand_like(self.grid_centers) - 0.5) * self.grid_size,
        ).view(-1, 3) # [R*R*R, 3]
        sdf_values = self.geom_decoder(self.trilinear_query(queries)) # [R*R*R, 1]
        flexicubes = FlexiCubes(
            vertices=queries,
            sdf_values=sdf_values + self.rand_sdf_values,
            indices=self.sampled_indices,
            resolution=self.geometric_repr.resolution - 1,
        )
        assert flexicubes.num_vertices == self.resolution ** 3
        assert flexicubes.num_cubes == (self.resolution - 1) ** 3
        mesh, L_dev = flexicubes.dual_marching_cubes()
        if self.sdf_weight > 0:
            geom_reg = flexicubes.compute_entropy() * self.sdf_weight
        else:
            geom_reg = 0.0
        return mesh, geom_reg + L_dev.mean() * 0.25

    def render_report(self, inputs: Cameras) -> Tuple[RGBAImages, TriangleMesh, Tensor]:
        mesh, geom_reg = self.get_geometry()
        mesh = mesh.compute_vertex_normals(fix=True)
        assert inputs.is_cuda
        inputs = inputs.view(-1)

        ctx = dr.RasterizeCudaContext(inputs.device)

        vertices = torch.cat((
            mesh.vertices,
            torch.ones_like(mesh.vertices[..., :1]),
        ), dim=-1).view(-1, 4, 1)                          # [V, 4, 1]

        images = []
        for camera in inputs:
            resolution = [camera.height.item(), camera.width.item()]
            mvp = camera.projection_matrix @ camera.view_matrix                    # [4, 4]
            projected = (mvp @ vertices).view(1, -1, 4)                            # [1, V, 4]
            indices = mesh.indices.int()
            with dr.DepthPeeler(ctx, projected, indices, resolution=resolution) as peeler:
                rast, _ = peeler.rasterize_next_layer()
            apps = self.app_decoder(self.trilinear_query(mesh.vertices)) # [V, 3]
            colors, _ = dr.interpolate(apps[None], rast, indices) # [1, H, W, 3]
            alphas = (rast[..., -1:] > 0).float() # [1, H, W, 1]
            rgba = torch.cat((colors, alphas), dim=-1)
            if self.antialias:
                rgba = dr.antialias(rgba, rast, projected, indices)
            images.append(rgba.squeeze(0))
        return (
            RGBAImages(images),
            mesh,
            geom_reg,
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
    def as_module(self, *, field_name: Literal['grid']) -> Any:

        def parameters(self) -> Any:
            params = {
                'grid': self.feats,
            }[field_name]
            return [params]

        return parameters
