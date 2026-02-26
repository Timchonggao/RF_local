from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor, nn

from rfstudio.graphics import (
    Cameras,
    FeatureImages,
    Points,
    TriangleMesh,
)
from rfstudio.graphics.math import spectral_clustering
from rfstudio.model.density_field.components.encoding import HashEncoding
from rfstudio.nn import MLP, Module
from rfstudio.utils.colormap import RainbowColorMap
from rfstudio.utils.lazy_module import dr


@dataclass
class FeatureMesh(Module):

    antialias: bool = True

    mlp_texture: HashEncoding = HashEncoding(
        mlp=MLP(
            layers=[32, 32, 6],
            activation='none',
            bias=False,
            initialization='kaiming-uniform',
        ),
        max_res=4096,
        grad_scaling=16.0,
    )

    gt_mesh: Path = ...

    def __setup__(self) -> None:
        mesh = TriangleMesh.from_file(self.gt_mesh)
        self.vertices = nn.Parameter(mesh.vertices, requires_grad=False)
        self.indices = nn.Parameter(mesh.indices, requires_grad=False)

    def render_report(self, inputs: Cameras) -> Tuple[FeatureImages, TriangleMesh]:
        mesh = TriangleMesh(vertices=self.vertices, indices=self.indices)
        assert inputs.is_cuda
        inputs = inputs.view(-1)

        ctx = dr.RasterizeCudaContext(inputs.device)

        vertices = torch.cat((
            mesh.vertices,
            torch.ones_like(mesh.vertices[..., :1]),
        ), dim=-1).view(-1, 4, 1)                          # [V, 4, 1]
        scale = mesh.vertices.abs().max(0).values

        images = []
        for camera in inputs:
            resolution = [camera.height.item(), camera.width.item()]
            mvp = camera.projection_matrix @ camera.view_matrix                    # [4, 4]
            projected = (mvp @ vertices).view(1, -1, 4)                            # [1, V, 4]
            indices = mesh.indices.int()
            with dr.DepthPeeler(ctx, projected, indices, resolution=resolution) as peeler:
                rast, _ = peeler.rasterize_next_layer()
            frag_pos, _ = dr.interpolate(mesh.vertices[None], rast, indices) # [1, H, W, 3]
            feats = self.mlp_texture((frag_pos / scale).clamp(-1, 1)) # [1, H, W, 6]
            if self.antialias:
                feats = dr.antialias(feats, rast, projected, indices)
            images.append(torch.where(rast[..., -1:] > 0, feats, 0).squeeze(0))
        return (FeatureImages(images), mesh)

    @torch.no_grad()
    def get_segmented_mesh(self, num_clusters: int) -> Tuple[TriangleMesh, Tensor]:
        scale = self.vertices.abs().max(0).values
        mesh = TriangleMesh(vertices=self.vertices, indices=self.indices)
        vertex_features = self.mlp_texture(mesh.vertices / scale) # [V, 6]
        indices = spectral_clustering(
            vertex_features,
            downsample_to=1024,
            dim=-1,
            num_clusters=num_clusters,
        ).indices
        colors = RainbowColorMap()(indices / (num_clusters - 1))
        return mesh, colors

    @torch.no_grad()
    def get_segmented_points(self, num_clusters: int, *, num_samples: int = 8192) -> Points:
        scale = self.vertices.abs().max(0).values
        pts = TriangleMesh(vertices=self.vertices, indices=self.indices).uniformly_sample(num_samples, samples_per_face='uniform')
        vertex_features = self.mlp_texture(pts.positions / scale) # [V, 6]
        indices = spectral_clustering(
            vertex_features,
            downsample_to=1024,
            dim=-1,
            num_clusters=num_clusters,
        ).indices
        colors = RainbowColorMap()(indices / (num_clusters - 1))
        return pts.annotate(colors=colors)
