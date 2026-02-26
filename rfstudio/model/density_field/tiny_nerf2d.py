from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Float, Float32
from nerfacc import accumulate_along_rays
from torch import Tensor

from rfstudio.graphics import RGBAImages
from rfstudio.graphics._2d import Cameras2D, Rays2D, RaySamples2D, RGBA2DImages
from rfstudio.nn import MLP, Module

from .components.encoding import PosEncoding


@dataclass
class MLPField2D(Module):

    position_encoding: PosEncoding = PosEncoding(num_frequencies=10, max_freq_exp=8.0)
    base_mlp: MLP = MLP(layers=[-1, 256, 256, 256], activation='relu')
    head_mlp: MLP = MLP(layers=[-1, 128, 128], activation='relu')

    def __setup__(self) -> None:
        self.color_head = MLP(layers=[-1, 3], activation='sigmoid')
        self.density_head = MLP(layers=[-1, 1], activation='softplus')

    def get_densities(self, ray_samples: RaySamples2D) -> RaySamples2D:
        encoded_pos = self.position_encoding(ray_samples.positions)    # [..., S, E]
        return ray_samples.annotate(densities=self.density_head(self.base_mlp(encoded_pos)))

    def get_outputs(self, ray_samples: RaySamples2D) -> RaySamples2D:
        encoded_pos = self.position_encoding(ray_samples.positions)    # [..., S, E]
        density_embedding = self.base_mlp(encoded_pos)                 # [..., S, D]
        density = self.density_head(density_embedding)                 # [..., S, 1]
        color = self.color_head(self.head_mlp(density_embedding))      # [..., S, 3]
        return ray_samples.annotate(colors=color, densities=density).get_weighted()

    def get_rgbd(self, positions: Float32[Tensor, "... 3"]) -> Float32[Tensor, "... 4"]:
        encoded_pos = self.position_encoding(positions)                # [..., S, E]
        density_embedding = self.base_mlp(encoded_pos)                 # [..., S, D]
        density = self.density_head(density_embedding)                 # [..., S, 1]
        color = self.color_head(self.head_mlp(density_embedding))      # [..., S, 3]
        return torch.cat((color, density), dim=-1)

@dataclass
class UniformSampler2D(Module):

    num_samples_per_ray: int

    def __call__(self, inputs: Rays2D) -> RaySamples2D:
        ts = torch.linspace(0, 1, self.num_samples_per_ray + 1, device=self.device)
        if self.training:
            rand = torch.rand(*inputs.shape, self.num_samples_per_ray + 1, device=self.device)
            centers = (ts[..., 1:] + ts[..., :-1]) * 0.5
            ub = torch.cat((centers, ts[..., -1:]), dim=-1)
            lb = torch.cat((ts[..., :1], centers), dim=-1)
            ts = lb + (ub - lb) * rand
        else:
            ts = ts.expand(*inputs.shape, self.num_samples_per_ray + 1)
        return inputs.get_samples(t=ts)

@dataclass
class VolumetricRenderer2D:

    def get_color(self, ray_samples: RaySamples2D) -> Float[Tensor, "B... 3"]:
        assert ray_samples.weights is not None
        assert ray_samples.colors is not None
        return accumulate_along_rays(
            weights=ray_samples.weights.view(-1, ray_samples.num_samples),
            values=ray_samples.colors.view(-1, ray_samples.num_samples, 3),
        ).view(*ray_samples.shape, 3) # -> [..., 3]

    def get_accumulation(self, ray_samples: RaySamples2D) -> Float[Tensor, "B... 1"]:
        assert ray_samples.weights is not None
        return accumulate_along_rays(
            weights=ray_samples.weights.view(-1, ray_samples.num_samples),
            values=None,
        ).view(*ray_samples.shape, 1) # -> [..., 1]

@dataclass
class TinyNeRF2D(Module):

    field: MLPField2D = MLPField2D()

    sampler: UniformSampler2D = UniformSampler2D(num_samples_per_ray=64)

    renderer: VolumetricRenderer2D = VolumetricRenderer2D()

    def render_rgb_along_rays(self, rays: Rays2D) -> Float[Tensor, "*B 4"]:
        fine_outputs = self.field.get_outputs(self.sampler(rays))
        rgb = self.renderer.get_color(fine_outputs)
        acc = self.renderer.get_accumulation(fine_outputs)
        return torch.cat((rgb, acc), dim=-1)

    def render_rgba(self, inputs: Cameras2D) -> RGBA2DImages:
        return RGBA2DImages(self.render_rgb_along_rays(inputs.view(-1).generate_rays()))

    def visualize(self, *, width: int, height: int, scale: float = 1.) -> RGBAImages:
        xs = torch.linspace(-scale, scale, width, device=self.device) # [W]
        ys = torch.linspace(-scale, scale, height, device=self.device) # [H]
        Ys, Xs = torch.meshgrid(ys, xs, indexing='ij') # [H, W]
        queries = torch.stack((Xs, Ys), dim=-1).flip(0) # [H, W, 2]
        rgbd = self.field.get_rgbd(queries) # [H, W, 4]
        alphas = 1 - (rgbd[..., 3:] * -10).exp()
        rgba = torch.cat((rgbd[..., :3], alphas), dim=-1) # [H, W, 4]
        return RGBAImages([rgba])
