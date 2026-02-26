from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from jaxtyping import Float
from torch import Tensor

from rfstudio.graphics import Cameras, IsoCubes, Rays, RGBImages, TriangleMesh
from rfstudio.nn import Module
from rfstudio.utils.decorator import chunkify

from .components.encoding import PosEncoding
from .components.field import MLPField
from .components.renderer import VolumetricRenderer
from .components.sampler import NeuSSampler, UniformSampler
from .components.sdf_field import SDFField


@dataclass
class VanillaNeuS(Module):

    chunk_size: int = 4096

    sdf_field: SDFField = SDFField()
    sampler: NeuSSampler = NeuSSampler(num_samples_per_ray=64, num_importance_samples_per_ray=64)
    background: Literal['mlp', 'none'] = 'mlp'
    background_distance: float = 4.0
    renderer: VolumetricRenderer = VolumetricRenderer()

    def __setup__(self) -> None:
        self.cos_anneal_ratio = 1.0
        self.bg_field = MLPField(
            position_encoding=PosEncoding(num_frequencies=10, max_freq_exp=9),
            direction_encoding=PosEncoding(num_frequencies=4, max_freq_exp=3),
        )
        self.bg_sampler = UniformSampler(num_samples_per_ray=32, spacing='linear_disparity')

    @chunkify(prop='chunk_size')
    def render_rgb_along_rays(self, rays: Rays) -> Tuple[Float[Tensor, "*B 3"], Float[Tensor, "1"]]:
        if self.background == 'mlp':
            inside_rays = rays.replace(far=torch.empty_like(rays.far).fill_(self.background_distance))
            outside_rays = rays.replace(near=torch.empty_like(rays.far).fill_(self.background_distance))
        else:
            inside_rays = rays
            outside_rays = None
        samples = self.sampler(inside_rays, sdf_fn=self.sdf_field.get_sdf)
        outputs, eikonal_loss = self.sdf_field.get_outputs(samples, cos_anneal_ratio=self.cos_anneal_ratio)
        rgb = self.renderer.get_color(outputs)
        if self.background == 'none':
            return rgb, eikonal_loss
        if self.background == 'mlp':
            acc = self.renderer.get_accumulation(outputs)
            bg_samples = self.bg_sampler(outside_rays)
            bg_outputs = self.bg_field.get_outputs(bg_samples)
            bg_rgb = self.renderer.get_color(bg_outputs)
            return rgb + (1 - acc) * bg_rgb, eikonal_loss
        raise ValueError(self.background)

    def render_rgb(self, inputs: Cameras) -> RGBImages:
        return RGBImages([self.render_rgb_along_rays(camera.generate_rays())[0] for camera in inputs.view(-1)])

    @torch.no_grad()
    def extract_mesh(self, *, roi_scale: float = 1.0, resolution: int = 128) -> TriangleMesh:
        isocubes = IsoCubes.from_resolution(resolution, device=self.device, random_sdf=False, scale=roi_scale)
        sdf_values = chunkify(SDFField.get_sdf, chunk_size=self.chunk_size * 4)(
            self.sdf_field,
            isocubes.vertices.view(-1, 3),
        )
        isocubes.replace_(sdf_values=sdf_values.view_as(isocubes.sdf_values))
        return isocubes.marching_cubes()
