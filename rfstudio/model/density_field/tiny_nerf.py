from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float
from torch import Tensor

from rfstudio.graphics import Cameras, Rays, RGBImages
from rfstudio.nn import Module
from rfstudio.utils.decorator import chunkify

from .components.field import MLPField
from .components.renderer import VolumetricRenderer
from .components.sampler import UniformSampler


@dataclass
class TinyNeRF(Module):

    chunk_size: int = 32768

    background_color: Literal['black', 'white'] = 'black'

    field: MLPField = MLPField()

    sampler: UniformSampler = UniformSampler(num_samples_per_ray=64)

    renderer: VolumetricRenderer = VolumetricRenderer()

    def get_background_color(self) -> Float[Tensor, "3 "]:
        return torch.zeros(3) if self.background_color == 'black' else torch.ones(3)

    @chunkify(prop='chunk_size')
    def render_rgb_along_rays(self, rays: Rays) -> Float[Tensor, "*B 3"]:
        fine_outputs = self.field.get_outputs(self.sampler(rays))
        rgb = self.renderer.get_color(fine_outputs)
        acc = self.renderer.get_accumulation(fine_outputs)
        return rgb + (1 - acc) * self.get_background_color().to(rgb)

    def render_rgb(self, inputs: Cameras) -> RGBImages:
        return RGBImages([self.render_rgb_along_rays(camera.generate_rays()) for camera in inputs.view(-1)])
