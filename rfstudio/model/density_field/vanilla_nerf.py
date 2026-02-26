from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from jaxtyping import Float
from torch import Tensor

from rfstudio.graphics import Cameras, Rays, RGBImages
from rfstudio.nn import Module
from rfstudio.utils.decorator import chunkify

from .components.field import MLPField
from .components.renderer import VolumetricRenderer
from .components.sampler import PDFSampler, UniformSampler


@dataclass
class VanillaNeRF(Module):

    chunk_size: int = 32768

    background_color: Literal['black', 'white'] = 'black'

    coarse_field: MLPField = MLPField()
    fine_field: MLPField = MLPField()
    coarse_sampler: UniformSampler = UniformSampler(num_samples_per_ray=64)
    fine_sampler: PDFSampler = PDFSampler(num_samples_per_ray=64)
    renderer: VolumetricRenderer = VolumetricRenderer()

    def get_background_color(self) -> Float[Tensor, "3 "]:
        return torch.zeros(3) if self.background_color == 'black' else torch.ones(3)

    @chunkify(prop='chunk_size')
    def render_rgb_along_rays(self, rays: Rays) -> Tuple[Float[Tensor, "*B 3"], Float[Tensor, "*B 3"]]:
        coarse_outputs = self.coarse_field.get_outputs(self.coarse_sampler(rays))
        fine_outputs = self.fine_field.get_outputs(self.fine_sampler(rays, coarse_samples=coarse_outputs))
        coarse_rgb = self.renderer.get_color(coarse_outputs)
        coarse_acc = self.renderer.get_accumulation(coarse_outputs)
        fine_rgb = self.renderer.get_color(fine_outputs)
        fine_acc = self.renderer.get_accumulation(fine_outputs)
        return (
            coarse_rgb + (1 - coarse_acc) * self.get_background_color().to(coarse_rgb),
            fine_rgb + (1 - fine_acc) * self.get_background_color().to(fine_rgb),
        )

    def render_rgb(self, inputs: Cameras) -> RGBImages:
        return RGBImages([self.render_rgb_along_rays(camera.generate_rays())[1] for camera in inputs.view(-1)])
