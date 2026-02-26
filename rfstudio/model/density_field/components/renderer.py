from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float
from nerfacc import accumulate_along_rays
from torch import Tensor

from rfstudio.graphics import RaySamples


@dataclass
class VolumetricRenderer:

    def get_color(self, ray_samples: RaySamples) -> Float[Tensor, "B... 3"]:
        assert ray_samples.weights is not None
        assert ray_samples.colors is not None
        return accumulate_along_rays(
            weights=ray_samples.weights.view(-1, ray_samples.num_samples),
            values=ray_samples.colors.view(-1, ray_samples.num_samples, 3),
        ).view(*ray_samples.shape, 3) # -> [..., 3]

    def get_accumulation(self, ray_samples: RaySamples) -> Float[Tensor, "B... 1"]:
        assert ray_samples.weights is not None
        return accumulate_along_rays(
            weights=ray_samples.weights.view(-1, ray_samples.num_samples),
            values=None,
        ).view(*ray_samples.shape, 1) # -> [..., 1]
