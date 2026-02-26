from __future__ import annotations

from dataclasses import dataclass

import torch

from rfstudio.graphics import RaySamples
from rfstudio.nn import MLP, Module

from .encoding import PosEncoding


@dataclass
class MLPField(Module):

    """NeRF Field"""

    position_encoding: PosEncoding = PosEncoding(num_frequencies=10, max_freq_exp=8.0)
    direction_encoding: PosEncoding = PosEncoding(num_frequencies=4, max_freq_exp=4.0)
    base_mlp: MLP = MLP(layers=[-1, 256, 256, 256, 256, 256, 256, 256, 256], skip_connections=[4], activation='relu')
    head_mlp: MLP = MLP(layers=[-1, 128, 128], activation='relu')

    def __setup__(self) -> None:
        self.color_head = MLP(layers=[-1, 3], activation='sigmoid')
        self.density_head = MLP(layers=[-1, 1], activation='softplus')

    def get_densities(self, ray_samples: RaySamples) -> RaySamples:
        encoded_pos = self.position_encoding(ray_samples.positions)    # [..., S, E]
        return ray_samples.annotate(densities=self.density_head(self.base_mlp(encoded_pos)))

    def get_outputs(self, ray_samples: RaySamples) -> RaySamples:
        encoded_pos = self.position_encoding(ray_samples.positions)    # [..., S, E]
        encoded_dir = self.direction_encoding(
            ray_samples.directions /
            ray_samples.directions.norm(dim=-1, keepdim=True)
        ) # [..., E]
        density_embedding = self.base_mlp(encoded_pos)                 # [..., S, D]
        density = self.density_head(density_embedding)                 # [..., S, 1]
        encoded_dir = encoded_dir[..., None, :].expand(
            *encoded_dir.shape[:-1],
            density.shape[-2],
            encoded_dir.shape[-1],
        )                                                              # [..., S, E]
        color = self.color_head(self.head_mlp(torch.cat((
            encoded_dir,
            density_embedding,
        ), dim=-1)))                                                   # [..., S, 3]
        return ray_samples.annotate(colors=color, densities=density).get_weighted()
