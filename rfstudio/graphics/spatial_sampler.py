from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.nn import Module

from .spatial_aggregator import NearestGrid


class SpatialSampler(Protocol):

    def aggregate(
        self,
        positions: Float32[Tensor, "... 3"],
        *,
        importances: Optional[Float32[Tensor, "... 1"]] = None,
    ) -> None: ...

    def get_density(self, positions: Float32[Tensor, "... 3"]) -> Float32[Tensor, "... 1"]: ...
    def get_probability(self, positions: Float32[Tensor, "... 3"]) -> Float32[Tensor, "... 1"]: ...
    def sample(self, num_points: int) -> Float32[Tensor, "N 3"]: ...
    def reset(self) -> None: ...

@dataclass
class NearestGridSampler(Module):

    resolution: int = 64
    center: Tuple[float, float, float] = (0., 0., 0.)
    scale: float = 1.0

    def __setup__(self) -> None:
        self._grid = NearestGrid.from_resolution(
            self.resolution,
            feature_dim=1,
            center=self.center,
            size=2 * self.scale,
            device=self.device,
        )

    def aggregate(
        self,
        positions: Float32[Tensor, "... 3"],
        *,
        importances: Optional[Float32[Tensor, "... 1"]] = None,
    ) -> None:
        if importances is None:
            importances = torch.ones_like(positions[..., :1])
        self._grid.aggregate(positions, importances)

    def get_density(self, positions: Float32[Tensor, "... 3"]) -> Float32[Tensor, "... 1"]:
        return self._grid.query(positions)

    def get_probability(self, positions: Float32[Tensor, "... 3"]) -> Float32[Tensor, "... 1"]:
        return self._grid.query(positions) / self._grid.values.sum().clamp_min(1e-12)

    def sample(self, num_samples: int, *, replacement: bool = True) -> Float32[Tensor, "N 3"]:
        return self._grid.sample(num_samples, replacement=replacement, weights=self._grid.values)

    def reset(self) -> None:
        self._grid.reset()
        self._grid = self._grid.to(self.device)
