from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from jaxtyping import Float32, Int32
from torch import Tensor

from rfstudio.utils.lazy_module import o3d
from rfstudio.utils.tensor_dataclass import Bool, Float, Size, TensorDataclass

Self = TypeVar('Self', bound='Points')


@dataclass
class Points(TensorDataclass):

    positions: Tensor = Float[..., 3]

    colors: Optional[Tensor] = Float[..., 3]

    normals: Optional[Tensor] = Float[..., 3]

    def normalize(self, *, scale: float = 1.0) -> Points:
        max_bound = self.positions.view(-1, 3).max(0).values
        min_bound = self.positions.view(-1, 3).min(0).values
        assert (max_bound > min_bound).all()
        center = (max_bound + min_bound) / 2
        scale = 2 * scale / (max_bound - min_bound).max()
        return self.replace(positions=(self.positions - center) * scale)

    @torch.no_grad()
    def perturb_to_avoid_duplication_(self, *, threshold: float = 1e-7, sigma: float = 1e-3) -> Points:
        pts = self.positions.clone().view(-1, 3) # [N, 3]
        sorted_indices = pts[:, 0].argsort() # [N]
        points_sorted = pts[sorted_indices] # [N', 3]

        diffs = points_sorted[1:] - points_sorted[:-1] # [N'-1, 3]
        distances = diffs.norm(dim=-1) # [N'-1]

        close_mask = distances < threshold # [N'-1]
        duplicate_indices_1 = sorted_indices[:-1][close_mask]
        duplicate_indices_2 = sorted_indices[1:][close_mask]
        duplicate_indices = torch.cat((duplicate_indices_1, duplicate_indices_2)) # [D]
        if duplicate_indices.shape[0] == 0:
            return self
        pts[duplicate_indices] += torch.randn_like(pts[duplicate_indices]) * (sigma ** 2)
        return self.replace_(positions=pts.view(*self.shape, 3))

    @torch.no_grad()
    def k_nearest(self, *, k: int) -> Tuple[Float32[Tensor, "B... K"], Int32[Tensor, "B... K"]]:

        from sklearn.neighbors import NearestNeighbors

        points_np = self.positions.detach().cpu().view(-1, 3).numpy()
        model = NearestNeighbors(
            n_neighbors=k + 1,
            algorithm="auto",
            metric="euclidean",
        ).fit(points_np)
        distances, indices = model.kneighbors(points_np)
        distances = torch.from_numpy(distances[:, 1:]).view(*self.shape, k)
        indices = torch.from_numpy(indices[:, 1:]).int().view(*self.shape, k)
        return distances.to(self.positions), indices.to(self.device)

    @torch.no_grad()
    def k_nearest_in(
        self,
        queries: Float32[Tensor, "B... 3"],
        *,
        k: int,
    ) -> Tuple[Float32[Tensor, "B... K"], Int32[Tensor, "B... K"]]:

        from sklearn.neighbors import NearestNeighbors

        points_np = self.positions.detach().cpu().view(-1, 3).numpy()
        others_np = queries.detach().cpu().view(-1, 3).numpy()
        model = NearestNeighbors(
            n_neighbors=k,
            algorithm="auto",
            metric="euclidean",
        ).fit(others_np)
        distances, indices = model.kneighbors(points_np)
        distances = torch.from_numpy(distances).view(*self.shape, k)
        indices = torch.from_numpy(indices).int().view(*self.shape, k)
        return distances.to(self.positions), indices.to(self.device)

    @classmethod
    def rand(
        cls,
        size: Union[int, Tuple[int, ...]],
        *,
        device: Optional[torch.device] = None,
    ) -> Points:
        if not isinstance(size, (torch.Size, tuple)):
            size = (size, )
        return Points(positions=torch.rand((*size, 3), device=device))

    @classmethod
    def randn(
        cls,
        size: Union[int, Tuple[int, ...]],
        *,
        device: Optional[torch.device] = None,
    ) -> Points:
        if not isinstance(size, (torch.Size, tuple)):
            size = (size, )
        return Points(positions=torch.randn((*size, 3), device=device))

    def translate(self: Self, offset: Any) -> Self:
        return self.replace(positions=self.positions + offset)

    def scale(self: Self, scalings: Any) -> Self:
        return self.replace(positions=self.positions * scalings)

    @torch.no_grad()
    def export(self, path: Path) -> None:
        assert path.suffix in ['.obj', '.ply']
        path.parent.mkdir(exist_ok=True, parents=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.positions.detach().cpu().numpy())
        if self.colors is not None:
            colors = self.colors.clamp(0, 1)
            pcd.colors = o3d.utility.Vector3dVector(colors.detach().cpu().numpy())
        if self.normals is not None:
            normals = self.normals / self.normals.norm(dim=-1, keepdim=True)
            pcd.normals = o3d.utility.Vector3dVector(normals.detach().cpu().numpy())
        o3d.io.write_point_cloud(str(path), pcd)

    @classmethod
    def from_file(cls, path: Path) -> Points:
        assert path.exists()
        pcd = o3d.io.read_point_cloud(str(path))
        points = Points(positions=torch.from_numpy(np.asarray(pcd.points)).float())
        if pcd.has_normals and len(pcd.normals) > 0:
            points.annotate_(normals=torch.from_numpy(np.asarray(pcd.normals)).float())
        if pcd.has_colors and len(pcd.colors) > 0:
            points.annotate_(colors=torch.from_numpy(np.asarray(pcd.colors)).float())
        return points

    @torch.no_grad()
    def farthest_distance_sample(self, num_samples: int) -> Points:
        if self.colors is not None or self.normals is not None:
            raise NotImplementedError
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.positions.cpu().numpy())
        downsampled = pcd.farthest_point_down_sample(num_samples)
        return Points(positions=torch.from_numpy(np.asarray(downsampled.points)).to(self.positions))


@dataclass
class SfMPoints(TensorDataclass):

    num_images: int = Size.Dynamic

    positions: Tensor = Float[..., 3]

    colors: Tensor = Float[..., 3]

    visibilities: Tensor = Bool[..., num_images]

    def as_points(self) -> Points:
        return Points(positions=self.positions, colors=self.colors)

    def seen_by(self, image_indices: Int32[Tensor, "*bs"]) -> SfMPoints:
        indices = image_indices.view(-1)       # [S]
        valid = (self.visibilities[..., indices]).any(-1)
        return self[valid]
