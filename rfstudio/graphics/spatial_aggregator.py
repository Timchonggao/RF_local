from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Protocol, Tuple, Type, TypeVar

import numpy as np
import torch
from jaxtyping import Bool, Float32, Int64
from torch import Tensor

from rfstudio.utils.tensor_dataclass import Float, Long, Size, TensorDataclass

from ._rays import Rays
from .math import safe_normalize

_BLUE_NOISE: Path = files('rfstudio') / 'assets' / 'geometry' / 'bluenoise3d.npz'

@lru_cache(maxsize=16)
def _cached_std_bluenoise(device: Optional[torch.device]) -> BlueNoise:
    return BlueNoise.from_file(device=device)

@dataclass
class BlueNoise(TensorDataclass):

    num_samples: int = Size.Dynamic
    num_levels: int = Size.Dynamic
    data: Tensor = Float[num_samples, 3]
    levels: Tensor = Long[num_levels]

    @classmethod
    def from_file(cls, *, file: Path = _BLUE_NOISE, device: Optional[torch.device] = None, scale: float = 1.0) -> BlueNoise:
        npz = np.load(file)
        data = []
        levels = []
        total = 0
        for key in npz:
            data.append(torch.from_numpy(npz[key]).float().to(device))
            total += data[-1].shape[0]
            levels.append(total)
        return BlueNoise(
            data=(torch.cat(data) - 0.5) * (2 * scale),
            levels=torch.tensor(levels, dtype=torch.long, device=device),
        )

    @classmethod
    def from_predefined(cls, *, device: Optional[torch.device] = None) -> BlueNoise:
        return _cached_std_bluenoise(device)

    def _aggregate_noise(self, num_samples: int) -> Float32[Tensor, "N 3"]:
        level = torch.searchsorted(self.levels, num_samples, side='left').item()
        if level >= self.num_levels:
            raise ValueError(f"Number of points must be less than {self.levels[-1].item()}")
        return self.data[:self.levels[level], :]

    def aggregate_randomized_targeted_noise(self, num_samples: int) -> Float32[Tensor, "N 3"]:
        samples = self._aggregate_noise(num_samples)
        random_points_indx = torch.randperm(samples.shape[0], device=self.device)[:num_samples]
        return samples[random_points_indx]

    def aggregate_scaled_targeted_noise(self, num_samples: int) -> Float32[Tensor, "N 3"]:
        samples = self._aggregate_noise(num_samples) # [S, 3]
        max_abs_coords = samples.abs().max(-1).values # [S]
        scaling_factors = 1.0 / max_abs_coords # [S]
        kth_factor = scaling_factors.kthvalue(scaling_factors.shape[0] - num_samples + 1).values
        return samples[scaling_factors >= kth_factor] * kth_factor


class SpatialAggregator(Protocol):

    @property
    def feature_dim(self) -> int: ...
    def aggregate(self, positions: Float32[Tensor, "... 3"], values: Float32[Tensor, "... C"]) -> None: ...
    def query(self, positions: Float32[Tensor, "... 3"]) -> Float32[Tensor, "... C"]: ...
    def reset(self) -> None: ...

    def sample(
        self,
        num_samples: int,
        *,
        replacement: bool = False,
        weights: Optional[Tensor] = None,
    ) -> Float32[Tensor, "N 3"]: ...


class _intersection(NamedTuple):
    positions: Float32[Tensor, "... 3R+1 3"]
    valid: Bool[Tensor, "... 3R+1 1"]
    values: Optional[Int64[Tensor, "... 3R feature_dim"]]

@dataclass
class NearestGrid(TensorDataclass):
    feature_dim: int = Size.Dynamic
    num_grids: int = Size.Dynamic
    values: Tensor = Float[num_grids, feature_dim]
    min_bound: Tensor = Float[3]
    max_bound: Tensor = Float[3]
    resolution: Tensor = Long[1]

    @classmethod
    def from_resolution(
        cls,
        resolution: int,
        *,
        feature_dim: int,
        center: Tuple[float, float, float] = (0., 0., 0.),
        size: float = 2.0,
        device: Optional[torch.device] = None,
    ) -> NearestGrid:
        center = torch.tensor(center, device=device, dtype=torch.float32)
        return NearestGrid(
            values=torch.zeros(resolution * resolution * resolution, feature_dim, device=device),
            min_bound=center - size / 2,
            max_bound=center + size / 2,
            resolution=torch.tensor([resolution], device=device, dtype=torch.long),
        )

    def get_grid_indices(self, positions: Float32[Tensor, "... 3"], *, eps: float = 1e-8) -> Int64[Tensor, "... 3"]:
        normalized = (positions - self.min_bound) / (self.max_bound - self.min_bound)
        return torch.where(
            (normalized - 1).abs() < eps,
            self.resolution - 1,
            (self.resolution * normalized).floor().long(),
        )

    def get_grid_centers(self, flatten_indices: Optional[Int64[Tensor, "..."]] = None) -> Float32[Tensor, "... 3"]:
        R = self.resolution.item()
        if flatten_indices is None:
            flatten_indices = torch.arange(self.num_grids, device=self.device).view(R, R, R)
        indices = torch.stack((
            flatten_indices % self.resolution,
            (flatten_indices % self.resolution.square()) // self.resolution,
            flatten_indices // self.resolution.square(),
        ), dim=-1) # [..., 3]
        return ((indices + 0.5) / R) * (self.max_bound - self.min_bound) + self.min_bound

    def aggregate(self, positions: Float32[Tensor, "... 3"], values: Float32[Tensor, "... C"]) -> None:
        # 假设你把三维空间切成了小立方体格子（像一个大魔方），现在你有很多粒子散布在这个魔方中，每个粒子带有一个数值（比如密度=1.0）。你想知道每个小立方体里“有多少密度被加进来了”，就可以用这个 aggregate 方法。
        R = self.resolution
        flatten_pos = positions.view(-1, 3) # [N, 3]
        flatten_val = values.view(-1, values.shape[-1]) # [N, C]
        indices = self.get_grid_indices(flatten_pos) # [N, 3]
        valid = (indices >= 0).all(-1) & (indices < R).all(-1) # [N]

        flatten_pos = flatten_pos[valid] # [N', 3]
        flatten_val = flatten_val[valid] # [N', C]
        indices = indices[valid] # [N', 3]
        flatten_indices = (indices * (R ** torch.arange(3, device=R.device))).sum(-1, keepdim=True) # [N', 1]
        self.values.scatter_add_(dim=0, index=flatten_indices.expand_as(flatten_val), src=flatten_val) # 将多个点的值累加到各自所属体素格子
        # 把 src 的值按 index 指定的 target 位置加到 target 中，沿着 dim 这个维度进行。
        # self.values 表示每个体素格子中存储的特征值或权重累积；flatten_indices 每个点的体素索引（展开为一维索引）；将 flatten_indices 从 [N', 1] 扩展为 [N', C]，让它的形状和 flatten_val 匹配
        # scatter_add_：高效：不用 for 循环，一次性完成加法；支持重复索引：同一个体素可能会被多个点命中。

    def query(self, positions: Float32[Tensor, "... 3"]) -> Float32[Tensor, "... C"]:
        # 查询一组位置在体素网格上的数值（也就是对应体素格子的值）；是 aggregate 的“读”操作，aggregate 往 voxel 网格里“加东西”，而 query 是根据位置从 voxel 网格里“查东西”。
        R = self.resolution
        indices = self.get_grid_indices(positions.view(-1, 3)) # [N, 3]
        valid = (indices.min() >= 0) & (indices.max() < R) # [1]
        assert valid.item(), "Out of range."
        flatten_indices = (indices * (R ** torch.arange(3, device=R.device))).sum(-1) # [N]
        return self.values[flatten_indices, :].reshape(*positions.shape[:-1], -1)

    def compute_gradients(
        self,
        positions: Float32[Tensor, "... 3"],
        *,
        fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> Float32[Tensor, "... 3"]:
        if fn is None:
            assert self.feature_dim == 1
        R = self.resolution
        indices = self.get_grid_indices(positions).unsqueeze(-2) # [..., 1, 3]
        valid = (indices.min() >= 0) & (indices.max() < R) # [1]
        assert valid.item(), "Out of range."
        offsets = torch.eye(3, dtype=indices.dtype, device=indices.device) # [3, 3]
        x_plus = (indices + offsets).clamp(0, R.item() - 1) # [..., 3, 3]
        x_minus = (indices - offsets).clamp(0, R.item() - 1) # [..., 3, 3]
        all_inds = torch.cat((x_plus, x_minus), dim=-2) # [..., 6, 3]
        flatten_all_inds = (all_inds * (R ** torch.arange(3, device=R.device))).sum(-1) # [..., 6]
        values = self.values[flatten_all_inds.flatten(), :].reshape(
            *flatten_all_inds.shape,
            self.feature_dim,
        ) # [..., 6, C]
        if fn is not None:
            values = fn(values)
        assert values.shape[-1] == 1
        values = values.squeeze(-1)
        dx = ((x_plus - x_minus) * ((self.max_bound - self.min_bound) / R)).sum(-1) # [..., 3]
        df = values[..., :3] - values[..., 3:] # [..., 3]
        return df / dx

    def intersect(self, rays: Rays, *, return_values: bool = False) -> _intersection:
        R: int = self.resolution.item()
        scale = self.max_bound - self.min_bound # [3]
        ray_origins = (rays.origins.view(-1, 3) - self.min_bound) / scale # [N, 3]
        ray_directions = safe_normalize(rays.directions).view(-1, 3) / scale # [N, 3]
        three_d_permutation = ray_directions.abs().argsort(-1, descending=True) # [N, 3]
        ray_origins = ray_origins.gather(dim=-1, index=three_d_permutation) # [N, 3]
        ray_directions = ray_directions.gather(dim=-1, index=three_d_permutation) # [N, 3]

        main_ts = (
            (torch.linspace(0, 1, R + 1, device=self.device) - ray_origins[:, 0:1]) /
            ray_directions[:, 0:1]
        ) # [N, R+1]
        positions = ray_origins[:, 1:, None] + ray_directions[:, 1:, None] * main_ts.unsqueeze(1) # [N, 2, R+1]
        indices = (positions * R).floor().long() # [N, 2, R+1]
        bounds = (indices[..., :-1] + indices[..., 1:] + 1) / (2 * R) # [N, 2, R+1]
        secondary_ts = (bounds - ray_origins[:, 1:, None]) / ray_directions[:, 1:, None] # [N, 2, R+1]
        secondary_ts = torch.where(indices[..., :-1] == indices[..., 1:], -1.0, secondary_ts) # [N, 2, R+1]
        all_ts = torch.cat((main_ts, secondary_ts.flatten(1, 2)), dim=-1) # [N, 3R+1]

        t_min = -ray_origins / ray_directions # [N, 3]
        t_max = (1 - ray_origins) / ray_directions # [N, 3]
        t_enter_max = torch.minimum(t_min, t_max).max(-1).values # [N]
        t_exit_min = torch.maximum(t_min, t_max).min(-1).values # [N]
        valid = (t_enter_max.unsqueeze(-1) <= all_ts) & (all_ts <= t_exit_min.unsqueeze(-1)) # [N, 3R+1]
        valid_ts = torch.where(valid, all_ts, torch.inf) # [N, 3R+1]
        valid_ts, ts_sorted_indices = valid_ts.sort(dim=-1) # [N, 3R+1]
        positions = (
            rays.origins.unsqueeze(1) +
            safe_normalize(rays.directions).unsqueeze(1) * valid_ts.view(*rays.shape, -1, 1)
        ) # [..., 3R+1, 3]
        return _intersection(
            positions=positions.clip(self.min_bound, self.max_bound),
            valid=valid.gather(dim=-1, index=ts_sorted_indices).view(*rays.shape, -1, 1),
            values=None,
        )

    def sample(
        self,
        num_samples: int,
        *,
        replacement: bool = True,
        weights: Optional[Tensor] = None,
        use_blue_noise: bool = True,
    ) -> Float32[Tensor, "N 3"]:
        # 从整个 3D 空间中，按重要性/密度分布进行采样，得到 num_samples 个 3D 点。
        R = self.resolution
        if weights is None:
            weights = self.values.new_ones(1, 1).expand(self.num_grids, 1) # 默认每个体素的采样概率是 1（均匀采样）
        else:
            weights = weights.clamp_min(1e-12) # 为了避免除以 0
        assert weights.shape == (self.num_grids, 1)
        flatten_indices = torch.multinomial(weights.flatten(), num_samples=num_samples, replacement=replacement) # [S] 根据 weights 概率采样出 num_samples 个体素索引
        flatten_indices, num_samples_per_grid = torch.unique(flatten_indices, return_counts=True) # [S'], [S']
        indices = torch.stack((
            flatten_indices % R,
            (flatten_indices % R.square()) // R,
            flatten_indices // R.square(),
        ), dim=-1) # [S', 3] 把一维体素索引解码成三维 [i, j, k] 索引，得到每个体素的空间坐标

        if not use_blue_noise:
            indices = indices.float()
            return (indices + torch.rand_like(indices)) * (self.max_bound - self.min_bound) + self.min_bound
        points = []
        blue_noise = BlueNoise.from_predefined(device=self.device)
        indices = indices + 0.5
        for i in range(indices.shape[0]):
            points.append(indices[i] + blue_noise.aggregate_scaled_targeted_noise(num_samples_per_grid[i]) * 0.5)
        points = torch.cat(points) / self.resolution
        return points * (self.max_bound - self.min_bound) + self.min_bound + torch.rand_like(points) * 1e-6

    def reset(self) -> None:
        self.values.zero_()


T = TypeVar('T', bound='HierarchicalNearestGrid')

@dataclass
class HierarchicalNearestGrid(TensorDataclass):
    feature_dim: int = Size.Dynamic
    num_nodes: int = Size.Dynamic
    depth: int = Size.Dynamic
    values: Tensor = Float[num_nodes, feature_dim]
    weights: Tensor = Float[num_nodes, 1]
    min_bound: Tensor = Float[3]
    max_bound: Tensor = Float[3]
    layer_bases: Tensor = Long[depth]

    '''
    layer L = [0, 1, ..., depth - 1]
    num of nodes = [1, 8, ..., 8 ^ (depth - 1)]
    '''

    @classmethod
    def from_depth(
        cls: Type[T],
        depth: int,
        *,
        feature_dim: int,
        center: Tuple[float, float, float] = (0., 0., 0.),
        size: float = 2.0,
        device: Optional[torch.device] = None,
    ) -> T:
        assert 0 < depth < 17
        center = torch.tensor(center, device=device, dtype=torch.float32)
        N = ((1 << (3 * depth)) - 1) // 7
        layer_sizes = 1 << (3 * torch.arange(depth, device=device))
        return cls(
            values=torch.zeros(N, feature_dim, device=device),
            weights=torch.zeros(N, 1, device=device),
            min_bound=center - size / 2,
            max_bound=center + size / 2,
            layer_bases=torch.cat((layer_sizes.new_zeros(1), layer_sizes.cumsum(0)[:-1]), dim=0),
        )

    def as_nearest_grid(self) -> NearestGrid:
        R = 1 << (self.depth - 1)
        flatten_indices = torch.arange(R ** 3, device=self.device).view(R, R, R)
        indices = torch.stack((
            flatten_indices % R,
            (flatten_indices % (R * R)) // R,
            flatten_indices // (R * R),
        ), dim=-1).view(-1, 3) # [RRR, 3]
        return NearestGrid(
            values=self.query_indices(indices),
            min_bound=-self.max_bound,
            max_bound=self.max_bound,
            resolution=torch.tensor([R]).to(indices),
        )

    def clamp_depth(self: T, depth_min: Optional[int], depth_max: Optional[int]) -> T:
        if depth_min is None:
            depth_min = 0
        if depth_max is None:
            depth_max = self.depth - 1
        if depth_min < 0:
            depth_min = self.depth + depth_min
        if depth_max < 0:
            depth_max = self.depth + depth_max
        assert 0 <= depth_max < self.depth
        assert 0 <= depth_min < self.depth
        weights = self.weights.clone()
        if depth_min > 0:
            weights[:self.layer_bases[depth_min]] = -torch.inf
        if depth_max + 1 < self.depth:
            weights[self.layer_bases[depth_max + 1]:] = -torch.inf
        return self.replace(weights=weights)

    def get_grid_indices(self, positions: Float32[Tensor, "... 3"], *, eps: float = 1e-8) -> Int64[Tensor, "... 3"]:
        normalized = (positions - self.min_bound) / (self.max_bound - self.min_bound)
        R = 1 << (self.depth - 1)
        return torch.where(
            (normalized - 1).abs() < eps,
            R - 1,
            (R * normalized).floor().long(),
        )

    def query(self, positions: Float32[Tensor, "... 3"]) -> Float32[Tensor, "... C"]:
        return self.query_indices(self.get_grid_indices(positions))

    def query_indices(self, indices: Int64[Tensor, "... 3"]) -> Float32[Tensor, "... C"]:
        D = self.depth
        R = 1 << (D - 1)
        shape = indices.shape[:-1]
        indices = indices.view(-1, 3) # [N, 3]
        valid = (indices.min() >= 0) & (indices.max() < R) # [1]
        assert valid.item(), "Out of range."
        depths = torch.arange(D, device=valid.device) # [D]
        h_indices = indices.unsqueeze(1) >> (D - 1 - depths.view(-1, 1)) # [N, D, 3]
        h_index_bases = 1 << (depths.view(-1, 1) * torch.arange(3, device=valid.device)) # [D, 3]
        flatten_indices = (h_indices * h_index_bases).sum(-1) + self.layer_bases # [N, D]
        values = self.values[flatten_indices.flatten(), :].reshape(*flatten_indices.shape, -1) # [N, D, C]
        weights = self.weights[flatten_indices.flatten(), :].reshape(*flatten_indices.shape, 1) # [N, D, 1]
        return (values * weights.softmax(1)).sum(1).view(*shape, -1) # [..., C]

    @torch.no_grad()
    def reset(self, *, reset_value: Optional[Tensor] = None) -> None:
        self.weights.zero_()
        self.values.zero_()
        if reset_value is None:
            return
        R = 1 << (self.depth - 1)
        flatten_indices = torch.arange(R ** 3, device=self.device).view(R, R, R)
        indices = torch.stack((
            flatten_indices % R,
            (flatten_indices % (R * R)) // R,
            flatten_indices // (R * R),
        ), dim=-1).view(-1, 3) # [RRR, 3]
        reset_value = reset_value.view(indices.shape[0], self.feature_dim) * self.depth # [RRR, C]
        depths = torch.arange(self.depth, device=indices.device) # [D]
        h_index_bases = 1 << (depths.view(-1, 1) * torch.arange(3, device=depths.device)) # [D, 3]
        for d in range(self.depth):
            local_indices = indices >> (self.depth - d - 1) # [RRR, 3]
            local_slice = (
                slice(self.layer_bases[d], self.layer_bases[d+1])
                if d + 1 < self.depth
                else slice(self.layer_bases[d], None)
            )
            scatter_indices = (local_indices * h_index_bases[d]).sum(-1, keepdim=True) # [RRR, 1]
            self.values[local_slice, :] = (
                self.values[local_slice, :].scatter_add(
                    dim=0,
                    index=scatter_indices.expand_as(reset_value),
                    src=reset_value,
                ) / torch.zeros_like(self.values[local_slice, :], dtype=torch.long).scatter_add(
                    dim=0,
                    index=scatter_indices.expand_as(reset_value),
                    src=torch.ones(1, dtype=torch.long, device=depths.device).expand_as(reset_value),
                ).clamp_min(1)
            )
            reset_value = reset_value - self.values[local_slice, :][scatter_indices.squeeze(-1), :]
        if torch.is_anomaly_enabled():
            assert reset_value.allclose(torch.zeros_like(reset_value))


@dataclass
class HierarchicalKernelGrid(HierarchicalNearestGrid):

    def query_indices(self, indices: Int64[Tensor, "... 3"]) -> Float32[Tensor, "... C"]:
        D = self.depth
        R = 1 << (D - 1)
        shape = indices.shape[:-1]
        indices = indices.view(-1, 3) # [N, 3]
        valid = (indices.min() >= 0) & (indices.max() < R) # [1]
        assert valid.item(), "Out of range."
        depths = torch.arange(D, device=valid.device) # [D]
        inverse_depths = (D - 1 - depths.view(-1, 1)) # [D]
        h_indices = indices.unsqueeze(1) >> inverse_depths # [N, D, 3]
        h_distances = indices.unsqueeze(1) - (h_indices << inverse_depths) # [N, D, 3]
        h_distances = (2 * h_distances + 1) / (1 << inverse_depths) - 1 # [N, D, 3] \in [0, 1]
        h_distances = (1 - h_distances.square() / 4).prod(-1, keepdim=True) # [N, D, 1] \in [0, 1]
        if torch.is_anomaly_enabled():
            assert h_distances.max() <= 1 and h_distances.min() >= 0
        h_index_bases = 1 << (depths.view(-1, 1) * torch.arange(3, device=valid.device)) # [D, 3]
        flatten_indices = (h_indices * h_index_bases).sum(-1) + self.layer_bases # [N, D]
        values = self.values[flatten_indices.flatten(), :].reshape(*flatten_indices.shape, -1) # [N, D, C]
        weights = self.weights[flatten_indices.flatten(), :].reshape(*flatten_indices.shape, 1) # [N, D, 1]
        weighted_values = values * weights.softmax(1) * h_distances # [N, D, C]
        return weighted_values.sum(1).view(*shape, -1) # [..., C]
        # weighted_indices = weighted_values.detach().abs().argmin(-2, keepdim=True) # [N, 1, C]
        # return weighted_values.gather(dim=-2, index=weighted_indices).view(*shape, -1) # [..., C]

    @torch.no_grad()
    def reset(self, *, reset_value: Optional[Tensor] = None) -> None:
        self.weights.zero_()
        self.values.zero_()
        if reset_value is None:
            return
        R = 1 << (self.depth - 1)
        flatten_indices = torch.arange(R ** 3, device=self.device).view(R, R, R)
        indices = torch.stack((
            flatten_indices % R,
            (flatten_indices % (R * R)) // R,
            flatten_indices // (R * R),
        ), dim=-1).view(-1, 3) # [RRR, 3]
        reset_value = reset_value.view(indices.shape[0], self.feature_dim) * self.depth # [RRR, C]
        depths = torch.arange(self.depth, device=indices.device) # [D]
        h_index_bases = 1 << (depths.view(-1, 1) * torch.arange(3, device=depths.device)) # [D, 3]
        for d in range(self.depth):
            local_indices = indices >> (self.depth - d - 1) # [RRR, 3]
            local_distances = indices - (local_indices << (self.depth - d - 1)) # [RRR, 3]
            local_distances = (2 * local_distances + 1) / (1 << (self.depth - d - 1)) - 1  # [RRR, 3] \in [0, 1]
            local_distances = (1 - local_distances.square() / 4).prod(-1, keepdim=True) # [RRR, 1] \in [0, 1]
            local_slice = (
                slice(self.layer_bases[d], self.layer_bases[d+1])
                if d + 1 < self.depth
                else slice(self.layer_bases[d], None)
            )
            scatter_indices = (local_indices * h_index_bases[d]).sum(-1, keepdim=True) # [RRR, 1]
            self.values[local_slice, :] = (
                self.values[local_slice, :].scatter_add(
                    dim=0,
                    index=scatter_indices.expand_as(reset_value),
                    src=reset_value,
                ) / torch.zeros_like(self.values[local_slice, :]).scatter_add(
                    dim=0,
                    index=scatter_indices.expand_as(reset_value),
                    src=local_distances.expand_as(reset_value),
                ).clamp_min(1e-6)
            )
            reset_value = reset_value - self.values[local_slice, :][scatter_indices.squeeze(-1), :] * local_distances
        if torch.is_anomaly_enabled():
            assert reset_value.allclose(torch.zeros_like(reset_value))

    def as_nearest_grid(self) -> NearestGrid:
        R = 1 << (self.depth - 1)
        flatten_indices = torch.arange(R ** 3, device=self.device).view(R, R, R)
        indices = torch.stack((
            flatten_indices % R,
            (flatten_indices % (R * R)) // R,
            flatten_indices // (R * R),
        ), dim=-1).view(-1, 3) # [RRR, 3]
        return NearestGrid(
            values=self.query_indices(indices),
            min_bound=-self.max_bound,
            max_bound=self.max_bound,
            resolution=torch.tensor([R]).to(indices),
        )
