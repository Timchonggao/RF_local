from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Optional, Protocol, Tuple

import numpy as np
import torch
from jaxtyping import Float32, Int64
from torch import Tensor

from rfstudio.utils.tensor_dataclass import Float, Long, Size, TensorDataclass

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


@dataclass
class NearestGrid(TensorDataclass):
    feature_dim: int = Size.Dynamic
    num_grids: int = Size.Dynamic
    values: Tensor = Float[num_grids, feature_dim]
    counts: Tensor = Float[num_grids, 1]
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
            counts=torch.zeros(resolution * resolution * resolution, 1, device=device),
            min_bound=center - size / 2,
            max_bound=center + size / 2,
            resolution=torch.tensor([resolution], device=device, dtype=torch.long)
        )

    def get_voxel_indices(self, positions: Float32[Tensor, "... 3"]) -> Int64[Tensor, "... 3"]:
        # 三维空间被分成了一个个小立方体（体素 voxel），这个函数计算某个点落在哪个立方体里。
        # 使用 floor() 将浮点数坐标向下取整，得到合法的索引位置。转为 long 类型（PyTorch 中的 int64），以便用于索引张量。
        return (self.resolution * (positions - self.min_bound) / (self.max_bound - self.min_bound)).floor().long()

    def aggregate(self, positions: Float32[Tensor, "... 3"], values: Float32[Tensor, "... C"], average: bool = False) -> None:
        # 假设你把三维空间切成了小立方体格子（像一个大魔方），现在你有很多粒子散布在这个魔方中，每个粒子带有一个数值（比如密度=1.0）。你想知道每个小立方体里“有多少密度被加进来了”，就可以用这个 aggregate 方法。
        R = self.resolution
        flatten_pos = positions.view(-1, 3) # [N, 3]
        flatten_val = values.view(-1, values.shape[-1]) # [N, C]
        indices = self.get_voxel_indices(flatten_pos) # [N, 3]
        valid = (indices >= 0).all(-1) & (indices < R).all(-1) # [N]

        flatten_pos = flatten_pos[valid] # [N', 3]
        flatten_val = flatten_val[valid] # [N', C]
        indices = indices[valid] # [N', 3]
        flatten_indices = (indices * (R ** torch.arange(3, device=R.device))).sum(-1, keepdim=True) # [N', 1] 把3D index 映射成1D，用于 scatter_add，注意这里的计算是默认x first的
        self.values.scatter_add_(dim=0, index=flatten_indices.expand_as(flatten_val), src=flatten_val) # 将多个点的值累加到各自所属体素格子
        # 把 src 的值按 index 指定的 target 位置加到 target 中，沿着 dim 这个维度进行。
        # self.values 表示每个体素格子中存储的特征值或权重累积；flatten_indices 每个点的体素索引（展开为一维索引）；将 flatten_indices 从 [N', 1] 扩展为 [N', C]，让它的形状和 flatten_val 匹配
        # scatter_add_：高效：不用 for 循环，一次性完成加法；支持重复索引：同一个体素可能会被多个点命中。

        if average:
            ones = torch.ones_like(flatten_val[:, :1])  # [N', 1] 每个点计数1
            self.counts.scatter_add_(dim=0, index=flatten_indices, src=ones)
            # 避免除以0
            valid_counts = self.counts.clamp(min=1.0)
            average_values = self.values / valid_counts  # 逐 voxel 平均 

             # ====== 识别异常 voxel ======
            count_threshold = 50
            low_count = self.counts.squeeze(-1) <= count_threshold  # 计数少的 voxel 往往是一些异常值，比如mesh内部的position，出现的概率很低，但是error value很大。
            average_values[low_count] = 0.0

            # 替换
            self.replace_(values=average_values, counts=valid_counts)

    def query(self, positions: Float32[Tensor, "... 3"]) -> Float32[Tensor, "... C"]:
        # 查询一组位置在体素网格上的数值（也就是对应体素格子的值）；aggregate 往 voxel 网格里“加东西”，而 query 是根据位置从 voxel 网格里“查东西”。
        R = self.resolution
        indices = self.get_voxel_indices(positions.view(-1, 3)) # [N, 3]
        valid = (indices >= 0).all(-1) & (indices < R).all(-1) # [N]
        assert valid.all().item(), "Out of range."
        flatten_indices = (indices * (R ** torch.arange(3, device=R.device))).sum(-1) # [N]
        return self.values[flatten_indices, :].reshape(*positions.shape[:-1], -1)

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
