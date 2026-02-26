from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional, Type, TypeVar

import numpy as np
import torch
import torch.utils.dlpack
from jaxtyping import Float32, Int32
from torch import Tensor

from rfstudio.utils.colormap import RainbowColorMap
from rfstudio.utils.decorator import chains
from rfstudio.utils.lazy_module import o3d
from rfstudio.utils.tensor_dataclass import Float, Size, TensorDataclass

from ._mesh import TriangleMesh
from ._points import Points
from .math import (
    get_random_quaternion,
    principal_component_analysis,
    quat2rot,
    rgb2sh,
    sh2rgb,
    sh_deg2dim,
    sh_dim2deg,
    spectral_clustering,
)
from .spatial_aggregator import NearestGrid

T = TypeVar('T', bound='Splats')
U = TypeVar('U', bound='FeatureSplats')

@dataclass
class Splats(TensorDataclass):

    sh_dim: int = Size.Dynamic

    means: torch.Tensor = Float.Trainable[..., 3]
    scales: torch.Tensor = Float.Trainable[..., 3]
    quats: torch.Tensor = Float.Trainable[..., 4]
    colors: torch.Tensor = Float.Trainable[..., 3]
    shs: torch.Tensor = Float.Trainable[..., sh_dim, 3]
    opacities: torch.Tensor = Float.Trainable[..., 1]

    last_width: Optional[torch.Tensor] = Float[1]
    last_height: Optional[torch.Tensor] = Float[1]
    xys_grad_norm: Optional[torch.Tensor] = Float[...]
    vis_counts: Optional[torch.Tensor] = Float[...]

    @property
    def sh_degree(self) -> Literal[0, 1, 2, 3, 4]:
        return sh_dim2deg(self.sh_dim + 1)

    @classmethod
    def from_points(
        cls: Type[T],
        points: Points,
        *,
        sh_degree: int,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> T:
        assert points.colors is not None
        points = points.flatten()
        distances = points.k_nearest(k=3)[0]   # [N, K]
        size = distances.shape[0]

        return cls(
            means=points.positions.clone().to(device),
            scales=distances.mean(dim=-1, keepdim=True).log().repeat(1, 3),
            quats=get_random_quaternion(size, device=device),
            opacities=torch.logit(0.1 * torch.ones((size, 1), device=device)),
            colors=points.colors.clone().to(device),
            shs=torch.zeros((size, sh_deg2dim(sh_degree) - 1, 3), device=device)
        ).requires_grad_(requires_grad)

    @classmethod
    def random(
        cls: Type[T],
        size: int,
        *,
        sh_degree: int,
        random_scale: float,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
    ) -> T:
        points = Points.rand((size, ), device=device).translate(-0.5).scale(2 * random_scale)
        distances = points.k_nearest(k=3)[0]   # [N, K]

        return cls(
            means=points.positions,
            scales=distances.mean(dim=-1, keepdim=True).log().repeat(1, 3),
            quats=get_random_quaternion(size, device=device),
            opacities=torch.logit(0.1 * torch.ones((size, 1), device=device)),
            colors=(torch.ones((size, 3), device=device) * 0.5),
            shs=torch.zeros((size, sh_deg2dim(sh_degree) - 1, 3), device=device)
        ).requires_grad_(requires_grad)

    @classmethod
    def from_file(cls: Type[T], path: Path, *, device: Optional[torch.device] = None) -> T:
        pcd = o3d.t.io.read_point_cloud(str(path)).point
        means = torch.utils.dlpack.from_dlpack(pcd.positions.to_dlpack()).to(device).float()
        opacities = torch.utils.dlpack.from_dlpack(pcd.opacity.to_dlpack()).to(device).float()
        scales = torch.cat((
            torch.utils.dlpack.from_dlpack(pcd.scale_0.to_dlpack()),
            torch.utils.dlpack.from_dlpack(pcd.scale_1.to_dlpack()),
            torch.utils.dlpack.from_dlpack(pcd.scale_2.to_dlpack()),
        ), dim=-1).to(device).float()
        quats = torch.cat((
            torch.utils.dlpack.from_dlpack(pcd.rot_0.to_dlpack()),
            torch.utils.dlpack.from_dlpack(pcd.rot_1.to_dlpack()),
            torch.utils.dlpack.from_dlpack(pcd.rot_2.to_dlpack()),
            torch.utils.dlpack.from_dlpack(pcd.rot_3.to_dlpack()),
        ), dim=-1).to(device).float()
        colors = torch.cat((
            torch.utils.dlpack.from_dlpack(pcd.f_dc_0.to_dlpack()),
            torch.utils.dlpack.from_dlpack(pcd.f_dc_1.to_dlpack()),
            torch.utils.dlpack.from_dlpack(pcd.f_dc_2.to_dlpack()),
        ), dim=-1).to(device).float()
        shs_lst = []
        shs = colors.new_empty(colors.shape[0], 0, 3)
        if hasattr(pcd, 'f_rest_8'):
            for i in range(0, 9):
                shs_lst.append(torch.utils.dlpack.from_dlpack(getattr(pcd, f'f_rest_{i}').to_dlpack()))
        if hasattr(pcd, 'f_rest_23'):
            for i in range(9, 24):
                shs_lst.append(torch.utils.dlpack.from_dlpack(getattr(pcd, f'f_rest_{i}').to_dlpack()))
        if hasattr(pcd, 'f_rest_44'):
            for i in range(24, 45):
                shs_lst.append(torch.utils.dlpack.from_dlpack(getattr(pcd, f'f_rest_{i}').to_dlpack()))
        if shs_lst:
            shs = torch.cat(shs_lst, dim=-1).view(colors.shape[0], -1, 3).to(device).float()
        return cls(means=means, scales=scales, quats=quats, colors=sh2rgb(colors), shs=shs, opacities=opacities)

    @torch.no_grad()
    def export(self, path: Path) -> None:
        map_to_tensors = {}
        map_to_tensors["positions"] = self.means.cpu().numpy()
        map_to_tensors["normals"] = np.zeros_like(map_to_tensors["positions"], dtype=np.float32)

        features_dc = rgb2sh(self.colors.cpu().numpy())      # [N, 3]
        for i in range(3):
            map_to_tensors[f"f_dc_{i}"] = features_dc[:, i, None]
        map_to_tensors["opacity"] = self.opacities.cpu().numpy()
        scales = self.scales.cpu().numpy()
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i, None]

        quats = (self.quats / self.quats.norm(dim=-1, keepdim=True)).cpu().numpy()
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i, None]

        pcd = o3d.t.geometry.PointCloud(map_to_tensors)
        o3d.t.io.write_point_cloud(str(path), pcd)

    @torch.no_grad()
    def reset_opacities(self, *, reset_value: float) -> None:
        # Reset value is set to be twice of the cull_alpha_thresh
        self.opacities.data.clamp_(max=torch.logit(torch.tensor(reset_value)).item())

    @torch.no_grad()
    def split(self: T, num_splits: int, scale_factor: float = 1 / 1.6) -> T:
        new_gaussians = self.__class__.stack([self.clear_extras()] * num_splits, dim=0)
        expaneded_shape = (num_splits, *self.shape)
        # sample new means
        randn = torch.randn((*expaneded_shape, 3), device=self.device)      # [S, ..., 3]
        scaled_offsets = self.scales.exp() * randn                          # [S, ..., 3]
        quats = self.quats / self.quats.norm(dim=-1, keepdim=True)          # [..., 4]
        rots = quat2rot(quats.view(-1, 4)).view(1, *self.shape, 3, 3) # [1, ..., 3, 3]
        rotated_offsets = (rots @ scaled_offsets[..., None])[..., 0]        # [S, ..., 3]

        new_means = rotated_offsets + self.means                            # [S, ..., 3]
        new_scales = (self.scales.exp() * scale_factor).log()
        new_scales = new_scales.expand(*expaneded_shape, 3).contiguous()    # [S, ..., 3]
        return new_gaussians.replace(means=new_means, scales=new_scales).requires_grad_(self.requires_grad)

    @torch.no_grad()
    def densify_and_cull(
        self,
        *,
        densify_grad_thresh: float,
        densify_size_thresh: float,
        num_splits: int,
        cull_alpha_thresh: float,
        cull_scale_thresh: Optional[float]
    ) -> Int32[Tensor, "N ndim"]:
        assert self.xys_grad_norm is not None and self.vis_counts is not None
        scale_max = self.scales.exp().max(dim=-1).values               # [...]
        avg_grad_norm = 0.5 * max(
            self.last_width.item(),
            self.last_height.item(),
        ) * (self.xys_grad_norm / self.vis_counts)                     # [...]
        high_grads = (avg_grad_norm > densify_grad_thresh)             # [...]
        splits = (scale_max > densify_size_thresh)                     # [...]
        dups = high_grads & (scale_max <= densify_size_thresh)         # [...]
        splits = high_grads & splits                                   # [...]

        culls = (self.opacities.sigmoid()[..., 0] < cull_alpha_thresh) # [...]
        if cull_scale_thresh is not None:
            toobigs = (scale_max > cull_scale_thresh)                  # [...]
            culls = culls | toobigs
        selected = ~(culls | splits)                                   # [...]

        new_gaussians = self.__class__.cat([
            self[selected].clear_extras_(),
            self[dups].clear_extras_(),
            self[splits].split(num_splits).view(-1),
        ], dim=0)
        self.swap_(new_gaussians.requires_grad_(self.requires_grad))
        splits_nonzero = -splits.nonzero() - 1

        return torch.cat([selected.nonzero(), -dups.nonzero() - 1] + [splits_nonzero] * num_splits, dim=0) # [N', ndim]

    @torch.no_grad()
    def cull(
        self,
        *,
        cull_alpha_thresh: float,
        cull_scale_thresh: Optional[float],
    ) -> Int32[Tensor, "N ndim"]:
        culls = (self.opacities.sigmoid()[..., 0] < cull_alpha_thresh) # [...]
        if cull_scale_thresh is not None:
            scale_max = self.scales.exp().max(dim=-1).values           # [...]
            toobigs = (scale_max > cull_scale_thresh)                  # [...]
            culls = culls | toobigs
        selected = ~culls
        self.swap_(self[selected].requires_grad_(self.requires_grad))
        return selected.nonzero()

    def clear_extras_(self: T) -> T:
        return self.replace_(
            xys_grad_norm=None,
            vis_counts=None,
            last_width=None,
            last_height=None,
        )

    def clear_extras(self: T) -> T:
        return self.replace(
            xys_grad_norm=None,
            vis_counts=None,
            last_width=None,
            last_height=None,
        )

    @chains
    def as_module(self, *, field_name: str) -> Any:

        def parameters(self) -> Any:
            return [getattr(self, field_name)]

        return parameters

    def get_cov3d_half(self) -> Float32[Tensor, "*bs 3 3"]:
        R = quat2rot(self.quats / self.quats.norm(dim=-1, keepdim=True)) # [..., 3, 3]
        S = self.scales.exp()                  # [..., 3]
        M = R * S[..., None, :]                # [..., 3, 3]
        return M

    def get_cov3d_inv_half(self) -> Float32[Tensor, "*bs 3 3"]:
        R = quat2rot(self.quats / self.quats.norm(dim=-1, keepdim=True)) # [..., 3, 3]
        S_inv = (-self.scales).exp()           # [..., 3]
        M_inv = R * S_inv[..., None, :]        # R @ (1/S) -> [..., 3, 3]
        # Cov3d.-1 = R.-T @ S.-T @ S.-1 @ R.-1 = R @ (1/S) @ (1/S).T @ R.T = M_inv @ M_inv.T
        return M_inv

    @torch.no_grad()
    def get_bounding_box(self, threshold: float) -> Float32[Tensor, "*bs 2 3"]:
        M_inv = self.get_cov3d_inv_half()      # [..., 3, 3]

        # solve equation:
        #     scaling_coeff * exp(-0.5 * x.T @ cov3d_inv @ x) < threshold
        #  -> scaling_coeff_log - 0.5 * x.T @ cov3d_inv @ x < threshold_log
        #  -> x.T @ cov3d_inv @ x > 2 * (scaling_coeff_log - threshold_log)
        # where:
        #       [scaling_coeff_log]
        #     = log(1 / (2pi ** 1.5 * gs.scales.exp()))
        #     = -1.5 * log(2pi) - gs.scales
        #       [cov3d_inv]
        #     = M_inv @ M_inv.T
        # thus:
        #       (x.T @ M_inv) @ (x.T @ M_inv).T > 2 * (scaling_coeff_log - threshold_log)

        xcxt_min_bound = -2 * (1.5 * np.log(2 * np.pi) + np.log(threshold) + self.scales) # [..., 1]
        assert (xcxt_min_bound > 0).all()
        offset_min_bound = (xcxt_min_bound / M_inv.square().sum(-1)).sqrt()               # [..., 3]
        return torch.stack((
            self.means - offset_min_bound,
            self.means + offset_min_bound,
        ), dim=-2)                                                                        # [..., 2, 3]

    @torch.no_grad()
    def get_cov3d_shape(self, *, iso_values: float = 1.0) -> TriangleMesh:
        eye = torch.eye(3, device=self.device)                              # [3, 3]
        scales = self.scales.exp().view(-1, 3) * iso_values                 # [N, 3]
        local_vertices = scales[:, None, :] * torch.cat((eye, -eye), dim=0) # [N, 6, 3]
        offsets = torch.matmul(
            quat2rot((self.quats / self.quats.norm(dim=-1, keepdim=True)).view(-1, 1, 4)),
            local_vertices[..., None],
        ).squeeze(-1)                                                       # [N, 6, 3]
        vertices = self.means.view(-1, 3)[:, None, :] + offsets             # [N, 6, 3]
        local_indices = torch.tensor([
            [0, 1, 2],
            [1, 3, 2],
            [3, 4, 2],
            [4, 0, 2],
            [0, 5, 1],
            [1, 5, 3],
            [3, 5, 4],
            [4, 5, 0],
        ]).long().to(self.device)                                           # [8, 3]

        indices = local_indices + (torch.arange(vertices.shape[0], device=self.device) * 6).view(-1, 1, 1) # [N, 8, 3]
        return TriangleMesh(vertices=vertices.view(-1, 3), indices=indices.view(-1, 3))

    @torch.no_grad()
    def sample_points(self, num_samples: int) -> Points:
        volumes = self.scales.sum(-1).exp().view(-1)                                     # [N]
        probs = volumes / volumes.sum()
        indices = torch.multinomial(probs, num_samples, replacement=True)                # [S]
        randn = torch.randn(num_samples, 3, device=self.device)
        offsets = randn * self.scales.view(-1, 3)[indices, :].exp()                      # [S, 3]
        rotated_offsets = quat2rot(
            (self.quats / self.quats.norm(dim=-1, keepdim=True)).view(-1, 4)[indices]
        ) @ offsets[..., None] # [S, 3, 1]
        positions = self.means.view(-1, 3)[indices, :] + rotated_offsets.squeeze(-1)     # [S, 3]
        colors = self.colors.view(-1, 3)[indices, :]
        return Points(positions=positions, colors=colors)

    @torch.no_grad()
    def as_points(self) -> Points:
        return Points(positions=self.means, colors=self.colors)

@dataclass
class FeatureSplats(Splats):

    feature_dim: int = Size.Dynamic
    features: torch.Tensor = Float.Trainable[..., feature_dim]

    @classmethod
    def from_points(
        cls: Type[U],
        points: Points,
        *,
        sh_degree: int,
        feature_dim: int,
        initial_feature: Literal['zero', 'random'] = 'random',
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> U:
        assert points.colors is not None
        points = points.flatten()
        distances = points.k_nearest(k=3)[0]   # [N, K]
        size = distances.shape[0]

        return cls(
            means=points.positions.clone().to(device),
            scales=distances.mean(dim=-1, keepdim=True).log().repeat(1, 3),
            quats=get_random_quaternion(size, device=device),
            opacities=torch.logit(0.1 * torch.ones((size, 1), device=device)),
            colors=points.colors.clone().to(device),
            shs=torch.zeros((size, sh_deg2dim(sh_degree) - 1, 3), device=device),
            features=(
                torch.randn(size, feature_dim, device=device)
                if initial_feature == 'random'
                else torch.zeros(size, feature_dim, device=device)
            ),
        ).requires_grad_(requires_grad)

    @classmethod
    def random(
        cls: Type[U],
        size: int,
        *,
        sh_degree: int,
        feature_dim: int,
        initial_feature: Literal['zero', 'random'] = 'random',
        random_scale: float,
        device: Optional[torch.device] = None,
        requires_grad: bool = False
    ) -> U:
        points = Points.rand((size, ), device=device).translate(-0.5).scale(2 * random_scale)
        distances = points.k_nearest(k=3)[0]   # [N, K]

        return cls(
            means=points.positions,
            scales=distances.mean(dim=-1, keepdim=True).log().repeat(1, 3),
            quats=get_random_quaternion(size, device=device),
            opacities=torch.logit(0.1 * torch.ones((size, 1), device=device)),
            colors=(torch.ones((size, 3), device=device) * 0.5),
            shs=torch.zeros((size, sh_deg2dim(sh_degree) - 1, 3), device=device),
            features=(
                torch.randn(size, feature_dim, device=device)
                if initial_feature == 'random'
                else torch.zeros(size, feature_dim, device=device)
            ),
        ).requires_grad_(requires_grad)

    def as_splats(self: U, *, gamma: float = 2.2, num_clusters: Optional[int] = None) -> Splats:
        if num_clusters is None:
            pca = principal_component_analysis(self.features, dim=-1, num_components=3)
            colors = (pca - pca.min()) / (pca.max() - pca.min()).clamp_min(1e-6)
        else:
            indices = spectral_clustering(
                self.features,
                downsample_to=1024,
                dim=-1,
                num_clusters=num_clusters,
            ).indices
            colors = RainbowColorMap()(indices / (num_clusters - 1))
        return Splats(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            opacities=self.opacities,
            colors=colors ** (1 / gamma),
            shs=torch.empty_like(self.shs[:, :0, :]),
        )

    def as_clustered_splats(self: U, *, gamma: float = 2.2, num_clusters: int) -> List[Splats]:
        indices = spectral_clustering(
            self.features,
            downsample_to=1024,
            dim=-1,
            num_clusters=num_clusters,
        ).indices
        colors = RainbowColorMap()(indices / (num_clusters - 1))
        splats = Splats(
            means=self.means,
            scales=self.scales,
            quats=self.quats,
            opacities=self.opacities,
            colors=colors ** (1 / gamma),
            shs=torch.empty_like(self.shs[:, :0, :]),
        )
        return [splats[(indices == i).nonzero().flatten()] for i in range(num_clusters)]

    def clustering(self: U, num_clusters: int) -> U:
        return self.replace(
            features=spectral_clustering(
                self.features,
                downsample_to=1024,
                dim=-1,
                num_clusters=num_clusters,
                return_values=True,
            ).values,
        )

    @torch.no_grad()
    def spatially_aggregate_features(self: U, *, voxel_size: float, size: float = 2.0) -> U:
        means = self.means.clamp(-size / 2, size / 2)
        grid = NearestGrid.from_resolution(
            resolution=round(size / voxel_size),
            feature_dim=self.feature_dim,
            size=size,
            device=self.device,
        )
        grid.aggregate(
            positions=means,
            values=self.features,
        )
        features = grid.query(means)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return self.replace_(features=features).requires_grad_(self.requires_grad)
