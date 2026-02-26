from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
from jaxtyping import Float32, Int64
from torch import Tensor

from rfstudio.utils.colormap import IntensityColorMap
from rfstudio.utils.tensor_dataclass import Bool, Float, Long, Size, TensorDataclass

from ..math import get_connected_components
from ._common import IntensityImages, SegImages


@dataclass
class SegTree(TensorDataclass):

    num_clusters: int = Size.Dynamic
    num_masks: int = Size.Dynamic
    image_width: int = Size.Dynamic
    image_height: int = Size.Dynamic

    cluster_correlation: Tensor = Long[num_clusters, num_clusters]
    pixel2cluster: Tensor = Long[image_height, image_width]
    cluster2mask: Tensor = Long[num_clusters, num_masks]
    masks: Tensor = Bool[image_height, image_width, num_masks]
    image: Optional[Tensor] = Float[image_height, image_width, 4]

    def compute_correlation_map(
        self,
        x: Union[int, Int64[Tensor, "..."]],
        y: Union[int, Int64[Tensor, "..."]],
    ) -> IntensityImages:
        cluster_idx = self.pixel2cluster[y, x].flatten() # [B]
        correlation = (
            self.cluster_correlation[cluster_idx, :] /
            self.cluster_correlation[cluster_idx, cluster_idx].unsqueeze(-1)
        ) # [B, C]
        valid = self.pixel2cluster >= 0 # [H, W]
        pixel_correlation = correlation[:, self.pixel2cluster.flatten()].reshape(-1, *valid.shape) # [B, H, W]
        return IntensityImages(torch.stack((pixel_correlation, valid.float().expand_as(pixel_correlation)), dim=-1))

    def compute_correlation(self, xy1: Int64[Tensor, "... 2"], xy2: Int64[Tensor, "... 2"]) -> Float32[Tensor, "... 1"]:
        cluster_idx1 = self.pixel2cluster[xy1[..., 0], xy2[..., 1]] # [...]
        cluster_idx2 = self.pixel2cluster[xy2[..., 0], xy2[..., 1]] # [...]
        valid = (cluster_idx1 >= 0) & (cluster_idx2 >= 0) # [...]
        correlation = (
            (2 * self.cluster_correlation[cluster_idx1, cluster_idx2]) /
            (
                self.cluster_correlation[cluster_idx1, cluster_idx1] +
                self.cluster_correlation[cluster_idx2, cluster_idx2]
            ).clamp_min(1)
        ) # [...]
        return torch.where(valid, correlation, 0).unsqueeze(-1)

    def sample_from_patches(self, *, approximate_num_patches: int = 4096) -> Int64[Tensor, "N 2"]:
        assert approximate_num_patches > 0 and self.image is not None
        H, W = self.image.shape[:2]
        P = ((H * W) / approximate_num_patches) ** 0.5
        P: int = 2 ** round(torch.tensor(P).log2().item())
        assert P > 0 and P <= H and P <= W
        padded_alpha = F.pad(self.image[..., 3], pad=(0, -W % P, 0, -H % P), mode='constant', value=0)
        H_, W_ = padded_alpha.shape[0] // P, padded_alpha.shape[1] // P
        padded_alpha = padded_alpha.reshape(H_, P, W_, P).transpose(1, 2).reshape(H_ * W_, P * P)
        valid_patches = (padded_alpha > 0).any(-1) # [H_ * W_]
        num_valid_patches = valid_patches.sum().item()
        assert num_valid_patches > 0
        num_samples_per_patch = round(approximate_num_patches / num_valid_patches)
        patch_indices = torch.multinomial(
            padded_alpha.clamp_min(1e-12),
            num_samples_per_patch,
            replacement=True,
        ) # [H_ * W_, S]
        rng = torch.arange(patch_indices.shape[0], device=patch_indices.device).view(-1, 1)
        Xs = (patch_indices % P) + (rng % W_) * P # [H_ * W_, S]
        Ys = (patch_indices // P) + (rng // W_) * P # [H_ * W_, S]
        return torch.stack((Xs[valid_patches, :], Ys[valid_patches, :]), dim=-1).view(-1, 2).unique(dim=0)

    def merge(self) -> SegImages:
        connected_components = get_connected_components(self.cluster_correlation > 0) # [K, N]
        merged_codes = (self.cluster2mask & connected_components.unsqueeze(-1)).any(1) # [K, M]
        assert merged_codes.any(0).all()
        results = torch.zeros_like(self.pixel2cluster) # [H, W]
        for i in range(merged_codes.shape[0]):
            to_fill = (self.masks & merged_codes[i]).any(-1) # [H, W]
            results[to_fill] = i + 1
        return SegImages(results.unsqueeze(-1))

    def visualize_clusters(self) -> SegImages:
        return SegImages(self.pixel2cluster.unsqueeze(-1) + 1)

    def visualize_masks(self, *, num_cols: int = 4) -> Tensor:

        from rfstudio.visualization import TabularFigures

        imgs = {}
        imgs['input'] = self.image[..., :3] * self.image[..., 3:] + (1 - self.image[..., 3:])
        imgs['clusters'] = self.visualize_clusters().visualize().item()
        for i in range(self.num_masks):
            imgs[f'mask{i:03d}'] = (
                (0.3 * self.masks[..., i:i+1] + 0.3) * imgs['input'] +
                (0.4 * self.masks[..., i:i+1] * torch.tensor([0, 1, 0]).to(imgs['input']))
            )
        tf = TabularFigures(
            num_rows=(len(imgs) + num_cols - 1) // num_cols,
            num_cols=num_cols,
            device=self.device
        )
        for i, (name, img) in enumerate(imgs.items()):
            tf[i // num_cols, i % num_cols].load(img, info=name)
        return tf.draw()

    def visualize_correlation(self, *, num_cols: int = 4) -> Tensor:

        from rfstudio.visualization import TabularFigures

        imgs = {}
        imgs['input'] = self.image[..., :3] * self.image[..., 3:] + (1 - self.image[..., 3:])
        imgs['clusters'] = self.visualize_clusters().visualize().item()
        for i in range(self.num_clusters):
            mask = (self.pixel2cluster == i).unsqueeze(-1)
            imgs[f'cluster{i:03d}'] = (
                (0.3 * mask + 0.3) * imgs['input'] +
                (0.4 * mask * torch.tensor([0, 1, 0]).to(imgs['input']))
            )
            correlation = self.cluster_correlation[i, :] / self.cluster_correlation[i, i].clamp_min(1) # [C]
            pixel_correlation = torch.where(
                self.pixel2cluster >= 0,
                correlation[self.pixel2cluster],
                0,
            ).unsqueeze(-1)
            imgs[f'correlation{i:03d}'] = IntensityColorMap().from_scaled(pixel_correlation)
        tf = TabularFigures(
            num_rows=(len(imgs) + num_cols - 1) // num_cols,
            num_cols=num_cols,
            device=self.device
        )
        for i, (name, img) in enumerate(imgs.items()):
            tf[i // num_cols, i % num_cols].load(img, info=name)
        return tf.draw()
