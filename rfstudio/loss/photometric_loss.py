from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Union

import torch
from jaxtyping import Float
from torch import Tensor

from rfstudio.graphics import FeatureImages, PBRAImages, PBRImages, RGBAImages, RGBImages, SegImages, SegTree
from rfstudio.utils.lazy_module import torchmetrics_F

from .base_loss import BaseLoss, L1Loss, L2Loss


@dataclass
class BasePhotometricLoss(BaseLoss):

    data_range: float = 1.0

    def __call__(
        self,
        outputs: Union[RGBImages, PBRImages],
        gt_outputs: Union[RGBImages, PBRImages],
    ) -> Float[Tensor, "1"]:
        results = []
        for output, gt_output in zip(outputs, gt_outputs, strict=True):
            results.append(self._impl(output, gt_output))
        return torch.stack(results).mean()

    def _impl(self, output: Float[Tensor, "H W 3"], gt_output: Float[Tensor, "H W 3"]) -> Float[Tensor, "1"]:
        raise NotImplementedError


@dataclass
class ImageL1Loss(BasePhotometricLoss):

    def _impl(
        self,
        output: Float[Tensor, "H W 3"],
        gt_output: Float[Tensor, "H W 3"],
    ) -> Float[Tensor, "1"]:
        return L1Loss.__call__(self, output, gt_output) / self.data_range ** 2


@dataclass
class ImageL2Loss(BasePhotometricLoss):

    def _impl(
        self,
        output: Float[Tensor, "H W 3"],
        gt_output: Float[Tensor, "H W 3"],
    ) -> Float[Tensor, "1"]:
        return L2Loss.__call__(self, output, gt_output) / self.data_range ** 2


@dataclass
class PSNRLoss(BasePhotometricLoss):

    def _impl(
        self,
        output: Float[Tensor, "H W 3"],
        gt_output: Float[Tensor, "H W 3"],
    ) -> Float[Tensor, "1"]:
        return -10 * (L2Loss.__call__(self, output, gt_output) / self.data_range ** 2).log10()


@dataclass
class SSIMLoss(BasePhotometricLoss):

    def _impl(
        self,
        output: Float[Tensor, "H W 3"],
        gt_output: Float[Tensor, "H W 3"],
    ) -> Float[Tensor, "1"]:
        return 1 - torchmetrics_F.structural_similarity_index_measure(
            gt_output.permute(2, 0, 1)[None],
            output.permute(2, 0, 1)[None],
            data_range=self.data_range,
        )


@dataclass
class LPIPSLoss(BasePhotometricLoss):

    def _impl(
        self,
        output: Float[Tensor, "H W 3"],
        gt_output: Float[Tensor, "H W 3"],
    ) -> Float[Tensor, "1"]:
        return torchmetrics_F.learned_perceptual_image_patch_similarity(
            gt_output.permute(2, 0, 1)[None] / self.data_range,
            output.permute(2, 0, 1)[None] / self.data_range,
            normalize=True,
        )

@dataclass
class SSIML1Loss(BasePhotometricLoss):

    ssim_lambda: float = 0.2

    def _impl(
        self,
        output: Float[Tensor, "H W 3"],
        gt_output: Float[Tensor, "H W 3"],
    ) -> Float[Tensor, "1"]:
        ssim_loss = SSIMLoss._impl(self, output, gt_output)
        l1_loss = L1Loss.__call__(self, output, gt_output)
        return ssim_loss * self.ssim_lambda + l1_loss * (1.0 - self.ssim_lambda)


@dataclass
class MaskedPhotometricLoss(BaseLoss):

    photometric_term: BasePhotometricLoss = ...
    coverage_coeff: float = 1.0
    coverage_loss: BaseLoss = L2Loss()

    def __call__(
        self,
        *,
        outputs: Union[RGBAImages, PBRAImages],
        gt_outputs: RGBAImages,
    ) -> Float[Tensor, "1"]:
        results = []
        if isinstance(outputs, PBRAImages):
            gt_outputs = gt_outputs.srgb2rgb()
        for output, gt_output in zip(outputs, gt_outputs, strict=True):
            results.append(
                torch.add(
                    self.photometric_term._impl(
                        output[..., :3] * gt_output[..., 3:],
                        gt_output[..., :3] * gt_output[..., 3:]
                    ),
                    self.coverage_coeff * self.coverage_loss(output[..., 3:], gt_output[..., 3:])
                )
            )
        return torch.stack(results).mean()


@dataclass
class HDRLoss(BasePhotometricLoss):

    def _rgb2srgb(self, f: Tensor) -> Tensor:
        return torch.where(
            f > 0.0031308,
            torch.pow(torch.clamp(f, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * f,
        )

    def _impl(
        self,
        output: Float[Tensor, "H W 3"],
        gt_output: Float[Tensor, "H W 3"],
    ) -> Float[Tensor, "1"]:

        return L1Loss()(
            self._rgb2srgb(torch.log(output.clamp(0, 65535) + 1)) / self.data_range,
            self._rgb2srgb(torch.log(gt_output.clamp(0, 65535) + 1)) / self.data_range,
        )


@dataclass
class HierarchicalSegContrastiveLoss(BaseLoss):

    chunk_size: int = 16
    inter_mask_weight: float = 0.1

    def __call__(
        self,
        *,
        outputs: FeatureImages,
        gt_outputs: Iterable[SegTree],
    ) -> Float[Tensor, "1"]:
        results = []
        for output, gt_output in zip(outputs, gt_outputs, strict=True):

            Nc = gt_output.num_clusters
            intra_loss = 0
            all_cluster_features = []
            for i in range(0, Nc, self.chunk_size):
                CK = min(Nc - i, self.chunk_size)
                cluster_indices = torch.arange(i, i + CK, device=gt_output.device) # [CK]
                cluster_masks = (
                    gt_output.pixel2cluster ==
                    cluster_indices.view(-1, 1, 1)
                ).flatten(1, 2).long() # [CK, H*W]
                if gt_output.image is not None:
                    cluster_masks = torch.where((gt_output.image[..., 3] > 0.5).flatten(), cluster_masks, 0)
                masked_features = (output.flatten(0, 1) * cluster_masks.unsqueeze(-1)) # [CK, H*W, C]
                cluster_features = masked_features.sum(1) / cluster_masks.sum(-1, keepdim=True).clamp_min(1) # [CK, C]
                intra_loss = intra_loss + (masked_features - cluster_features.unsqueeze(1)).square().sum()
                all_cluster_features.append(cluster_features)
            all_cluster_features = torch.cat(all_cluster_features) # [Nc, C]
            # cluster_differences = (
            #     -5 * (all_cluster_features - all_cluster_features.unsqueeze(1)).square().sum(-1)
            # ).softmax(dim=-1) # [Nc, Nc]
            # inter_loss = F.binary_cross_entropy(cluster_differences, gt_output.cluster_correlation)
            cluster_differences = (all_cluster_features - all_cluster_features.unsqueeze(1)).square().sum(-1) # [Nc, Nc]
            assert cluster_differences.shape == (Nc, Nc)
            inter_loss = ((1 - torch.eye(Nc).to(cluster_differences)) / (1 + cluster_differences)).mean()
            results.append(intra_loss / Nc + self.inter_mask_weight * inter_loss)
        return torch.stack(results).mean()

@dataclass
class SegContrastiveLoss(BaseLoss):

    chunk_size: int = 16
    intra_mask_weight: float = 0.01

    def __call__(
        self,
        *,
        outputs: FeatureImages,
        gt_outputs: SegImages,
    ) -> Float[Tensor, "1"]:

        results = []
        for output, gt_output in zip(outputs, gt_outputs, strict=True):
            Nc = gt_output.max().item()
            intra_loss = 0
            all_cluster_features = []
            for i in range(0, Nc, self.chunk_size):
                CK = min(Nc - i, self.chunk_size)
                cluster_indices = torch.arange(i + 1, i + CK + 1, device=gt_output.device) # [CK]
                cluster_masks = (
                    gt_output.squeeze(-1) ==
                    cluster_indices.view(-1, 1, 1)
                ).flatten(1, 2).long() # [CK, H*W]
                masked_features = (output.flatten(0, 1) * cluster_masks.unsqueeze(-1)) # [CK, H*W, C]
                cluster_features = masked_features.sum(1) / cluster_masks.sum(-1, keepdim=True).clamp_min(1) # [CK, C]
                dist = (masked_features - cluster_features.unsqueeze(1)).norm(dim=-1) * cluster_masks # [CK, H*W]
                intra_loss = intra_loss + (dist.sum(-1) / cluster_masks.sum(-1).clamp_min(1)).sum()
                all_cluster_features.append(cluster_features)
            all_cluster_features = torch.cat(all_cluster_features) # [Nc, C]
            cluster_differences = (all_cluster_features - all_cluster_features.unsqueeze(1)).square().sum(-1) # [Nc, Nc]
            assert cluster_differences.shape == (Nc, Nc)
            inter_loss = ((1 - torch.eye(Nc).to(cluster_differences)) / (1 + cluster_differences)).sum() # [1]
            results.append(self.intra_mask_weight * intra_loss / Nc + inter_loss / max(1, Nc * Nc - Nc))
        return torch.stack(results).mean()
