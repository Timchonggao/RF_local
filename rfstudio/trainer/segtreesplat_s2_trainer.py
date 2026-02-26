from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from rfstudio.data import SegTreeDataset
from rfstudio.graphics import Cameras, SegTree
from rfstudio.loss import HierarchicalSegContrastiveLoss, SegContrastiveLoss
from rfstudio.model.density_primitives.featsplat_s2 import FeatureSplatterS2
from rfstudio.optim import ModuleOptimizers, Optimizer
from rfstudio.utils.colormap import IntensityColorMap

from .base_trainer import BaseTrainer


@dataclass
class SegTreeSplatS2Trainer(BaseTrainer):

    base_lr: float = 1e-3

    base_eps: float = 1e-15

    accumulate_seg_loss: int = 1
    use_merged: bool = True

    def setup(
        self,
        model: FeatureSplatterS2,
        dataset: SegTreeDataset,
    ) -> ModuleOptimizers:
        assert isinstance(dataset, SegTreeDataset)

        self._segimages = dataset.get_meta(split='train') if self.use_merged else None

        optim_dict = {
            'codebook': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='codebook'),
                lr=self.base_lr * 0.1,
                eps=self.base_eps
            ),
            'features': Optimizer(
                category=torch.optim.Adam,
                modules=model.as_module(field_name='features'),
                lr=self.base_lr * 2.5,
                eps=self.base_eps
            ),
        }
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def step(
        self,
        model: FeatureSplatterS2,
        inputs: Cameras,
        gt_outputs: Iterable[SegTree],
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        features = model.render_feature(inputs)
        if self.use_merged:
            seg_loss = SegContrastiveLoss(intra_mask_weight=1e-3)(outputs=features, gt_outputs=self._segimages[indices])
        else:
            seg_loss = HierarchicalSegContrastiveLoss()(outputs=features, gt_outputs=gt_outputs)
        norm_loss = sum([
            ((output.norm(dim=-1) - 1).abs() * gt_output.image[..., 3]).mean()
            for output, gt_output in zip(features, gt_outputs)
        ])
        loss = seg_loss + norm_loss * 10
        metrics = {
            'seg-loss': seg_loss.detach(),
            'norm-loss': norm_loss.detach(),
        }
        image = None
        if visual:
            with torch.no_grad():
                colormap = IntensityColorMap()
                segtree = next(iter(gt_outputs))
                vis = []
                raw_alpha = segtree.image[..., 3:] # [H, W, 1]
                raw_image = segtree.image[..., :3] * raw_alpha + (1 - raw_alpha) # [H, W, 3]
                feat_map = features[0].item() # [H, W, C]
                for px, py in segtree.sample_from_patches(approximate_num_patches=8)[:2]:
                    vis_raw = raw_image.clone()
                    vis_raw[py-7:py+8, px-7:px+8, :] = 0
                    vis_raw[py-7:py+8, px-7:px+8, 1] = 1
                    vis_corr = colormap(
                        F.cosine_similarity(feat_map[py, px, :], feat_map, dim=-1).unsqueeze(-1) * 0.5 + 0.5
                    )
                    vis_corr_gt = segtree.compute_correlation_map(px, py).visualize(colormap).blend((0, 0, 0)).item()
                    vis.append(torch.cat((vis_raw, vis_corr * raw_alpha, vis_corr_gt), dim=1))
                gt_vis = (
                    self._segimages[indices[0]].visualize().item()
                    if self.use_merged
                    else segtree.visualize_clusters().visualize().item()
                )
                image = torch.cat((
                    torch.cat((gt_vis, features[0].visualize(gamma=1.0).item()), dim=0),
                    torch.cat(vis),
                ), dim=1)
        return loss, metrics, image

    def before_update(
        self,
        model: FeatureSplatterS2,
        optimizers: ModuleOptimizers,
        *,
        curr_step: int,
    ) -> None:
        if curr_step % self.accumulate_seg_loss > 0:
            optimizers['codebook'].skip_once()
            optimizers['features'].skip_once()
