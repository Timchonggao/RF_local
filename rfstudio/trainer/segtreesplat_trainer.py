from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from rfstudio.data import SegTreeDataset
from rfstudio.graphics import Cameras, RGBAImages, RGBImages, SegTree
from rfstudio.loss import HierarchicalSegContrastiveLoss, PSNRLoss, SegContrastiveLoss, SSIML1Loss
from rfstudio.model import FeatureSplatter
from rfstudio.optim import ModuleOptimizers, Optimizer
from rfstudio.utils.colormap import IntensityColorMap

from .base_trainer import BaseTrainer


@dataclass
class SegTreeSplatTrainer(BaseTrainer):

    base_lr: float = 1e-3

    base_eps: float = 1e-15

    pos_lr_decay: int = 4500

    seg_reg_end: float = 0.1
    seg_reg_warmup: int = 2000
    seg_reg_decay: int = 10000

    entropy_reg_end: float = 0.1
    entropy_reg_warmup: int = 5000
    entropy_reg_decay: int = 15000

    accumulate_seg_loss: int = 1
    use_merged: bool = True

    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    num_splits: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    stop_split_at: int = 15000
    """stop splitting at this step"""

    normal_weight: float = 5e-2
    normal_weight_start: int = 7000
    distort_weight: float = 1e-2
    distort_weight_start: int = 3000

    rgb_loss: SSIML1Loss = SSIML1Loss(ssim_lambda=0.2)

    def setup(
        self,
        model: FeatureSplatter,
        dataset: SegTreeDataset,
    ) -> ModuleOptimizers:
        assert isinstance(dataset, SegTreeDataset)

        self._dataset_size = dataset.get_size(split='train')
        self._segimages = dataset.get_meta(split='train') if self.use_merged else None
        self._normal_weight_enable = False
        self._distort_weight_enable = False
        self._seg_loss_weight = self.seg_reg_end

        optim_dict = {
            'means': Optimizer(
                category=torch.optim.Adam,
                modules=model.gaussians.as_module(field_name='means'),
                lr=self.base_lr * 0.16,
                eps=self.base_eps,
                lr_decay=self.pos_lr_decay
            ),
            'scales': Optimizer(
                category=torch.optim.Adam,
                modules=model.gaussians.as_module(field_name='scales'),
                lr=self.base_lr * 5,
                eps=self.base_eps
            ),
            'quats': Optimizer(
                category=torch.optim.Adam,
                modules=model.gaussians.as_module(field_name='quats'),
                lr=self.base_lr,
                eps=self.base_eps
            ),
            'colors': Optimizer(
                category=torch.optim.Adam,
                modules=model.gaussians.as_module(field_name='colors'),
                lr=self.base_lr * 2.5,
                eps=self.base_eps
            ),
            'opacities': Optimizer(
                category=torch.optim.Adam,
                modules=model.gaussians.as_module(field_name='opacities'),
                lr=self.base_lr * 50,
                eps=self.base_eps
            ),
            'features': Optimizer(
                category=torch.optim.Adam,
                modules=model.gaussians.as_module(field_name='features'),
                lr=self.base_lr * 2.5,
                eps=self.base_eps
            ),
        }
        if model.sh_degree > 0:
            optim_dict['shs'] = Optimizer(
                category=torch.optim.Adam,
                modules=model.gaussians.as_module(field_name='shs'),
                lr=self.base_lr * 0.125,
                eps=self.base_eps
            )
        return ModuleOptimizers(
            mixed_precision=self.mixed_precision,
            optim_dict=optim_dict,
        )

    def step(
        self,
        model: FeatureSplatter,
        inputs: Cameras,
        gt_outputs: Iterable[SegTree],
        *,
        indices: Optional[Tensor],
        training: bool,
        visual: bool,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        bg = torch.rand(3, device=inputs.device)
        gt_rgb = RGBAImages([seg.image for seg in gt_outputs]).blend(bg)
        rgba = model.render_rgba(inputs)
        rgb = RGBImages([img[..., :3] + (1 - img[..., 3:]) * bg for img in rgba])
        features = model.render_feature(inputs)
        rgb_loss = self.rgb_loss(gt_rgb, rgb)
        if self.use_merged:
            seg_loss = SegContrastiveLoss(intra_mask_weight=1e-3)(outputs=features, gt_outputs=self._segimages[indices])
        else:
            seg_loss = HierarchicalSegContrastiveLoss()(outputs=features, gt_outputs=gt_outputs)
        norm_loss = sum([
            ((output.norm(dim=-1) - 1).abs() * gt_output.image[..., 3]).mean()
            for output, gt_output in zip(features, gt_outputs)
        ])
        entropy_loss = F.binary_cross_entropy_with_logits(
            torch.where(model.gaussians.opacities > -2, model.gaussians.opacities, 1),
            model.gaussians.opacities.new_ones(1).expand_as(model.gaussians.opacities),
        )
        loss = rgb_loss + seg_loss * self._seg_loss_weight
        loss = loss + norm_loss * self._seg_loss_weight * 10
        loss = loss + entropy_loss * self._entropy_loss_weight
        if model.rasterize_mode == '2dgs':
            if self._normal_weight_enable:
                loss = loss + model.update_info.normal_loss * self.normal_weight
            if self._distort_weight_enable:
                loss = loss + model.update_info.distort_loss * self.distort_weight
        metrics = {
            'rgb-loss': rgb_loss.detach(),
            'seg-loss': seg_loss.detach(),
            'psnr': PSNRLoss()(gt_rgb, rgb.detach()),
            'norm-loss': norm_loss.detach(),
            'entropy-loss': entropy_loss.detach(),
            '#gaussians': model.gaussians.shape[0]
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
                depth = model.render_depth(inputs[0:1])
                normal = depth.compute_pseudo_normals(inputs[0]).visualize((1, 1, 1))
                gt_vis = (
                    self._segimages[indices[0]].visualize().item()
                    if self.use_merged
                    else segtree.visualize_clusters().visualize().item()
                )
                row1 = torch.cat((rgba[0].item()[..., :3] + 1 - rgba[0].item()[..., 3:], gt_vis), dim=1).clamp(0, 1)
                row2 = torch.cat((normal.item(), features[0].visualize(gamma=1.0).item()), dim=1).clamp(0, 1)
                image = torch.cat((torch.cat((row1, row2)), torch.cat(vis)), dim=1)
        return loss, metrics, image

    def before_update(
        self,
        model: FeatureSplatter,
        optimizers: ModuleOptimizers,
        *,
        curr_step: int,
    ) -> None:
        self._seg_loss_weight = (
            self.seg_reg_end *
            min(1.0, max(0, curr_step - self.seg_reg_warmup) / (self.seg_reg_decay - self.seg_reg_warmup))
        )
        self._entropy_loss_weight = (
            self.entropy_reg_end *
            min(1.0, max(0, curr_step - self.entropy_reg_warmup) / (self.entropy_reg_decay - self.entropy_reg_warmup))
        )
        model.set_max_sh_degree(curr_step // self.sh_degree_interval)
        if curr_step > self.normal_weight_start:
            self._normal_weight_enable = True
        if curr_step > self.distort_weight_start:
            self._distort_weight_enable = True
        if curr_step > self.warmup_length and curr_step % self.refine_every == 0:
            pass
        elif curr_step % self.accumulate_seg_loss > 0:
            optimizers['features'].skip_once()

    def after_update(
        self,
        model: FeatureSplatter,
        optimizers: ModuleOptimizers,
        *,
        curr_step: int,
    ) -> None:
        model.set_max_sh_degree(None)

        if curr_step < self.stop_split_at:
            model.update_grad_norm()

        if curr_step > self.warmup_length and curr_step % self.refine_every == 0:
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            reset_interval = self.reset_alpha_every * self.refine_every

            # only split/cull if we've seen every image since opacity reset
            if all([
                curr_step < self.stop_split_at,
                curr_step % reset_interval > self._dataset_size + self.refine_every,
            ]):
                # ui.log('densify and cull @ step ', curr_step)
                indices = model.gaussians.densify_and_cull(
                    densify_grad_thresh=self.densify_grad_thresh,
                    densify_size_thresh=self.densify_size_thresh,
                    num_splits=self.num_splits,
                    cull_alpha_thresh=self.cull_alpha_thresh,
                    cull_scale_thresh=(
                        self.cull_scale_thresh
                        if curr_step > self.refine_every * self.reset_alpha_every
                        else None
                    )
                )
                optimizers.mutate_params(indices=indices)

            if all([
                curr_step >= self.stop_split_at,
                self.continue_cull_post_densification,
            ]):
                # ui.log('post cull @ step ', curr_step)
                indices = model.gaussians.cull(
                    cull_alpha_thresh=self.cull_alpha_thresh,
                    cull_scale_thresh=(
                        self.cull_scale_thresh
                        if curr_step > self.refine_every * self.reset_alpha_every
                        else None
                    )
                )
                optimizers.mutate_params(indices=indices)

            if all([
                curr_step < self.stop_split_at,
                curr_step % reset_interval == self.refine_every
            ]):
                # ui.log('reset opacities @ step ', curr_step)
                model.gaussians.reset_opacities(reset_value=self.cull_alpha_thresh * 2.0)
                model.gaussians.spatially_aggregate_features(voxel_size=0.1)
                optimizers['opacities'].mutate_params(clear=True)
                optimizers['features'].mutate_params(clear=True)

            model.gaussians.clear_extras_()
