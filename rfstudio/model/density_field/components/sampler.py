from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional, Tuple

import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.graphics import Rays, RaySamples
from rfstudio.nn import Module


@dataclass
class UniformSampler(Module):

    num_samples_per_ray: int
    train_stratified: bool = True
    single_jitter: bool = False
    spacing: Literal['uniform', 'linear_disparity'] = 'uniform'

    def __call__(self, inputs: Rays) -> RaySamples:
        ts = torch.linspace(0, 1, self.num_samples_per_ray + 1, device=self.device)
        if self.training and self.train_stratified:
            if self.single_jitter:
                rand = torch.rand(*inputs.shape, 1, device=self.device)
            else:
                rand = torch.rand(*inputs.shape, self.num_samples_per_ray + 1, device=self.device)
            centers = (ts[..., 1:] + ts[..., :-1]) * 0.5
            ub = torch.cat((centers, ts[..., -1:]), dim=-1)
            lb = torch.cat((ts[..., :1], centers), dim=-1)
            ts = lb + (ub - lb) * rand
        return inputs.get_samples(t=ts, spacing=self.spacing)


@dataclass
class PDFSampler(Module):

    ''' Sample linearly in disparity along a ray '''

    num_samples_per_ray: int = ...
    histogram_padding: float = 0.01
    single_jitter: bool = True
    eps: float = 1e-5
    include_original: bool = True

    def __call__(
        self,
        inputs: Rays,
        *,
        coarse_samples: RaySamples,
    ) -> RaySamples:

        coarse_samples = coarse_samples.detach()
        assert coarse_samples.weights is not None

        weights = coarse_samples.weights.squeeze(-1) + self.histogram_padding # [..., S]
        weights_sum = weights.sum(dim=-1, keepdim=True) # [..., 1]

        # Add small offset to rays with zero weight to prevent NaNs
        padding = (self.eps - weights_sum).relu() # [..., 1]
        weights = weights + padding / weights.shape[-1] # [..., S]

        pdf = weights / (weights_sum + padding) # [..., S]
        cdf = pdf.cumsum(-1).clamp_max(1) # [..., S]
        cdf = torch.cat((torch.zeros_like(cdf[..., :1]), cdf), dim=-1) # [..., S+1]

        num_bins = self.num_samples_per_ray + 1 # S'+1

        u = torch.linspace(0, 1 - 1 / num_bins, num_bins, device=cdf.device) # [S'+1]
        if self.training:
            if self.single_jitter:
                u = u + torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins # [..., S'+1]
            else:
                u = u + torch.rand((*cdf.shape[:-1], num_bins), device=cdf.device) / num_bins # [..., S'+1]
        else:
            u = u + 0.5 / num_bins # [S'+1]
            u = u.expand(*cdf.shape[:-1], num_bins).contiguous() # [..., S'+1]

        existing_bins = torch.cat((
            coarse_samples.bins,
            coarse_samples.max_bin.unsqueeze(-1),
        ), dim=-2).squeeze(-1) # [..., S+1]

        indices = torch.searchsorted(cdf, u, side='right') # [..., S'+1]
        below = (indices - 1).clamp(0, cdf.shape[-1] - 1) # [..., S'+1]
        above = indices.clamp(0, cdf.shape[-1] - 1) # [..., S'+1]
        cdf_g0 = cdf.gather(dim=-1, index=below) # [..., S'+1]
        cdf_g1 = cdf.gather(dim=-1, index=above) # [..., S'+1]
        bins_g0 = existing_bins.gather(dim=-1, index=below) # [..., S'+1]
        bins_g1 = existing_bins.gather(dim=-1, index=above) # [..., S'+1]
        ts = ((u - cdf_g0) / (cdf_g1 - cdf_g0)).nan_to_num(nan=0).clamp(0, 1) # [..., S'+1]
        new_bins = bins_g0 + ts * (bins_g1 - bins_g0) # [..., S'+1]
        if self.include_original:
            # discard last one as t=1 probably repeats twice
            new_bins = torch.cat((existing_bins, new_bins), dim=-1).sort(dim=-1).values[..., :-1] # [..., S+S'+1]
        return RaySamples(
            origins=inputs.origins,
            directions=inputs.directions,
            bins=new_bins[..., :-1].unsqueeze(-1),
            max_bin=new_bins[..., -1:],
        )

@dataclass
class NeuSSampler(Module):

    num_samples_per_ray: int
    num_importance_samples_per_ray: int
    num_upsample_steps: int = 4
    base_variance: float = 64.0
    single_jitter: bool = True

    @torch.no_grad()
    def _merge_ray_samples(
        self,
        ray_samples_1: RaySamples,
        ray_samples_2: RaySamples,
    ) -> Tuple[RaySamples, Tensor]:
        """
        Merge two set of ray samples and return sorted index which can be used to merge sdf values
        Args:
            ray_samples_1 : ray_samples to merge
            ray_samples_2 : ray_samples to merge
        """

        assert ray_samples_1.shape == ray_samples_2.shape
        assert ray_samples_1.origins.allclose(ray_samples_2.origins)
        assert ray_samples_1.directions.allclose(ray_samples_2.directions)

        bins, sorted_index = torch.sort(torch.cat((ray_samples_1.bins, ray_samples_2.bins), dim=-2), -2) # [..., S, 1]
        ray_samples = RaySamples(
            origins=ray_samples_1.origins,
            directions=ray_samples_1.directions,
            bins=bins,
            max_bin=torch.maximum(ray_samples_1.max_bin, ray_samples_2.max_bin),
        )
        return ray_samples, sorted_index

    def _rendering_sdf_with_fixed_inv_s(
        self,
        ray_samples: RaySamples,
        *,
        sdf: Float32[Tensor, "... num_samples 1"],
        inv_s: int,
    ) -> Float32[Tensor, "... num_samples 1"]:
        """
        rendering given a fixed inv_s as NeuS

        Args:
            ray_samples: samples along ray
            sdf: sdf values along ray
            inv_s: fixed variance value
        Returns:
            alpha value
        """
        prev_sdf, next_sdf = sdf[..., :-1, 0], sdf[..., 1:, 0]
        dist = ray_samples.distances[..., :-1, 0]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (dist + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat((torch.zeros(*ray_samples.shape, 1, device=sdf.device), cos_val[..., :-1]), dim=-1)
        cos_val = torch.stack((prev_cos_val, cos_val), dim=-1).min(-1).values.clip(-1e3, 0.0)

        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = (prev_esti_sdf * inv_s).sigmoid()
        next_cdf = (next_esti_sdf * inv_s).sigmoid()
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        return torch.cat((alpha, torch.zeros_like(alpha[..., -1:])), dim=-1).unsqueeze(-1)

    def __call__(self, inputs: Rays, *, sdf_fn: Callable[[Tensor], Tensor]) -> RaySamples:
        total_iters = 0
        sorted_index = None
        sdf: Optional[torch.Tensor] = None
        ray_samples = UniformSampler(
            num_samples_per_ray=self.num_samples_per_ray,
            single_jitter=self.single_jitter,
        ).to(self.device)(inputs)
        pdf_sampler = PDFSampler(
            num_samples_per_ray=self.num_importance_samples_per_ray // self.num_upsample_steps,
            include_original=False,
            single_jitter=self.single_jitter,
            histogram_padding=1e-5,
        ).to(self.device)
        new_samples = ray_samples

        base_variance = self.base_variance

        while total_iters < self.num_upsample_steps:
            with torch.no_grad():
                starts = new_samples.origins[..., None, :] + new_samples.bins * new_samples.directions[..., None, :]
                new_sdf = sdf_fn(starts)

            # merge sdf predictions
            if sorted_index is not None:
                assert sdf is not None
                sdf = torch.cat((sdf, new_sdf), dim=-2).gather(dim=-2, index=sorted_index)
            else:
                sdf = new_sdf

            # compute with fix variances
            alphas = self._rendering_sdf_with_fixed_inv_s(
                ray_samples,
                sdf=sdf,
                inv_s=base_variance * (2 ** total_iters),
            )

            new_samples = pdf_sampler(
                inputs,
                coarse_samples=ray_samples.annotate(alphas=alphas).get_weighted(),
            )

            ray_samples, sorted_index = self._merge_ray_samples(ray_samples, new_samples)

            total_iters += 1

        return ray_samples
