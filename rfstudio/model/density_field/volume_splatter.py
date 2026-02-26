from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float32
from nerfacc import accumulate_along_rays
from torch import Tensor, nn

from rfstudio.graphics import Cameras, IsoCubes, Rays, RGBImages, TriangleMesh
from rfstudio.graphics.math import safe_normalize
from rfstudio.graphics.spatial_aggregator import HierarchicalKernelGrid, HierarchicalNearestGrid, NearestGrid
from rfstudio.model.density_field.components import SDFField
from rfstudio.model.density_field.components.sdf_field import SDFMLP
from rfstudio.nn import Module
from rfstudio.nn.mlp import MLP
from rfstudio.utils.decorator import chains, chunkify

from .components.renderer import VolumetricRenderer


@dataclass
class VolumeSplatter(Module):

    scale: float = 1.05
    resolution: int = 128
    chunk_size: int = 4096
    renderer: VolumetricRenderer = VolumetricRenderer()
    bg: float = 1.0
    field_type: Literal['density', 'vacancy', 'sdf'] = 'density'
    reparameterization: Literal['none', 'hierarchical', 'mlp'] = 'none'
    use_kernel: bool = False
    z_up: bool = False
    deviation_init: float = 0.1

    def __setup__(self) -> None:
        if self.reparameterization == 'hierarchical':
            D = torch.tensor(self.resolution).log2().round().long().item()
            assert (1 << D) == self.resolution
            grid = (
                HierarchicalKernelGrid
                if self.use_kernel
                else HierarchicalNearestGrid
            ).from_depth(D + 1, feature_dim=4, size=self.scale * 2)
            self.colors = nn.Parameter(grid.values[..., :3].clone())
            self.weights = nn.Parameter(grid.weights)
            if self.field_type == 'density':
                self.scalars = nn.Parameter(grid.values[..., 3:].clone())
                self.deviation = nn.Parameter(grid.values.new_empty(0))
            elif self.field_type in ['sdf', 'vacancy']:
                R = self.resolution
                flatten_indices = torch.arange(R ** 3, device=self.device).view(R, R, R)
                indices = torch.stack((
                    flatten_indices % R,
                    (flatten_indices % (R * R)) // R,
                    flatten_indices // (R * R),
                ), dim=-1).view(-1, 3) # [RRR, 3]
                grid_centers = ((indices + 0.5) / R) * (grid.max_bound - grid.min_bound) + grid.min_bound # [RRR, 3]
                ball_sdf = grid_centers.norm(dim=-1, keepdim=True) - 0.5 # [RRR, 1]
                grid.reset(reset_value=torch.cat((torch.zeros_like(ball_sdf).expand(-1, 3), ball_sdf), dim=-1))
                if torch.is_anomaly_enabled():
                    assert grid.query(grid_centers)[..., 3:].allclose(ball_sdf, rtol=1e-3, atol=1e-3)
                self.scalars = nn.Parameter(grid.values[..., 3:].clone())
                self.deviation = nn.Parameter(torch.ones(1) * self.deviation_init)
            else:
                raise ValueError(self.field_type)
            self._resolution = nn.Parameter(torch.tensor([self.resolution]).long(), requires_grad=False)
            self._layer_bases = nn.Parameter(grid.layer_bases, requires_grad=False)
            self.max_bound = nn.Parameter(grid.max_bound, requires_grad=False)
        elif self.reparameterization == 'mlp':
            assert self.field_type == 'sdf'
            self.colors = nn.Parameter(torch.empty(0))
            self.weights = nn.Parameter(torch.empty(0))
            self.scalars = nn.Parameter(torch.empty(0))
            self.deviation = nn.Parameter(torch.ones(1) * self.deviation_init)
            self._resolution = nn.Parameter(torch.tensor([self.resolution]).long(), requires_grad=False)
            self.sdf_field = SDFField(
                sdf_mlp=SDFMLP(layers=[-1, 64, 64, 65]),
                color_mlp=MLP(layers=[-1, 64, 3], activation='sigmoid', weight_norm=True),
            )
            self.max_bound = nn.Parameter(torch.tensor([self.scale] * 3).float(), requires_grad=False)
        elif self.reparameterization == 'none':
            grid = NearestGrid.from_resolution(self.resolution, feature_dim=4, size=self.scale * 2)
            self.colors = nn.Parameter(grid.values[..., :3].clone())
            self.weights = nn.Parameter(grid.values.new_empty(0))
            if self.field_type == 'density':
                self.scalars = nn.Parameter(grid.values[..., 3:].clone())
                self.deviation = nn.Parameter(grid.values.new_empty(0))
            elif self.field_type in ['sdf', 'vacancy']:
                ball_sdf = grid.get_grid_centers().view(grid.num_grids, 3).norm(dim=-1, keepdim=True) - 0.5
                self.scalars = nn.Parameter(ball_sdf)
                self.deviation = nn.Parameter(torch.ones(1) * self.deviation_init)
            else:
                raise ValueError(self.field_type)
            self._resolution = nn.Parameter(grid.resolution, requires_grad=False)
            self._layer_bases = nn.Parameter(grid.values.new_empty(0), requires_grad=False)
            self.max_bound = nn.Parameter(grid.max_bound, requires_grad=False)
        else:
            raise ValueError(self.reparameterization)
        self.anisotropy = 1.0

    def get_grid(self, *, depth_idx: Optional[int] = None) -> NearestGrid:
        if self.reparameterization == 'none':
            if self.use_kernel:
                rng = torch.tensor([1, 0, 1], dtype=torch.float32, device=self.device)
                kernel = (rng + rng.view(-1, 1) + rng.view(-1, 1, 1))
                kernel = (-kernel.flatten()).softmax(0).view(1, 1, 3, 3, 3)
                scalars = self.scalars.view(1, self.resolution, self.resolution, self.resolution) # [1, R, R, R]
                scalars = F.conv3d(scalars, kernel, padding=1).view_as(self.scalars)
            else:
                scalars = self.scalars
            return NearestGrid(
                values=torch.cat((self.colors, scalars), dim=-1),
                min_bound=-self.max_bound,
                max_bound=self.max_bound,
                resolution=self._resolution,
            )
        if self.reparameterization == 'mlp':
            grid = NearestGrid(
                values=torch.empty(1, device=self.device).expand(self.resolution ** 3, 4),
                min_bound=-self.max_bound,
                max_bound=self.max_bound,
                resolution=self._resolution,
            )
            centers = grid.get_grid_centers().view(-1, 3) # [RRR, 3]
            geo_feats = chunkify(self.sdf_field.sdf_mlp.__call__.__func__, chunk_size=self.chunk_size)(
                self.sdf_field.sdf_mlp,
                centers,
            ) # [RRR, 65]
            colors = chunkify(self.sdf_field.color_mlp.__call__.__func__, chunk_size=self.chunk_size)(
                self.sdf_field.color_mlp,
                torch.cat((centers, geo_feats[..., 1:]), dim=-1), # [RRR, 64 + 3]
            ) # [RRR, 3]
            return grid.replace(values=torch.cat((colors, geo_feats[..., :1]), dim=-1))
        if self.reparameterization == 'hierarchical':
            h_grid = (
                HierarchicalKernelGrid
                if self.use_kernel
                else HierarchicalNearestGrid
            )(
                values=torch.cat((self.colors, self.scalars), dim=-1),
                weights=torch.zeros_like(self.weights) if self.use_kernel else self.weights,
                min_bound=-self.max_bound,
                max_bound=self.max_bound,
                layer_bases=self._layer_bases,
            )
            if depth_idx is not None:
                h_grid = h_grid.clamp_depth(0, depth_idx)
            return h_grid.as_nearest_grid()
        raise ValueError(self.reparameterization)

    def get_variance(self) -> Tensor:
        return (self.deviation * 10.0).exp().clamp(1e-6, 1e6)

    def extract_mesh(self, transmittance: float = 0.5, *, depth_idx: Optional[int] = None) -> TriangleMesh:
        assert self.field_type in ['sdf', 'vacancy']
        grid = self.get_grid(depth_idx=depth_idx)
        scalars = grid.values[..., 3:]
        R = grid.resolution.item()
        # if self.field_type == 'vacancy':
        #     sdf_values = scalars - (1 - transmittance) / self.get_variance()
        # elif self.field_type == 'sdf':
        #     sdf_values = scalars - torch.tensor(transmittance).logit(eps=1e-9).item()
        # else:
        #     raise ValueError(self.field_type)
        sdf_values = scalars - torch.tensor(transmittance).logit(eps=1e-9).item()
        isocubes = IsoCubes.from_resolution(
            R - 1,
            device=self.device,
            scale=self.scale * (R - 1) / R,
            random_sdf=False,
        ).replace(sdf_values=sdf_values)
        return isocubes.marching_cubes()

    def render_rgb_along_rays(self, rays: Rays) -> Float32[Tensor, "*B 4"]:
        rgba = self.render_rgba_along_rays(rays)
        return rgba[..., :3] + self.bg * (1 - rgba[..., 3:])

    @chunkify(prop='chunk_size')
    def render_rgba_along_rays(self, rays: Rays) -> Float32[Tensor, "*B 4"]:
        grid = self.get_grid()
        positions, masks, _ = grid.intersect(rays.flatten()) # [N, 3R+1, 3], [N, 3R+1, 1]
        positions = torch.where(masks, positions, 0)
        distances = (positions[..., 1:, :] - positions[..., :-1, :]).norm(dim=-1, keepdim=True) # [N, 3R, 1]
        centers = (positions[..., 1:, :] + positions[..., :-1, :]) / 2 # [N, 3R, 3]
        valid = masks[..., 1:, :] & masks[..., :-1, :] # [N, 3R, 1]
        rgbvs = grid.query(centers) # [N, 3R, 4]
        if self.field_type == 'density':
            densities = F.softplus(rgbvs[..., 3:] * (0.1 * self.resolution)) # [N, 3R, 1]
            density_rendering = True
        elif self.field_type == 'vacancy':
            variance = self.get_variance() # [1]
            grad_f = grid.compute_gradients(
                centers,
                fn=lambda x: x[..., 3:],
            ) # [N, 3R, 3]
            omega = safe_normalize(rays.directions).view(-1, 1, 3) # [N, 1, 3]
            densities = (
                (omega * grad_f).sum(-1, keepdim=True).abs() * self.anisotropy +
                grad_f.norm(dim=-1, keepdim=True) * (0.5 - 0.5 * self.anisotropy)
            ) * (variance * (rgbvs[..., 3:] * -variance).sigmoid()) # [N, 3R, 1]
            density_rendering = True
        elif self.field_type == 'sdf':
            variance = self.get_variance() # [1]
            grad_f = grid.compute_gradients(
                centers,
                fn=lambda x: x[..., 3:],
            ) # [N, 3R, 3]
            omega = safe_normalize(rays.directions).view(-1, 1, 3) # [N, 1, 3]
            true_cos = (omega * grad_f).sum(-1, keepdim=True) # [N, 3R, 1]
            iter_cos = -(
                F.relu(-true_cos * 0.5 + 0.5) * min(1, 2 - 2 * self.anisotropy) +
                F.relu(-true_cos) * max(0, 2 * self.anisotropy - 1)
            )  # always non-positive, [N, 3R, 1]
            sdf_offsets = iter_cos * distances * 0.5 # [N, 3R, 1]
            estimated_prev_sdf = rgbvs[..., 3:] - sdf_offsets # [N, 3R, 1]
            estimated_next_sdf = rgbvs[..., 3:] + sdf_offsets # [N, 3R, 1]
            prev_cdf = (estimated_prev_sdf * variance).sigmoid() # [N, 3R, 1]
            next_cdf = (estimated_next_sdf * variance).sigmoid() # [N, 3R, 1]
            alphas = ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).clamp(0, 1) # [N, 3R, 1]
            density_rendering = False
        else:
            raise ValueError(self.field_type)
        if density_rendering:
            delta_densities = distances * densities # [N, 3R, 1]
            alphas = 1 - (-delta_densities).exp()                    # [N, 3R, 1]
            transmittance = delta_densities.cumsum(dim=-2)           # [N, 3R, 1]
            transmittance = torch.cat((
                torch.zeros_like(transmittance[..., :1, :]),
                delta_densities.cumsum(dim=-2)[..., :-1, :],
            ), dim=-2)
            transmittance = (-transmittance).exp()                   # [N, 3R, 1]
        else:
            transmittance = torch.cat((
                torch.zeros_like(alphas[..., :1, :]),
                (1.0 - alphas[..., :-1, :] + 1e-7).cumprod(dim=-2),
            ), dim=-2) # [N, 3R, 1]
        weights = torch.nan_to_num(alphas * transmittance)           # [N, 3R, 1]
        ray_indices = torch.arange(weights.shape[0], device=weights.device).view(-1, 1, 1).expand_as(valid)[valid] # [N]
        colors = accumulate_along_rays(
            weights=weights[valid],
            values=rgbvs[..., :3][valid.expand_as(rgbvs[..., :3])].view(-1, 3).sigmoid(),
            ray_indices=ray_indices,
            n_rays=weights.shape[0],
        ) # [N, 3]
        alphas = accumulate_along_rays(
            weights=weights[valid],
            ray_indices=ray_indices,
            n_rays=weights.shape[0],
        ) # [N, 1]
        return torch.cat((colors, alphas), dim=-1).view(*rays.shape, 4)

    def render_rgb(self, inputs: Cameras) -> RGBImages:
        return RGBImages(self.render_rgb_along_rays(camera.generate_rays()) for camera in inputs.view(-1))

    def compute_eikonal_loss(self) -> Tensor:
        assert self.field_type in ['sdf', 'vacancy']
        grid = self.get_grid()
        grad_f = grid.compute_gradients(grid.get_grid_centers(), fn=lambda x: x[..., 3:])
        grad_norm = grad_f.norm(dim=-1, keepdim=True)
        return (grad_norm - 1).square().mean()

    def compute_entropy(self) -> Tensor:
        assert self.field_type in ['sdf', 'vacancy']
        grid = self.get_grid()
        R = grid.resolution.item()
        flatten_indices = torch.arange(grid.num_grids, device=grid.device).view(R, R, R) # [R, R, R]
        indices = torch.stack((
            flatten_indices % grid.resolution,
            (flatten_indices % grid.resolution.square()) // grid.resolution,
            flatten_indices // grid.resolution.square(),
        ), dim=-1) # [R, R, R, 3]
        base = (grid.resolution ** torch.arange(3, device=grid.resolution.device))
        vacancy_logits_plus = grid.values[
            torch.cat((
                (indices[1:, :, :, :] * base).sum(-1).flatten(),
                (indices[:, 1:, :, :] * base).sum(-1).flatten(),
                (indices[:, :, 1:, :] * base).sum(-1).flatten(),
            ), dim=0),
            3:,
        ] # [K, 1]
        vacancy_logits = grid.values[
            torch.cat((
                (indices[:-1, :, :, :] * base).sum(-1).flatten(),
                (indices[:, :-1, :, :] * base).sum(-1).flatten(),
                (indices[:, :, :-1, :] * base).sum(-1).flatten(),
            ), dim=0),
            3:,
        ] # [K, 1]
        return torch.add(
            F.binary_cross_entropy_with_logits(vacancy_logits, (vacancy_logits_plus > 0).float()),
            F.binary_cross_entropy_with_logits(vacancy_logits_plus, (vacancy_logits > 0).float()),
        )
        return torch.add(
            F.binary_cross_entropy_with_logits(vacancy_logits, vacancy_logits_plus.detach().sigmoid()),
            F.binary_cross_entropy_with_logits(vacancy_logits_plus, vacancy_logits.detach().sigmoid()),
        )

    @chains
    def as_module(self, *, field_name: str) -> Any:

        def parameters(self) -> Any:
            return [getattr(self, field_name)]

        return parameters
