from __future__ import annotations

# import module
from dataclasses import dataclass, replace, field
from typing import Literal, Optional, List, Union, Sequence, Collection, Iterable, Tuple, Dict, Any
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped
from torch import Tensor, nn

# import rfstudio modules
from rfstudio.nn import Module, ParameterModule

# import rfstudio_ds modules
from .hashencoder.hashgrid import _hash_encode, HashEncoder



@dataclass
class Grid4d_HashEncoding(Module):

    """
    Hash encoding ref to Grid4D paper.

    Args:
        TODOS: add args
    """

    # shared settings
    decoder: Module = ...
    grad_scaling: Optional[float] = 10
    backend: Literal['tcnn', 'grid4d'] = 'tcnn'

    # xyz canonical space settings
    canonical_base_resolution: Union[int, List[int]] = 16
    canonical_num_levels: int = 16
    canonical_per_level_scale: int = 2 # control scale, if provide desired_resolution, per_level_scale will be ignored.
    canonical_desired_resolution: Union[int, List[int]] = 2048
    canonical_features_per_level: int = 2
    canonical_log2_hashmap_size: int = 19
    canonical_interpolation: Literal["nearest", "linear", "smoothstep"] = "linear"

    # x, y, z, t deformed space setting
    deform_base_resolution: Union[int, List[int]] = field(default_factory=lambda: [8,8,8])
    deform_num_levels: int = 32
    deform_per_level_scale: int = 2 # if provide desired_resolution, per_level_scale will be ignored.
    deform_desired_resolution: Union[int, List[int]] = field(default_factory=lambda: [2048,2048,32])
    deform_features_per_level: int = 2
    deform_log2_hashmap_size: int = 19
    deform_interpolation: Literal["nearest", "linear", "smoothstep"] = "linear"
    

    def reset(self) -> None:
        device = self.device
        self.decoder = replace(self.decoder)
        self.decoder.__setup__()
        self.__setup__()
        self.to(device)
        if self.backend == "tcnn":
            import tinycudann as tcnn
            tcnn.free_temporary_memory()

    # define hash grid encoder
    def __setup__(self) -> None:
        assert self.grad_scaling is None or self.grad_scaling > 0
        if self.backend == 'grid4d':
            self.xyz_encoder = HashEncoder(
                input_dim=3, 
                base_resolution=self.canonical_base_resolution,
                desired_resolution=self.canonical_desired_resolution,
                num_levels=self.canonical_num_levels, 
                per_level_scale=self.canonical_per_level_scale,
                level_dim=self.canonical_features_per_level,
                log2_hashmap_size=self.canonical_log2_hashmap_size,
            )
            self.xyt_encoder = HashEncoder(
                input_dim=3, 
                base_resolution=self.deform_base_resolution,
                desired_resolution=self.deform_desired_resolution,
                num_levels=self.deform_num_levels, 
                per_level_scale=self.deform_per_level_scale,
                level_dim=self.deform_features_per_level,
                log2_hashmap_size=self.deform_log2_hashmap_size,
            )
            self.yzt_encoder = HashEncoder(
                input_dim=3, 
                base_resolution=self.deform_base_resolution,
                desired_resolution=self.deform_desired_resolution,
                num_levels=self.deform_num_levels, 
                per_level_scale=self.deform_per_level_scale,
                level_dim=self.deform_features_per_level,
                log2_hashmap_size=self.deform_log2_hashmap_size,
            )
            self.xzt_encoder = HashEncoder(
                input_dim=3, 
                base_resolution=self.deform_base_resolution,
                desired_resolution=self.deform_desired_resolution,
                num_levels=self.deform_num_levels, 
                per_level_scale=self.deform_per_level_scale,
                level_dim=self.deform_features_per_level,
                log2_hashmap_size=self.deform_log2_hashmap_size,
            )
        elif self.backend == 'tcnn':
            import tinycudann as tcnn

            if self.canonical_desired_resolution is not None:
                self.canonical_per_level_scale = (
                    np.exp((np.log(self.canonical_desired_resolution) - np.log(self.canonical_base_resolution)) / (self.canonical_num_levels - 1))
                    if self.canonical_num_levels > 1
                    else 1
                ) 
            xyz_encoding_config = {
                "otype": "HashGrid",
                "n_levels": self.canonical_num_levels,
                "n_features_per_level": self.canonical_features_per_level,
                "log2_hashmap_size": self.canonical_log2_hashmap_size,
                "base_resolution": self.canonical_base_resolution,
                "per_level_scale": self.canonical_per_level_scale,
                "interpolation": self.canonical_interpolation.capitalize(),
            }
            self.xyz_encoder = tcnn.Encoding(
                n_input_dims=3, # 输入维度为 3
                encoding_config=xyz_encoding_config,
            )

            self.deform_base_resolution = self.deform_base_resolution[0] if isinstance(self.deform_base_resolution, list) else self.deform_base_resolution
            self.deform_desired_resolution = self.deform_desired_resolution[0] if isinstance(self.deform_desired_resolution, list) else self.deform_desired_resolution
            if self.deform_desired_resolution is not None:
                self.deform_per_level_scale = (
                    np.exp((np.log(self.deform_desired_resolution) - np.log(self.deform_base_resolution)) / (self.deform_num_levels - 1))
                    if self.deform_num_levels > 1
                    else 1
                )
            deform_encoding_config = {
                "otype": "HashGrid",
                "n_levels": self.deform_num_levels,
                "n_features_per_level": self.deform_features_per_level,
                "log2_hashmap_size": self.deform_log2_hashmap_size,
                "base_resolution": self.deform_base_resolution,
                "per_level_scale": self.deform_per_level_scale,
                "interpolation": self.deform_interpolation.capitalize(),
            }
            self.xyt_encoder = tcnn.Encoding(
                n_input_dims=3, # 输入维度为 3
                encoding_config=deform_encoding_config,
            )
            self.yzt_encoder = tcnn.Encoding(
                n_input_dims=3, # 输入维度为 3
                encoding_config=deform_encoding_config,
            )
            self.xzt_encoder = tcnn.Encoding(
                n_input_dims=3, # 输入维度为 3
                encoding_config=deform_encoding_config,
            )
        else:
            raise ValueError('The argument `implementation` must be one of torch and tcnn.')

    def encode_spatial(self, xyz):
        if self.backend == 'grid4d':
            spatial_feats = self.xyz_encoder((xyz + 1) / (1 * 2))
        elif self.backend == 'tcnn':
            spatial_feats = self.xyz_encoder((xyz + 1) / (1 * 2))
        
        return spatial_feats
    
    def encode_temporal(self, xyzt):
        xyt = torch.cat([xyzt[..., :2], xyzt[..., 3:]], dim=-1)
        yzt = xyzt[..., 1:]
        xzt = torch.cat([xyzt[..., :1], xyzt[..., 2:]], dim=-1)

        xyt_feat = self.xyt_encoder((xyt + 1) / (1 * 2))  # [N, num_leval*2]
        yzt_feat = self.yzt_encoder((yzt + 1) / (1 * 2))  # [N, num_leval*2]
        xzt_feat = self.xzt_encoder((xzt + 1) / (1 * 2))  # [N, num_leval*2]

        # temporal_feats = xyt_feat * yzt_feat * xzt_feat     # todo try this [N, 64]
        temporal_feats = torch.cat([xyt_feat, yzt_feat, xzt_feat], dim=-1) # [N, num_leval*6]

        return temporal_feats
    
    def decode_features(self, spatial_feats: Optional[Float[Tensor, "*bs 64"]] = None, temporal_feats: Optional[Float[Tensor, "*bs 64"]] = None):
        return self.decoder(spatial_feats, temporal_feats)

    def __call__(self, in_tensor: Float[Tensor, "*bs xyzt"]) -> Float[Tensor, "*bs output_dim"]:
        # notion in_tensor should be in [-1, 1]
        if self.grad_scaling is not None:
            in_tensor = in_tensor * (1 / self.grad_scaling) + in_tensor.detach() * (1 - 1 / self.grad_scaling)
        
        temporal_feats = self.encode_temporal(in_tensor.flatten(end_dim=-2)).view(*in_tensor.shape[:-1], -1).float()
        
        if self.grad_scaling is not None:
            temporal_feats = temporal_feats * self.grad_scaling + temporal_feats.detach() * (1 - self.grad_scaling)
        
        return self.decode_features(None, temporal_feats), temporal_feats



@dataclass
class KplaneEncoding(Module):
    """
    单尺度 K-Planes 编码器（稳健版）
    支持 3D/4D 输入:
      3D -> planes: XY, XZ, YZ (3 planes)
      4D -> planes: XY, XZ, YZ, XT, YT, ZT (6 planes)
    """

    decoder: Module = ...

    # Resolutions
    spatial_resolution: Union[int, List[int]] = 1024
    time_resolution: int = 64

    # Single-scale only
    num_components: int = 32
    reduce: Literal["sum", "product", "concat"] = "product"
    input_dim: int = 4  # 3 or 4

    # Normalization bbox: if provided, pts will be mapped from [bbox_min, bbox_max] -> [-1,1]
    # bbox should be a tuple (min_tensor, max_tensor) of shape (input_dim,)
    bbox: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    # If True: initialize time-containing planes to ones (default behaviour)
    init_time_ones: bool = True
    
    def reset(self) -> None:
        device = self.device
        self.decoder = replace(self.decoder)
        self.decoder.__setup__()
        self.__setup__()
        self.to(device)

    def __setup__(self):
        # build resolution list
        if isinstance(self.spatial_resolution, int):
            spatial_reso = [self.spatial_resolution] * 3
        else:
            spatial_reso = list(self.spatial_resolution)

        if self.input_dim == 4:
            reso = spatial_reso + [self.time_resolution]
        else:
            reso = spatial_reso

        # create plane coefficients (2D planes)
        self.plane_coefs = self.init_grid_param(
            grid_nd=2,
            in_dim=self.input_dim,
            out_dim=self.num_components,
            reso=reso
        )

        self.num_planes = len(self.plane_coefs)
        if self.reduce in ("sum", "product"):
            self.feature_dim = self.num_components
        elif self.reduce == "concat":
            self.feature_dim = self.num_components * self.num_planes
        else:
            raise ValueError("Unsupported reduce mode: choose from 'sum','product','concat'")

    def init_grid_param(self, grid_nd: int, in_dim: int, out_dim: int, reso: Sequence[int],
                        a: float = 0.1, b: float = 0.5):
        # reso length must equal in_dim (e.g., [W,H,D,T] for 4D)
        assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
        has_time = (in_dim == 4)
        coo_combs = list(itertools.combinations(range(in_dim), grid_nd))  # e.g., (0,1),(0,2),(1,2)...
        plane_list = nn.ParameterList()
        for comb in coo_combs:
            # reverse order to match grid_sample convention (W last)
            shape = [1, out_dim] + [int(reso[c]) for c in comb[::-1]]
            param = nn.Parameter(torch.empty(*shape))
            if has_time and self.init_time_ones and 3 in comb:
                nn.init.ones_(param)
            else:
                nn.init.uniform_(param, a=a, b=b)
            plane_list.append(param)
        return plane_list

    def _normalize_pts_to_grid(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Normalize pts to grid_sample coordinate system [-1, 1].
        If self.bbox is provided, do per-dimension mapping from [min, max] -> [-1, 1].
        Otherwise assume pts are already in [-1, 1] and return as-is.
        pts: (..., input_dim)
        """
        if self.bbox is None:
            return pts
        bbox_min, bbox_max = self.bbox
        # ensure shapes
        bbox_min = bbox_min.to(pts.device).view(*([1] * (pts.dim() - 1)), -1)
        bbox_max = bbox_max.to(pts.device).view(*([1] * (pts.dim() - 1)), -1)
        # avoid division by zero
        denom = (bbox_max - bbox_min).clamp_min(1e-6)
        # map: val -> (val - min) / (max - min) in [0,1], then to [-1,1]
        normalized = (pts - bbox_min) / denom
        normalized = normalized * 2.0 - 1.0
        return normalized

    def _grid_sample_wrapper(self, grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True):
        """
        Robust wrapper around F.grid_sample to sample arbitrarily many coords.
        - grid: Tensor with shape [C,H,W] or [1,C,H,W] or [B,C,H,W] for 2D
                For 3D grids (if used), shapes are [C,D,H,W] / [1,C,D,H,W] / [B,C,D,H,W]
        - coords: Tensor of normalized coordinates in [-1,1], shape (B, N, grid_dim) or (N, grid_dim)
        Returns: Tensor shape (B, N, C) or (N, C) if no batch dim initially.
        """
        # coords expected shape (..., grid_dim)
        grid_dim = coords.shape[-1]
        # Expected grid dims = grid_dim + 2 (batch + channel + spatial...)
        expected_grid_dims = grid_dim + 2

        # Ensure grid has batch dim
        if grid.dim() == expected_grid_dims - 1:
            # e.g., grid: (C, H, W) -> make (1, C, H, W)
            grid = grid.unsqueeze(0)
        elif grid.dim() == expected_grid_dims:
            # already has batch (B, C, H, W)
            pass
        else:
            # unexpected shape
            raise RuntimeError(f"Unexpected grid dim {grid.dim()} for grid_dim {grid_dim}")

        # coords -> ensure batch dim
        had_batch = True
        if coords.dim() == 2:
            # (N, grid_dim) -> treat as no batch, make batch=1
            coords = coords.unsqueeze(0)  # (1, N, grid_dim)
            had_batch = False
        elif coords.dim() == 3:
            # (B, N, grid_dim) -> ok
            had_batch = True
        else:
            # support more general leading dims by flattening prior to call
            raise RuntimeError("coords must be (N, D) or (B, N, D)")

        B = coords.shape[0]
        N = coords.shape[1]

        # reshape coords into F.grid_sample expected shape:
        # For 2D: (B, H_out, W_out, 2) -> choose H_out=1, W_out=N -> shape (B,1,N,2)
        # For 3D: (B, D_out, H_out, W_out, 3) -> choose D_out=1,H_out=1,W_out=N -> (B,1,1,N,3)
        if grid_dim == 2:
            grid_coords = coords.view(B, 1, N, 2)
        elif grid_dim == 3:
            grid_coords = coords.view(B, 1, 1, N, 3)
        else:
            raise NotImplementedError("grid_sample_wrapper only implements 2D/3D sampling")

        # ensure grid batch size matches coords batch size
        # If grid has batch size 1 and coords B>1, expand grid to B
        if grid.shape[0] == 1 and B > 1:
            grid = grid.expand(B, -1, *grid.shape[2:])

        # Do sampling
        # grid: (B, C, H, W) or (B, C, D, H, W)
        sampled = F.grid_sample(
            grid,
            grid_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=align_corners
        )
        # sampled shape:
        #  - 2D: (B, C, 1, N)
        #  - 3D: (B, C, 1, 1, N)
        # reshape to (B, N, C)
        if grid_dim == 2:
            B_out, C, _, N_out = sampled.shape
            sampled = sampled.view(B_out, C, N_out).permute(0, 2, 1).contiguous()  # (B, N, C)
        else:
            # 3D
            B_out, C, _, _, N_out = sampled.shape
            sampled = sampled.view(B_out, C, N_out).permute(0, 2, 1).contiguous()

        if not had_batch:
            # return (N, C)
            return sampled.squeeze(0)
        return sampled  # (B, N, C)

    def __call__(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: shape (..., input_dim)
          - (N, D) -> returns (N, feature_dim)
          - (B, N, D) -> returns (B, N, feature_dim)
          - (B, D) -> returns (B, feature_dim)
        """
        assert pts.shape[-1] == self.input_dim, f"Expected last dim {self.input_dim} (got {pts.shape[-1]})"

        orig_prefix = pts.shape[:-1]  # may be (N,) or (B, N) or (B,)
        total_points = int(torch.prod(torch.tensor(orig_prefix))) if len(orig_prefix) > 0 else 1
        # flatten leading dims to (P, D)
        flat_pts = pts.reshape(-1, self.input_dim)  # (P, input_dim)

        # Normalize coords to [-1,1] if bbox provided
        flat_pts_norm = self._normalize_pts_to_grid(flat_pts)

        # We'll sample each plane: produce sampled shape (P, C)
        # But _grid_sample_wrapper expects either (N,D) or (B,N,D). We'll call with (N,D) and it returns (N,C)
        P = flat_pts_norm.shape[0]

        fused = None  # initialize lazily to ensure correct device/dtype
        sampled_shape_device = None

        # Precompute coordinate combinations in same order as plane_coefs
        coo_combs = list(itertools.combinations(range(self.input_dim), 2))

        # Sanity check plane counts
        assert len(coo_combs) == self.num_planes, "Plane count mismatch."

        for idx, comb in enumerate(coo_combs):
            plane = self.plane_coefs[idx]  # Parameter with shape [1, C, H, W] (or similar)
            # pick coords for this plane: use columns corresponding to comb
            coords_for_plane = flat_pts_norm[:, list(comb)]  # shape (P, 2) or (P, 3)

            # sample -> returns (P, C)
            sampled = self._grid_sample_wrapper(plane, coords_for_plane)  # (P, C)
            # ensure contiguous
            sampled = sampled.view(P, -1)

            # lazy init fused with correct device/dtype
            if fused is None:
                if self.reduce == "product" or self.reduce == "sum":
                    # init as first sampled
                    fused = sampled.clone()
                elif self.reduce == "concat":
                    fused = [sampled]
                else:
                    raise ValueError("Unsupported reduce mode.")
            else:
                if self.reduce == "product":
                    fused = fused * sampled
                elif self.reduce == "sum":
                    fused = fused + sampled
                elif self.reduce == "concat":
                    fused.append(sampled)

        # finalize fused
        if self.reduce == "concat":
            fused = torch.cat(fused, dim=-1)  # (P, C_total)
        # fused is (P, feature_dim)

        # restore original prefix shape
        if len(orig_prefix) == 0:
            out = fused.view(self.feature_dim)  # scalar-case (unlikely)
        else:
            out = fused.view(*orig_prefix, fused.shape[-1])  # (..., feature_dim)

        # pass through decoder if available
        if self.decoder is not None:
            # decoder should accept shape (..., feature_dim)
            return self.decoder(temporal_h=out)
        return out
