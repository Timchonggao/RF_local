from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.utils.colormap import IntensityColorMap, RainbowColorMap
from rfstudio.utils.decorator import lazy
from rfstudio.utils.lazy_module import torchvision
from rfstudio.utils.typing import IntArrayLike, IntLike

from .._cameras import Cameras
from .._points import Points
from .._rays import Rays
from ..math import principal_component_analysis

T = TypeVar('T', bound='BaseImages')


def _soft_clamp(t: Tensor, min: Optional[float], max: Optional[float]) -> Tensor:
    assert min is not None or max is not None
    softplus = torch.nn.Softplus(beta=100)
    if min is not None and max is not None:
        return torch.where(
            t < (min + max) / 2,
            min + softplus(t - min),
            max - softplus(max - t),
        )
    if max is not None:
        return max - softplus(max - t)
    return min + softplus(t - min)


class BaseImages(ABC):

    @abstractmethod
    def get_num_channels(cls_or_self = None) -> int:
        ...

    def __init__(self, tensors: Union[Tensor, Iterator[Tensor]]) -> None:
        C = self.__class__.get_num_channels()
        if isinstance(tensors, Tensor):
            assert tensors.ndim in [3, 4] and (C == -1 or tensors.shape[-1] == C), (
                f"Tensors must have shape (B, H, W, {C}) or (H, W, {C})."
            )
            if tensors.ndim == 3:
                tensors = tensors.unsqueeze(0)
            self._tensors = tensors
            self._batch = True
        else:
            self._tensors = list(tensors)
            assert all(t.ndim == 3 and (C == -1 or t.shape[2] == C) for t in self._tensors), (
                f"Tensors must have shape (H, W, {C})."
            )
            if all(t.shape == self._tensors[0].shape for t in self._tensors[1:]):
                self._tensors = torch.stack(self._tensors, dim=0)
                self._batch = True
            else:
                self._batch = False

    def __getitem__(self: T, indices: Union[IntArrayLike, IntLike]) -> T:
        if self._batch:
            return self.__class__(self._tensors[indices])
        if not isinstance(indices, Iterable):
            indices = (indices, )
        return self.__class__(self._tensors[i] for i in indices)

    def get(self, idx: int) -> Tensor:
        return self._tensors[idx]

    def query_pixels(self, rays: Rays) -> Float32[Tensor, "... C"]:
        if self._batch:
            return self._tensors[
                rays.pixel_indices[..., 0],
                rays.pixel_indices[..., 1],
                rays.pixel_indices[..., 2],
                :
            ]
        results = []
        for i, image in enumerate(self._tensors):
            mask = rays.pixel_indices[..., 0] == i # [...]
            results.append(image[rays.pixel_indices[..., 1][mask], rays.pixel_indices[..., 2][mask], :])
        return torch.cat(results, dim=0).view(*rays.shape, -1)

    def __len__(self) -> int:
        return len(self._tensors)

    def __iter__(self) -> Iterator[Tensor]:
        return iter(self._tensors)

    def to(self: T, device: torch.device) -> T:
        if self._batch:
            return self.__class__(self._tensors.to(device))
        return self.__class__(t.to(device) for t in self._tensors)

    def detach(self: T) -> T:
        if self._batch:
            return self.__class__(self._tensors.detach())
        return self.__class__(t.detach() for t in self._tensors)

    def cpu(self: T) -> T:
        if self._batch:
            return self.__class__(self._tensors.cpu())
        return self.__class__(t.cpu() for t in self._tensors)

    @property
    def device(self) -> torch.device:
        return self._tensors[0].device

    def item(self) -> Tensor:
        assert len(self._tensors) == 1
        return self._tensors[0]

    def clamp(self: T, min: Optional[float] = None, max: Optional[float] = None) -> T:
        self.int8_to_float32()
        if self._batch:
            return self.__class__(self._tensors.clamp(min=min, max=max))
        return self.__class__(t.clamp(min=min, max=max) for t in self._tensors)

    def soft_clamp(self: T, min: Optional[float] = None, max: Optional[float] = None) -> T:
        self.int8_to_float32()
        if self._batch:
            return self.__class__(_soft_clamp(self._tensors, min=min, max=max))
        return self.__class__(_soft_clamp(t, min=min, max=max) for t in self._tensors)

    def resize_to(self: T, width: int, height: int) -> T:
        resizer = torchvision.transforms.Resize((height, width), antialias=True)
        if self._batch:
            return self.__class__(resizer(self._tensors.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous())
        return self.__class__(torch.stack([
            resizer(t.permute(2, 0, 1)).permute(1, 2, 0).contiguous()
            for t in self._tensors
        ])).to(self.device)

    def concat(self: T, other: T) -> T:
        assert other.__class__ is self.__class__
        if self._batch and other._batch and self._tensors.shape[1:3] == other._tensors.shape[1:3]:
            return self.__class__(torch.cat((self._tensors, other._tensors)))
        return self.__class__(list(self._tensors) + list(other._tensors))

    def int8_to_float32(self: T):
        if self._tensors.dtype == torch.uint8:
            self._tensors = self._tensors.float() / 255.0

class RGBImages(BaseImages):

    def get_num_channels(cls_or_self = None) -> int:
        return 3

    def srgb2rgb(self) -> PBRImages:
        self.int8_to_float32()
        if self._batch:
            colors = self._tensors
            return PBRImages(
                torch.where(
                    colors <= 0.04045,
                    colors / 12.92,
                    torch.pow((colors.clamp_min(0.04045) + 0.055) / 1.055, 2.4),
                ),
            )
        return PBRImages(
            torch.where(
                rgb <= 0.04045,
                rgb / 12.92,
                torch.pow((rgb.clamp_min(0.04045) + 0.055) / 1.055, 2.4),
            )
            for rgb in self._tensors
        )

class FeatureImages(BaseImages):

    def __init__(self, tensors: Union[Tensor, Iterator[Tensor]]) -> None:
        super().__init__(tensors)
        self._feature_dim = tensors[0].shape[-1]

    def get_num_channels(cls_or_self = None) -> int:
        if isinstance(cls_or_self, FeatureImages):
            return cls_or_self._feature_dim
        return -1

    def visualize(self, *, gamma: Optional[float] = 2.2) -> RGBImages:
        results = []
        for t in self._tensors:
            pca = principal_component_analysis(t, dim=-1, num_components=3)
            pca = (pca - pca.min()) / (pca.max() - pca.min()).clamp_min(1e-6)
            if gamma:
                pca = pca ** (1 / gamma)
            results.append(pca.clamp(0, 1))
        if self._batch:
            results = torch.stack(results)
        return RGBImages(results)

class SegImages(BaseImages):

    def __init__(self, tensors: Union[Tensor, Iterator[Tensor]]) -> None:
        if isinstance(tensors, Tensor):
            assert tensors.dtype == torch.long
            super().__init__(tensors)
        else:
            tensors = list(tensors)
            assert all([t.dtype == torch.long for t in tensors]), "Tensors must be torch.Long type."
            super().__init__(tensors)

    def get_num_channels(cls_or_self = None) -> int:
        return 1

    def clamp(self: T, min: Optional[int] = None, max: Optional[int] = None) -> T:
        raise NotImplementedError

    def soft_clamp(self: T, min: Optional[int] = None, max: Optional[int] = None) -> T:
        raise NotImplementedError

    def resize_to(self: T, width: int, height: int) -> T:
        resizer = torchvision.transforms.Resize(
            (height, width),
            antialias=False,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )
        if self._batch:
            return self.__class__(resizer(self._tensors.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous())
        return self.__class__(torch.stack([
            resizer(t.permute(2, 0, 1)).permute(1, 2, 0).contiguous()
            for t in self._tensors
        ])).to(self.device)

    def visualize(self) -> RGBImages:
        return RGBImages([
            torch.where(
                seg > 0,
                RainbowColorMap()((seg - 1).clamp_min(0) / (seg.max() - 1).clamp_min(1)),
                0.5,
            )
            for seg in self._tensors
        ])

    @property
    @lazy
    def num_segments(self) -> List[int]:
        return [t.max().item() for t in self._tensors]


class PBRImages(BaseImages):

    def get_num_channels(cls_or_self = None) -> int:
        return 3

    def rgb2srgb(self) -> RGBImages:
        self.int8_to_float32()
        if self._batch:
            colors = self._tensors[..., :3]
            return RGBImages(
                torch.where(
                    colors <= 0.0031308,
                    colors * 12.92,
                    torch.clamp(colors, min=0.0031308).pow(1.0 / 2.4) * 1.055 - 0.055,
                )
            )
        return RGBImages(
            torch.where(
                rgb <= 0.0031308,
                rgb * 12.92,
                torch.clamp(rgb, min=0.0031308).pow(1.0 / 2.4) * 1.055 - 0.055,
            )
            for rgb in self._tensors
        )


class PBRAImages(BaseImages):

    def get_num_channels(cls_or_self = None) -> int:
        return 4

    def blend(self, background_color: Union[Tensor, Tuple[float, float, float]]) -> PBRImages:
        self.int8_to_float32()
        if not isinstance(background_color, Tensor):
            r, g, b = background_color
            background_color = torch.tensor([r, g, b])
        background_color = background_color.to(self._tensors[0])
        if self._batch:
            return PBRImages(
                self._tensors[..., 3:] * self._tensors[..., :3] +
                background_color * (1 - self._tensors[..., 3:])
            )
        return PBRImages(
            rgba[..., 3:] * rgba[..., :3] + background_color * (1 - rgba[..., 3:])
            for rgba in self._tensors
        )

    def blend_random(self) -> PBRImages:
        self.int8_to_float32()
        if self._batch:
            return PBRImages(
                self._tensors[..., 3:] * self._tensors[..., :3] +
                torch.rand_like(self._tensors[..., :3]) * (1 - self._tensors[..., 3:])
            )
        return PBRImages(
            rgba[..., 3:] * rgba[..., :3] + torch.rand_like(rgba[..., :3]) * (1 - rgba[..., 3:])
            for rgba in self._tensors
        )

    def rgb2srgb(self) -> RGBAImages:
        self.int8_to_float32()
        if self._batch:
            colors = self._tensors[..., :3]
            alphas = self._tensors[..., 3:]
            return RGBAImages(
                torch.cat((
                    torch.where(
                        colors <= 0.0031308,
                        colors * 12.92,
                        torch.clamp(colors, min=0.0031308).pow(1.0 / 2.4) * 1.055 - 0.055,
                    ),
                    alphas,
                ), dim=-1)
            )
        return RGBAImages(
            torch.cat((
                torch.where(
                    rgba[..., :3] <= 0.0031308,
                    rgba[..., :3] * 12.92,
                    torch.clamp(rgba[..., :3], min=0.0031308).pow(1.0 / 2.4) * 1.055 - 0.055,
                ),
                rgba[..., 3:],
            ), dim=-1)
            for rgba in self._tensors
        )


class RGBAImages(BaseImages):

    def get_num_channels(cls_or_self = None) -> int:
        return 4

    def blend(self, background_color: Union[Tensor, Tuple[float, float, float]]) -> RGBImages:
        self.int8_to_float32()
        if not isinstance(background_color, Tensor):
            r, g, b = background_color
            background_color = torch.tensor([r, g, b])
        background_color = background_color.to(self._tensors[0])
        if self._batch:
            return RGBImages(
                self._tensors[..., 3:] * self._tensors[..., :3] +
                background_color * (1 - self._tensors[..., 3:])
            )
        return RGBImages(
            rgba[..., 3:] * rgba[..., :3] + background_color * (1 - rgba[..., 3:])
            for rgba in self._tensors
        )

    def blend_background(self, bg: RGBImages) -> RGBImages:
        self.int8_to_float32()
        if self._batch and bg._batch:
            return RGBImages(
                self._tensors[..., 3:] * self._tensors[..., :3] +
                bg._tensors * (1 - self._tensors[..., 3:])
            )
        return RGBImages(
            rgba[..., 3:] * rgba[..., :3] + bg_rgb * (1 - rgba[..., 3:])
            for bg_rgb, rgba in zip(bg, self._tensors, strict=True)
        )

    def blend_random(self) -> RGBImages:
        self.int8_to_float32()
        if self._batch:
            return RGBImages(
                self._tensors[..., 3:] * self._tensors[..., :3] +
                torch.rand_like(self._tensors[..., :3]) * (1 - self._tensors[..., 3:])
            )
        return RGBImages(
            rgba[..., 3:] * rgba[..., :3] + torch.rand_like(rgba[..., :3]) * (1 - rgba[..., 3:])
            for rgba in self._tensors
        )

    def srgb2rgb(self) -> PBRAImages:
        self.int8_to_float32()
        if self._batch:
            colors = self._tensors[..., :3]
            alphas = self._tensors[..., 3:]
            return PBRAImages(
                torch.cat((
                    torch.where(
                        colors <= 0.04045,
                        colors / 12.92,
                        torch.pow((colors.clamp_min(0.04045) + 0.055) / 1.055, 2.4),
                    ),
                    alphas,
                ), dim=-1)
            )
        return PBRAImages(
            torch.cat((
                torch.where(
                    rgba[..., :3] <= 0.04045,
                    rgba[..., :3] / 12.92,
                    torch.pow((rgba[..., :3].clamp_min(0.04045) + 0.055) / 1.055, 2.4),
                ),
                rgba[..., 3:],
            ), dim=-1)
            for rgba in self._tensors
        )


class IntensityImages(BaseImages):

    def get_num_channels(cls_or_self = None) -> int:
        return 2

    @torch.no_grad()
    def visualize(
        self,
        color_map: IntensityColorMap = IntensityColorMap('gist_heat'),
    ) -> RGBAImages:
        if self._batch:
            max_bound = self._tensors[..., :1].max()
            min_bound = self._tensors[..., :1].min()
            colors = color_map(
                torch.where(
                    self._tensors[..., 1:] > 0.5,
                    self._tensors[..., :1],
                    (max_bound + min_bound) / 2,
                )
            )
            return RGBAImages(torch.cat((colors, self._tensors[..., 1:]), dim=-1))
        images = []
        for ia in self._tensors:
            max_bound = ia[..., :1].max()
            min_bound = ia[..., :1].min()
            colors = color_map(
                torch.where(
                    ia[..., 1:] > 0.5,
                    ia[..., :1],
                    (max_bound + min_bound) / 2,
                )
            )
            images.append(torch.cat((colors, ia[..., 1:]), dim=-1))
        return RGBAImages(images)



class DepthImages(BaseImages):

    def get_num_channels(cls_or_self = None) -> int:
        return 2

    @torch.no_grad()
    def visualize(
        self,
        color_map: IntensityColorMap = IntensityColorMap(),
        *,
        min_bound: float = 0.0,
        max_bound: Optional[float] = None,
    ) -> RGBImages:
        if self._batch:
            if max_bound is None:
                non_empty = self._tensors[..., :1] * self._tensors[..., 1:] / self._tensors[..., 1:].clamp_min(1e-10)
                max_bound = non_empty.max().item()
            scale = max(1e-10, max_bound - min_bound)
            scaled = (self._tensors[..., :1] - max_bound) / scale * self._tensors[..., 1:] + 1
            colors = color_map.from_scaled(scaled.clamp(0, 1))
            return RGBImages(colors)
        if max_bound is None:
            max_bound = min_bound
            for da in self._tensors:
                non_empty = da[..., :1] * da[..., 1:] / da[..., 1:].clamp_min(1e-10)
                max_bound = max(max_bound, non_empty.max().item())
        scale = max(1e-10, max_bound - min_bound)
        return RGBImages(
            color_map.from_scaled(((da[..., :1] - max_bound) / scale * da[..., 1:] + 1).clamp(0, 1))
            for da in self._tensors
        )

    def compute_pseudo_normals(self, cameras: Cameras) -> VectorImages:
        if self._batch:
            tensors = [self._tensors]
        else:
            tensors = self._tensors

        assert cameras.shape == ()
        pose = cameras.c2w                                             # [4, 4]
        xy = cameras.pixel_coordinates                                 # [H, W, 2]
        offset_y = (0.5 - cameras.cy + xy[:, :, 0]) / cameras.fy       # [H, W]
        offset_x = (0.5 - cameras.cx + xy[:, :, 1]) / cameras.fx       # [H, W]
        xyz_image_space = torch.stack((
            offset_x,
            -offset_y,
            -torch.ones_like(offset_x),
        ), dim=-1)                                                     # [H, W, 3]

        results = []
        for da in tensors:
            depth_values = da[..., :1]                                 # [..., H, W, 1]
            alpha_mask = da[..., 1:] > 0                               # [..., H, W, 1]
            xyz_camera_space = xyz_image_space * depth_values          # [..., H, W, 3]
            xyz_world_space = torch.add(
                pose[:3, :3] @ xyz_camera_space[..., None],
                pose[:3, 3:],
            ).squeeze(-1)                                              # [..., H, W, 3]
            dy = torch.sub(
                xyz_world_space[..., 1:, :-1, :],
                xyz_world_space[..., :-1, :-1, :]
            )                                                          # [..., H-1, W-1, 3]
            dx = torch.sub(
                xyz_world_space[..., :-1, 1:, :],
                xyz_world_space[..., :-1, :-1, :]
            )                                                          # [..., H-1, W-1, 3]
            directions = dy.cross(dx, dim=-1)                          # [..., H-1, W-1, 3]
            valid_mask = (
                alpha_mask[..., :-1, :-1, :] &
                alpha_mask[..., 1:, :-1, :] &
                alpha_mask[..., :-1, 1:, :]
            )                                                          # [..., H-1, W-1, 1]
            results.append(torch.nn.functional.pad(
                torch.cat((directions, valid_mask.float()), dim=-1),
                pad=(0, 0, 0, 1, 0, 1),
            ))                                                         # [..., H, W, 4]

        if self._batch:
            return VectorImages(results[0])
        return VectorImages(results)

    def deproject(self, cameras: Cameras, *, alpha_threshold: Optional[float] = None) -> Points:
        assert cameras.shape == ()
        points = []
        pose = cameras.c2w                                               # [4, 4]
        xy = cameras.pixel_coordinates                                   # [H, W, 2]
        offset_y = (0.5 - cameras.cy + xy[:, :, 0]) / cameras.fy         # [H, W]
        offset_x = (0.5 - cameras.cx + xy[:, :, 1]) / cameras.fx         # [H, W]
        xyz_base = torch.stack((
            offset_x,
            -offset_y,
            -torch.ones_like(offset_x),
        ), dim=-1).view(-1, 3)                                           # [H * W, 3]
        for da in self._tensors:
            assert da.shape == xy.shape                                  # [H, W, 2]
            if alpha_threshold is None:
                valid_mask = (da[:, :, 1:] > 0).flatten()                # [H * W]
            else:
                valid_mask = (da[:, :, 1:] >= alpha_threshold).flatten() # [H * W]
            depth_values = da[:, :, :1].view(-1, 1)[valid_mask, :]       # [N, 1]
            xyz_image_space = xyz_base[valid_mask, :].view(-1, 3)        # [N, 3]
            xyz_camera_space = xyz_image_space * depth_values            # [N, 3]
            xyz_world_space = torch.add(
                pose[:3, :3] @ xyz_camera_space[..., None],
                pose[:3, 3:],
            )                                                            # [N, 3, 1]
            points.append(xyz_world_space.squeeze(-1))
        return Points(positions=torch.cat(points, dim=0))


class VectorImages(BaseImages):

    def get_num_channels(cls_or_self = None) -> int:
        return 4

    @overload
    def visualize(self) -> RGBAImages:
        ...

    @overload
    def visualize(self, background_color: Union[Tensor, Tuple[float, float, float]]) -> RGBImages:
        ...

    @torch.no_grad()
    def visualize(
        self,
        background_color: Union[Tensor, Tuple[float, float, float]] = None,
    ) -> Union[RGBImages, RGBAImages]:
        if background_color is None:
            if self._batch:
                vectors = self._tensors[..., :3]
                normalized = vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                return RGBAImages(torch.cat((normalized[..., :3] * 0.5 + 0.5, self._tensors[..., 3:]), dim=-1))
            return RGBAImages(
                torch.cat((
                    (xyz[..., :3] / xyz[..., :3].norm(dim=-1, keepdim=True).clamp(min=1e-8)) * 0.5 + 0.5,
                    xyz[..., 3:],
                ), dim=-1)
                for xyz in self._tensors
            )

        if not isinstance(background_color, Tensor):
            r, g, b = background_color
            background_color = torch.tensor([r, g, b])
        background_color = background_color.to(self._tensors[0])
        if self._batch:
            vectors = self._tensors[..., :3]
            normalized = vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            return RGBImages(
                torch.add(
                    (normalized * 0.5 + 0.5) * self._tensors[..., 3:],
                    (1 - self._tensors[..., 3:]) * background_color,
                )
            )
        return RGBImages(
            xyz[..., 3:] * torch.add(
                torch.div(
                    xyz[..., :3],
                    xyz[..., :3].norm(dim=-1, keepdim=True).clamp(min=1e-8),
                ) * 0.5,
                0.5,
            ) + (1 - xyz[..., 3:]) * background_color
            for xyz in self._tensors
        )


class RGBDImages(BaseImages):

    def get_num_channels(cls_or_self = None) -> int:
        return 5

    def split(self) -> Tuple[RGBAImages, DepthImages]:
        if self._batch:
            return (
                RGBAImages(self._tensors[..., [0, 1, 2, 4]].contiguous()),
                DepthImages(self._tensors[..., 3:].contiguous()),
            )
        return (
            RGBAImages(rgbda[..., [0, 1, 2, 4]].contiguous() for rgbda in self._tensors),
            DepthImages(rgbda[..., 3:].contiguous() for rgbda in self._tensors),
        )

    @torch.no_grad()
    def warping(self, camera_from: Cameras, camera_to: Cameras) -> RGBAImages:
        assert (
            camera_from.shape == () and camera_to.shape == () and
            camera_from.width.item() == camera_to.width.item() and
            camera_from.height.item() == camera_to.height.item()
        )
        # (x1, y1, d1) = I1 @ E1 @ global_xyz
        # (x2, y2, d2) = I2 @ E2 @ global_xyz = I2 @ (E2 @ E1^{-1}) @ I1^{-1} @ (x1, y1, d1)
        xy = camera_from.pixel_coordinates                                      # [H, W, 2]
        pad = torch.tensor([0, 0, 0, 1]).to(camera_from.c2w).view(-1, 4)
        relative_pose = torch.matmul(
            torch.cat((camera_to.c2w, pad), dim=-2).inverse(),
            torch.cat((camera_from.c2w, pad), dim=-2),
        )               # [4, 4]
        images = []
        for rgbda in self._tensors:
            assert rgbda.shape == (xy.shape[0], xy.shape[1], 5)             # [H, W, 5]
            valid_mask = (rgbda[:, :, 4:] > 0).flatten()                # [H * W]
            rgbd_values = rgbda[:, :, :4].view(-1, 4)[valid_mask, :]      # [N, 4]
            scaled_xy = xy.view(-1, 2)[valid_mask, :]                       # [N, 2]
            offset_y = (0.5 - camera_from.cy + scaled_xy[:, 0]) / camera_from.fy    # [N]
            offset_x = (0.5 - camera_from.cx + scaled_xy[:, 1]) / camera_from.fx    # [N]
            xyz_camera_space = torch.stack((
                offset_x,
                -offset_y,
                -torch.ones_like(offset_x),
            ), dim=-1) * rgbd_values[..., 3:]                                       # [N, 3]
            xyz_camera_space = (relative_pose[:3, :3] @ xyz_camera_space[..., None] + relative_pose[:3, 3:]).squeeze(-1)
            xy_projected = xyz_camera_space[..., :2] / xyz_camera_space[..., 2:] # [N, 2]
            warped_xy = torch.stack((
                -xy_projected[..., 0] * camera_to.fx + (camera_to.cx - 0.5),
                xy_projected[..., 1] * camera_to.fy + (camera_to.cy - 0.5),
            ), dim=-1).round().long() # [N, 2]
            valid_xy = (warped_xy >= 0).all(-1) & (warped_xy[..., 0] < xy.shape[1]) & (warped_xy[..., 1] < xy.shape[0])
            indices = warped_xy[valid_xy, 0] + warped_xy[valid_xy, 1] * xy.shape[1] # [N']
            depth_projected = -xyz_camera_space[valid_xy, 2] # [N']
            depth_buffer = torch.empty_like(rgbda[..., :1].flatten()).fill_(torch.inf) # [H * W]
            rgb_buffer = torch.zeros_like(rgbda[..., :3].view(-1, 3)) # [H * W, 3]
            depth_buffer.scatter_reduce_(
                dim=0,
                index=indices,
                src=depth_projected,
                reduce='min',
                include_self=True,
            )
            visible = (depth_projected == depth_buffer[indices]) # [N']
            rgb_buffer.scatter_(
                dim=0,
                index=indices[visible, None].repeat(1, 3),
                src=rgbd_values[valid_xy, :3][visible, :3],
            )
            alpha = ((depth_buffer.isfinite()) & (depth_buffer > 0)).float().unsqueeze(-1)
            images.append(torch.cat((rgb_buffer * alpha, alpha), dim=-1).view_as(rgbda[..., :4]))
        return RGBAImages(images)


    def deproject(self, cameras: Cameras, *, alpha_threshold: Optional[float] = None) -> Points:
        assert cameras.shape == ()
        xy = cameras.pixel_coordinates                                      # [H, W, 2]
        pose = cameras.c2w                                                  # [4, 4]
        points = []
        colors = []
        for rgbda in self._tensors:
            assert rgbda.shape == (xy.shape[0], xy.shape[1], 5)             # [H, W, 5]
            if alpha_threshold is None:
                valid_mask = (rgbda[:, :, 4:] > 0).flatten()                # [H * W]
            else:
                valid_mask = (rgbda[:, :, 4:] >= alpha_threshold).flatten() # [H * W]
            depth_values = rgbda[:, :, 3:4].view(-1, 1)[valid_mask, :]      # [N, 1]
            scaled_xy = xy.view(-1, 2)[valid_mask, :]                       # [N, 2]
            offset_y = (0.5 - cameras.cy + scaled_xy[:, 0]) / cameras.fy    # [N]
            offset_x = (0.5 - cameras.cx + scaled_xy[:, 1]) / cameras.fx    # [N]
            xyz_camera_space = torch.stack((
                offset_x,
                -offset_y,
                -torch.ones_like(offset_x),
            ), dim=-1) * depth_values                                       # [N, 3]
            xyz_world_space = torch.add(
                pose[:3, :3] @ xyz_camera_space[..., None],
                pose[:3, 3:],
            )                                                               # [N, 3, 1]
            rgb = rgbda[:, :, :3].view(-1, 3)[valid_mask, :]                # [N, 3]
            points.append(xyz_world_space.squeeze(-1))
            colors.append(rgb)
        return Points(
            positions=torch.cat(points, dim=0),
            colors=torch.cat(colors, dim=0),
        )


class RGBDNImages(BaseImages):

    def get_num_channels(cls_or_self = None) -> int:
        return 8

    def split(self) -> Tuple[RGBAImages, DepthImages, VectorImages]:
        if self._batch:
            return (
                RGBAImages(self._tensors[..., [0, 1, 2, 7]].contiguous()),
                DepthImages(self._tensors[..., [3, 7]].contiguous()),
                VectorImages(self._tensors[..., 4:].contiguous()),
            )
        return (
            RGBAImages(rgbda[..., [0, 1, 2, 7]].contiguous() for rgbda in self._tensors),
            DepthImages(rgbda[..., [3, 7]].contiguous() for rgbda in self._tensors),
            VectorImages(rgbda[..., 4:].contiguous() for rgbda in self._tensors),
        )

    @torch.no_grad()
    def warping(self, camera_from: Cameras, camera_to: Cameras) -> RGBAImages:
        assert (
            camera_from.shape == () and camera_to.shape == () and
            camera_from.width.item() == camera_to.width.item() and
            camera_from.height.item() == camera_to.height.item()
        )
        # (x1, y1, d1) = I1 @ E1 @ global_xyz
        # (x2, y2, d2) = I2 @ E2 @ global_xyz = I2 @ (E2 @ E1^{-1}) @ I1^{-1} @ (x1, y1, d1)
        xy = camera_from.pixel_coordinates                                      # [H, W, 2]
        pad = torch.tensor([0, 0, 0, 1]).to(camera_from.c2w).view(-1, 4)
        relative_pose = torch.matmul(
            torch.cat((camera_to.c2w, pad), dim=-2).inverse(),
            torch.cat((camera_from.c2w, pad), dim=-2),
        ) # [4, 4]
        images = []
        for rgbdxyza in self._tensors:
            assert rgbdxyza.shape == (xy.shape[0], xy.shape[1], 8)             # [H, W, 5]
            valid_mask = (rgbdxyza[:, :, 7:] > 0).flatten()                # [H * W]
            rgbdxyz_values = rgbdxyza[:, :, :7].view(-1, 7)[valid_mask, :]      # [N, 7]
            scaled_xy = xy.view(-1, 2)[valid_mask, :]                       # [N, 2]
            offset_y = (0.5 - camera_from.cy + scaled_xy[:, 0]) / camera_from.fy    # [N]
            offset_x = (0.5 - camera_from.cx + scaled_xy[:, 1]) / camera_from.fx    # [N]
            xyz_camera_space = torch.stack((
                offset_x,
                -offset_y,
                -torch.ones_like(offset_x),
            ), dim=-1) * rgbdxyz_values[..., 3:4]                                       # [N, 3]
            xyz_camera_space = (relative_pose[:3, :3] @ xyz_camera_space[..., None] + relative_pose[:3, 3:]).squeeze(-1)
            xy_projected = xyz_camera_space[..., :2] / xyz_camera_space[..., 2:] # [N, 2]
            warped_xy = torch.stack((
                -xy_projected[..., 0] * camera_to.fx + (camera_to.cx - 0.5),
                xy_projected[..., 1] * camera_to.fy + (camera_to.cy - 0.5),
            ), dim=-1).round().long() # [N, 2]
            valid_xy = (
                (warped_xy >= 0).all(-1) &
                (warped_xy[..., 0] < xy.shape[1]) &
                (warped_xy[..., 1] < xy.shape[0])
            ) # [N]
            indices = warped_xy[valid_xy, 0] + warped_xy[valid_xy, 1] * xy.shape[1] # [N']
            depth_projected = -xyz_camera_space[valid_xy, 2] # [N']
            depth_buffer = torch.empty_like(rgbdxyza[..., :1].flatten()).fill_(torch.inf) # [H * W]
            rgbxyz_buffer = torch.zeros_like(rgbdxyza[..., :6].view(-1, 6)) # [H * W, 6]
            depth_buffer.scatter_reduce_(
                dim=0,
                index=indices,
                src=depth_projected,
                reduce='min',
                include_self=True,
            )
            visible = (depth_projected == depth_buffer[indices]) # [N']
            rgbxyz_buffer.scatter_(
                dim=0,
                index=indices[visible, None].repeat(1, 6),
                src=rgbdxyz_values[valid_xy, :][visible, :][:, [0, 1, 2, 4, 5, 6]],
            )
            point_to_camera = (camera_to.c2w[:, 2] * rgbxyz_buffer[..., 3:]).sum(-1) > 0 # [H * W]
            alpha = ((depth_buffer.isfinite()) & (depth_buffer > 0) & point_to_camera).float().unsqueeze(-1)
            rgb = rgbxyz_buffer[..., :3] * alpha
            images.append(torch.cat((rgb, alpha), dim=-1).view_as(rgbdxyza[..., :4]))
        return RGBAImages(images)

    @torch.no_grad()
    def warping_with_warpability(self, camera_from: Cameras, camera_to: Cameras) -> Tuple[RGBAImages, RGBAImages]:
        assert (
            camera_from.shape == () and camera_to.shape == () and
            camera_from.width.item() == camera_to.width.item() and
            camera_from.height.item() == camera_to.height.item()
        )
        # (x1, y1, d1) = I1 @ E1 @ global_xyz
        # (x2, y2, d2) = I2 @ E2 @ global_xyz = I2 @ (E2 @ E1^{-1}) @ I1^{-1} @ (x1, y1, d1)
        xy = camera_from.pixel_coordinates                                      # [H, W, 2]
        pad = torch.tensor([0, 0, 0, 1]).to(camera_from.c2w).view(-1, 4)
        relative_pose = torch.matmul(
            torch.cat((camera_to.c2w, pad), dim=-2).inverse(),
            torch.cat((camera_from.c2w, pad), dim=-2),
        ) # [4, 4]
        images = []
        ref_inds_images = []
        for rgbdxyza in self._tensors:
            assert rgbdxyza.shape == (xy.shape[0], xy.shape[1], 8)             # [H, W, 5]
            valid_mask = (rgbdxyza[:, :, 7:] > 0).flatten()                # [H * W]
            ref_indices = valid_mask.nonzero().flatten() # [N]
            rgbdxyz_values = rgbdxyza[:, :, :7].view(-1, 7)[valid_mask, :]      # [N, 7]
            scaled_xy = xy.view(-1, 2)[valid_mask, :]                       # [N, 2]
            offset_y = (0.5 - camera_from.cy + scaled_xy[:, 0]) / camera_from.fy    # [N]
            offset_x = (0.5 - camera_from.cx + scaled_xy[:, 1]) / camera_from.fx    # [N]
            xyz_camera_space = torch.stack((
                offset_x,
                -offset_y,
                -torch.ones_like(offset_x),
            ), dim=-1) * rgbdxyz_values[..., 3:4]                                       # [N, 3]
            xyz_camera_space = (relative_pose[:3, :3] @ xyz_camera_space[..., None] + relative_pose[:3, 3:]).squeeze(-1)
            xy_projected = xyz_camera_space[..., :2] / xyz_camera_space[..., 2:] # [N, 2]
            warped_xy = torch.stack((
                -xy_projected[..., 0] * camera_to.fx + (camera_to.cx - 0.5),
                xy_projected[..., 1] * camera_to.fy + (camera_to.cy - 0.5),
            ), dim=-1).round().long() # [N, 2]
            valid_xy = (
                (warped_xy >= 0).all(-1) &
                (warped_xy[..., 0] < xy.shape[1]) &
                (warped_xy[..., 1] < xy.shape[0])
            ) # [N]
            indices = warped_xy[valid_xy, 0] + warped_xy[valid_xy, 1] * xy.shape[1] # [N']
            ref_indices = ref_indices[valid_xy] # [N']
            depth_projected = -xyz_camera_space[valid_xy, 2] # [N']
            depth_buffer = torch.empty_like(rgbdxyza[..., :1].flatten()).fill_(torch.inf) # [H * W]
            rgbxyz_buffer = torch.zeros_like(rgbdxyza[..., :6].view(-1, 6)) # [H * W, 6]
            ref_inds_buffer = torch.zeros_like(rgbdxyza[..., :1].flatten(), dtype=torch.long).fill_(-1) # [H * W]
            depth_buffer.scatter_reduce_(
                dim=0,
                index=indices,
                src=depth_projected,
                reduce='min',
                include_self=True,
            )
            visible = depth_projected == depth_buffer[indices] # [N']
            rgbxyz_buffer.scatter_(
                dim=0,
                index=indices[visible, None].repeat(1, 6),
                src=rgbdxyz_values[valid_xy, :][visible, :][:, [0, 1, 2, 4, 5, 6]],
            )
            ref_inds_buffer.scatter_(
                dim=0,
                index=indices[visible],
                src=ref_indices[visible],
            )
            point_to_camera = (camera_to.c2w[:, 2] * rgbxyz_buffer[..., 3:]).sum(-1) > 0 # [H * W]
            alpha_mask = (depth_buffer.isfinite()) & (depth_buffer > 0) & point_to_camera # [H * W]
            rgb = rgbxyz_buffer[..., :3] * alpha_mask.float().unsqueeze(-1)
            images.append(torch.cat((rgb, alpha_mask.float().unsqueeze(-1)), dim=-1).view_as(rgbdxyza[..., :4]))
            ref_inds_buffer = torch.where(alpha_mask, ref_inds_buffer, -1)
            ref_inds = ref_inds_buffer[ref_inds_buffer > -1].unique()
            ref_inds_rgb = rgbdxyza[..., :3].view(-1, 3).clone().contiguous()
            ref_inds_rgb[ref_inds, :] = (torch.tensor([32, 32, 224]).to(rgb) / 255)
            ref_inds_images.append(torch.cat((ref_inds_rgb.view_as(rgbdxyza[..., :3]), rgbdxyza[..., 7:]), dim=-1))
        return RGBAImages(images), RGBAImages(ref_inds_images)
