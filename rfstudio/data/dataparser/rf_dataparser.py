from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple

import torch

from rfstudio.graphics import Cameras, RGBAImages, SegImages, SegTree
from rfstudio.io import dump_float32_image
from rfstudio.ui import console
from rfstudio.utils.tensor_container import TensorLikeList
from rfstudio.utils.typing import Indexable

from .base_dataparser import BaseDataparser
from .utils import load_masked_image_batch_lazy


@dataclass
class RFMaskedRealDataparser(BaseDataparser[Cameras, RGBAImages, Any]):

    scale_factor: Optional[float] = None
    """
    scale factor for resizing image
    """

    def parse(
        self,
        path: pathlib.Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[Cameras], Indexable[RGBAImages], Any]:

        image_filenames = list((path / 'images').glob("*.png"))
        image_filenames = [path / 'images' / f'{i:04d}.png' for i in range(len(image_filenames))]

        camera_data = torch.load(path / 'cameras.pkl', map_location='cpu')
        cameras = Cameras(
            c2w=camera_data['c2w'],
            fx=camera_data['fx'],
            fy=camera_data['fy'],
            cx=camera_data['cx'],
            cy=camera_data['cy'],
            width=camera_data['width'],
            height=camera_data['height'],
            near=camera_data['near'],
            far=camera_data['far'],
        ).to(device)

        images = load_masked_image_batch_lazy(
            image_filenames,
            device=device,
            scale_factor=self.scale_factor,
        )

        return cameras, images, None

    @staticmethod
    def dump(
        inputs: Cameras,
        gt_outputs: RGBAImages,
        meta: Any,
        *,
        path: pathlib.Path,
        split: Literal['all'],
    ) -> None:

        assert inputs.ndim == 1
        assert split == 'all'
        assert len(inputs) == len(gt_outputs)
        inputs = inputs.detach().cpu()

        assert path.exists()
        (path / 'images').mkdir(exist_ok=True)

        camera_data = {
            'c2w': inputs.c2w,
            'fx': inputs.fx,
            'fy': inputs.fy,
            'cx': inputs.cx,
            'cy': inputs.cy,
            'width': inputs.width,
            'height': inputs.height,
            'near': inputs.near,
            'far': inputs.far,
        }
        torch.save(camera_data, path / 'cameras.pkl')
        for idx, image in enumerate(gt_outputs):
            dump_float32_image(path / 'images' / f'{idx:04d}.png', image)

    @staticmethod
    def recognize(path: pathlib.Path) -> bool:
        paths = [
            path / 'images' / '0000.png',
            path / 'cameras.pkl',
        ]
        return all([p.exists() for p in paths])


@dataclass
class RFSegTreeDataparser(BaseDataparser[Cameras, SegTree, SegImages]):

    scale_factor: Optional[float] = None
    """
    scale factor for resizing image
    """

    def __post_init__(self) -> None:
        self._cache = None

    def parse(
        self,
        path: pathlib.Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[Cameras], Indexable[SegTree], SegImages]:
        if self._cache is None or device != self._cache[-1]:
            image_filenames = list((path / 'images').glob("*.png"))
            image_filenames = [path / 'images' / f'{i:04d}.png' for i in range(len(image_filenames))]

            camera_data = torch.load(path / 'cameras.pkl', map_location='cpu')
            cameras = Cameras(
                c2w=camera_data['c2w'],
                fx=camera_data['fx'],
                fy=camera_data['fy'],
                cx=camera_data['cx'],
                cy=camera_data['cy'],
                width=camera_data['width'],
                height=camera_data['height'],
                near=camera_data['near'],
                far=camera_data['far'],
            ).to(device)

            images = load_masked_image_batch_lazy(
                image_filenames,
                device=device,
                scale_factor=self.scale_factor,
            )[...]

            segtree_data = torch.load(path / 'segtree.pkl', map_location=device)
            segtree = TensorLikeList(
                SegTree(
                    cluster_correlation=item['cluster_correlation'],
                    pixel2cluster=item['pixel2cluster'],
                    cluster2mask=item['cluster2mask'],
                    masks=item['masks'],
                    image=img,
                )
                for item, img in zip(segtree_data, images, strict=True)
            )

            self._cache = cameras, segtree, SegImages(item['merge'] for item in segtree_data), device
        return self._cache[:3]

    @staticmethod
    def dump(
        inputs: Cameras,
        gt_outputs: List[SegTree],
        meta: Optional[SegImages] = None,
        *,
        path: pathlib.Path,
        split: Literal['all'],
        vis: bool = False,
        progress: Optional[str] = None,
    ) -> None:

        assert inputs.ndim == 1
        assert split == 'all'
        assert len(inputs) == len(gt_outputs)
        inputs = inputs.detach().cpu()

        assert path.exists()
        (path / 'images').mkdir(exist_ok=True)
        if vis:
            (path / 'visual').mkdir(exist_ok=True)

        camera_data = {
            'c2w': inputs.c2w,
            'fx': inputs.fx,
            'fy': inputs.fy,
            'cx': inputs.cx,
            'cy': inputs.cy,
            'width': inputs.width,
            'height': inputs.height,
            'near': inputs.near,
            'far': inputs.far,
        }
        torch.save(camera_data, path / 'cameras.pkl')
        segtree_data = []
        with console.progress(desc=progress, transient=True, enabled=progress is not None) as ptrack:
            for idx, segtree in enumerate(ptrack(gt_outputs)):
                assert segtree.image is not None
                dump_float32_image(path / 'images' / f'{idx:04d}.png', segtree.image)
                seg_img = segtree.merge() if meta is None else meta[idx]
                segtree_data.append({
                    'cluster_correlation': segtree.cluster_correlation,
                    'pixel2cluster': segtree.pixel2cluster,
                    'cluster2mask': segtree.cluster2mask,
                    'masks': segtree.masks,
                    'merge': seg_img.item(),
                })
                if vis:
                    dump_float32_image(path / 'visual' / f'{idx:04d}.png', seg_img.visualize().item())
                    # dump_float32_image(path / 'visual' / f'{idx:04d}.mask.png', segtree.visualize_masks())
                    # dump_float32_image(path / 'visual' / f'{idx:04d}.corr.png', segtree.visualize_correlation())
        torch.save(segtree_data, path / 'segtree.pkl')

    @staticmethod
    def recognize(path: pathlib.Path) -> bool:
        paths = [
            path / 'images' / '0000.png',
            path / 'cameras.pkl',
            path / 'segtree.pkl',
        ]
        return all([p.exists() for p in paths])
