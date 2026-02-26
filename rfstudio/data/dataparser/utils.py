from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import torch

from rfstudio.graphics import PBRAImages, RGBAImages, RGBImages
from rfstudio.io import load_float32_image, load_float32_masked_image
from rfstudio.utils.typing import IntArrayLike, IntLike


def _call(x: partial[Any]) -> Any:
    return x()


class LazyImageBatchProxy:

    def __init__(
        self,
        promises: List[partial[RGBImages]],
        *,
        num_workers: Optional[int],
        device: torch.device,
    ) -> None:
        self._batch: Union[List[partial[RGBImages]], RGBImages]
        self._batch = promises
        self._num_workers = num_workers
        self._loaded = False
        self._device = device

    def load(self) -> None:
        if self._loaded:
            return
        assert isinstance(self._batch, list)
        if self._num_workers is not None and self._num_workers <= 0:
            self._batch = RGBImages(promise() for promise in self._batch).to(self._device)
        else:
            with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
                result = pool.map(_call, self._batch, chunksize=4)
            self._batch = RGBImages(result).to(self._device)
        self._loaded = True

    def __getitem__(self, index: Union[IntArrayLike, IntLike]) -> RGBImages:
        if not self._loaded:
            self.load()
        assert isinstance(self._batch, RGBImages)
        results = self._batch[index]
        results.int8_to_float32()
        return results

    def __len__(self) -> int:
        return len(self._batch)


class LazyMaskedImageBatchProxy:

    def __init__(
        self,
        promises: List[partial[RGBAImages]],
        *,
        num_workers: Optional[int],
        device: torch.device,
        pbra: bool = False,
    ) -> None:
        self._batch: Union[List[partial[RGBAImages]], RGBAImages]
        self._batch = promises
        self._num_workers = num_workers
        self._loaded = False
        self._device = device
        self._pbra = pbra

    def load(self) -> None:
        if self._loaded:
            return
        assert isinstance(self._batch, list)
        img_class = PBRAImages if self._pbra else RGBAImages
        if self._num_workers is not None and self._num_workers <= 0:
            self._batch = img_class(promise() for promise in self._batch).to(self._device)
        else:
            with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
                result = pool.map(_call, self._batch, chunksize=4)
            self._batch = img_class(result).to(self._device)
        if self._pbra:
            self._batch = self._batch.rgb2srgb()
        self._loaded = True

    def __getitem__(self, index: Union[IntArrayLike, IntLike]) -> RGBAImages:
        if not self._loaded:
            self.load()
        assert isinstance(self._batch, RGBAImages)
        results = self._batch[index]
        results.int8_to_float32()
        return results

    def __len__(self) -> int:
        return len(self._batch)


def load_image_batch_lazy(
    filenames: Iterable[Path],
    *,
    device: torch.device,
    scale_factor: Optional[float] = None,
    alpha_color: Optional[Tuple[float, float, float]] = None,
    num_workers: Optional[int] = 4,
) -> LazyImageBatchProxy:
    promises = [
        partial(load_float32_image, filename, scale_factor=scale_factor, alpha_color=alpha_color)
        for filename in filenames
    ]
    return LazyImageBatchProxy(promises, num_workers=num_workers, device=device)


def _load_image_and_mask(
    img_file: Path,
    mask_file: Path,
    *,
    scale_factor: Optional[float] = None,
    read_uint8: bool = False,
    read_mask_uint8: bool = False,
) -> torch.Tensor:
    img = load_float32_image(img_file, scale_factor=scale_factor, read_uint8=read_uint8)
    mask = load_float32_image(mask_file, scale_factor=scale_factor, read_uint8=read_mask_uint8)
    assert img.shape == mask.shape
    if read_uint8:
        mask = mask.float() / 255.0
        mask = (mask > 0.5).astype(img.dtype) * 255
        imga = torch.cat((img, mask), dim=-1)
        return imga
    else:
        return torch.cat((img, (mask > 0.5).any(-1, keepdim=True).float()), dim=-1)


def load_masked_image_batch_lazy(
    filenames: Iterable[Path],
    *,
    device: torch.device,
    scale_factor: Optional[float] = None,
    num_workers: Optional[int] = 4,
    masks: Optional[Iterable[Path]] = None,
    pbra: bool = False,
    read_uint8: bool = False,
    read_mask_uint8: bool = False,
) -> LazyMaskedImageBatchProxy:
    if masks is None:
        promises = [
            partial(load_float32_masked_image, filename, scale_factor=scale_factor, read_uint8=read_uint8)
            for filename in filenames
        ]
    else:
        promises = [
            partial(_load_image_and_mask, img, mask, scale_factor=scale_factor, read_uint8=read_uint8, read_mask_uint8=read_mask_uint8)
            for img, mask in zip(filenames, masks, strict=True)
        ]
    return LazyMaskedImageBatchProxy(promises, num_workers=num_workers, device=device, pbra=pbra)
