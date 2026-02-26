from __future__ import annotations

import pathlib
from typing import Optional, Tuple

import cv2
import numpy as np
import pyexr
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor


def load_float32_image(
    filename: pathlib.Path,
    *,
    scale_factor: Optional[float] = None,
    alpha_color: Optional[Tuple[float, float, float]] = None,
    read_uint8: bool = False,
) -> Float[Tensor, "H W C"]:
    if filename.suffix == '.hdr':
        image = cv2.imread(str(filename), cv2.IMREAD_UNCHANGED)[..., [2, 1, 0]]
        assert image.dtype == np.float32
        assert scale_factor is None and image.shape[-1] == 3
    elif filename.suffix == '.exr':
        with pyexr.open(filename) as file:
            image = file.get()[..., :3]
        assert image.dtype == np.float32
        assert scale_factor is None and image.shape[-1] == 3
    else:
        pil_image = Image.open(filename)
        if scale_factor is not None:
            width, height = pil_image.size
            newsize = (int(width * scale_factor), int(height * scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        if read_uint8:
            image = np.array(pil_image, dtype="uint8")
        else:
            image = np.array(pil_image, dtype="uint8").astype(np.float32) / 255
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1) # [H, W, 3]
    if alpha_color is not None and image.shape[-1] == 4:
        if read_uint8:
            image = image.astype(np.float32) / 255
            image = image[..., :3] * image[..., 3:] + (1 - image[..., 3:]) * np.asarray(alpha_color).astype(np.float32)
            image = (image * 255).astype(np.uint8)
        else:
            image = image[..., :3] * image[..., 3:] + (1 - image[..., 3:]) * np.asarray(alpha_color).astype(np.float32)
    if alpha_color is None and image.shape[-1] == 4: # load four channel mask as binary
        alpha = image[..., 0]
        if read_uint8:
            alpha = alpha.astype(np.float32) / 255  # 转换为0-1范围
            mask = (alpha < 0.5).astype(np.uint8)[..., None] * 255  # 增加一个维度
        else:
            mask = (alpha < 0.5).astype(np.float32)[..., None]  # 增加一个维度
        image = np.repeat(mask, 3, axis=-1)
    assert image.shape[-1] == 3
    return torch.from_numpy(image).contiguous()

def load_float32_masked_image(
    filename: pathlib.Path,
    *,
    scale_factor: Optional[float] = None,
    read_uint8: bool = False
) -> Float[Tensor, "H W C"]:
    assert filename.suffix == '.png'
    pil_image = Image.open(filename)
    if scale_factor is not None:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
    if read_uint8:
        image = np.array(pil_image, dtype="uint8")
    else:
        image = np.array(pil_image, dtype="uint8").astype(np.float32) / 255
    assert image.shape[-1] == 4
    return torch.from_numpy(image)

def dump_float32_image(filename: pathlib.Path, image: Float[Tensor, "H W 3"]) -> None:
    assert filename.parent.exists()
    if filename.suffix == '.hdr':
        assert image.shape[-1] == 3
        cv2.imwrite(filename, image.detach().float().cpu().numpy()[..., [2, 1, 0]])
    elif filename.suffix == '.exr':
        pyexr.write(filename, image.detach().float().cpu().numpy())
    else:
        assert image.shape[-1] in [3, 4]
        mode = 'RGB' if image.shape[-1] == 3 else 'RGBA'
        pil_image = Image.fromarray(image.detach().mul(255).byte().cpu().numpy().astype(np.uint8), mode)
        pil_image.save(filename)
