from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from rfstudio.graphics import FeatureImages, RGBImages
from rfstudio.nn import Module
from rfstudio.utils.download import download_model_weights
from rfstudio.utils.lazy_module import torchvision


def _load(
    name: Literal['vit_b', 'vit_l', 'vit_g'],
    *,
    with_registers: bool,
    no_xformers: bool,
    pretrained: bool = True,
) -> nn.Module:
    # Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    # This source code is licensed under the Apache License, Version 2.0
    # found in the LICENSE file in the root directory of this source tree.

    from . import vit

    constructor = {
        'vit_b': vit.vit_base,
        'vit_l': vit.vit_large,
        'vit_g': vit.vit_giant2,
    }[name]
    img_size: int = 518
    patch_size: int = 14
    init_values: float = 1.0
    ffn_layer: str = "mlp" if name != 'vit_g' else "swiglufused"
    block_chunks: int = 0
    num_register_tokens: int = 4 if with_registers else 0
    interpolate_antialias: bool = not with_registers
    interpolate_offset: float = 0.0 if with_registers else 0.1

    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
        no_xformers=no_xformers,
    )
    model = constructor(**vit_kwargs)

    if pretrained:
        model_base_name = f'dinov2_vit{name[-1]}14'
        model_full_name = f'{model_base_name}_reg4' if with_registers else model_base_name
        url = f"https://dl.fbaipublicfiles.com/dinov2/{model_base_name}/{model_full_name}_pretrain.pth"
        path = download_model_weights(url)
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    return model

@dataclass
class DINOv2(Module):

    model_type: Literal['vit_g', 'vit_l', 'vit_b'] = 'vit_b'
    with_registers: bool = True
    no_xformers: bool = True

    def __setup__(self) -> None:
        self._basemodel = None

    def _load(self) -> None:
        if self._basemodel is not None:
            return
        self._basemodel = _load(self.model_type, with_registers=self.with_registers, no_xformers=self.no_xformers)
        self._basemodel.to(self.device)

    @torch.no_grad()
    def __call__(self, images: RGBImages) -> FeatureImages:
        self._load()
        transforms = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        assert images._batch and images._tensors.shape[1:] == (518, 518, 3)
        return FeatureImages(self._basemodel(
            transforms(images._tensors.permute(0, 3, 1, 2)),
            is_training=True,
        )["x_norm_patchtokens"].view(images._tensors.shape[0], 37, 37, -1))
