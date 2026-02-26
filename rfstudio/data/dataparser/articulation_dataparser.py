from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from rfstudio.graphics import Cameras, RGBAImages
from rfstudio.utils.typing import Indexable

from .base_dataparser import BaseDataparser
from .utils import load_masked_image_batch_lazy


@dataclass
class ArticulationDataparser(BaseDataparser[Cameras, RGBAImages, Tensor]):

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
    ) -> Tuple[Indexable[Cameras], Indexable[RGBAImages], Tensor]:

        IMAGE_WIDTH = 800
        IMAGE_HEIGHT = 800

        if self.scale_factor is not None:
            IMAGE_WIDTH = int(IMAGE_WIDTH * self.scale_factor)
            IMAGE_HEIGHT = int(IMAGE_HEIGHT * self.scale_factor)

        total_cameras = []
        total_image_filenames = []

        for state in ['start', 'end']:
            with open(path / f"transforms_train_{state}.json", 'r') as f:
                meta = json.loads(f.read())
            total_image_filenames.append([path / (frame['file_path'] + ".png") for frame in meta['frames']])
            poses = np.array([frame['transform_matrix'] for frame in meta['frames']], dtype=np.float32)

            camera_angle_x = float(meta["camera_angle_x"])
            camera_angle_y = float(meta["camera_angle_y"])
            focal_length_x = 0.5 * IMAGE_WIDTH / np.tan(0.5 * camera_angle_x)
            focal_length_y = 0.5 * IMAGE_HEIGHT / np.tan(0.5 * camera_angle_y)

            c2w = torch.from_numpy(poses[:, :3, :])            # camera to world transform
            c2w[:, :, 3] *= 2 / 3                              # scale to bbox [-1, 1]^3

            N = c2w.shape[0]
            cameras = Cameras(
                c2w=c2w,
                fx=torch.ones(N) * focal_length_x,
                fy=torch.ones(N) * focal_length_y,
                cx=torch.ones(N) * IMAGE_WIDTH * 0.5,
                cy=torch.ones(N) * IMAGE_HEIGHT * 0.5,
                width=torch.ones(N, dtype=torch.long) * IMAGE_WIDTH,
                height=torch.ones(N, dtype=torch.long) * IMAGE_HEIGHT,
                near=torch.ones(N) * (4 / 3),
                far=torch.ones(N) * 4
            ).to(device)
            total_cameras.append(cameras)

        images = load_masked_image_batch_lazy(
            total_image_filenames[0] + total_image_filenames[1],
            device=device,
            scale_factor=self.scale_factor,
        )
        cameras = Cameras.cat(total_cameras, dim=0)
        timestamps = torch.cat((
            torch.tensor([0]).repeat(len(total_image_filenames[0])),
            torch.tensor([1]).repeat(len(total_image_filenames[1])),
        ), dim=0).float().to(device)
        return cameras, images, timestamps

    @staticmethod
    def recognize(path: pathlib.Path) -> bool:
        paths = [
            path / 'start' / 'train' / 'rgba',
            path / 'end' / 'train' / 'rgba',
            path / 'gt' / 'start',
            path / 'gt' / 'end',
            path / 'transforms_train_start.json',
            path / 'transforms_train_end.json',
        ]
        return all([p.exists() for p in paths])
