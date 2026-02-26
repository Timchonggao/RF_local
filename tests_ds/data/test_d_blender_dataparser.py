from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras, RGBAImages, Texture2D, TriangleMesh
from rfstudio.graphics.shaders import NormalShader
from rfstudio.io import dump_float32_image, load_float32_masked_image


def parse(
    path: Path,
    *,
    idx: int,
    device: torch.device,
    split: Literal['train', 'test', 'val'] = 'test',
) -> Tuple[Cameras, RGBAImages]:

    IMAGE_WIDTH = 800
    IMAGE_HEIGHT = 800

    with open(path / f"transforms_{split}.json", 'r') as f:
        meta = json.loads(f.read())
    image_filenames = [path / (frame['file_path'] + ".png") for frame in meta['frames']]
    poses = np.array([frame['transform_matrix'] for frame in meta['frames']], dtype=np.float32)

    camera_angle_x = float(meta["camera_angle_x"])
    focal_length = 0.5 * IMAGE_WIDTH / np.tan(0.5 * camera_angle_x)

    c2w = torch.from_numpy(poses[:, :3, :])            # camera to world transform
    c2w[:, :, 3] *= 2 / 3                              # scale to bbox [-1, 1]^3

    N = c2w.shape[0]
    cameras = Cameras(
        c2w=c2w,
        fx=torch.ones(N) * focal_length,
        fy=torch.ones(N) * focal_length,
        cx=torch.ones(N) * IMAGE_WIDTH * 0.5,
        cy=torch.ones(N) * IMAGE_HEIGHT * 0.5,
        width=torch.ones(N, dtype=torch.long) * IMAGE_WIDTH,
        height=torch.ones(N, dtype=torch.long) * IMAGE_HEIGHT,
        near=torch.ones(N) * (4 / 3),
        far=torch.ones(N) * 4
    ).to(device)[idx]

    images = load_float32_masked_image(image_filenames[idx])

    return cameras, RGBAImages([images]).to(device)


@dataclass
class Script(Task):

    data: Path = Path('data/beagle')

    idx: int = 0

    output: Path = Path('temp.png')

    def run(self) -> None:
        camera, gt_img = parse(self.data, idx=self.idx, device=self.device)
        gt_mesh_path = self.data / 'mesh_gt' / f'{self.data.name.capitalize()}{self.idx}.obj'
        mesh = TriangleMesh.from_file(gt_mesh_path, kd_texture=Texture2D.from_constants((0, 0, 0))).to(self.device)
        trans = torch.tensor([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ]).to(mesh.vertices)
        mesh.replace_(
            vertices=(trans @ mesh.vertices.unsqueeze(-1)).squeeze(-1) * (2 / 3),
        )
        img = mesh.render(camera, shader=NormalShader()).visualize()
        dump_float32_image(self.output, torch.cat((img.item(), gt_img.item()), dim=1).clamp(0, 1))


if __name__ == '__main__':
    Script(cuda=0).run()