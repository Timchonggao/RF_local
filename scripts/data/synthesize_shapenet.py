from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras, TextureLatLng, TriangleMesh
from rfstudio.graphics.shaders import DepthShader, MCShader, NormalShader
from rfstudio.io import dump_float32_image
from rfstudio.ui import console


@dataclass
class Synthesis(Task):

    shapenet_path: Path = ...

    output: Path = ...

    chunksize: int = 128

    num_views: int = 64

    resolution: int = 256

    max_instances: int = 64

    envmap: Path = Path('data') / 'tensoir' / 'bridge.hdr'

    debug: bool = False

    @torch.no_grad()
    def run(self) -> None:

        all_models: List[Path] = []
        for cate_id in self.shapenet_path.glob("*"):
            if not cate_id.name.isdecimal() or cate_id.is_file():
                continue
            all_models.extend(list(cate_id.glob("*"))[:self.max_instances])

        cameras = Cameras.from_sphere(
            center=(0, 0, 0),
            up=(0, 1, 0),
            radius=3.,
            num_samples=self.num_views,
            resolution=(self.resolution, self.resolution),
            near=1e-2,
            far=1e2,
            hfov_degree=45,
            device=self.device,
        )
        envmap = TextureLatLng.from_image_file(self.envmap, device=self.device).compute_pdf_()

        with console.progress('Rendering') as ptrack:
            content = None
            for idx, path in enumerate(ptrack(all_models)):
                chunk_idx = idx // self.chunksize
                output_path = self.output / f'{chunk_idx:04d}.torch'
                if output_path.exists():
                    continue
                if content is None:
                    content = {
                        'cameras': {
                            'c2w': cameras.c2w,
                            'fxfycxcy': torch.stack((cameras.fx, cameras.fy, cameras.cx, cameras.cy), dim=-1),
                            'resolution': self.resolution,
                        },
                        'objects': [],
                    }
                try:
                    mesh = TriangleMesh.from_file(path / 'models' / 'model_normalized.obj').to(self.device).normalize()
                except Exception:
                    continue
                pbr = mesh.render(
                    cameras,
                    shader=MCShader(antialias=False, culling=False, bend_back_normal=True, envmap=envmap),
                ).rgb2srgb().clamp(0, 1)
                normal = mesh.render(
                    cameras,
                    shader=NormalShader(antialias=False, culling=False, bend_back_normal=True),
                )
                depth = mesh.render(
                    cameras,
                    shader=DepthShader(antialias=False, culling=False),
                )
                content['objects'].append({
                    'category': path.parent.name,
                    'rgbdna': torch.stack([
                        torch.cat((pbr[i].item()[..., :3], depth[i].item()[..., :1], normal[i].item()), dim=-1)
                        for i in range(cameras.shape[0])
                    ], dim=0),
                })

                gc.collect()
                torch.cuda.empty_cache()

                if self.debug:
                    vis = torch.cat([
                        torch.cat((
                            pbr[i].blend((1, 1, 1)).item(),
                            depth[i].visualize().item(),
                            normal[i].visualize((1, 1, 1)).item(),
                        ), dim=0)
                        for i in range(4)
                    ], dim=1)
                    self.output.mkdir(exist_ok=True, parents=True)
                    dump_float32_image(self.output / f'{path.parent.name}-{path.name}.jpg', vis.clamp(0, 1))

                if idx % self.chunksize == (self.chunksize - 1):
                    if not self.debug:
                        self.output.mkdir(exist_ok=True, parents=True)
                        torch.save(content, output_path)
                    content = None

if __name__ == '__main__':
    Synthesis(cuda=0, seed=1).run()
