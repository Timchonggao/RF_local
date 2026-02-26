from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras, TextureLatLng, TriangleMesh
from rfstudio.graphics.shaders import MCShader, PrettyShader
from rfstudio.io import open_video_renderer
from rfstudio.ui import console


@dataclass
class Vis(Task):

    input: Path = ...
    z_up: bool = False
    envmap: Path = Path('data') / 'tensoir' / 'bridge.hdr'
    output: Optional[Path] = None
    fps: float = 30.0
    num_frames: int = 180
    target_mb: float = 16.0
    geometry: bool = True

    def run(self) -> None:

        with console.status('Loading'):
            mesh = TriangleMesh.from_file(self.input).normalize().to(self.device)
            cameras = Cameras.from_orbit(
                center=(0, 0, 0),
                up=(0, 0, 1) if self.z_up else (0, 1, 0),
                radius=2.5,
                pitch_degree=45,
                hfov_degree=45,
                resolution=(800, 800),
                num_samples=self.num_frames,
                device=self.device,
            )
            envmap = TextureLatLng.from_image_file(self.envmap, device=self.device)
            if self.z_up:
                envmap.z_up_to_y_up_()

        with console.progress('Rendering Appearance', transient=True) as ptrack:
            images = mesh.render(
                cameras,
                shader=MCShader(envmap=envmap, num_samples_per_ray=16),
                progress_handle=ptrack,
            ).rgb2srgb().clamp(0, 1).blend((1, 1, 1))

        if self.geometry:
            with console.progress('Rendering Geometry', transient=True) as ptrack:
                geometries = mesh.render(
                    cameras,
                    shader=PrettyShader(z_up=self.z_up),
                    progress_handle=ptrack,
                ).rgb2srgb().clamp(0, 1).blend((1, 1, 1))

        output_path = self.input.parent / 'vis.mp4' if self.output is None else self.output
        with open_video_renderer(output_path, fps=self.fps, target_mb=self.target_mb) as renderer:
            with console.progress('Fusing Video', transient=True) as ptrack:
                if self.geometry:
                    for img, geo_img in zip(ptrack(images), geometries):
                        renderer.write(torch.cat((img, geo_img), dim=1))
                else:
                    for img in ptrack(images):
                        renderer.write(img)

if __name__ == '__main__':
    Vis(cuda=0).run()
