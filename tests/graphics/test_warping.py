from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras, RGBDNImages, TriangleMesh
from rfstudio.graphics.shaders import DepthShader, MCShader, NormalShader, TextureLatLng
from rfstudio.io import dump_float32_image


@dataclass
class Tester(Task):

    model: Optional[Path] = ...
    output: Path = ...
    aa: bool = True
    z_up: bool = False
    flip: bool = False
    wireframe: bool = True

    def run(self) -> None:
        if self.model is None:
            mesh = TriangleMesh.create_sphere().to(self.device)
        else:
            mesh = TriangleMesh.from_file(self.model).normalize().to(self.device)
        flip_coeff = -1.0 if self.flip else 1.0
        camera_from = Cameras.from_lookat(
            eye=(-1.7 * flip_coeff, 1.7 * flip_coeff, 1.7) if self.z_up else (1.7 * flip_coeff, 1.7, -1.7 * flip_coeff),
            resolution=(800, 800),
            up=(0, 0, 1) if self.z_up else (0, 1, 0),
            hfov_degree=45,
            near=1e-2,
            far=1e2,
            device=mesh.device,
        ).to(mesh.device)
        camera_to = Cameras.from_lookat(
            eye=(1.7 * flip_coeff, 1.7 * flip_coeff, 1.7) if self.z_up else (1.7 * flip_coeff, 1.7, 1.7 * flip_coeff),
            resolution=(800, 800),
            up=(0, 0, 1) if self.z_up else (0, 1, 0),
            hfov_degree=45,
            near=1e-2,
            far=1e2,
            device=mesh.device,
        ).to(mesh.device)
        envmap = TextureLatLng.from_image_file(Path('data') / 'tensoir' / 'bridge.hdr', device=self.device)
        rgb = mesh.render(
            camera_from,
            shader=MCShader(antialias=self.aa, envmap=envmap),
        ).rgb2srgb().clamp(0, 1).item()[..., :3]
        depth = mesh.render(
            camera_from,
            shader=DepthShader(antialias=self.aa),
        ).item()[..., :1]
        normal = mesh.render(
            camera_from,
            shader=NormalShader(antialias=self.aa),
        ).item()
        rgbd = RGBDNImages([torch.cat((rgb, depth, normal), dim=-1)])
        same_warp = rgbd.warping(camera_from, camera_from).item()
        diff_warp = rgbd.warping(camera_from, camera_to).item()
        dump_float32_image(self.output, torch.cat((same_warp, diff_warp), dim=1))


if __name__ == '__main__':
    Tester(
        model=Path('data') / 'spot' / 'spot.obj',
        output=Path('temp.png'),
        cuda=0,
    ).run()
