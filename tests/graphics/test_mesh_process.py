from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.engine.task import Task, TaskGroup
from rfstudio.graphics import Cameras, TriangleMesh
from rfstudio.graphics.shaders import PrettyShader, VertexAttrShader
from rfstudio.io import dump_float32_image
from rfstudio.utils.colormap import IntensityColorMap
from rfstudio.visualization import Visualizer


@dataclass
class Curvature(Task):

    mesh: Path = ...

    output: Path = Path('temp.png')

    factor: float = 0.05

    pretty: bool = False

    def run(self) -> None:
        camera = Cameras.from_lookat(
            eye=(-2, 2, 2),
            resolution=(800, 800),
            up=(0, 1, 0),
            hfov_degree=45,
            near=1e-2,
            far=1e2,
            device=self.device,
        ).to(self.device)
        mesh = TriangleMesh.from_file(self.mesh).normalize().to(self.device)
        # mesh = TriangleMesh.create_cube().to(self.device)
        if self.pretty:
            img1 = mesh.render(camera, shader=PrettyShader()).rgb2srgb().clamp(0, 1).blend((1, 1, 1)).item()
        else:
            img1 = (
                mesh.render(camera, shader=VertexAttrShader(vertex_attrs=mesh.compute_curvature().clamp(-1e2, 1e2)))
                    .visualize(IntensityColorMap('plasma'))
                    .blend((1, 1, 1))
                    .item()
            )
        for _ in range(10):
            mesh = mesh.replace(vertices=mesh.vertices + self.factor * mesh.cotangent_laplace(mesh.vertices))
        if self.pretty:
            img2 = mesh.render(camera, shader=PrettyShader()).rgb2srgb().clamp(0, 1).blend((1, 1, 1)).item()
        else:
            img2 = (
                mesh.render(camera, shader=VertexAttrShader(vertex_attrs=mesh.compute_curvature().clamp(-1e2, 1e2)))
                    .visualize(IntensityColorMap('plasma'))
                    .blend((1, 1, 1))
                    .item()
            )
        for _ in range(90):
            mesh = mesh.replace(vertices=mesh.vertices + self.factor * mesh.cotangent_laplace(mesh.vertices))
        if self.pretty:
            img3 = mesh.render(camera, shader=PrettyShader()).rgb2srgb().clamp(0, 1).blend((1, 1, 1)).item()
        else:
            img3 = (
                mesh.render(camera, shader=VertexAttrShader(vertex_attrs=mesh.compute_curvature().clamp(-1e2, 1e2)))
                    .visualize(IntensityColorMap('plasma'))
                    .blend((1, 1, 1))
                    .item()
            )
        dump_float32_image(self.output, torch.cat((img1, img2, img3), dim=1))


@dataclass
class Subdivision(Task):

    mesh: Path = ...

    output: Path = Path('temp.png')

    def run(self) -> None:
        mesh1 = TriangleMesh.from_file(self.mesh).normalize().to(self.device)
        mesh2 = mesh1.subdivide()
        mesh3 = mesh2.subdivide()
        camera = Cameras.from_lookat(
            eye=(-2, 2, 2),
            target=(0.3, -0.3, 0),
            resolution=(1600, 1600),
            up=(0, 1, 0),
            hfov_degree=27,
            near=1e-2,
            far=1e2,
            device=self.device,
        ).to(self.device)
        img1 = (
            mesh1.render(camera, shader=PrettyShader(normal_type='flat', wireframe=True))
                .rgb2srgb()
                .clamp(0, 1)
                .blend((1, 1, 1))
                .resize_to(800, 800)
                .item()
        )
        img2 = (
            mesh2.render(camera, shader=PrettyShader(normal_type='flat', wireframe=True))
                .rgb2srgb()
                .clamp(0, 1)
                .blend((1, 1, 1))
                .resize_to(800, 800)
                .item()
        )
        img3 = (
            mesh3.render(camera, shader=PrettyShader(normal_type='flat', wireframe=True))
                .rgb2srgb()
                .clamp(0, 1)
                .blend((1, 1, 1))
                .resize_to(800, 800)
                .item()
        )
        dump_float32_image(self.output, torch.cat((img1, img2, img3), dim=1))

@dataclass
class Laplace(Task):

    mesh: Path = ...

    viser: Visualizer = Visualizer()

    factor: float = 0.2

    def run(self) -> None:
        mesh = TriangleMesh.from_file(self.mesh).to(self.device)
        # mesh = TriangleMesh.create_cube().to(self.device)
        with self.viser.customize() as handle:
            handle['input'].show(mesh)
            dummy = mesh
            for i in range(5):
                laplace = dummy.uniform_laplace
                dummy = dummy.replace(vertices=dummy.vertices + self.factor * laplace(dummy.vertices))
                handle[f'uniform_laplace_{i+1}'].show(dummy)
            dummy = mesh
            for i in range(5):
                laplace = dummy.cotangent_laplace
                dummy = dummy.replace(vertices=dummy.vertices + self.factor * laplace(dummy.vertices))
                handle[f'cotangent_laplace_{i+1}'].show(dummy)


if __name__ == '__main__':
    TaskGroup(
        laplace=Laplace(mesh=Path('data/spot/spot.obj'), cuda=0),
        curvature=Curvature(mesh=Path('data/bun_zipper.ply'), cuda=0),
        subdiv=Subdivision(mesh=Path('data/dinosaur.obj'), cuda=0),
    ).run()
