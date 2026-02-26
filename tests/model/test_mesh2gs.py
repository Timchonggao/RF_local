from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from torch.nn.functional import avg_pool2d

from rfstudio.engine.task import Task, TaskGroup
from rfstudio.graphics import Cameras, TextureLatLng, TriangleMesh, VectorImages
from rfstudio.graphics.shaders import (
    MCShader,
    NormalShader,
    PrettyShader,
)
from rfstudio.io import dump_float32_image
from rfstudio.model import GSplatter
from rfstudio.model.density_primitives.geosplat import MGAdapter
from rfstudio.visualization import highlight


@dataclass
class TestNormal(Task):

    load: Path = ...

    output: Path = Path('exports') / 'mesh_sampler'

    @torch.no_grad()
    def run(self) -> None:
        gsplat = GSplatter(
            num_random=4,
            sh_degree=0,
            background_color='white',
            rasterize_mode='antialiased',
        ).to(self.device)
        gsplat.__setup__()
        mesh = TriangleMesh.from_file(self.load).to(self.device).normalize()
        mesh.replace_(vertices=torch.stack((-mesh.vertices[..., 2], -mesh.vertices[..., 0], mesh.vertices[..., 1]), dim=-1))
        gsplat.gaussians = MGAdapter().make(mesh.compute_vertex_normals(fix=True), normal_interpolation=True)[0]
        camera = Cameras.from_lookat(eye=(2, -2, 2), up=(0, 0, 1), resolution=(3200, 3200), hfov_degree=45, device=self.device)
        gs_n = VectorImages(gsplat.render_rgba(camera[None])).visualize()
        mesh_n = mesh.render(camera, shader=NormalShader(antialias=True)).visualize()
        mesh_pretty = mesh.render(camera, shader=PrettyShader(wireframe=True, z_up=True, normal_type='flat')).rgb2srgb()
        window = (620*2, 550*2, 620*2+600, 550*2+600)
        self.output.mkdir(parents=True, exist_ok=True)
        _, gs_window = highlight(gs_n, window=window, border_width=40, border_color=(90, 116, 143), padding=20)
        mesh_n, mesh_window = highlight(mesh_n, window=window, border_width=40, border_color=(90, 116, 143), padding=20)
        mesh_pretty, _ = highlight(mesh_pretty, window=window, border_width=40, border_color=(90, 116, 143), padding=20)
        dump_float32_image(self.output / 'gs_window.png', gs_window.resize_to(800, 800).item())
        dump_float32_image(self.output / 'mesh_window.png', mesh_window.resize_to(800, 800).item())
        dump_float32_image(self.output / 'gt.png', mesh_pretty.resize_to(800, 800).item().clamp(0, 1))

@dataclass
class TestSpot(Task):

    load: Path = Path('data') / 'spot' / 'spot.obj'

    output: Path = Path('temp.png')

    mode: Literal['mesh', 'gs', 'pbr'] = 'gs'

    @torch.no_grad()
    def run(self) -> None:
        mesh = TriangleMesh.from_file(self.load).normalize().to(self.device)
        camera = Cameras.from_lookat(
            eye=(1.7, 1.7, -1.7),
            resolution=(1600, 1600),
            up=(0, 1, 0),
            hfov_degree=45,
            near=1e-2,
            far=1e2,
            device=mesh.device,
        ).to(mesh.device)
        if self.mode == 'mesh':
            output = mesh.render(
                camera,
                shader=PrettyShader(antialias=True, wireframe=True, normal_type='flat'),
            ).rgb2srgb().item()
        elif self.mode == 'pbr':
            envmap = TextureLatLng.from_image_file(
                Path('data') / 'irrmaps' / 'aerodynamics_workshop_2k.hdr',
                device=self.device,
            ).compute_pdf_()
            output = mesh.render(camera, shader=MCShader(antialias=True, envmap=envmap)).rgb2srgb().item()
        else:
            mask = mesh.visible_faces(camera) # [N]
            mask = torch.cat([mask + i * mesh.num_faces for i in range(6)])
            gsplat = GSplatter(
                num_random=4,
                sh_degree=0,
                background_color='white',
                rasterize_mode='antialiased',
            ).to(self.device)
            gsplat.__setup__()
            gsplat.gaussians = MGAdapter().make(mesh.compute_vertex_normals(fix=True), normal_interpolation=False)[0]
            gsplat.gaussians = gsplat.gaussians[mask]
            color = torch.tensor([119/512, 150/512, 170/512]).to(mesh.vertices) ** (1 / 2.2)
            gsplat.gaussians.replace_(
                colors=color.expand_as(gsplat.gaussians.colors),
                scales=gsplat.gaussians.scales - 1.5,
            )
            attached = gsplat.render_rgba(camera[None]).item()
            output = mesh.subdivide().render(
                camera,
                shader=PrettyShader(antialias=True),
            ).rgb2srgb().item()
            output[..., :3] = output[..., :3] * (1 - attached[..., 3:]) + attached[..., :3]
        output = avg_pool2d(output[None].permute(0, 3, 1, 2), 2).permute(0, 2, 3, 1).squeeze(0)
        dump_float32_image(self.output, output.clamp(0, 1))

@dataclass
class Profiling(Task):

    load: Path = Path('data') / 'spot' / 'spot.obj'

    output: Path = Path('temp.png')

    def run(self) -> None:
        mesh = TriangleMesh.from_file(self.load).to(self.device).subdivide().normalize()
        mesh.replace_(vertices=torch.stack((-mesh.vertices[..., 2], -mesh.vertices[..., 0], mesh.vertices[..., 1]), dim=-1))
        gsplat = GSplatter(
            num_random=4,
            sh_degree=0,
            background_color='white',
            rasterize_mode='antialiased',
        ).to(self.device)
        gsplat.__setup__()
        camera = Cameras.from_lookat(eye=(2, -2, 2), up=(0, 0, 1), resolution=(800, 800), hfov_degree=45, device=self.device)
        mesh_vis = mesh.render(camera, shader=NormalShader(antialias=True, normal_type='flat')).item()

        for i in range(3):
            mesh.vertices.grad = None
            mesh.vertices.requires_grad_()
            pre_mem = torch.cuda.memory_allocated(mesh.device)
            gsplat.gaussians = MGAdapter().make(mesh.compute_vertex_normals(fix=True), normal_interpolation=False)[0]
            assert gsplat.gaussians.means.grad_fn is not None
            post_mem = torch.cuda.memory_allocated(mesh.device)
            vis = VectorImages(gsplat.render_rgba(camera[None])).item()
            loss = torch.nn.functional.mse_loss(vis, mesh_vis)
            loss.backward()

        print(f'CUDA memory per million Gaussians: {(post_mem - pre_mem) / 10.24**3 / gsplat.gaussians.shape[0]:.2f} GB')

if __name__ == '__main__':
    TaskGroup(
        normal=TestNormal(cuda=0),
        spot=TestSpot(cuda=0),
        profiling=Profiling(cuda=0),
    ).run()
