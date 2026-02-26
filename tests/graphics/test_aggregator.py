from __future__ import annotations

from dataclasses import dataclass

import torch

from rfstudio.engine.task import Task, TaskGroup
from rfstudio.graphics import Cameras, Points, TriangleMesh
from rfstudio.graphics.spatial_aggregator import NearestGrid
from rfstudio.graphics.spatial_sampler import NearestGridSampler
from rfstudio.visualization import Visualizer


@dataclass
class TestSample(Task):

    def run(self) -> None:
        sphere = TriangleMesh.create_sphere().to(self.device)
        pts = sphere.uniformly_sample(512)
        pts.positions.add_(torch.randn_like(pts.positions) * 0.1)
        sampler = NearestGridSampler()
        sampler.__setup__()
        sampler.to(self.device)
        sampler.reset()
        sampler.aggregate(pts.positions)
        resampled = Points(positions=sampler.sample(512))
        Visualizer().show(
            mesh=sphere,
            source=pts,
            resampled=resampled,
        )

@dataclass
class TestIntersect(Task):

    def run(self) -> None:
        grid = NearestGrid.from_resolution(4, feature_dim=1, center=(0.2, 0.4, -0.6), device=self.device)
        rays = Cameras.from_lookat(
            eye=(1, 1, 1),
            device=self.device,
            resolution=(4, 6),
            far=3,
        ).generate_rays().flatten()
        results = grid.intersect(rays)
        positions = results.positions[results.valid.expand_as(results.positions)].view(-1, 3)
        grid_centers = grid.get_grid_centers() # [R, R, R, 3]
        template_cube = TriangleMesh.create_cube(size=2.0 / 4)
        meshes = TriangleMesh.merge(*[
            template_cube.translate(x, y, z)
            for x, y, z in grid_centers.view(-1, 3)
        ], only_geometry=True)
        ray_vis = rays.get_samples(torch.linspace(0, 1, 256, device=rays.device)).as_points()
        with Visualizer().customize() as handle:
            handle['pts'].show(Points(positions=positions)).configurate(point_size=0.05)
            handle['grid'].show(meshes).configurate(shade='wireframe')
            handle['rays'].show(ray_vis).configurate(point_shape='square', point_size=0.01)

if __name__ == '__main__':
    TaskGroup(
        sample=TestSample(cuda=0),
        intersect=TestIntersect(cuda=0),
    ).run()
