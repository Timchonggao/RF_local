from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Literal
import gc

import torch
from torch import Tensor

from rfstudio.engine.task import Task, TaskGroup
from rfstudio.ui import console
from rfstudio.utils.pretty import P
from rfstudio.loss import ChamferDistanceMetric
from rfstudio.graphics.shaders import PrettyShader, DepthShader, NormalShader
from rfstudio.graphics import Points, TriangleMesh

from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh
from rfstudio_ds.visualization import Visualizer

from rfstudio.io import load_float32_image, open_video_renderer, dump_float32_image
from natsort import natsorted



@dataclass
class MeshSequence2Video(Task):

    input: Path = Path('/data/gaochong/project/DG-Mesh/outputs/dg-mesh/cat/cat-2025-08-01_21-23-33/test_results/dynamic_mesh')

    output: Path = Path('/data/gaochong/project/DG-Mesh/outputs/dg-mesh/cat/cat-2025-08-01_21-23-33/test_results/dynamic_mesh.mp4')

    fps: float = 20

    duration = 5

    downsample: Optional[float] = None

    target_mb: Optional[float] = None

    extension: Literal['obj', 'ply',] = 'ply'

    def run(self) -> None:
        mesh_list = list(self.input.glob(f"*.{self.extension}"))
        mesh_list = natsorted(mesh_list)

        if self.duration is not None:
            self.fps = max(1, int(len(mesh_list) / self.duration))
        
        vis_cameras = DS_Cameras.from_orbit(
            center=(0, 0, 0),
            up=(0, 0, 1),
            radius=1.5,
            pitch_degree=15.0,
            num_samples=len(mesh_list),
            resolution=(800, 800),
            near=1e-2,
            far=1e2,
            hfov_degree=60,
            device=self.device,
        )

        pretty_shader = PrettyShader(antialias=True, wireframe=False, z_up=True)
        pretty_images = []
        for i in range(len(mesh_list)):
            mesh = DS_TriangleMesh.from_file(mesh_list[i], read_mtl=False).to(self.device)
            pretty_images.append(mesh.render(vis_cameras[i].to(mesh.device), shader=pretty_shader).blend((1, 1, 1)).item().clamp(0, 1))

        self.output.parent.mkdir(parents=True, exist_ok=True)
        with open_video_renderer(
            self.output,
            fps=self.fps,
            downsample=self.downsample,
            target_mb=self.target_mb,
        ) as renderer:
            with console.progress('Exporting...') as ptrack:
                for pretty_image in ptrack(pretty_images):
                    renderer.write(pretty_image)



if __name__ == '__main__':
    MeshSequence2Video(cuda=0).run()
