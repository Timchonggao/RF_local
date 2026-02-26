from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch

import numpy as np

from rfstudio.engine.task import Task
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh
from rfstudio.graphics.shaders import (DepthShader, NormalShader)
from rfstudio.io import dump_float32_image
from rfstudio_ds.visualization import Visualizer


@dataclass
class Tester(Task):

    model: Optional[Path] = ...
    output: Path = ...
    aa: bool = False

    def run(self) -> None:

        mesh = DS_TriangleMesh.from_file(self.model, read_mtl=False).to(self.device)
        trans = torch.tensor([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ]).to(mesh.vertices)
        mesh.replace_(
            vertices=((trans @ mesh.vertices.unsqueeze(-1)).squeeze(-1)),
        ) 
        # mesh.export(path=Path('temp.obj'),only_geometry=True)
        mesh.replace_(
            vertices=mesh.vertices * (2/3),
        )

        camera_frames_pose = [
        [
          1.0,
          0.0,
          0.0,
          0.0
        ],
        [
          0.0,
          -4.371138828673793e-08,
          -1.0,
          -2.0
        ],
        [
          0.0,
          1.0,
          -4.371138828673793e-08,
          0.0
        ],
        [
          0.0,
          0.0,
          0.0,
          1.0
        ]]
        camera_frames_pose = np.array(camera_frames_pose)
        c2w = torch.from_numpy(camera_frames_pose[:3, :]).to(dtype=torch.float32).unsqueeze(0)
        c2w[:, :, 3] *= 2 / 3
        camera_angle_x = float(0.6911112070083618)
        camera_focal_length = 0.5 * 800 / np.tan(0.5 * camera_angle_x)
        breakpoint()

        N = c2w.shape[0]
        cameras = DS_Cameras(
            c2w=c2w,
            fx=torch.ones(N) * camera_focal_length,
            fy=torch.ones(N) * camera_focal_length,
            cx=torch.ones(N) * 800 * 0.5,
            cy=torch.ones(N) * 800 * 0.5,
            width=torch.ones(N, dtype=torch.long) * 800,
            height=torch.ones(N, dtype=torch.long) * 800,
            near=torch.ones(N) * (4 / 3),
            far=torch.ones(N) * 6,
            times=torch.ones(N,1) * 0,
            dts=torch.ones(N,1) * 1,
        ).to(self.device)   


        images = torch.cat((
            mesh.render(cameras=cameras,shader=NormalShader()).visualize((1, 1, 1)).item(),
            mesh.render(cameras=cameras,shader=DepthShader()).visualize().item(),
        ), dim=1)
        dump_float32_image(self.output, images)


if __name__ == '__main__':
    Tester(
        # model=Path('temp.obj'),
        model=Path('/data3/gaochong/project/RadianceFieldStudio/data/multiview_dynamic_blender/toy/obj/frame0.obj'),
        output=Path('temp.png'),
        cuda=0,
    ).run()


# from __future__ import annotations

# from dataclasses import dataclass
# from pathlib import Path

# import torch

# from rfstudio.engine.task import Task
# from rfstudio.graphics import Cameras, TriangleMesh
# from rfstudio.graphics.shaders import NormalShader
# from rfstudio.io import dump_float32_image
# from rfstudio.visualization import Visualizer


# @dataclass
# class Script(Task):

#     vis: bool = False
#     output: Path = Path('temp1.png')

#     def run(self) -> None:
#         mesh = TriangleMesh.from_file(Path('temp.obj')).to(self.device)
#         mesh.replace_(vertices=mesh.vertices * (2/3))
#         camera = Cameras.from_lookat(
#             eye=(1, 1, 1),
#             target=(0, 0, 0),
#             resolution=(800, 800),
#             hfov_degree=40,
#         ).to(self.device)
#         pose = [
#             [1.0, 0.0, 0.0, 0.0],
#             [0.0, -4.371138828673793e-08, -1.0, -2.0],
#             [0.0, 1.0, -4.371138828673793e-08, 0.0],
#             [0.0, 0.0, 0.0, 1.0],
#         ]
#         pose = torch.tensor(pose)[:3, :4]
#         pose[:, 3] *= 2 / 3
#         camera.replace_(c2w=pose.to(camera.c2w))
#         breakpoint()

#         img = mesh.render(camera, shader=NormalShader())
#         dump_float32_image(self.output, img.visualize().item().clamp(0, 1))
#         if self.vis:
#             Visualizer().show(mesh=mesh, camera=camera)


# if __name__ == '__main__':
#     Script(cuda=0).run()
