from __future__ import annotations

# import modules
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Literal, NamedTuple, Tuple, List, Any, Union, Optional
import random

import torch
import numpy as np

# import rfstudio modules
from rfstudio.graphics import DepthImages
from rfstudio.utils.context import create_random_seed_context
from rfstudio.utils.typing import Indexable
from rfstudio.io import dump_float32_image

# import rfstudio_ds modules
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh
from .base_dataparser import DS_BaseDataparser


class _KnownDynamicModel(NamedTuple):
    animation_model_name: str              # 动画名称，例如 "horse"

    reflect_padding: bool                  # 是否需要反射padding，一般是True
    padding_size: int                     # 反射padding的大小，一般是16

    blender_mesh_sequence_dir: str             # 原始mesh文件所在的文件夹名称，例如 "mesh_gt"
    blender_mesh_name: str                    # 原始mesh文件的文件名，例如 "Beagle{}.obj"
    blender_num_frames: int                   # 原始mesh文件总帧数
    blender_begin_frame: int                  # 原始mesh文件开始帧数

    blender_val_pitch_degree: float
    blender_view_radius: float
    blender_z_up: bool

    process_mesh_sequence_dir: str                 # 存放预处理网格的文件夹名称，例如 "mesh_gt_preprocess"
    process_mesh_name: str                         # 预处理网格的文件名，例如 "{}.obj"
    process_num_frames: int                        # 动画帧数
    process_begin_frame: int                       # 动画开始帧

    process_val_pitch_degree: float
    process_view_radius: float
    process_z_up: bool



_KNOWN_MODELS = {

    'beagle': _KnownDynamicModel(
        animation_model_name='beagle',
        reflect_padding=True,
        padding_size=20,

        blender_mesh_sequence_dir='mesh_gt', # 原始mesh文件，这个mesh的center和scale都不标准，需要特殊的cameras参数才能渲染，一般在d_nvdiffrec pipeline下用
        blender_mesh_name='Beagle{}.obj',
        blender_num_frames=200,
        blender_begin_frame=0,

        blender_val_pitch_degree=15.,
        blender_view_radius=2.5,
        blender_z_up=True,

        process_mesh_sequence_dir='mesh_gt_preprocess/mesh_files', # 与sdf fit 的 gt sdf 对其，用这个mesh 训练，方便和sdf fit pipeline下的 gt sdf curve 进行对比
        process_mesh_name='{}.obj',
        process_num_frames=200,
        process_begin_frame=20, # 20帧开始，因为导出这个gt mesh的时候有padding，所以从20帧开始才是真正的gt mesh

        process_val_pitch_degree=15.,
        process_view_radius=2.5,
        process_z_up=False,
    ),

    'bird': _KnownDynamicModel(
        animation_model_name='bird',
        reflect_padding=True,
        padding_size=20,

        blender_mesh_sequence_dir='mesh_gt', # 原始mesh文件，这个mesh的center和scale都不标准，需要特殊的cameras参数才能渲染，一般在d_nvdiffrec pipeline下用
        blender_mesh_name='bluebird_animated{}.obj',
        blender_num_frames=200,
        blender_begin_frame=0,

        blender_val_pitch_degree=15.,
        blender_view_radius=2.5,
        blender_z_up=True,

        process_mesh_sequence_dir='mesh_gt_preprocess/mesh_files', # 与sdf fit 的 gt sdf 对其，用这个mesh 训练，方便和sdf fit pipeline下的 gt sdf curve 进行对比
        process_mesh_name='{}.obj',
        process_num_frames=200,
        process_begin_frame=20, # 20帧开始，因为导出这个gt mesh的时候有padding，所以从20帧开始才是真正的gt mesh

        process_val_pitch_degree=15.,
        process_view_radius=2.5,
        process_z_up=False,
    ),

    'duck': _KnownDynamicModel(
        animation_model_name='duck',
        reflect_padding=True,
        padding_size=20,

        blender_mesh_sequence_dir='mesh_gt', # 原始mesh文件，这个mesh的center和scale都不标准，需要特殊的cameras参数才能渲染，一般在d_nvdiffrec pipeline下用
        blender_mesh_name='Duck{}.obj',
        blender_num_frames=200,
        blender_begin_frame=0,

        blender_val_pitch_degree=15.,
        blender_view_radius=2.5,
        blender_z_up=True,

        process_mesh_sequence_dir='mesh_gt_preprocess/mesh_files', # 与sdf fit 的 gt sdf 对其，用这个mesh 训练，方便和sdf fit pipeline下的 gt sdf curve 进行对比
        process_mesh_name='{}.obj',
        process_num_frames=200,
        process_begin_frame=20, # 20帧开始，因为导出这个gt mesh的时候有padding，所以从20帧开始才是真正的gt mesh

        process_val_pitch_degree=15.,
        process_view_radius=2.5,
        process_z_up=False,
    ),

    'girlwalk': _KnownDynamicModel(
        animation_model_name='girlwalk',
        reflect_padding=True,
        padding_size=20,

        blender_mesh_sequence_dir='mesh_gt', # 原始mesh文件，这个mesh的center和scale都不标准，需要特殊的cameras参数才能渲染，一般在d_nvdiffrec pipeline下用
        blender_mesh_name='Cartoon Teen Girl Character_01_Anim{}.obj',
        blender_num_frames=200,
        blender_begin_frame=0,

        blender_val_pitch_degree=15.,
        blender_view_radius=2.5,
        blender_z_up=True,
        
        process_mesh_sequence_dir='mesh_gt_preprocess/mesh_files', # 与sdf fit 的 gt sdf 对其，用这个mesh 训练，方便和sdf fit pipeline下的 gt sdf curve 进行对比
        process_mesh_name='{}.obj',
        process_num_frames=200,
        process_begin_frame=20, # 20帧开始，因为导出这个gt mesh的时候有padding，所以从20帧开始才是真正的gt mesh

        process_val_pitch_degree=15.,
        process_view_radius=2.5,
        process_z_up=False,
    ),

    'horse': _KnownDynamicModel(
        animation_model_name='horse',
        reflect_padding=True,
        padding_size=20,

        blender_mesh_sequence_dir='mesh_gt', # 原始mesh文件，这个mesh的center和scale都不标准，需要特殊的cameras参数才能渲染，一般在d_nvdiffrec pipeline下用
        blender_mesh_name='horse{}.obj',
        blender_num_frames=200,
        blender_begin_frame=0,

        blender_val_pitch_degree=15.,
        blender_view_radius=2.5,
        blender_z_up=True,

        process_mesh_sequence_dir='mesh_gt_preprocess/mesh_files', # 与sdf fit 的 gt sdf 对其，用这个mesh 训练，方便和sdf fit pipeline下的 gt sdf curve 进行对比
        process_mesh_name='{}.obj',
        process_num_frames=200,
        process_begin_frame=20, # 20帧开始，因为导出这个gt mesh的时候有padding，所以从20帧开始才是真正的gt mesh

        process_val_pitch_degree=15.,
        process_view_radius=2,
        process_z_up=False,
    ),

    'torus2sphere': _KnownDynamicModel(
        animation_model_name='torus2sphere',
        reflect_padding=True,
        padding_size=20,

        blender_mesh_sequence_dir='mesh_gt', # 原始mesh文件，这个mesh的center和scale都不标准，需要特殊的cameras参数才能渲染，一般在d_nvdiffrec pipeline下用
        blender_mesh_name='deform{}.obj',
        blender_num_frames=200,
        blender_begin_frame=1,

        blender_val_pitch_degree=15.,
        blender_view_radius=2.5,
        blender_z_up=True,

        process_mesh_sequence_dir='mesh_gt_preprocess/mesh_files', # 与sdf fit 的 gt sdf 对其，用这个mesh 训练，方便和sdf fit pipeline下的 gt sdf curve 进行对比
        process_mesh_name='{}.obj',
        process_num_frames=200,
        process_begin_frame=20, # 20帧开始，因为导出这个gt mesh的时候有padding，所以从20帧开始才是真正的gt mesh

        process_val_pitch_degree=60.,
        process_view_radius=2.,
        process_z_up=False,
    ),

    'cat': _KnownDynamicModel(
        animation_model_name='cat',
        reflect_padding=True,
        padding_size=20,

        blender_mesh_sequence_dir='obj', # 原始mesh文件，这个mesh的center和scale都不标准，需要特殊的cameras参数才能渲染，一般在d_nvdiffrec pipeline下用
        blender_mesh_name='frame{}.obj',
        blender_num_frames=240,
        blender_begin_frame=0,

        blender_val_pitch_degree=15.,
        blender_view_radius=2.5,
        blender_z_up=False,

        process_mesh_sequence_dir='obj', # 与sdf fit 的 gt sdf 对其，用这个mesh 训练，方便和sdf fit pipeline下的 gt sdf curve 进行对比
        process_mesh_name='frame{}.obj',
        process_num_frames=240,
        process_begin_frame=0, # 20帧开始，因为导出这个gt mesh的时候有padding，所以从20帧开始才是真正的gt mesh

        process_val_pitch_degree=15.,
        process_view_radius=2.5,
        process_z_up=False,
    ),

    'deer': _KnownDynamicModel(
        animation_model_name='deer',
        reflect_padding=True,
        padding_size=20,

        blender_mesh_sequence_dir='obj', # 原始mesh文件，这个mesh的center和scale都不标准，需要特殊的cameras参数才能渲染，一般在d_nvdiffrec pipeline下用
        blender_mesh_name='frame{}.obj',
        blender_num_frames=241,
        blender_begin_frame=0,

        blender_val_pitch_degree=15.,
        blender_view_radius=2.5,
        blender_z_up=False,

        process_mesh_sequence_dir='obj', # 与sdf fit 的 gt sdf 对其，用这个mesh 训练，方便和sdf fit pipeline下的 gt sdf curve 进行对比
        process_mesh_name='frame{}.obj',
        process_num_frames=241,
        process_begin_frame=0, # 20帧开始，因为导出这个gt mesh的时候有padding，所以从20帧开始才是真正的gt mesh

        process_val_pitch_degree=15.,
        process_view_radius=2.5,
        process_z_up=False,
    ),

    'girlwalk': _KnownDynamicModel(
        animation_model_name='girlwalk',
        reflect_padding=True,
        padding_size=20,

        blender_mesh_sequence_dir='obj', # 原始mesh文件，这个mesh的center和scale都不标准，需要特殊的cameras参数才能渲染，一般在d_nvdiffrec pipeline下用
        blender_mesh_name='frame{}.obj',
        blender_num_frames=148,
        blender_begin_frame=0,

        blender_val_pitch_degree=15.,
        blender_view_radius=2.5,
        blender_z_up=False,

        process_mesh_sequence_dir='obj', # 与sdf fit 的 gt sdf 对其，用这个mesh 训练，方便和sdf fit pipeline下的 gt sdf curve 进行对比
        process_mesh_name='frame{}.obj',
        process_num_frames=148,
        process_begin_frame=0, # 20帧开始，因为导出这个gt mesh的时候有padding，所以从20帧开始才是真正的gt mesh

        process_val_pitch_degree=15.,
        process_view_radius=2.5,
        process_z_up=False,
    ),

    'lego': _KnownDynamicModel(
        animation_model_name='lego',
        reflect_padding=True,
        padding_size=20,

        blender_mesh_sequence_dir='obj', # 原始mesh文件，这个mesh的center和scale都不标准，需要特殊的cameras参数才能渲染，一般在d_nvdiffrec pipeline下用
        blender_mesh_name='frame{}.obj',
        blender_num_frames=169,
        blender_begin_frame=0,

        blender_val_pitch_degree=15.,
        blender_view_radius=2.5,
        blender_z_up=False,

        process_mesh_sequence_dir='obj', # 与sdf fit 的 gt sdf 对其，用这个mesh 训练，方便和sdf fit pipeline下的 gt sdf curve 进行对比
        process_mesh_name='frame{}.obj',
        process_num_frames=169,
        process_begin_frame=0, # 20帧开始，因为导出这个gt mesh的时候有padding，所以从20帧开始才是真正的gt mesh

        process_val_pitch_degree=15.,
        process_view_radius=2.5,
        process_z_up=False,
    ),

    'spidermanwalk': _KnownDynamicModel(
        animation_model_name='spidermanwalk',
        reflect_padding=True,
        padding_size=10,

        blender_mesh_sequence_dir='obj', # 原始mesh文件，这个mesh的center和scale都不标准，需要特殊的cameras参数才能渲染，一般在d_nvdiffrec pipeline下用
        blender_mesh_name='frame{}.obj',
        blender_num_frames=25,
        blender_begin_frame=0,

        blender_val_pitch_degree=15.,
        blender_view_radius=2.5,
        blender_z_up=False,

        process_mesh_sequence_dir='obj', # 与sdf fit 的 gt sdf 对其，用这个mesh 训练，方便和sdf fit pipeline下的 gt sdf curve 进行对比
        process_mesh_name='frame{}.obj',
        process_num_frames=25,
        process_begin_frame=0, # 20帧开始，因为导出这个gt mesh的时候有padding，所以从20帧开始才是真正的gt mesh

        process_val_pitch_degree=15.,
        process_view_radius=2.5,
        process_z_up=False,
    ),

    'toy': _KnownDynamicModel(
        animation_model_name='toy',
        reflect_padding=True,
        padding_size=15,

        blender_mesh_sequence_dir='obj', # 原始mesh文件，这个mesh的center和scale都不标准，需要特殊的cameras参数才能渲染，一般在d_nvdiffrec pipeline下用
        blender_mesh_name='frame{}.obj',
        blender_num_frames=71,
        blender_begin_frame=0,

        blender_val_pitch_degree=15.,
        blender_view_radius=2.5,
        blender_z_up=False,

        process_mesh_sequence_dir='mesh_gt_preprocess/mesh_files', # 与sdf fit 的 gt sdf 对其，用这个mesh 训练，方便和sdf fit pipeline下的 gt sdf curve 进行对比
        process_mesh_name='{}.obj',
        process_num_frames=71,
        process_begin_frame=20, # 20帧开始，因为导出这个gt mesh的时候有padding，所以从20帧开始才是真正的gt mesh

        process_val_pitch_degree=15.,
        process_view_radius=2.5,
        process_z_up=False,
    ),
}

"""
相机采样策略
    单目解析器：
        Blender：从JSON文件读取预定义相机位姿。
        Costume：使用球面或轨道采样生成相机。
    多视角解析器：
        Blender：从多视角JSON文件读取相机，或随机分配训练/验证/测试视角。
        Costume：生成多视角相机，训练时每帧采样多个视角。
"""


@dataclass
class SyntheticDynamicMonocularBlenderDepthDataparser(DS_BaseDataparser[DS_Cameras, DepthImages, List[DS_TriangleMesh]]):

    resolution: Tuple[int, int] = (800, 800)

    antialias: bool = True

    view_sampling_seed: int = 123

    def parse(
        self,
        path: Path,
        *,
        split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis'],
        device: torch.device,
        parse_meshes: bool = False,
    ) -> Tuple[Indexable[DS_Cameras], Any, Any]:
        model = _KNOWN_MODELS[path.name]
        self.padding_enabled = model.reflect_padding
        self.padding_size = model.padding_size

        # parse cameras
        if split in ['train', 'val', 'test']:
            IMAGE_WIDTH = self.resolution[0]
            IMAGE_HEIGHT = self.resolution[1]

            with open(path / f"transforms_{split}.json", 'r') as f:
                meta = json.loads(f.read())

            poses = np.array([frame['transform_matrix'] for frame in meta['frames']], dtype=np.float32)

            try:
                cam_origin = torch.tensor(meta["camera_origin"]).to(device)
            except:
                cam_origin = torch.zeros(3).to(device)
            
            camera_angle_x = float(meta["camera_angle_x"])
            focal_length = 0.5 * IMAGE_WIDTH / np.tan(0.5 * camera_angle_x)

            c2w = torch.from_numpy(poses[:, :3, :])          # camera to world transform
            c2w[:, :, 3] *= 2 / 3                              # scale to bbox [-1, 1]^3

            if self.padding_enabled:
                c2w = self._apply_reflection_padding(data=c2w)

                total_num_frames = model.blender_num_frames + self.padding_size * 2
                camera_times = torch.arange(0, total_num_frames) / total_num_frames
                camera_times = camera_times.unsqueeze(-1)
                dt = 1 / (total_num_frames - 1)
                camera_dts = torch.ones(total_num_frames, 1) * dt
                if split != 'train':
                    c2w = c2w[self.padding_size:-self.padding_size]
                    camera_times = camera_times[self.padding_size:-self.padding_size]
                    camera_dts = camera_dts[self.padding_size:-self.padding_size]
            else:
                total_num_frames = model.blender_num_frames
                camera_times = torch.arange(0, total_num_frames) / total_num_frames
                camera_times = camera_times.unsqueeze(-1)
                dt = 1 / (total_num_frames - 1)
                camera_dts = torch.ones(total_num_frames, 1) * dt

            N = c2w.shape[0]
            cameras = DS_Cameras(
                c2w=c2w,
                fx=torch.ones(N) * focal_length,
                fy=torch.ones(N) * focal_length,
                cx=torch.ones(N) * IMAGE_WIDTH * 0.5,
                cy=torch.ones(N) * IMAGE_HEIGHT * 0.5,
                width=torch.ones(N, dtype=torch.long) * IMAGE_WIDTH,
                height=torch.ones(N, dtype=torch.long) * IMAGE_HEIGHT,
                near=torch.ones(N) * (4 / 3),
                far=torch.ones(N) * 4,
                times=camera_times,
                dts=camera_dts,
            ).to(device)
        elif split in ['orbit_vis', 'fix_vis']:
            if self.padding_enabled:
                total_num_frames = model.blender_num_frames + self.padding_size * 2
                camera_times = torch.arange(0, total_num_frames, device=device) / total_num_frames
                camera_times = camera_times[self.padding_size:-self.padding_size].unsqueeze(-1)
                dt = 1 / (total_num_frames - 1)
                camera_dts = torch.ones(total_num_frames, 1, device=device) * dt
                camera_dts = camera_dts[self.padding_size:-self.padding_size]
            else:
                total_num_frames = model.blender_num_frames
                camera_times = torch.arange(0, total_num_frames, device=device) / total_num_frames
                camera_times = camera_times.unsqueeze(-1)
                dt = 1 / (total_num_frames - 1)
                camera_dts = torch.ones(total_num_frames, 1, device=device) * dt

            with open(path / f"transforms_train.json", 'r') as f:
                meta = json.loads(f.read())
            try:
                cam_origin = torch.tensor(meta["camera_origin"]).to(device)
            except:
                cam_origin = torch.zeros(3).to(device)
            
            if split == 'orbit_vis':
                cameras = DS_Cameras.from_orbit(
                    center=(0, 0, 0),
                    up=(0, 0, 1) if model.blender_z_up else (0, 1, 0), # need set z-up camera for dg-mesh dataset
                    radius=model.blender_view_radius,
                    pitch_degree=model.blender_val_pitch_degree,
                    num_samples=model.blender_num_frames,
                    resolution=self.resolution,
                    near=1e-2,
                    far=1e2,
                    hfov_degree=60,
                    device=device,
                )
            elif split == 'fix_vis':
                cameras = DS_Cameras.from_orbit(
                    center=(0, 0, 0),
                    up=(0, 0, 1) if model.blender_z_up else (0, 1, 0),
                    radius=model.blender_view_radius,
                    pitch_degree=model.blender_val_pitch_degree,
                    num_samples=model.blender_num_frames,
                    resolution=self.resolution,
                    near=1e-2,
                    far=1e2,
                    hfov_degree=60,
                    device=device,
                )
            cameras = cameras.set_times(camera_times, camera_dts)
            if split == 'fix_vis':
                cameras = cameras.reset_c2w_to_ref_camera_pose()

        # parse mesh
        meshes = []
        for i in range(model.blender_begin_frame, model.blender_begin_frame+model.blender_num_frames):
            obj_path = path / model.blender_mesh_sequence_dir / model.blender_mesh_name.format(i)
            pkl_path = obj_path.with_suffix('.pkl')

            if pkl_path.exists():
                mesh = DS_TriangleMesh.deserialize(pkl_path).to(device)
            else:
                if model.animation_model_name == 'torus2sphere':
                    mesh = DS_TriangleMesh.from_file(obj_path, read_mtl=False).simplify(target_num_faces=60000).to(device)
                else:
                    mesh = DS_TriangleMesh.from_file(obj_path, read_mtl=False).to(device)
                mesh.serialize(pkl_path)
            
            trans = torch.tensor([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ]).to(mesh.vertices)
            mesh.replace_(
                vertices=((trans @ mesh.vertices.unsqueeze(-1)).squeeze(-1) - cam_origin) * (2 / 3),
            ) # transform to y-up and shift to origin
            meshes.append(mesh)
        if self.padding_enabled and split == 'train':
            meshes = self._apply_reflection_padding(data=meshes)
        if parse_meshes and split == 'train':
            return meshes

        return cameras, None, None

    def _apply_reflection_padding(self, data: Union[torch.Tensor, List]):
        if isinstance(data, torch.Tensor):
            start_padding = torch.flip(data[1:self.padding_size + 1], dims=[0])
            end_padding = torch.flip(data[-self.padding_size - 1:-1], dims=[0])
            data = torch.cat([start_padding, data, end_padding], dim=0).to(data.device)
        elif isinstance(data, list): # padding mesh
            start_padding = data[1:self.padding_size + 1][::-1]
            end_padding = data[-self.padding_size - 1:-1][::-1]
            data = start_padding + data + end_padding
        return data

    @staticmethod
    def recognize(path: Path) -> bool:
        if path.exists() and path.name in _KNOWN_MODELS:
            model = _KNOWN_MODELS[path.name]
            # Check if the mesh file exists.
            if (path / model.blender_mesh_sequence_dir).exists():
                return all((path / model.blender_mesh_sequence_dir / model.blender_mesh_name.format(i)).exists() for i in range(model.blender_begin_frame, model.blender_begin_frame+model.blender_num_frames))
        return False



@dataclass
class SyntheticDynamicMonocularCostumeDepthDataparser(DS_BaseDataparser[DS_Cameras, DepthImages, List[DS_TriangleMesh]]):

    resolution: Tuple[int, int] = (800, 800)

    antialias: bool = True

    view_sampling_seed: int = 123

    def parse(
        self,
        path: Path,
        *,
        split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis'],
        device: torch.device,
        parse_meshes: bool = False,
    ) -> Tuple[Indexable[DS_Cameras], Any, Any]:
        model = _KNOWN_MODELS[path.name]
        self.padding_enabled = model.reflect_padding
        self.padding_size = model.padding_size

        # 加载 mesh 序列
        meshes = self._load_mesh_sequence(path, model, device)
        if self.padding_enabled and split == 'train':
            meshes = self._apply_reflection_padding(meshes)

        # 若只解析 mesh
        if parse_meshes and split == 'train':
            return meshes

        # 构造相机采样
        with create_random_seed_context(self.view_sampling_seed):
            cameras, camera_times, camera_dts = self._create_cameras_for_split(split, model, device)
            cameras = cameras.set_times(camera_times, camera_dts)
            if split == 'fix_vis':
                cameras = cameras.reset_c2w_to_ref_camera_pose()

        return cameras, None, None

    def _load_mesh_sequence(self, path: Path, model: _KnownDynamicModel, device: torch.device) -> List[DS_TriangleMesh]:
        meshes = []
        for i in range(model.process_begin_frame, model.process_begin_frame + model.process_num_frames):
            obj_path = path / model.process_mesh_sequence_dir / model.process_mesh_name.format(i)
            pkl_path = obj_path.with_suffix('.pkl')
            if pkl_path.exists():
                mesh = DS_TriangleMesh.deserialize(pkl_path).to(device)
            else:
                mesh = DS_TriangleMesh.from_file(obj_path, read_mtl=False).to(device)
                if model.animation_model_name == 'torus2sphere':
                    mesh = mesh.simplify(target_num_faces=60000)
                mesh.serialize(pkl_path)
            meshes.append(mesh)
        return meshes

    def _create_cameras_for_split(self, split: str, model: _KnownDynamicModel, device: torch.device) -> DS_Cameras:
        resolution = self.resolution
        up = (0, 0, 1) if model.process_z_up else (0, 1, 0)
        radius = model.process_view_radius

        def _get_total_frames():
            return model.process_num_frames + self.padding_size * 2 if self.padding_enabled else model.process_num_frames

        def _get_time_data(split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis']) -> Tuple[torch.Tensor, torch.Tensor]:
            total_frames = _get_total_frames()
            dt = 1 / (total_frames - 1)
            times = torch.linspace(0, 1, total_frames, device=device)

            camera_dts = torch.full((total_frames,), dt, device=device).unsqueeze(-1)
            camera_times = times.unsqueeze(-1)
            if self.padding_enabled and split != 'train':
                camera_dts = camera_dts[self.padding_size:-self.padding_size]
                camera_times = camera_times[self.padding_size:-self.padding_size]
            
            return camera_times, camera_dts
        
        if split == 'train':
            cameras = DS_Cameras.from_sphere(
                center=(0, 0, 0), up=up, radius=radius,
                num_samples=_get_total_frames(), resolution=resolution,
                near=1e-2, far=1e2, hfov_degree=60, device=device, uniform=False
            )
            times, dts = _get_time_data(split)
        elif split in ['val', 'test']:
            cameras = DS_Cameras.from_sphere(
                center=(0, 0, 0), up=up, radius=radius,
                num_samples=model.process_num_frames, resolution=resolution,
                near=1e-2, far=1e2, hfov_degree=60, device=device
            )
            times, dts = _get_time_data(split)
        elif split == 'fix_vis':
            cameras = DS_Cameras.from_orbit(
                center=(0, 0, 0), up=up, radius=radius,
                pitch_degree=model.process_val_pitch_degree, num_samples=model.process_num_frames,
                resolution=resolution, near=1e-2, far=1e2, hfov_degree=60, device=device
            )
            times, dts = _get_time_data(split)
        elif split == 'orbit_vis':
            vis_frames = model.process_num_frames
            cameras = DS_Cameras.from_orbit(
                center=(0, 0, 0), up=up, radius=radius,
                pitch_degree=model.process_val_pitch_degree, num_samples=vis_frames,
                resolution=resolution, near=1e-2, far=1e2, hfov_degree=60, device=device
            )
            times, dts = _get_time_data(split=split)
        return cameras, times, dts

    def _apply_reflection_padding(self, data: Union[torch.Tensor, List]):
        if isinstance(data, torch.Tensor):
            start_padding = torch.flip(data[1:self.padding_size + 1], dims=[0])
            end_padding = torch.flip(data[-self.padding_size - 1:-1], dims=[0])
            data = torch.cat([start_padding, data, end_padding], dim=0).to(data.device)
        elif isinstance(data, list): # padding mesh
            start_padding = data[1:self.padding_size + 1][::-1]
            end_padding = data[-self.padding_size - 1:-1][::-1]
            data = start_padding + data + end_padding
        return data

    @staticmethod
    def recognize(path: Path) -> bool:
        if path.exists() and path.name in _KNOWN_MODELS:
            model = _KNOWN_MODELS[path.name]
            # Check if the mesh file exists.
            if (path / model.process_mesh_sequence_dir).exists():
                return all((path / model.process_mesh_sequence_dir / model.process_mesh_name.format(i)).exists() for i in range(model.process_begin_frame, model.process_begin_frame+model.process_num_frames))
        return False



@dataclass
class SyntheticDynamicMultiViewBlenderDepthDataparser(DS_BaseDataparser[DS_Cameras, DepthImages, List[DS_TriangleMesh]]):
    """
    Objverse-select dataset
    """
    resolution: Tuple[int, int] = (800, 800)

    scale_factor: Optional[float] = None

    antialias: bool = True

    view_sampling_seed: int = 123

    def parse(
        self,
        path: Path,
        *,
        split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis'],
        device: torch.device,
        parse_meshes: bool = False,
        costume_sample_frames: Optional[List[int]] = None,
        costume_padding_size: Optional[int] = None,
    ) -> Tuple[Indexable[DS_Cameras], Any, Any]:
        model = _KNOWN_MODELS[path.name]
        self.padding_enabled = model.reflect_padding
        self.padding_size = model.padding_size
        if costume_padding_size is not None:
            self.padding_size = costume_padding_size

        # 解析 mesh
        if parse_meshes and split == 'train':
            return self._load_mesh_sequence(model, path, device)

        # 采样相机，和图像
        with create_random_seed_context(self.view_sampling_seed):
            cameras = self._get_data_for_split(split, model, path, device)

        return cameras, None, None

    def _load_mesh_sequence(self, model: _KnownDynamicModel, path: Path, device: torch.device) -> List[DS_TriangleMesh]:
        meshes = []

        start = model.blender_begin_frame
        end = model.blender_begin_frame + model.blender_num_frames

        for i in range(start, end):
            obj_path = path / model.blender_mesh_sequence_dir / model.blender_mesh_name.format(i)
            pkl_path = obj_path.with_suffix('.pkl')
            if pkl_path.exists():
                mesh = DS_TriangleMesh.deserialize(pkl_path).to(device)
            else:
                mesh = DS_TriangleMesh.from_file(obj_path, read_mtl=False).to(device)
                mesh.serialize(pkl_path)
            trans = torch.tensor([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ]).to(mesh.vertices)
            mesh.replace_(
                vertices=((trans @ mesh.vertices.unsqueeze(-1)).squeeze(-1)) * (2 / 3),
            ) 
            meshes.append(mesh)
        if self.padding_enabled:
            meshes = self._apply_reflection_padding(meshes)
        return meshes

    def _get_data_for_split(self, split: str, model: _KnownDynamicModel, path: Path, device: torch.device) -> DS_Cameras:
        
        resolution = self.resolution
        IMAGE_WIDTH = resolution[0]
        IMAGE_HEIGHT = resolution[1]

        if self.scale_factor is not None:
            IMAGE_WIDTH = int(IMAGE_WIDTH * self.scale_factor)
            IMAGE_HEIGHT = int(IMAGE_HEIGHT * self.scale_factor)
        
        # 解析相机参数
        if split in ['train', 'val', 'test']:
            with open(path / 'transforms' / f"multiview_transforms.json", 'r') as f:
                meta = json.loads(f.read())
        elif split == 'orbit_vis':
            with open(path / 'transforms' / f"orbit_transforms.json", 'r') as f:
                meta = json.loads(f.read())
        elif split == 'fix_vis':
            with open(path / 'transforms' / f"front_transforms.json", 'r') as f:
                meta = json.loads(f.read())
        
        camera_angle_x = float(meta["camera_angle_x"])
        camera_focal_length = 0.5 * IMAGE_WIDTH / np.tan(0.5 * camera_angle_x)
        
        camera_frame_num = len(meta['frames'])
        camera_frames_pose = []
        for camera_frame in meta['frames']:
            camera_frames_pose.append(camera_frame['transform_matrix'])
        
        camera_frames_pose = np.array(camera_frames_pose, dtype=np.float32)
        c2w = torch.from_numpy(camera_frames_pose[:, :3, :])
        c2w[:, :, 3] *= 2 / 3  # scale to bbox [-1, 1]^3
            
        total_time_frames = (model.blender_num_frames + self.padding_size * 2) if self.padding_enabled else model.blender_num_frames
        dt = 1 / (total_time_frames - 1)
        times = torch.linspace(0, 1, total_time_frames) # 均匀的时间序列

        # 根据split获得raw data，主要关注：c2w的变化；padding 对 time 和 image path 的影响
        if split in ['train', 'val', 'test']:
            camera_frame_indices = list(range(camera_frame_num))
            # val_camera_frame_idx, test_camera_frame_idx = 0, 1
            random.seed(self.view_sampling_seed)
            val_camera_frame_idx, test_camera_frame_idx = random.sample(camera_frame_indices, 2)
            train_camera_frame_idx = [i for i in camera_frame_indices if i not in [val_camera_frame_idx, test_camera_frame_idx]]
            train_camera_frame_num = len(train_camera_frame_idx)        

            if split == 'train':
                c2w = c2w[train_camera_frame_idx] # 得到训练集的相机c2w
                camera_dts = torch.full((train_camera_frame_num, total_time_frames), dt)
                camera_times = times.unsqueeze(0).expand(train_camera_frame_num, -1)
            elif split in ['val', 'test']:
                c2w = c2w[val_camera_frame_idx].unsqueeze(0) if split == 'val' else c2w[test_camera_frame_idx].unsqueeze(0)
                camera_dts = torch.full((1, total_time_frames), dt)
                camera_times = times.unsqueeze(0)
                if self.padding_enabled:
                    camera_dts = camera_dts[:, self.padding_size:-self.padding_size]
                    camera_times = camera_times[:, self.padding_size:-self.padding_size]
        elif split == 'orbit_vis':
            c2w = c2w
            camera_dts = torch.full((total_time_frames, 1), dt)
            camera_times = times.unsqueeze(-1)
            if self.padding_enabled:
                camera_dts = camera_dts[self.padding_size:-self.padding_size]
                camera_times = camera_times[self.padding_size:-self.padding_size]
        elif split == 'fix_vis':
            c2w = c2w
            camera_dts = torch.full((1, total_time_frames), dt)
            camera_times = times.unsqueeze(0)
            if self.padding_enabled:
                camera_dts = camera_dts[:, self.padding_size:-self.padding_size]
                camera_times = camera_times[:, self.padding_size:-self.padding_size]

        # 构造cameras对象
        N = c2w.shape[0]
        cameras = DS_Cameras(
            c2w=c2w,
            fx=torch.ones(N) * camera_focal_length,
            fy=torch.ones(N) * camera_focal_length,
            cx=torch.ones(N) * IMAGE_WIDTH * 0.5,
            cy=torch.ones(N) * IMAGE_HEIGHT * 0.5,
            width=torch.ones(N, dtype=torch.long) * IMAGE_WIDTH,
            height=torch.ones(N, dtype=torch.long) * IMAGE_HEIGHT,
            near=torch.ones(N) * 0.001,
            far=torch.ones(N) * 1000.0,
            times=camera_times,
            dts=camera_dts,
        ).to(device)        
        
        return cameras

    def _apply_reflection_padding(self, data: Union[torch.Tensor, List]):
        if isinstance(data, torch.Tensor):
            start_padding = torch.flip(data[1:self.padding_size + 1], dims=[0])
            end_padding = torch.flip(data[-self.padding_size - 1:-1], dims=[0])
            data = torch.cat([start_padding, data, end_padding], dim=0).to(data.device)
        elif isinstance(data, list):
            start_padding = data[1:self.padding_size + 1][::-1]
            end_padding = data[-self.padding_size - 1:-1][::-1]
            data = start_padding + data + end_padding
        return data
  
    @staticmethod
    def recognize(path: Path) -> bool:
        paths = [
            path / 'transforms'/ 'multiview_transforms.json',
            path / 'transforms'/ 'orbit_transforms.json',
            path / 'obj' / 'frame1.obj',
        ]
        return all([p.exists() for p in paths])


@dataclass
class SyntheticDynamicMultiViewCostumeDepthDataparser(DS_BaseDataparser[DS_Cameras, DepthImages, List[DS_TriangleMesh]]):
    """
    Objverse-select dataset or DG-Mesh dataset
    """
    resolution: Tuple[int, int] = (800, 800)

    antialias: bool = True

    view_sampling_seed: int = 123

    train_views_per_frame: int = 32

    def parse(
        self,
        path: Path,
        *,
        split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis'],
        device: torch.device,
        parse_meshes: bool = False,
        costume_sample_frames: Optional[List[int]] = None,
        costume_padding_size: Optional[int] = None,
    ) -> Tuple[Indexable[DS_Cameras], Any, Any]:
        model = _KNOWN_MODELS[path.name]
        self.padding_enabled = model.reflect_padding
        self.padding_size = model.padding_size
        if costume_padding_size is not None:
            self.padding_size = costume_padding_size

        # 解析 mesh
        if parse_meshes and split == 'train':
            return self._load_mesh_sequence(model, path, device)

        # 构造相机采样
        with create_random_seed_context(self.view_sampling_seed):
            cameras = self._create_cameras_for_split(split, model, device)
            
        return cameras, None, None

    def _load_mesh_sequence(self, model: _KnownDynamicModel, path: Path, device: torch.device) -> List[DS_TriangleMesh]:
        meshes = []
        start = model.process_begin_frame
        end = model.process_begin_frame + model.process_num_frames
        for i in range(start, end):
            obj_path = path / model.process_mesh_sequence_dir / model.process_mesh_name.format(i)
            pkl_path = obj_path.with_suffix('.pkl')
            if pkl_path.exists():
                mesh = DS_TriangleMesh.deserialize(pkl_path).to(device)
            else:
                mesh = DS_TriangleMesh.from_file(obj_path, read_mtl=False).to(device)
                if model.animation_model_name == 'torus2sphere':
                    mesh = mesh.simplify(target_num_faces=60000)
                mesh.serialize(pkl_path)
            meshes.append(mesh)
        if self.padding_enabled:
            meshes = self._apply_reflection_padding(meshes)
        return meshes
    
    def _create_cameras_for_split(self, split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis'], model: _KnownDynamicModel, device: torch.device) -> Tuple[DS_Cameras, torch.Tensor, torch.Tensor]:
        resolution = self.resolution
        up = (0, 0, 1) if model.process_z_up else (0, 1, 0)
        radius = model.process_view_radius

        def _get_total_frames():
            return model.process_num_frames + self.padding_size * 2 if self.padding_enabled else model.process_num_frames

        def _get_time_data(split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis']) -> Tuple[torch.Tensor, torch.Tensor]:
            total_frames = _get_total_frames()
            dt = 1 / (total_frames - 1)
            times = torch.linspace(0, 1, total_frames, device=device)
            if split == 'train':
                camera_dts = torch.full((self.train_views_per_frame, total_frames), dt, device=device)
                camera_times = times.unsqueeze(0).expand(self.train_views_per_frame, -1)
            else:
                camera_dts = torch.full((total_frames,), dt, device=device).unsqueeze(-1)
                camera_times = times.unsqueeze(-1)
                if self.padding_enabled:
                    camera_dts = camera_dts[self.padding_size:-self.padding_size]
                    camera_times = camera_times[self.padding_size:-self.padding_size]
            
            return camera_times, camera_dts

        if split == 'train':
            cameras = DS_Cameras.from_sphere(
                center=(0, 0, 0), up=up, radius=radius,
                num_samples=self.train_views_per_frame, resolution=resolution,
                near=1e-2, far=1e2, hfov_degree=60, device=device,
                num_frames_per_view=_get_total_frames()
            )
        elif split in ['val', 'test']:
            cameras = DS_Cameras.from_sphere(
                center=(0, 0, 0), up=up, radius=radius, uniform=False,
                num_samples=model.process_num_frames, resolution=resolution,
                near=1e-2, far=1e2, hfov_degree=60, device=device
            )
        elif split in ['fix_vis', 'orbit_vis']:
            cameras = DS_Cameras.from_orbit(
                center=(0, 0, 0), up=up, radius=radius,
                pitch_degree=model.process_val_pitch_degree, num_samples=model.process_num_frames,
                resolution=resolution, near=1e-2, far=1e2, hfov_degree=60, device=device
            )
        times, dts = _get_time_data(split)
        cameras = cameras.set_times(times, dts)
        if split == 'fix_vis':
            cameras = cameras.reset_c2w_to_ref_camera_pose()

        return cameras

    def _apply_reflection_padding(self, data: Union[torch.Tensor, List]):
        if isinstance(data, torch.Tensor):
            start_padding = torch.flip(data[1:self.padding_size + 1], dims=[0])
            end_padding = torch.flip(data[-self.padding_size - 1:-1], dims=[0])
            data = torch.cat([start_padding, data, end_padding], dim=0).to(data.device)
        elif isinstance(data, list):
            start_padding = data[1:self.padding_size + 1][::-1]
            end_padding = data[-self.padding_size - 1:-1][::-1]
            data = start_padding + data + end_padding
        return data
       
    @staticmethod
    def recognize(path: Path) -> bool:
        if path.exists() and path.name in _KNOWN_MODELS:
            model = _KNOWN_MODELS[path.name]
            # Check if the mesh file exists.
            if (path / model.process_mesh_sequence_dir).exists():
                return all((path / model.process_mesh_sequence_dir / model.process_mesh_name.format(i)).exists() for i in range(model.process_begin_frame, model.process_begin_frame+model.process_num_frames))
        return False
