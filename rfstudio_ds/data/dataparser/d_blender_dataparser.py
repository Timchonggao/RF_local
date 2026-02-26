from __future__ import annotations

# import modules
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Literal, NamedTuple, Tuple, List, Any, Union, Optional

import torch
import numpy as np
# import rfstudio modules
from rfstudio.graphics import RGBAImages, RGBImages
from rfstudio.utils.typing import Indexable
from rfstudio.data.dataparser.utils import load_image_batch_lazy, load_masked_image_batch_lazy, LazyMaskedImageBatchProxy
from rfstudio.io import dump_float32_image
from rfstudio.utils.context import create_random_seed_context

# import rfstudio_ds modules
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh
from .base_dataparser import DS_BaseDataparser


class _KnownDynamicModel(NamedTuple):
    animation_model_name: str              # 动画名称，例如 "horse"
    reflect_padding: bool                  # 是否需要反射padding，一般是True
    padding_size: int                     # 反射padding的大小，一般是16
    num_frames: int                        # 动画帧数
    begin_frame: int                       # 动画开始帧
    mesh_sequence_dir: str                 # 存放原始网格的文件夹名称，例如 "raw_mesh"
    mesh_name: str                         # 原始网格的文件名，例如 "Duck0.obj"
    val_pitch_degree: float
    view_radius: float
    z_up: bool


_KNOWN_MODELS = {

    'bouncingballs': _KnownDynamicModel(
        animation_model_name='bouncingballs',
        num_frames=150,
        begin_frame=0,
        mesh_sequence_dir=None,
        mesh_name=None,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=True,
        reflect_padding=True,
        padding_size=15,
    ),

    'hellwarrior': _KnownDynamicModel(
        animation_model_name='hellwarrior',
        num_frames=100,
        begin_frame=0,
        mesh_sequence_dir=None,
        mesh_name=None,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=True,
        reflect_padding=True,
        padding_size=10,
    ),

    'hook': _KnownDynamicModel(
        animation_model_name='hook',
        num_frames=100,
        begin_frame=0,
        mesh_sequence_dir=None,
        mesh_name=None,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=True,
        reflect_padding=True,
        padding_size=10,
    ),

    'jumpingjacks': _KnownDynamicModel(
        animation_model_name='jumpingjacks',
        num_frames=200,
        begin_frame=0,
        mesh_sequence_dir=None,
        mesh_name=None,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=True,
        reflect_padding=True,
        padding_size=20,
    ),

    'DNerf_lego': _KnownDynamicModel(
        animation_model_name='DNerf_lego',
        num_frames=50,
        begin_frame=0,
        mesh_sequence_dir=None,
        mesh_name=None,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=True,
        reflect_padding=True,
        padding_size=5,
    ),

    'mutant': _KnownDynamicModel(
        animation_model_name='mutant',
        num_frames=150,
        begin_frame=0,
        mesh_sequence_dir=None,
        mesh_name=None,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=True,
        reflect_padding=True,
        padding_size=15,
    ),

    'standup': _KnownDynamicModel(
        animation_model_name='standup',
        num_frames=150,
        begin_frame=0,
        mesh_sequence_dir=None,
        mesh_name=None,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=True,
        reflect_padding=True,
        padding_size=15,
    ),

    'trex': _KnownDynamicModel(
        animation_model_name='trex',
        num_frames=200,
        begin_frame=0,
        mesh_sequence_dir=None,
        mesh_name=None,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=True,
        reflect_padding=True,
        padding_size=20,
    ),

    'beagle': _KnownDynamicModel(
        animation_model_name='beagle',
        num_frames=200,
        begin_frame=0,
        mesh_sequence_dir='mesh_gt',
        mesh_name='Beagle{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=True,
        reflect_padding=True,
        padding_size=15,
    ),

    'bird': _KnownDynamicModel(
        animation_model_name='bird',
        num_frames=200,
        begin_frame=0,
        mesh_sequence_dir='mesh_gt',
        mesh_name='bluebird_animated{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=True,
        reflect_padding=True,
        padding_size=15,
    ),

    'duck': _KnownDynamicModel(
        animation_model_name='duck',
        num_frames=200,
        begin_frame=0,
        mesh_sequence_dir='mesh_gt',
        mesh_name='Duck{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=True,
        reflect_padding=True,
        padding_size=15,
    ),

    'girlwalk': _KnownDynamicModel(
        animation_model_name='girlwalk',
        num_frames=200,
        begin_frame=0,
        mesh_sequence_dir='mesh_gt',
        mesh_name='Cartoon Teen Girl Character_01_Anim{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=True,
        reflect_padding=True,
        padding_size=15,
    ),
    
    'horse': _KnownDynamicModel(
        animation_model_name='horse',
        num_frames=200,
        begin_frame=0,
        mesh_sequence_dir='mesh_gt',
        mesh_name='horse{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=True,
        reflect_padding=True,
        padding_size=15,
    ),

    'torus2sphere': _KnownDynamicModel(
        animation_model_name='torus2sphere',
        num_frames=200,
        begin_frame=1,
        mesh_sequence_dir='mesh_gt',
        mesh_name='deform{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=True,
        reflect_padding=True,
        padding_size=15,
    ),

    'toy': _KnownDynamicModel(
        animation_model_name='toy',
        reflect_padding=True,
        padding_size=15,
        num_frames=71,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=1.5,
        z_up=True,
    ),

    'excavator': _KnownDynamicModel(
        animation_model_name='excavator',
        reflect_padding=True,
        padding_size=15,
        num_frames=81,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=1.5,
        z_up=True,
    ),

    'spiderman_fight': _KnownDynamicModel(
        animation_model_name='spiderman_fight',
        reflect_padding=True,
        padding_size=20,
        num_frames=96,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=1.5,
        z_up=True,
    ),

    'cat': _KnownDynamicModel(
        animation_model_name='cat',
        reflect_padding=True,
        padding_size=20,
        num_frames=240,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=1.5,
        z_up=True,
    ),

    'football_player': _KnownDynamicModel(
        animation_model_name='football_player',
        reflect_padding=True,
        padding_size=15,
        num_frames=153,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=1.5,
        z_up=True,
    ),

    'rabbit': _KnownDynamicModel(
        animation_model_name='rabbit',
        reflect_padding=True,
        padding_size=5,
        num_frames=253,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=1.5,
        z_up=True,
    ),

    'deer': _KnownDynamicModel(
        animation_model_name='deer',
        reflect_padding=True,
        padding_size=20,
        num_frames=241,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=1.5,
        z_up=True,
    ),

    'lego': _KnownDynamicModel(
        animation_model_name='lego',
        reflect_padding=True,
        padding_size=20,
        num_frames=169,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=1.5,
        z_up=True,
    ),
}

@dataclass
class SyntheticDynamicMonocularBlenderRGBDataparser(DS_BaseDataparser[DS_Cameras, RGBImages, Any]):

    alpha_color: Literal['white', 'black'] = 'black'
    """
    alpha color of background
    """

    scale_factor: Optional[float] = None
    """
    scale factor for resizing image
    """

    def parse(
        self,
        path: Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
    ) -> Tuple[Indexable[DS_Cameras], Indexable[RGBImages], Any]:

        IMAGE_WIDTH = 800
        IMAGE_HEIGHT = 800

        if self.scale_factor is not None:
            IMAGE_WIDTH = int(IMAGE_WIDTH * self.scale_factor)
            IMAGE_HEIGHT = int(IMAGE_HEIGHT * self.scale_factor)

        with open(path / f"transforms_{split}.json", 'r') as f:
            meta = json.loads(f.read())
        image_filenames = [path / (frame['file_path'] + ".png") for frame in meta['frames']]
        frame_times = [frame['time'] for frame in meta['frames']]
        poses = np.array([frame['transform_matrix'] for frame in meta['frames']], dtype=np.float32)

        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * IMAGE_WIDTH / np.tan(0.5 * camera_angle_x)

        c2w = torch.from_numpy(poses[:, :3, :])            # camera to world transform
        c2w[:, :, 3] *= 2 / 3                              # scale to bbox [-1, 1]^3

        alpha_color = {
            'white': (1., 1., 1.),
            'black': (0., 0., 0.),
        }[self.alpha_color]

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
            times=torch.tensor(frame_times, dtype=torch.float32)
        ).to(device)

        images = load_image_batch_lazy(
            image_filenames,
            device=device,
            alpha_color=alpha_color,
            scale_factor=self.scale_factor,
        )

        return cameras, images, None

    @staticmethod
    def recognize(path: Path) -> bool:
        paths = [
            path / 'transforms_test.json',
            path / 'transforms_train.json',
            path / 'transforms_val.json',
        ]
        return all([p.exists() for p in paths])


@dataclass
class DNerfSyntheticDynamicMonocularBlenderRGBADataparser(DS_BaseDataparser[DS_Cameras, RGBAImages, Any]):

    resolution: Tuple[int, int] = (800, 800)
    
    scale_factor: Optional[float] = None

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
    ) -> Tuple[Indexable[DS_Cameras], Indexable[RGBAImages], Any]:
        
        if 'lego' in path.name:
            model = _KNOWN_MODELS['DNerf_lego']
        else:
            model = _KNOWN_MODELS[path.name]
        
        self.padding_enabled = model.reflect_padding
        self.padding_size = model.padding_size
        if costume_padding_size is not None:
            self.padding_size = costume_padding_size

        with open(path / f"transforms_train.json", 'r') as f:
            meta = json.loads(f.read())
        ori_full_times = [frame['time'] for frame in meta['frames']]
        total_num_frames = model.num_frames
        if self.padding_enabled:
            total_num_frames = model.num_frames + 2 * self.padding_size
            padding_times = np.linspace(0, 1, total_num_frames)
            
            # 构建原_time → 新_time 映射
            time_mapping = {
                ori_full_times[i]: padding_times[i + self.padding_size]
                for i in range(len(ori_full_times))
            }
        else:
            time_mapping = {t: t for t in ori_full_times}
        
        # parse meshes
        if parse_meshes and split == 'train':
            return None
        
        # parse cameras
        if split in ['train', 'val', 'test']:
            resolution = self.resolution
            IMAGE_WIDTH = resolution[0]
            IMAGE_HEIGHT = resolution[1]

            if self.scale_factor is not None:
                IMAGE_WIDTH = int(IMAGE_WIDTH * self.scale_factor)
                IMAGE_HEIGHT = int(IMAGE_HEIGHT * self.scale_factor)

            with open(path / f"transforms_{split}.json", 'r') as f:
                meta = json.loads(f.read())
            image_filenames = [path / (frame['file_path'] + ".png") for frame in meta['frames']]
            poses = np.array([frame['transform_matrix'] for frame in meta['frames']], dtype=np.float32)

            camera_angle_x = float(meta["camera_angle_x"])
            focal_length = 0.5 * IMAGE_WIDTH / np.tan(0.5 * camera_angle_x)

            c2w = torch.from_numpy(poses[:, :3, :])          # camera to world transform
            c2w[:, :, 3] *= 2 / 3                              # scale to bbox [-1, 1]^3

            times = np.array([frame['time'] for frame in meta['frames']])
            dt = times[1] - times[0]
            
            if self.padding_enabled:
                if split == 'train':
                    padding_times = np.linspace(0, 1, total_num_frames)
                    dt = padding_times[1] - padding_times[0]
                    times = padding_times
                    c2w = self._apply_reflection_padding(data=c2w)
                    image_filenames = self._apply_reflection_padding(data=image_filenames)
                if split != 'train':
                    padding_times = [time_mapping[t] for t in times]
                    dt = padding_times[1] - padding_times[0]
                    times = padding_times
            camera_times = torch.tensor(times, dtype=torch.float32).unsqueeze(-1)
            camera_dts = torch.tensor([dt] * len(times), dtype=torch.float32).unsqueeze(-1)

            N = c2w.shape[0]
            cameras = DS_Cameras(
                c2w=c2w,
                fx=torch.ones(N) * focal_length,
                fy=torch.ones(N) * focal_length,
                cx=torch.ones(N) * IMAGE_WIDTH * 0.5,
                cy=torch.ones(N) * IMAGE_HEIGHT * 0.5,
                width=torch.ones(N, dtype=torch.long) * IMAGE_WIDTH,
                height=torch.ones(N, dtype=torch.long) * IMAGE_HEIGHT,
                near=torch.ones(N) * 0.001,
                far=torch.ones(N) * 1000.0,
                times=camera_times,
                dts=camera_dts,
            ).to(device)
        elif split in ['orbit_vis', 'fix_vis']:
            times = np.linspace(0, 1, total_num_frames)
            dt = times[1] - times[0]
            if self.padding_enabled:
                times = times[self.padding_size:-self.padding_size]
            camera_times = torch.tensor(times, dtype=torch.float32).unsqueeze(-1)
            camera_dts = torch.tensor([dt] * len(times), dtype=torch.float32).unsqueeze(-1)
            
            if split == 'orbit_vis':
                cameras = DS_Cameras.from_orbit(
                    center=(0, 0, 0),
                    up=(0, 0, 1) if model.z_up else (0, 1, 0), # need set z-up camera for dg-mesh dataset
                    radius=model.view_radius,
                    pitch_degree=model.val_pitch_degree,
                    num_samples=int(model.num_frames),
                    resolution=self.resolution,
                    near=1e-2,
                    far=1e2,
                    hfov_degree=60,
                    device=device,
                )
            elif split == 'fix_vis':
                cameras = DS_Cameras.from_orbit(
                    center=(0, 0, 0),
                    up=(0, 0, 1) if model.z_up else (0, 1, 0),
                    radius=model.view_radius,
                    pitch_degree=model.val_pitch_degree,
                    num_samples=model.num_frames,
                    resolution=self.resolution,
                    near=1e-2,
                    far=1e2,
                    hfov_degree=60,
                    device=device,
                )
            cameras = cameras.set_times(camera_times, camera_dts)
            if split == 'fix_vis':
                cameras = cameras.reset_c2w_to_ref_camera_pose(ref_camera_idx=int(model.num_frames * 3 / 4))
        
        # parse images (outputs)
        if split in ['train', 'val', 'test']:
            images = load_masked_image_batch_lazy(
                image_filenames,
                device=device,
                scale_factor=self.scale_factor,
            )
        elif split in ['orbit_vis', 'fix_vis']:
            images = None

        return cameras, images, None

    def _apply_reflection_padding(self, data: Union[torch.Tensor, List, LazyMaskedImageBatchProxy]):
        if isinstance(data, torch.Tensor):
            start_padding = torch.flip(data[1:self.padding_size + 1], dims=[0])
            end_padding = torch.flip(data[-self.padding_size - 1:-1], dims=[0])
            data = torch.cat([start_padding, data, end_padding], dim=0).to(data.device)
        elif isinstance(data, list):
            start_padding = data[1:self.padding_size + 1][::-1]
            end_padding = data[-self.padding_size - 1:-1][::-1]
            data = start_padding + data + end_padding
        elif isinstance(data, LazyMaskedImageBatchProxy):
            data.load()
            images_tensor = torch.cat([data[i].item().unsqueeze(0) for i in range(len(data))])
            start_padding = torch.flip(images_tensor[1:self.padding_size + 1], dims=[0])
            end_padding = torch.flip(images_tensor[-self.padding_size - 1:-1], dims=[0])
            images_tensor = torch.cat([start_padding, images_tensor, end_padding], dim=0).to(data._device)
            data._batch = RGBAImages(images_tensor)
        return data

    @staticmethod
    def recognize(path: Path) -> bool:
        paths = [
            path / 'transforms_test.json',
            path / 'transforms_train.json',
            path / 'transforms_val.json',
        ]
        return all([p.exists() for p in paths]) and 'DNerf' in str(path)


@dataclass
class DGMeshSyntheticDynamicMonocularBlenderRGBADataparser(DS_BaseDataparser[DS_Cameras, RGBAImages, Any]):

    resolution: Tuple[int, int] = (800, 800)
    
    scale_factor: Optional[float] = None

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
    ) -> Tuple[Indexable[DS_Cameras], Indexable[RGBAImages], Any]:
        model = _KNOWN_MODELS[path.name]
        self.padding_enabled = model.reflect_padding
        self.padding_size = model.padding_size
        
        if costume_padding_size is not None:
            self.padding_size = costume_padding_size

        with open(path / f"transforms_train.json", 'r') as f:
            meta = json.loads(f.read())
        try:
            self.cam_origin = torch.tensor(meta["camera_origin"]).to(device)
        except:
            self.cam_origin = torch.zeros(3).to(device)
        ori_full_times = [frame['time'] for frame in meta['frames']]
        total_num_frames = model.num_frames
        if self.padding_enabled:
            total_num_frames = model.num_frames + 2 * self.padding_size
            padding_times = np.linspace(0, 1, total_num_frames)
            
            # 构建原_time → 新_time 映射
            time_mapping = {
                ori_full_times[i]: padding_times[i + self.padding_size]
                for i in range(len(ori_full_times))
            }
        else:
            time_mapping = {t: t for t in ori_full_times}

        # parse mesh
        if parse_meshes and split == 'train':
            return self._load_mesh_sequence(model, path, device)

        # parse cameras
        if split in ['train', 'val', 'test']:
            resolution = self.resolution
            IMAGE_WIDTH = resolution[0]
            IMAGE_HEIGHT = resolution[1]

            if self.scale_factor is not None:
                IMAGE_WIDTH = int(IMAGE_WIDTH * self.scale_factor)
                IMAGE_HEIGHT = int(IMAGE_HEIGHT * self.scale_factor)

            with open(path / f"transforms_{split}.json", 'r') as f:
                meta = json.loads(f.read())
            image_filenames = [path / (frame['file_path'] + ".png") for frame in meta['frames']]
            poses = np.array([frame['transform_matrix'] for frame in meta['frames']], dtype=np.float32)
            
            camera_angle_x = float(meta["camera_angle_x"])
            focal_length = 0.5 * IMAGE_WIDTH / np.tan(0.5 * camera_angle_x)

            c2w = torch.from_numpy(poses[:, :3, :])          # camera to world transform
            c2w[:, :, 3] *= 2 / 3                              # scale to bbox [-1, 1]^3

            times = np.array([frame['time'] for frame in meta['frames']])
            dt = times[1] - times[0]

            if self.padding_enabled:
                if split == 'train':
                    padding_times = np.linspace(0, 1, total_num_frames)
                    dt = padding_times[1] - padding_times[0]
                    times = padding_times
                    c2w = self._apply_reflection_padding(data=c2w)
                    image_filenames = self._apply_reflection_padding(data=image_filenames)
                if split != 'train':
                    padding_times = [time_mapping[t] for t in times]
                    dt = padding_times[1] - padding_times[0]
                    times = padding_times
            camera_times = torch.tensor(times, dtype=torch.float32).unsqueeze(-1)
            camera_dts = torch.tensor([dt] * len(times), dtype=torch.float32).unsqueeze(-1)

            N = c2w.shape[0]
            cameras = DS_Cameras(
                c2w=c2w,
                fx=torch.ones(N) * focal_length,
                fy=torch.ones(N) * focal_length,
                cx=torch.ones(N) * IMAGE_WIDTH * 0.5,
                cy=torch.ones(N) * IMAGE_HEIGHT * 0.5,
                width=torch.ones(N, dtype=torch.long) * IMAGE_WIDTH,
                height=torch.ones(N, dtype=torch.long) * IMAGE_HEIGHT,
                near=torch.ones(N) * 0.001,
                far=torch.ones(N) * 1000.0,
                times=camera_times,
                dts=camera_dts,
            ).to(device)
        elif split in ['orbit_vis', 'fix_vis']:
            times = np.linspace(0, 1, total_num_frames)
            dt = times[1] - times[0]
            if self.padding_enabled:
                times = times[self.padding_size:-self.padding_size]
            camera_times = torch.tensor(times, dtype=torch.float32).unsqueeze(-1)
            camera_dts = torch.tensor([dt] * len(times), dtype=torch.float32).unsqueeze(-1)
            
            if split == 'orbit_vis':
                cameras = DS_Cameras.from_orbit(
                    center=(0, 0, 0),
                    up=(0, 0, 1) if model.z_up else (0, 1, 0), # need set z-up camera for dg-mesh dataset
                    radius=model.view_radius,
                    pitch_degree=model.val_pitch_degree,
                    num_samples=int(model.num_frames),
                    resolution=self.resolution,
                    near=1e-2,
                    far=1e2,
                    hfov_degree=60,
                    device=device,
                )
            elif split == 'fix_vis':
                cameras = DS_Cameras.from_orbit(
                    center=(0, 0, 0),
                    up=(0, 0, 1) if model.z_up else (0, 1, 0),
                    radius=model.view_radius,
                    pitch_degree=model.val_pitch_degree,
                    num_samples=model.num_frames,
                    resolution=self.resolution,
                    near=1e-2,
                    far=1e2,
                    hfov_degree=60,
                    device=device,
                )
            cameras = cameras.set_times(camera_times, camera_dts)
            if split == 'fix_vis':
                cameras = cameras.reset_c2w_to_ref_camera_pose(ref_camera_idx=int(model.num_frames * 3 / 4))
        
        # parse images (outputs)
        if split in ['train', 'val', 'test']:
            images = load_masked_image_batch_lazy(
                image_filenames,
                device=device,
                scale_factor=self.scale_factor,
            )
        elif split in ['orbit_vis', 'fix_vis']:
            images = None

        return cameras, images, None

    def _load_mesh_sequence(self, model: _KnownDynamicModel, path: Path, device: torch.device) -> List[DS_TriangleMesh]:
        meshes = []
        start = model.begin_frame
        end = model.begin_frame + model.num_frames
        
        for i in range(start, end):
            obj_path = path / model.mesh_sequence_dir / model.mesh_name.format(i)
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
                vertices=((trans @ mesh.vertices.unsqueeze(-1)).squeeze(-1) - self.cam_origin) * (2 / 3),
            ) # transform to y-up and shift to origin
            meshes.append(mesh)
        if self.padding_enabled:
            meshes = self._apply_reflection_padding(meshes)
        return meshes

    def _apply_reflection_padding(self, data: Union[torch.Tensor, List, LazyMaskedImageBatchProxy]):
        if isinstance(data, torch.Tensor):
            start_padding = torch.flip(data[1:self.padding_size + 1], dims=[0])
            end_padding = torch.flip(data[-self.padding_size - 1:-1], dims=[0])
            data = torch.cat([start_padding, data, end_padding], dim=0).to(data.device)
        elif isinstance(data, list):
            start_padding = data[1:self.padding_size + 1][::-1]
            end_padding = data[-self.padding_size - 1:-1][::-1]
            data = start_padding + data + end_padding
        elif isinstance(data, LazyMaskedImageBatchProxy):
            data.load()
            images_tensor = torch.cat([data[i].item().unsqueeze(0) for i in range(len(data))])
            start_padding = torch.flip(images_tensor[1:self.padding_size + 1], dims=[0])
            end_padding = torch.flip(images_tensor[-self.padding_size - 1:-1], dims=[0])
            images_tensor = torch.cat([start_padding, images_tensor, end_padding], dim=0).to(data._device)
            data._batch = RGBAImages(images_tensor)
        return data
    
    @staticmethod
    def recognize(path: Path) -> bool:
        paths = [
            path / 'transforms_test.json',
            path / 'transforms_train.json',
            path / 'transforms_val.json',
        ]
        return all([p.exists() for p in paths]) and 'dg-mesh' in str(path)



@dataclass
class ObjSelSyntheticDynamicMonocularBlenderRGBADataparser(DS_BaseDataparser[DS_Cameras, RGBAImages, Any]):

    resolution: Tuple[int, int] = (800, 800)
    
    scale_factor: Optional[float] = None

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
    ) -> Tuple[Indexable[DS_Cameras], Indexable[RGBAImages], Any]:
        model = _KNOWN_MODELS[path.name]
        self.padding_enabled = model.reflect_padding
        self.padding_size = model.padding_size
        
        if costume_padding_size is not None:
            self.padding_size = costume_padding_size

        with open(path / f"transforms_train.json", 'r') as f:
            meta = json.loads(f.read())
        try:
            self.cam_origin = torch.tensor(meta["camera_origin"]).to(device)
        except:
            self.cam_origin = torch.zeros(3).to(device)
        ori_full_times = [frame['time'] for frame in meta['frames']]
        total_num_frames = model.num_frames
        if self.padding_enabled:
            total_num_frames = model.num_frames + 2 * self.padding_size
            padding_times = np.linspace(0, 1, total_num_frames)
            
            # 构建原_time → 新_time 映射
            time_mapping = {
                ori_full_times[i]: padding_times[i + self.padding_size]
                for i in range(len(ori_full_times))
            }
        else:
            time_mapping = {t: t for t in ori_full_times}

        # parse mesh
        if parse_meshes and split == 'train':
            return self._load_mesh_sequence(model, path, device)

        # parse cameras
        if split in ['train', 'val', 'test']:
            resolution = self.resolution
            IMAGE_WIDTH = resolution[0]
            IMAGE_HEIGHT = resolution[1]

            if self.scale_factor is not None:
                IMAGE_WIDTH = int(IMAGE_WIDTH * self.scale_factor)
                IMAGE_HEIGHT = int(IMAGE_HEIGHT * self.scale_factor)

            with open(path / f"transforms_{split}.json", 'r') as f:
                meta = json.loads(f.read())
            image_filenames = [path / (frame['file_path'] + ".png") for frame in meta['frames']]
            poses = np.array([frame['transform_matrix'] for frame in meta['frames']], dtype=np.float32)
            
            camera_angle_x = float(meta["camera_angle_x"])
            focal_length = 0.5 * IMAGE_WIDTH / np.tan(0.5 * camera_angle_x)

            c2w = torch.from_numpy(poses[:, :3, :])          # camera to world transform
            c2w[:, :, 3] *= 2 / 3                              # scale to bbox [-1, 1]^3

            times = np.array([frame['time'] for frame in meta['frames']])
            dt = times[1] - times[0]

            if self.padding_enabled:
                if split == 'train':
                    padding_times = np.linspace(0, 1, total_num_frames)
                    dt = padding_times[1] - padding_times[0]
                    times = padding_times
                    c2w = self._apply_reflection_padding(data=c2w)
                    image_filenames = self._apply_reflection_padding(data=image_filenames)
                if split != 'train':
                    padding_times = [time_mapping[t] for t in times]
                    dt = padding_times[1] - padding_times[0]
                    times = padding_times
            camera_times = torch.tensor(times, dtype=torch.float32).unsqueeze(-1)
            camera_dts = torch.tensor([dt] * len(times), dtype=torch.float32).unsqueeze(-1)

            N = c2w.shape[0]
            cameras = DS_Cameras(
                c2w=c2w,
                fx=torch.ones(N) * focal_length,
                fy=torch.ones(N) * focal_length,
                cx=torch.ones(N) * IMAGE_WIDTH * 0.5,
                cy=torch.ones(N) * IMAGE_HEIGHT * 0.5,
                width=torch.ones(N, dtype=torch.long) * IMAGE_WIDTH,
                height=torch.ones(N, dtype=torch.long) * IMAGE_HEIGHT,
                near=torch.ones(N) * 0.001,
                far=torch.ones(N) * 1000.0,
                times=camera_times,
                dts=camera_dts,
            ).to(device)
        elif split in ['orbit_vis', 'fix_vis']:
            times = np.linspace(0, 1, total_num_frames)
            dt = times[1] - times[0]
            if self.padding_enabled:
                times = times[self.padding_size:-self.padding_size]
            camera_times = torch.tensor(times, dtype=torch.float32).unsqueeze(-1)
            camera_dts = torch.tensor([dt] * len(times), dtype=torch.float32).unsqueeze(-1)
            
            if split == 'orbit_vis':
                cameras = DS_Cameras.from_orbit(
                    center=(0, 0, 0),
                    up=(0, 0, 1) if model.z_up else (0, 1, 0), # need set z-up camera for dg-mesh dataset
                    radius=model.view_radius,
                    pitch_degree=model.val_pitch_degree,
                    num_samples=int(model.num_frames),
                    resolution=self.resolution,
                    near=1e-2,
                    far=1e2,
                    hfov_degree=60,
                    device=device,
                )
            elif split == 'fix_vis':
                cameras = DS_Cameras.from_orbit(
                    center=(0, 0, 0),
                    up=(0, 0, 1) if model.z_up else (0, 1, 0),
                    radius=model.view_radius,
                    pitch_degree=model.val_pitch_degree,
                    num_samples=model.num_frames,
                    resolution=self.resolution,
                    near=1e-2,
                    far=1e2,
                    hfov_degree=60,
                    device=device,
                )
            cameras = cameras.set_times(camera_times, camera_dts)
            if split == 'fix_vis':
                cameras = cameras.reset_c2w_to_ref_camera_pose(ref_camera_idx=int(model.num_frames * 3 / 4))
        
        # parse images (outputs)
        if split in ['train', 'val', 'test']:
            images = load_masked_image_batch_lazy(
                image_filenames,
                device=device,
                scale_factor=self.scale_factor,
            )
        elif split in ['orbit_vis', 'fix_vis']:
            images = None

        return cameras, images, None
    
    def _load_mesh_sequence(self, model: _KnownDynamicModel, path: Path, device: torch.device) -> List[DS_TriangleMesh]:
        
        meshes = []

        start = model.begin_frame
        end = model.begin_frame + model.num_frames
        
        # for i in range(model.begin_frame, model.begin_frame + model.num_frames):
        for i in range(start, end):
            obj_path = path / model.mesh_sequence_dir / model.mesh_name.format(i)
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

    def _apply_reflection_padding(self, data: Union[torch.Tensor, List, LazyMaskedImageBatchProxy]):
        if isinstance(data, torch.Tensor):
            start_padding = torch.flip(data[1:self.padding_size + 1], dims=[0])
            end_padding = torch.flip(data[-self.padding_size - 1:-1], dims=[0])
            data = torch.cat([start_padding, data, end_padding], dim=0).to(data.device)
        elif isinstance(data, list):
            start_padding = data[1:self.padding_size + 1][::-1]
            end_padding = data[-self.padding_size - 1:-1][::-1]
            data = start_padding + data + end_padding
        elif isinstance(data, LazyMaskedImageBatchProxy):
            data.load()
            images_tensor = torch.cat([data[i].item().unsqueeze(0) for i in range(len(data))])
            start_padding = torch.flip(images_tensor[1:self.padding_size + 1], dims=[0])
            end_padding = torch.flip(images_tensor[-self.padding_size - 1:-1], dims=[0])
            images_tensor = torch.cat([start_padding, images_tensor, end_padding], dim=0).to(data._device)
            data._batch = RGBAImages(images_tensor)
        return data
    
    @staticmethod
    def recognize(path: Path) -> bool:
        paths = [
            path / 'transforms_test.json',
            path / 'transforms_train.json',
            path / 'transforms_val.json',
        ]
        return all([p.exists() for p in paths]) and 'ObjSel-Dyn-monocular' in str(path)
