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
from rfstudio.graphics import RGBAImages, RGBImages
from rfstudio.utils.typing import Indexable
from rfstudio.utils.context import create_random_seed_context
from rfstudio.data.dataparser.utils import load_image_batch_lazy, load_masked_image_batch_lazy, LazyMaskedImageBatchProxy
from rfstudio.io import dump_float32_image

# import rfstudio_ds modules
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh
from .base_dataparser import DS_BaseDataparser
from natsort import natsorted


class _KnownDynamicModel(NamedTuple):
    animation_model_name: str              # 动画名称，例如 "horse"
    reflect_padding: bool                  # 是否需要反射padding，一般是True
    padding_size: int                     # 反射padding的大小，一般是16
    num_frames: int                        # 动画帧数
    image_dir: str                        # 图像文件夹
    begin_frame: int                       # 动画开始帧

    val_pitch_degree: float
    view_radius: float
    z_up: bool


_KNOWN_MODELS = { 
    'blue_car': _KnownDynamicModel(
        animation_model_name='blue_car',
        reflect_padding=True,
        padding_size=15,
        num_frames=76,
        image_dir='select_frames_1',
        begin_frame=17,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),

    'dog': _KnownDynamicModel(
        animation_model_name='dog',
        reflect_padding=True,
        padding_size=15,
        num_frames=134,
        image_dir='select_frames_1',
        begin_frame=0,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),

    'k1_double_punch': _KnownDynamicModel(
        animation_model_name='k1_double_punch',
        reflect_padding=True,
        padding_size=15,
        num_frames=121,
        image_dir='select_frames_1',
        begin_frame=262,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),
    
    'penguin': _KnownDynamicModel(
        animation_model_name='penguin',
        reflect_padding=True,
        padding_size=15,
        num_frames=71,
        image_dir='select_frames_1',
        begin_frame=10,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),

    'wolf': _KnownDynamicModel(
        animation_model_name='wolf',
        reflect_padding=True,
        padding_size=15,
        num_frames=72,
        image_dir='select_frames_1',
        begin_frame=0,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),
}



@dataclass
class RealDynamicMultiviewObjectRGBADataparser(DS_BaseDataparser[DS_Cameras, RGBAImages, Any]):

    scale_factor: Optional[float] = None

    view_sampling_seed: int = 123

    def parse(
        self,
        path: Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
        parse_meshes: bool = False,
        costume_sample_frames: Optional[List[int]] = None,
        costume_padding_size: Optional[int] = None,
    ) -> Tuple[Indexable[DS_Cameras], Indexable[RGBAImages], Any]:
        model = _KNOWN_MODELS[path.name]
        self.model = model
        self.padding_enabled = model.reflect_padding
        self.padding_size = model.padding_size
        
        if costume_padding_size is not None:
            self.padding_size = costume_padding_size

        # 解析 mesh
        if parse_meshes and split == 'train':
            return None

        # 采样相机，和图像
        with create_random_seed_context(self.view_sampling_seed):
            cameras, images = self._get_data_for_split(split, path, device)

        return cameras, images, None


    def _get_data_for_split(self, split: str, path: Path, device: torch.device) -> DS_Cameras:
        # 解析相机参数
        with open(path / f'transforms_{split}.json', 'r') as f:
            meta = json.loads(f.read())

        camera_frames_pose = []
        camera_frames_images_path = []
        camera_fx = []
        camera_fy = []
        camera_cx = []
        camera_cy = []
        caemra_width = []
        camera_height = []
        for camera_frame in meta['frames']:
            camera_frames_pose.append(camera_frame['transform_matrix'])

            camera_fx.append(float(camera_frame['fl_x']))
            camera_fy.append(float(camera_frame['fl_y']))
            camera_cx.append(float(camera_frame['cx']))
            camera_cy.append(float(camera_frame['cy']))
            caemra_width.append(int(camera_frame['w']) if self.scale_factor is None else int(camera_frame['w'] * self.scale_factor))
            camera_height.append(int(camera_frame['h']) if self.scale_factor is None else int(camera_frame['h'] * self.scale_factor))

            camera_name = camera_frame['file_path'].split('/')[1]
            camera_frames_images_dir = Path(path / self.model.image_dir / camera_name)
            camera_image_list = list(camera_frames_images_dir.glob('*.png'))
            camera_image_list = natsorted(camera_image_list)
            camera_frames_images_path.append([Path(p) for p in camera_image_list])

        camera_frames_pose = np.array(camera_frames_pose, dtype=np.float32)
        c2w = torch.from_numpy(camera_frames_pose[:, :3, :])
        c2w[:, :, 3] *= 1 / 3  # scale to bbox [-1, 1]^3
        camera_fx = torch.from_numpy(np.array(camera_fx, dtype=np.float32))
        camera_fy = torch.from_numpy(np.array(camera_fy, dtype=np.float32))
        camera_cx = torch.from_numpy(np.array(camera_cx, dtype=np.float32))
        camera_cy = torch.from_numpy(np.array(camera_cy, dtype=np.float32))
        caemra_width = torch.from_numpy(np.array(caemra_width, dtype=np.int32))
        camera_height = torch.from_numpy(np.array(camera_height, dtype=np.int32))

        camera_frame_num = len(meta['frames'])
        total_time_frames = (self.model.num_frames + self.padding_size * 2) if self.padding_enabled else self.model.num_frames
        dt = 1 / (total_time_frames - 1)
        times = torch.linspace(0, 1, total_time_frames) # 均匀的时间序列(已处理padding)
        
        camera_times = times.unsqueeze(0).expand(camera_frame_num, -1)
        camera_dts = torch.full((camera_frame_num, total_time_frames), dt)
        camera_frames_images_path = camera_frames_images_path
        if split == 'train':
            padding_camera_frames_images_path = []
            padding_camera_frames_images_path.extend(self._apply_reflection_padding(p) for p in camera_frames_images_path)
            camera_frames_images_path = padding_camera_frames_images_path # padding train image
        else:
            camera_times = camera_times[:, self.padding_size:-self.padding_size]
            camera_dts = camera_dts[:, self.padding_size:-self.padding_size]

        # 构造cameras对象
        N = c2w.shape[0]
        cameras = DS_Cameras(
            c2w=c2w,
            fx=camera_fx,
            fy=camera_fy,
            cx=camera_cx,
            cy=camera_cy,
            width=caemra_width,
            height=camera_height,
            near=torch.ones(N) * 0.001,
            far=torch.ones(N) * 1000.0,
            times=camera_times,
            dts=camera_dts,
        ).to(device)        
        
        # === 根据选中的帧索引，提取图像路径和时间 ===
        images = load_masked_image_batch_lazy(
            [item for sublist in camera_frames_images_path for item in sublist], # 逐个camera去读取image
            device='cpu',
            scale_factor=self.scale_factor,
            read_uint8=True,
        )
        
        return cameras, images

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
            path / 'transforms_train.json',
            path / 'transforms_test.json',
            path / 'transforms_val.json',
            path / 'select_frames_1',
        ]
        return all([p.exists() for p in paths])
    
