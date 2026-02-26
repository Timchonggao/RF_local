from __future__ import annotations

# import modules
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Literal, NamedTuple, Tuple, List, Any, Union, Optional, Dict
import random

import torch
import cv2
import numpy as np
# import rfstudio modules
from rfstudio.graphics import RGBAImages, RGBImages, Points
from rfstudio.utils.typing import Indexable
from rfstudio.utils.context import create_random_seed_context
from rfstudio.data.dataparser.utils import load_image_batch_lazy, load_masked_image_batch_lazy, LazyMaskedImageBatchProxy
from rfstudio.io import dump_float32_image, load_float32_image

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
    '171204_pose1_sample': _KnownDynamicModel(
        animation_model_name='171204_pose1_sample',
        reflect_padding=True,
        padding_size=15,
        num_frames=101,
        image_dir='rgba_images',
        begin_frame=0,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),

    'cello1': _KnownDynamicModel(
        animation_model_name='cello1',
        reflect_padding=True,
        padding_size=5,
        num_frames=24,
        image_dir='image',
        begin_frame=1,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),

    'band1': _KnownDynamicModel(
        animation_model_name='band1',
        reflect_padding=True,
        padding_size=5,
        num_frames=24,
        image_dir='image',
        begin_frame=1,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),

    'hanggling_b2': _KnownDynamicModel(
        animation_model_name='hanggling_b2',
        reflect_padding=True,
        padding_size=5,
        num_frames=24,
        image_dir='image',
        begin_frame=1,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),

    'ian3': _KnownDynamicModel(
        animation_model_name='ian3',
        reflect_padding=True,
        padding_size=5,
        num_frames=24,
        image_dir='image',
        begin_frame=1,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),

    'pizza1': _KnownDynamicModel(
        animation_model_name='pizza1',
        reflect_padding=True,
        padding_size=5,
        num_frames=24,
        image_dir='image',
        begin_frame=1,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),
    
    'CoreView_394_blender': _KnownDynamicModel(
        animation_model_name='CoreView_394_blender',
        reflect_padding=True,
        padding_size=30,
        num_frames=300,
        image_dir='image',
        begin_frame=0,
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),

}

def _load_K_Rt_from_P(P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # This function is borrowed from IDR: https://github.com/lioryariv/idr

    K, R, t, *_ = cv2.decomposeProjectionMatrix(P.detach().cpu().numpy())

    K = K / K[2, 2]

    pose = np.eye(4)
    pose[:3, :3] = R.T
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return torch.from_numpy(K).to(P).flatten(), torch.from_numpy(pose).to(P)[:3, :4]



@dataclass
class CMUPanonicRGBADataparser(DS_BaseDataparser[DS_Cameras, RGBAImages, Any]):

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

            camera_name = camera_frame['camera_name']
            camera_frames_images_dir = Path(path / self.model.image_dir / camera_name)
            camera_image_list = list(camera_frames_images_dir.glob('*.png'))
            camera_image_list = natsorted(camera_image_list)
            camera_frames_images_path.append([Path(p) for p in camera_image_list])

        camera_frames_pose = np.array(camera_frames_pose, dtype=np.float32)
        c2w = torch.from_numpy(camera_frames_pose[:, :3, :])
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
            path / 'rgba_images',
        ]
        return all([p.exists() for p in paths])
    

@dataclass
class SDFFlowRGBADataparser(DS_BaseDataparser[DS_Cameras, RGBAImages, Any]):
    scale_factor: Optional[float] = None

    view_sampling_seed: int = 123

    IMAGE_WIDTH: int = 1920
    IMAGE_HEIGHT: int = 1080

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
            return self._load_mesh_sequence(model, path, device)

        # 采样相机，和图像
        with create_random_seed_context(self.view_sampling_seed):
            cameras, images = self._get_data_for_split(split, path, device)

        return cameras, images, None

    def _load_mesh_sequence(self, model: _KnownDynamicModel, path: Path, device: torch.device) -> List[DS_TriangleMesh]:
        cam_file = path / 'cameras.npz'
        camera_dict = np.load(cam_file)
        camera_frame_num = len(camera_dict.keys())//2
        
        meshes = []
        scale_mat = camera_dict['scale_mat_0'].astype(np.float32)
        scale_mat = torch.from_numpy(scale_mat).to(device)
        scale_mat_inv = torch.inverse(scale_mat)

        for frame_idx in range(1, self.model.num_frames+1):
            
            pointclouds_path = path/ 'mesh' / f"{frame_idx:04d}.ply"
            pkl_path = pointclouds_path.with_suffix('.pkl')
            if pkl_path.exists():
                pointclouds = Points.deserialize(pkl_path).to(device)
            else:
                pointclouds = Points.from_file(pointclouds_path).to(device)
                pointclouds.serialize(pkl_path)
            points = pointclouds.positions / 100 # alin ori cm to m
            ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
            points_homo = torch.cat([points, ones], dim=1)  # (N, 4)
            transformed = points_homo @ scale_mat_inv.T  # (N, 4)
            scaled_points = transformed[:, :3]
            pointclouds.replace_(
                positions=scaled_points,
            ) 
            meshes.append(pointclouds)
        if self.padding_enabled:
            meshes = self._apply_reflection_padding(meshes)
        return meshes

    def _get_data_for_split(self, split: str, path: Path, device: torch.device) -> DS_Cameras:
        # 解析相机参数
        cam_file = path / 'cameras.npz'
        camera_dict = np.load(cam_file)
        camera_frame_num = len(camera_dict.keys())//2

        c2w = torch.empty((camera_frame_num, 3, 4)) # [N, 3, 4]
        fxfycxcy = torch.empty((camera_frame_num, 4)) # [N, 4]
        camera_frames_images_path = []
        camera_frames_masks_path = []
        for camera_idx in range(camera_frame_num):
            scale_mat = camera_dict['scale_mat_%d' % camera_idx].astype(np.float32)
            world_mat = camera_dict['world_mat_%d' % camera_idx].astype(np.float32)
            P = torch.from_numpy(world_mat @ scale_mat).float()
            ixt, inv_ext = _load_K_Rt_from_P(P[:3, :4])
            c2w[camera_idx, :3, :4] = inv_ext
            fxfycxcy[camera_idx] = ixt[[0, 4, 2, 5]]
            camera_frames_images_path.append(
                [path / 'image' / f"{camera_idx:03d}_{i:04d}.jpg"
                for i in range(1, self.model.num_frames+1)]
            )
            camera_frames_masks_path.append(
                [path /'motion_mask' / f"{camera_idx:03d}_{i:04d}.png"
                for i in range(1, self.model.num_frames+1)]
            )
        c2w[:, :, 1:3] *= -1 # from COLMAP to Blender

        total_time_frames = (self.model.num_frames + self.padding_size * 2) if self.padding_enabled else self.model.num_frames
        dt = 1 / (total_time_frames - 1)
        times = torch.linspace(0, 1, total_time_frames) # 均匀的时间序列(已处理padding)
        
        camera_times = times.unsqueeze(0).expand(camera_frame_num, -1)
        camera_dts = torch.full((camera_frame_num, total_time_frames), dt)
        if split == 'train':
            padding_camera_frames_images_path = []
            padding_camera_frames_images_path.extend(self._apply_reflection_padding(p) for p in camera_frames_images_path)
            camera_frames_images_path = padding_camera_frames_images_path # padding train image
            padding_camera_frames_masks_path = []
            padding_camera_frames_masks_path.extend(self._apply_reflection_padding(p) for p in camera_frames_masks_path)
            camera_frames_masks_path = padding_camera_frames_masks_path # padding train image
        else:
            camera_times = camera_times[:, self.padding_size:-self.padding_size]
            camera_dts = camera_dts[:, self.padding_size:-self.padding_size]

        N = c2w.shape[0]
        cameras = DS_Cameras(
            c2w=c2w,
            fx=fxfycxcy[:, 0],
            fy=fxfycxcy[:, 1],
            cx=fxfycxcy[:, 2],
            cy=fxfycxcy[:, 3],
            width=torch.empty(N, dtype=torch.long).fill_(int(self.scale_factor * self.IMAGE_WIDTH) if self.scale_factor is not None else self.IMAGE_WIDTH),
            height=torch.empty(N, dtype=torch.long).fill_(int(self.scale_factor * self.IMAGE_HEIGHT) if self.scale_factor is not None else self.IMAGE_HEIGHT),
            near=torch.ones(N) * 0.001,
            far=torch.ones(N) * 1000.0,
            times=camera_times,
            dts=camera_dts,
        ).to(device)

        images = load_masked_image_batch_lazy(
            filenames=[item for sublist in camera_frames_images_path for item in sublist], # 逐个camera去读取image
            masks=[item for sublist in camera_frames_masks_path for item in sublist], # 逐个camera去读取mask
            device=device,
            scale_factor=self.scale_factor,
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
            path / 'cameras.npz',
            path / 'image',
            path / 'motion_mask',
        ]
        return all([p.exists() for p in paths])

    @staticmethod
    def dump(
        inputs: DS_Cameras,
        gt_outputs: RGBImages,
        meta: Any,
        *,
        path: Path,
        split: Literal['train', 'test', 'val'],
    ) -> None:
        """
        导出为 Dnerf 格式：
        transform.json
        """
        if split == 'test':
            cam_file = path / 'cameras.npz'
            camera_dict = np.load(cam_file)
            camera_frame_num = len(camera_dict.keys())//2
            nerf_cameras = []
            concat_rgba_dir = path / 'rgba_images'
            concat_rgba_dir.mkdir(exist_ok=True, parents=True)

            for camera_idx in range(camera_frame_num):
                scale_mat = camera_dict['scale_mat_%d' % camera_idx].astype(np.float32)
                world_mat = camera_dict['world_mat_%d' % camera_idx].astype(np.float32)
                P = torch.from_numpy(world_mat @ scale_mat).float()
                ixt, inv_ext = _load_K_Rt_from_P(P[:3, :4])
                
                T = torch.eye(4, dtype=torch.float32) 
                T[:3, :4] = inv_ext
                T[:, 1:3] *= -1 # from COLMAP to Blender
                fl_x = ixt[0]
                fl_y = ixt[4]
                cx = ixt[2]
                cy = ixt[5]
                
                for time_idx in range(inputs.times.shape[1]):
                    image_path = path / 'image' / f"{camera_idx:03d}_{time_idx+1:04d}.jpg"
                    mask_path = path /'motion_mask' / f"{camera_idx:03d}_{time_idx+1:04d}.png"
                    if not image_path.exists() or not mask_path.exists():
                        raise ValueError(f"Image or mask not found: {image_path}, {mask_path}")
                    image = load_float32_image(filename=image_path)
                    mask = load_float32_image(filename=mask_path)
                    rgba = torch.cat((image, (mask > 0.5).any(-1, keepdim=True).float()), dim=-1)
                    rgba_path = concat_rgba_dir / f"{camera_idx:03d}_{time_idx+1:04d}.png"
                    dump_float32_image(filename=rgba_path,image=rgba)

                    nerf_cam = {
                        "transform_matrix": T.tolist(),
                        "file_path": f'./rgba_images/{camera_idx:03d}_{time_idx+1:04d}.png',
                        "fl_x": float(fl_x),
                        "fl_y": float(fl_y),
                        "cx": float(cx),
                        "cy": float(cy),
                        "w": 1920,
                        "h": 1080,
                    }
                    nerf_cameras.append(nerf_cam)
            
            nerf_camera_json_path = path / 'transforms.json'
            with open(nerf_camera_json_path, "w") as f:
                json.dump({"cameras": nerf_cameras}, f, indent=2)
        else:
            return None

@dataclass
class ZJUMOCAPRGBADataparser(DS_BaseDataparser[DS_Cameras, RGBAImages, Any]):

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


    def _get_data_for_split(self, split: str, path: Path, device: torch.device) -> Tuple[DS_Cameras, Any]:
        # --- 1. 读取 JSON 文件 ---
        if split in ['test', 'val']:
            json_path = path / 'transforms_test_custom_crop300.json'
        elif split == 'train':
            json_path = path / 'transforms_train_custom_crop300.json'
        else:
            raise ValueError(f"Unsupported split: {split}")

        with open(json_path, 'r') as f:
            meta = json.loads(f.read())

        # --- 2. 按 Camera Name 分组数据 ---
        # 结构: {camera_name: [frame1, frame2, ...], ...}
        grouped_data: Dict[str, List[Dict[str, Any]]] = {}
        
        # 预处理：从 image_path 中提取 camera_name
        for frame in meta['frames']:
            # 提取相机名称 (例如，从 './Camera_B22/000318.jpg' 提取 'Camera_B22')
            file_path_parts = Path(frame['file_path']).parts
            # 相机目录名通常是倒数第二部分
            camera_name = file_path_parts[-2]

            if camera_name not in grouped_data:
                grouped_data[camera_name] = []
            
            grouped_data[camera_name].append(frame)

        # --- 3. 提取和转换数据 (按相机分组) ---
        
        # 存储所有相机的所有帧数据（展平后）
        camera_frames_pose = []
        camera_frames_images_path = []
        camera_frames_masks_path = []
        camera_fx = []
        camera_fy = []
        camera_cx = []
        camera_cy = []
        caemra_width = []
        camera_height = []
  
        camera_names = sorted(grouped_data.keys()) # 确保顺序一致

        for camera_name in camera_names:
            frames = grouped_data[camera_name]
            camera_frames_pose.append(frames[0]['transform_matrix'])
            
            camera_fx.append(float(frames[0]['fl_x']))
            camera_fy.append(float(frames[0]['fl_y']))
            camera_cx.append(float(frames[0]['cx']))
            camera_cy.append(float(frames[0]['cy']))
            caemra_width.append(int(frames[0]['w']) if self.scale_factor is None else int(frames[0]['w'] * self.scale_factor))
            camera_height.append(int(frames[0]['h']) if self.scale_factor is None else int(frames[0]['h'] * self.scale_factor))

            frame_images_path = [Path(path / frame['file_path']) for frame in frames]
            frame_masks_path = [path.parent / 'CoreView_394'/ 'mask_cihp'/ frame['file_path'].replace('jpg', 'png') for frame in frames]
            camera_frames_images_path.append(frame_images_path)
            camera_frames_masks_path.append(frame_masks_path)
            
        camera_frames_pose = np.array(camera_frames_pose, dtype=np.float32)
        c2w = torch.from_numpy(camera_frames_pose[:, :3, :])
        camera_fx = torch.from_numpy(np.array(camera_fx, dtype=np.float32))
        camera_fy = torch.from_numpy(np.array(camera_fy, dtype=np.float32))
        camera_cx = torch.from_numpy(np.array(camera_cx, dtype=np.float32))
        camera_cy = torch.from_numpy(np.array(camera_cy, dtype=np.float32))
        caemra_width = torch.from_numpy(np.array(caemra_width, dtype=np.int32))
        camera_height = torch.from_numpy(np.array(camera_height, dtype=np.int32))

        camera_frame_num = len(camera_names)
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
            padding_camera_frames_masks_path = []
            padding_camera_frames_masks_path.extend(self._apply_reflection_padding(p) for p in camera_frames_masks_path)
            camera_frames_masks_path = padding_camera_frames_masks_path # padding train mask
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
        
        images = load_masked_image_batch_lazy(
            filenames=[item for sublist in camera_frames_images_path for item in sublist], # 逐个camera去读取image
            masks=[item for sublist in camera_frames_masks_path for item in sublist], # 逐个camera去读取mask
            device='cpu',
            scale_factor=self.scale_factor,
            read_mask_uint8=True,
            
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
            path / 'transforms_test_custom_crop300.json',
            path / 'transforms_train_custom_crop300.json',
        ]
        return all([p.exists() for p in paths])
  