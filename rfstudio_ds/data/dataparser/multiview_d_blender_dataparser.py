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
    
    'toy': _KnownDynamicModel(
        animation_model_name='toy',
        reflect_padding=True,
        padding_size=15,
        num_frames=71,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
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
        view_radius=2.5,
        z_up=False,
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
        view_radius=2.5,
        z_up=False,
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
        view_radius=2.5,
        z_up=False,
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
        view_radius=2.5,
        z_up=False,
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
        view_radius=2.5,
        z_up=False,
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
        view_radius=2.5,
        z_up=False,
    ),

}

_KNOWN_MODELS_full = {
    
    'toy': _KnownDynamicModel(
        animation_model_name='toy',
        reflect_padding=True,
        padding_size=15,
        num_frames=71,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),

    'spidermanwalk': _KnownDynamicModel(
        animation_model_name='spidermanwalk',
        reflect_padding=True,
        padding_size=20,
        num_frames=25,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
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
        view_radius=2.5,
        z_up=False,
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
        view_radius=2.5,
        z_up=False,
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
        view_radius=2.5,
        z_up=False,
    ),

    'eagle': _KnownDynamicModel(
        animation_model_name='eagle',
        reflect_padding=True,
        padding_size=20,
        num_frames=209,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
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
        view_radius=2.5,
        z_up=False,
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
        view_radius=2.5,
        z_up=False,
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
        view_radius=2.5,
        z_up=False,
    ),

    'girlwalk': _KnownDynamicModel(
        animation_model_name='girlwalk',
        reflect_padding=True,
        padding_size=20,
        num_frames=148,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
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
        view_radius=2.5,
        z_up=False,
    ),

    'monster_roar': _KnownDynamicModel(
        animation_model_name='monster_roar',
        reflect_padding=True,
        padding_size=15,
        num_frames=130,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),

    'boy_warrior': _KnownDynamicModel(
        animation_model_name='boy_warrior',
        reflect_padding=True,
        padding_size=15,
        num_frames=49,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),

    'refrigerator': _KnownDynamicModel(
        animation_model_name='refrigerator',
        reflect_padding=True,
        padding_size=15,
        num_frames=121,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),

    'dump_truck': _KnownDynamicModel(
        animation_model_name='dump_truck',
        reflect_padding=True,
        padding_size=15,
        num_frames=81,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
    ),
}


@dataclass
class SyntheticDynamicMultiviewBlenderRGBDataparser(DS_BaseDataparser[DS_Cameras, RGBImages, Any]):

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

        pass

    @staticmethod
    def recognize(path: Path) -> bool:
        pass


@dataclass
class SyntheticDynamicMultiviewBlenderRGBADataparser(DS_BaseDataparser[DS_Cameras, RGBAImages, Any]):

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

        # 解析 mesh
        if parse_meshes and split == 'train':
            return self._load_mesh_sequence(model, path, device)

        # 采样相机，和图像
        with create_random_seed_context(self.view_sampling_seed):
            cameras, images = self._get_data_for_split(split, model, path, device)

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
        if split == 'train':
            with open(path / 'transforms' / f"random_multiview_train_transforms_downsample_2x.json", 'r') as f:
                meta = json.loads(f.read())
        elif split == 'test':
            with open(path / 'transforms' / f"random_multiview_test_transforms.json", 'r') as f:
                meta = json.loads(f.read())
        elif split in ['val', 'fix_vis']:
            with open(path / 'transforms' / f"front_transforms.json", 'r') as f:
                meta = json.loads(f.read())
        elif split == 'orbit_vis':
            with open(path / 'transforms' / f"orbit_transforms.json", 'r') as f:
                meta = json.loads(f.read())

        camera_angle_x = float(meta["camera_angle_x"])
        camera_focal_length = 0.5 * IMAGE_WIDTH / np.tan(0.5 * camera_angle_x)
        
        camera_frame_num = len(meta['frames'])
        camera_frames_pose = []
        camera_frames_images_path = []
        for camera_frame in meta['frames']:
            camera_frames_pose.append(camera_frame['transform_matrix'])
            camera_frames_images_path.append([Path(path / 'transforms' / p) for p in camera_frame['file_path']])
        
        camera_frames_pose = np.array(camera_frames_pose, dtype=np.float32)
        c2w = torch.from_numpy(camera_frames_pose[:, :3, :])
        c2w[:, :, 3] *= 2 / 3  # scale to bbox [-1, 1]^3
         
        total_time_frames = (model.num_frames + self.padding_size * 2) if self.padding_enabled else model.num_frames
        dt = 1 / (total_time_frames - 1)
        times = torch.linspace(0, 1, total_time_frames) # 均匀的时间序列(已处理padding)

        # 根据split获得raw data，主要关注：c2w的变化；padding 对 time 和 image path 的影响
        if split in ['train', 'test']:
            camera_dts = torch.full((camera_frame_num, total_time_frames), dt)
            camera_times = times.unsqueeze(0).expand(camera_frame_num, -1)
            camera_frames_images_path = camera_frames_images_path
            if self.padding_enabled: # 如果padding了time，则需要调整image path，让model能够得到padding输入；c2w不用修改，因为padding是针对time进行的,对view不用做padding
                if split == 'train':
                    padding_camera_frames_images_path = []
                    padding_camera_frames_images_path.extend(self._apply_reflection_padding(p) for p in camera_frames_images_path)
                    camera_frames_images_path = padding_camera_frames_images_path # padding train image
                elif split == 'test': # 需要调整time，让model能够得到原本的time输入（test不用考虑padding的帧）
                    camera_dts = camera_dts[:, self.padding_size:-self.padding_size]
                    camera_times = camera_times[:, self.padding_size:-self.padding_size]
        elif split in ['val', 'fix_vis']:
            camera_dts = torch.full((1, total_time_frames), dt)
            camera_times = times.unsqueeze(0)
            camera_frames_images_path = camera_frames_images_path
            if self.padding_enabled: # 需要调整time，让model能够得到原本的time输入（test不用考虑padding的帧）
                camera_dts = camera_dts[:, self.padding_size:-self.padding_size]
                camera_times = camera_times[:, self.padding_size:-self.padding_size]
        elif split == 'orbit_vis':
            camera_dts = torch.full((total_time_frames, 1), dt) # 每一个camera只有1个time
            camera_times = times.unsqueeze(-1)
            camera_frames_images_path = camera_frames_images_path
            if self.padding_enabled: # # 需要调整time，让model能够得到原本的time输入（test不用考虑padding的帧）
                camera_dts = camera_dts[self.padding_size:-self.padding_size]
                camera_times = camera_times[self.padding_size:-self.padding_size]

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
            path / 'transforms'/ 'random_multiview_test_transforms.json',
            path / 'transforms'/ 'random_multiview_train_transforms.json',
            path / 'transforms'/ 'orbit_transforms.json',
            path / 'obj' / 'frame1.obj',
        ]
        return all([p.exists() for p in paths])
    
    @staticmethod
    def dump(
        inputs: DS_Cameras,
        gt_outputs: RGBImages,
        meta: Any,
        *,
        path: Path,
        split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis'],
    ) -> None:
        """
        导出为 IDR 格式：
        image/{cam_id:03d}_{frame_id:04d}.png
        motion_mask/{cam_id:03d}_{frame_id:04d}.png
        cameras.npz
        """
        assert path.exists()
        model = _KNOWN_MODELS[path.name]

        image_dir = path / "IDR" / split / "image"
        mask_dir = path / "IDR" / split / "motion_mask"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        N_cams = inputs.c2w.shape[0]
        N_frames = inputs.times.shape[1]  # 注意: times 是 [N_cams, N_frames]

        # === 构造相机矩阵 ===
        c2w = torch.cat((inputs.c2w, torch.tensor([0, 0, 0, 1]).to(inputs.c2w).expand(inputs.shape[0], 1, 4)), dim=1)
        # Blender -> COLMAP
        c2w[:, :, 1:3] *= -1 # from Blender to COLMAP
        K = inputs.intrinsic_matrix  # [N, 3, 3]
        P = torch.eye(4).to(K).view(1, 4, 4).repeat(inputs.shape[0], 1, 1)
        P[:, :3, :3] = K @ c2w[:, :3, :3].transpose(-1, -2)
        P[:, :3, 3:] = P[:, :3, :3] @ -c2w[:, :3, 3:]
        cam_file = {}
        for cam_id in range(N_cams):
            cam_file[f'world_mat_{cam_id}'] = P[cam_id].detach().cpu().numpy().astype(np.float64)
            cam_file[f'scale_mat_{cam_id}'] = np.eye(4)
            if split == 'train':
                start_frame = model.padding_size
                end_frame = model.num_frames + model.padding_size
            else:
                start_frame = 0
                end_frame = model.num_frames
            for frame_id in range(start_frame, end_frame):
                count_id = cam_id * N_frames + frame_id
                image = gt_outputs[count_id].blend((1, 1, 1)).item().clamp(0, 1)
                mask_ = gt_outputs[count_id].item()[..., -1]
                binary_mask = (mask_ <= 0.5).float()
                mask = binary_mask.unsqueeze(-1).repeat(1, 1, 3)
                if split == 'train':
                    frame_id = frame_id - model.padding_size
                image_path = image_dir / f'{cam_id:03d}_{frame_id:04d}.png'
                mask_path = mask_dir / f'{cam_id:03d}_{frame_id:04d}.png'
                dump_float32_image(image_path, image)
                dump_float32_image(mask_path, mask)
        np.savez_compressed(path / 'IDR'/ split / 'cameras.npz', **cam_file)


@dataclass
class SyntheticTemporalDynamicMultiviewBlenderRGBADataparser(DS_BaseDataparser[DS_Cameras, RGBAImages, Any]):

    resolution: Tuple[int, int] = (800, 800)
    
    scale_factor: Optional[float] = None

    view_sampling_seed: int = 123
    
    mode: Literal['uniform', 'random'] = 'uniform'
    
    test_ratio: Optional[float] = 0.2

    def parse(
        self,
        path: Path,
        *,
        split: Literal['train', 'test', 'val'],
        device: torch.device,
        parse_meshes: bool = False,
        costume_sample_frames: Optional[List[int]] = None,
        costume_padding_size: Optional[int] = None,
        test_ratio: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> Tuple[Indexable[DS_Cameras], Indexable[RGBAImages], Any]:
        model = _KNOWN_MODELS[path.name]
        self.padding_enabled = model.reflect_padding
        self.padding_size = model.padding_size
        self.total_frames = model.num_frames
        if costume_padding_size is not None:
            self.padding_size = costume_padding_size
        if self.padding_enabled:
            self.total_frames = self.total_frames + 2 * self.padding_size
        if test_ratio is not None:
            self.test_ratio = test_ratio
        if mode is not None:
            self.mode = mode
            
        if split in ['train', 'test', 'val']:
            test_frame_ids = self._generate_test_frame_ids(model.num_frames)

            # padding 的情况，需要把索引 shift 到 padded 区间中
            if self.padding_enabled:
                offset = self.padding_size
                test_frame_ids_padded = [t + offset for t in test_frame_ids]
                test_frame_ids = test_frame_ids_padded
            train_frame_ids = [
                f for f in range(self.total_frames)
                if f not in test_frame_ids
            ]
            
            if parse_meshes:
                meshes = self._load_mesh_sequence(model, path, device, train_frame_ids, test_frame_ids, split=split)
                return meshes
            
            cameras, images = self._load_temporal_splits(
                path, device,
                ["random_multiview_train_transforms_downsample_2x.json", "random_multiview_test_transforms.json"],
                train_frame_ids,
                test_frame_ids,
                split=split,
            )
            return cameras, images, None      

    def _load_temporal_splits(
        self,
        path: Path,
        device: torch.device,
        json_names: List[str],
        train_frame_ids: List[int],
        test_frame_ids: List[int],
        split: Literal['train', 'test'],
    ):
        """
        读取多个 transform JSON 文件，并直接合并为一个 unified cameras + images。

        例如 json_names = [
            "random_multiview_train_transforms_downsample_2x.json",
            "random_multiview_test_transforms.json"
        ]
        """
        # -------------------------------
        # 统一 resolution
        # -------------------------------
        W, H = self.resolution
        if self.scale_factor is not None:
            W = int(W * self.scale_factor)
            H = int(H * self.scale_factor)

        # -------------------------------
        # 收集所有视角
        # -------------------------------
        all_c2w = []
        all_img_paths = []

        for json_name in json_names:
            with open(path / "transforms" / json_name, "r") as f:
                meta = json.load(f)

            camera_angle_x = float(meta["camera_angle_x"])
            focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

            # ---- per-split ----
            split_c2w = []
            split_img_paths = []

            for frame in meta["frames"]:
                split_c2w.append(frame["transform_matrix"])
                split_img_paths.append(
                    [path / "transforms" / p for p in frame["file_path"]]
                )

            # 每个文件可能多视角
            split_c2w = torch.tensor(split_c2w, dtype=torch.float32)[:, :3, :]
            split_c2w[:, :, 3] *= 2 / 3  # scale to bbox

            all_c2w.append(split_c2w)
            all_img_paths.append(split_img_paths)

        # ----------------------------------
        # 合并所有视角
        # ----------------------------------
        all_c2w = torch.cat(all_c2w, dim=0)
        V = all_c2w.shape[0]  # total merged views

        # 展开 image path 列表
        flat_img_paths = [] # V个list，每个list下是Frames个图片路径（padding后）
        for split_images in all_img_paths: # all_img_paths = [split1[view1[], view2[]...], split2[view7[], view8[]...]]
            for view_imgs in split_images: # view_imgs = view1[]
                if self.padding_enabled:
                    padding_imgs = self._apply_reflection_padding(view_imgs)
                    view_imgs = padding_imgs # 这里的imgs已经是padded的了 padded_view1[]
                flat_img_paths.append(view_imgs)
            
        train_flats_imgs = []
        test_flats_imgs = []
        for view_imgs in flat_img_paths:
            train_flats_imgs.append([view_imgs[i] for i in train_frame_ids])
            test_flats_imgs.append([view_imgs[i] for i in test_frame_ids])

        # ----------------------------------
        # 构造 cameras
        # ----------------------------------
        T = self.total_frames 

        times = torch.linspace(0, 1, T)
        
        train_times = times[train_frame_ids]
        train_dts = []
        for i in range(len(train_frame_ids) - 1):
            train_dts.append(train_times[i+1] - train_times[i])
        train_dts.append(1 - train_times[-1])
        train_dts = torch.tensor(train_dts, dtype=torch.float32).unsqueeze(0).expand(V, -1)
        
        test_times = times[test_frame_ids]
        test_dts = []
        for i in range(len(test_frame_ids) - 1):
            test_dts.append(test_times[i+1] - test_times[i])
        test_dts.append(1 - test_times[-1])
        test_dts = torch.tensor(test_dts, dtype=torch.float32).unsqueeze(0).expand(V, -1)

        if split == 'train':
            cameras = DS_Cameras(
                c2w=all_c2w,
                fx=torch.full((V,), focal),
                fy=torch.full((V,), focal),
                cx=torch.full((V,), W * 0.5),
                cy=torch.full((V,), H * 0.5),
                width=torch.full((V,), W, dtype=torch.long),
                height=torch.full((V,), H, dtype=torch.long),
                near=torch.full((V,), 0.001),
                far=torch.full((V,), 1000.0),
                times=train_times.unsqueeze(0).expand(V, -1),
                dts=train_dts,
            ).to(device) 
            images = load_masked_image_batch_lazy(
                [item for sublist in train_flats_imgs for item in sublist],
                device='cpu',
                scale_factor=self.scale_factor,
                read_uint8=True,
            )
        else:
            cameras = DS_Cameras(
                c2w=all_c2w,
                fx=torch.full((V,), focal),
                fy=torch.full((V,), focal),
                cx=torch.full((V,), W * 0.5),
                cy=torch.full((V,), H * 0.5),
                width=torch.full((V,), W, dtype=torch.long),
                height=torch.full((V,), H, dtype=torch.long),
                near=torch.full((V,), 0.001),
                far=torch.full((V,), 1000.0),
                times=test_times.unsqueeze(0).expand(V, -1),
                dts=test_dts,
            ).to(device) 
            images = load_masked_image_batch_lazy(
                [item for sublist in test_flats_imgs for item in sublist],
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
    
    def _generate_test_frame_ids(self, total_frames: int):
        # determine K
        K = max(1, int(total_frames * self.test_ratio))
        K = min(K, total_frames)

        # uniform mode
        if self.mode == "uniform":
            if K == 1:
                return [0]
            return sorted(np.linspace(0, total_frames - 1, K, dtype=int).tolist())

        # random mode
        if self.mode == "random":
            np.random.seed(self.view_sampling_seed)
            return sorted(np.random.choice(total_frames, K, replace=False).tolist())

    def _load_mesh_sequence(self, model: _KnownDynamicModel, path: Path, device: torch.device, train_frame_ids: List[int], test_frame_ids: List[int], split: Literal['train', 'test', 'val']) -> List[DS_TriangleMesh]:
        
        meshes = []

        start = model.begin_frame
        end = model.begin_frame + model.num_frames

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
        if split == 'train':
            return [meshes[i] for i in train_frame_ids]
        else:
            return [meshes[i] for i in test_frame_ids]
    
    @staticmethod
    def recognize(path: Path) -> bool:
        paths = [
            path / 'transforms'/ 'random_multiview_test_transforms.json',
            path / 'transforms'/ 'random_multiview_train_transforms.json',
            path / 'transforms'/ 'orbit_transforms.json',
            path / 'obj' / 'frame1.obj',
        ]
        return all([p.exists() for p in paths])
    