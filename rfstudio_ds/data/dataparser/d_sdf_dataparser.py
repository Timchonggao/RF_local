from __future__ import annotations

# import modules
from dataclasses import dataclass
import pathlib
from pathlib import Path
from typing import Literal, NamedTuple, Tuple, List, Any

import torch
from torch import Tensor

# import rfstudio modules
from rfstudio.utils.context import create_random_seed_context
from rfstudio.utils.typing import Indexable
from rfstudio.io import dump_float32_image

# import rfstudio_ds modules
from .base_dsdf_dataparser import DSDF_BaseDataparser

from ._isovalues import IsoValue, DynamicMesh
from sklearn.model_selection import train_test_split


class _KnownDynamicModel(NamedTuple):
    animation_model_name: str              # 动画名称，例如 "horse"
    num_frames: int                        # 动画帧数
    begin_frame: int                       # 动画开始帧
    mesh_sequence_dir: str                # 存放原始网格的文件夹名称，例如 "raw_mesh"
    mesh_name: str                         # 原始网格的文件名，例如 "Duck0.obj"
    val_pitch_degree: float
    view_radius: float
    z_up: bool
    sdf_sequence_dir: str = None                  # 存放SDF的文件夹名称，例如 "sdf"
    sdf_name: str = None                          # SDF的文件名，例如 "Duck0.sdf"
    sdf_sequence_padding_enabled: bool = True             # 是否对SDF进行padding，默认True
    sdf_sequence_padding_size: int = 20             # padding的大小，默认20


_KNOWN_MODELS = {
    'beagle': _KnownDynamicModel(
        animation_model_name='beagle',
        num_frames=200,
        begin_frame=0,
        mesh_sequence_dir='mesh_gt',
        mesh_name='Beagle{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
        sdf_sequence_dir='mesh_gt_preprocess/sdf_files',
        sdf_name='Beagle{}.fixed.npy',
        sdf_sequence_padding_enabled=True,
        sdf_sequence_padding_size=20,
    ),
    
    'bird': _KnownDynamicModel(
        animation_model_name='bird',
        num_frames=200,
        begin_frame=0,
        mesh_sequence_dir='mesh_gt',
        mesh_name='bluebird_animated{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
        sdf_sequence_dir='mesh_gt_preprocess/sdf_files',
        sdf_name='bluebird_animated{}.fixed.npy',
        sdf_sequence_padding_enabled=True,
        sdf_sequence_padding_size=20,
    ),

    'duck': _KnownDynamicModel(
        animation_model_name='duck',
        num_frames=200,
        begin_frame=0,
        mesh_sequence_dir='mesh_gt',
        mesh_name='Duck{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
        sdf_sequence_dir='mesh_gt_preprocess/sdf_files',
        sdf_name='Duck{}.npy',
        sdf_sequence_padding_enabled=True,
        sdf_sequence_padding_size=20,
    ),

    'girlwalk': _KnownDynamicModel(
        animation_model_name='girlwalk',
        num_frames=200,
        begin_frame=0,
        mesh_sequence_dir='mesh_gt',
        mesh_name='Cartoon Teen Girl Character_01_Anim{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
        sdf_sequence_dir='mesh_gt_preprocess/sdf_files',
        sdf_name='Anim{}.npy',
        sdf_sequence_padding_enabled=True,
        sdf_sequence_padding_size=20,
    ),

    'horse': _KnownDynamicModel(
        animation_model_name='horse',
        num_frames=200,
        begin_frame=0,
        mesh_sequence_dir='mesh_gt',
        mesh_name='horse{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
        sdf_sequence_dir='mesh_gt_preprocess/sdf_files',
        sdf_name='Brown_Horse{}.fixed.npy',
        sdf_sequence_padding_enabled=True,
        sdf_sequence_padding_size=20,
    ),

    'torus2sphere': _KnownDynamicModel(
        animation_model_name='torus2sphere',
        num_frames=200,
        begin_frame=1,
        mesh_sequence_dir='mesh_gt',
        mesh_name='deform{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
        sdf_sequence_dir='mesh_gt_preprocess/sdf_files',
        sdf_name='deform{}.npy',
        sdf_sequence_padding_enabled=True,
        sdf_sequence_padding_size=20,
    ),
    
    'cat': _KnownDynamicModel(
        animation_model_name='cat',
        num_frames=240,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
        sdf_sequence_dir='mesh_gt_preprocess',
        sdf_name='frame{}.fixed.npy',
        sdf_sequence_padding_enabled=True,
        sdf_sequence_padding_size=20,
    ),

    'deer': _KnownDynamicModel(
        animation_model_name='deer',
        num_frames=241,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
        sdf_sequence_dir='mesh_gt_preprocess',
        sdf_name='frame{}.fixed.npy',
        sdf_sequence_padding_enabled=True,
        sdf_sequence_padding_size=20,
    ),

    'girlwalk_diffusion4d': _KnownDynamicModel(
        animation_model_name='girlwalk',
        num_frames=148,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
        sdf_sequence_dir='mesh_gt_preprocess',
        sdf_name='frame{}.fixed.npy',
        sdf_sequence_padding_enabled=True,
        sdf_sequence_padding_size=20,
    ),

    'lego': _KnownDynamicModel(
        animation_model_name='lego',
        num_frames=169,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
        sdf_sequence_dir='mesh_gt_preprocess',
        sdf_name='frame{}.fixed.npy',
        sdf_sequence_padding_enabled=True,
        sdf_sequence_padding_size=20,
    ),

    'spidermanwalk': _KnownDynamicModel(
        animation_model_name='spidermanwalk',
        num_frames=25,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
        sdf_sequence_dir='mesh_gt_preprocess',
        sdf_name='frame{}.fixed.npy',
        sdf_sequence_padding_enabled=True,
        sdf_sequence_padding_size=10,
    ),

    'toy': _KnownDynamicModel(
        animation_model_name='toy',
        num_frames=71,
        begin_frame=0,
        mesh_sequence_dir='obj',
        mesh_name='frame{}.obj',
        val_pitch_degree=15.,
        view_radius=2.5,
        z_up=False,
        sdf_sequence_dir='mesh_gt_preprocess',
        sdf_name='frame{}.fixed.npy',
        sdf_sequence_padding_enabled=True,
        sdf_sequence_padding_size=20,
    ),
}


@dataclass
class DynamicMeshSDFDataparser(DSDF_BaseDataparser[Tensor, Tensor, Any]):

    split_seed: int = 123

    dynamic_object: DynamicMesh = None

    def parse(
        self,
        path: Path,
        *,
        split: Literal['train', 'test', 'all'],
        device: torch.device,
    ) -> Tuple[Any, Indexable[IsoValue], Any]:
        self.model = _KNOWN_MODELS[path.name]
        model = self.model
        sdf_file_dir = path / model.sdf_sequence_dir

        if self.dynamic_object is None:
            self.dynamic_object = DynamicMesh(sdf_file_dir=str(sdf_file_dir), padding_enabled=model.sdf_sequence_padding_enabled, padding_size=model.sdf_sequence_padding_size)
            self.isovalues = self.dynamic_object.iso_values # numpy array
            self.times = self.dynamic_object.times
            self.num_points = self.isovalues.shape[1] * self.isovalues.shape[2] * self.isovalues.shape[3]
            self.time_step = self.isovalues.shape[0]

        isovalues = torch.tensor(self.isovalues, dtype=torch.float).to(device)
        times = torch.tensor(self.times, dtype=torch.float).to(device)

        time_train, time_test, isovalues_train, isovalues_test = train_test_split(
            times, isovalues, test_size=0.2, random_state=self.split_seed
        )

        if split == "train":
            return time_train, isovalues_train, None
        elif split == "test":
            return time_test, isovalues_test, None
        elif split == "all":
            return times, isovalues, None
        else:
            raise ValueError(f"Unsupported split type: {split}")
    
    @staticmethod
    def recognize(path: Path) -> bool:
        if path.exists() and path.name in _KNOWN_MODELS:
            model = _KNOWN_MODELS[path.name]
            # Check if the mesh file exists.
            if (path / model.sdf_sequence_dir).exists():
                return all((path / model.sdf_sequence_dir / model.sdf_name.format(i)).exists() for i in range(model.begin_frame, model.begin_frame+model.num_frames))
        return False
