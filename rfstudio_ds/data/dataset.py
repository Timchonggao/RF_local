from __future__ import annotations

# import modules
from dataclasses import dataclass
from pathlib import Path

from typing import (
    Optional,
    Dict,
    Tuple,
    Literal,
    Generator,
    Type,
    List,
    Any,
)

import torch

# import rfstudio modules
from abc import ABC, abstractmethod
from rfstudio.data.dataset import DT, GT, MT
from rfstudio.utils.typing import Indexable, IntArrayLike, Generic
from rfstudio.graphics import DepthImages, RGBAImages, RGBImages

# import rfstudio_ds modules
from .dataparser import (
    DS_BaseDataparser,
    SyntheticDynamicMonocularBlenderDepthDataparser, SyntheticDynamicMonocularCostumeDepthDataparser, 
    SyntheticDynamicMultiViewBlenderDepthDataparser, SyntheticDynamicMultiViewCostumeDepthDataparser,
    SyntheticDynamicMonocularBlenderRGBDataparser, DNerfSyntheticDynamicMonocularBlenderRGBADataparser, DGMeshSyntheticDynamicMonocularBlenderRGBADataparser, ObjSelSyntheticDynamicMonocularBlenderRGBADataparser,
    SyntheticDynamicMultiviewBlenderRGBDataparser, SyntheticDynamicMultiviewBlenderRGBADataparser,
    RealDynamicMultiviewObjectRGBADataparser,
    CMUPanonicRGBADataparser, SDFFlowRGBADataparser, ZJUMOCAPRGBADataparser,
    SyntheticTemporalDynamicMultiviewBlenderRGBADataparser,
)
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh



@dataclass
class DS_BaseDataset(ABC, Generic[DT, GT, MT]):
    '''
    TODO
    '''

    path: Path
    """
    path to data
    """

    dataparser: Optional[DS_BaseDataparser[DT, GT, MT]] = None

    costume_sample_frames: Optional[List[int]] = None # 用于自定义指定数据帧的范围

    costume_padding_size: Optional[int] = None # 用于指定数据帧的填充大小

    agumentation_sample_frames: Optional[List[int]] = None # 用于指定数据增强的采样帧范围，比如高频运动的帧范围
    
    mode: Optional[Literal['uniform', 'random']] = 'uniform'
    
    test_ratio: Optional[float] = 0.2

    @classmethod
    @abstractmethod
    def get_dataparser_list(cls) -> List[Type[DS_BaseDataparser[DT, GT, MT]]]:
        ...

    def __setup__(self) -> None:
        if self.dataparser is None:
            for dataparser_class in self.__class__.get_dataparser_list():
                if dataparser_class.recognize(self.path):
                    self.dataparser = dataparser_class()
                    break
            else:
                raise RuntimeError("Cannot decide dataparser automatically.")
        self._pairs: Dict[Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis'], Tuple[Indexable[DT], Indexable[GT], MT]] # 字典，用于存储不同划分（训练集、测试集、验证集）的数据及其对应的元数据; 键为划分名称，值为元组（数据、标签、元数据）
        self._meshes: List[DS_TriangleMesh]
        self._meshes_pairs: Dict[Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis'], MT] 

        self._pairs = {}
        self._meshes = []
        self._meshes_pairs = {}
        self._device = None

    def to(self, device: torch.device) -> None:
        self._device = device
    
    def load(self, *, split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis']) -> None:
        if split not in self._pairs:
            assert self.dataparser is not None
            if isinstance(self.dataparser, (SyntheticTemporalDynamicMultiviewBlenderRGBADataparser)):
                self._pairs[split] = self.dataparser.parse(self.path, split=split, device=self._device, costume_sample_frames=self.costume_sample_frames, costume_padding_size=self.costume_padding_size, mode=self.mode, test_ratio=self.test_ratio)
            else:
                self._pairs[split] = self.dataparser.parse(self.path, split=split, device=self._device, costume_sample_frames=self.costume_sample_frames, costume_padding_size=self.costume_padding_size)
    def get_inputs(self, *, split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis',]) -> Indexable[DT]:
        self.load(split=split)
        dt = self._pairs[split][0]
        return dt
    
    def get_gt_outputs(self, *, split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis'],) -> Indexable[GT]:
        self.load(split=split)
        gt = self._pairs[split][1]
        return gt

    def get_meta(self, *, split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis'],) -> MT:
        self.load(split=split)
        mt = self._pairs[split][2]
        return mt
    
    def get_meshes(self, *, split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis']) -> MT:
        if isinstance (self.dataparser, (SyntheticTemporalDynamicMultiviewBlenderRGBADataparser)):
            self._meshes_pairs[split] = self.dataparser.parse(self.path, split=split, device=self._device, costume_padding_size=self.costume_padding_size, parse_meshes=True, mode=self.mode, test_ratio=self.test_ratio)
            return self._meshes_pairs[split]
        else:
            if self._meshes == []:
                # only parse meshes when it's empty, only use train split to parse meshes
                self._meshes = self.dataparser.parse(self.path, split='train', device=self._device, parse_meshes=True, costume_sample_frames=self.costume_sample_frames, costume_padding_size=self.costume_padding_size)
            if self._meshes is None:
                return None
            else:
                if split not in self._meshes_pairs:
                    if self.dataparser.padding_enabled:
                        if split == 'train':
                            self._meshes_pairs[split] = self._meshes # use padding meshes
                        else:
                            self._meshes_pairs[split] = self._meshes[self.dataparser.padding_size:-self.dataparser.padding_size] # use original meshes
                    else:
                        self._meshes_pairs[split] = self._meshes
                return self._meshes_pairs[split]

    def get_size(self, eval_split:Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis']='train', *, split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis'],) -> int:
        if isinstance(self.dataparser, (SyntheticTemporalDynamicMultiviewBlenderRGBADataparser)) and split == 'fix_vis':
            if eval_split == 'train':
                data = self.get_inputs(split='train')
            elif eval_split == 'test':
                data = self.get_inputs(split='test')
            return data.times.shape[1]
        else:
            data = self.get_inputs(split=split)
            is_multi_timeframe = isinstance(data, DS_Cameras) and data.times.shape[1] > 1
            size = data.times.shape[0] * data.times.shape[1] if is_multi_timeframe else len(data)
        return size

    def get_iter(
        self,
        split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis'],
        *,
        shuffle: bool,
        infinite: bool,
        batch_size: Optional[int] = None,
        frame_batch_size: Optional[int] = None,
        camera_batch_size: Optional[int] = None,
        time_window_size: Optional[int] = None,
    ) -> Generator[Tuple[DT, GT, IntArrayLike], None, None]:
        self.load(split=split )
        data = self.get_inputs(split=split )
        gt = self.get_gt_outputs(split=split )
        is_multi_timeframe = isinstance(data, DS_Cameras) and data.times.shape[1] > 1 # 默认camera的time只有1个，如果一个camera有多个time，则视为多视角动态数据集，需要特殊处理

        if batch_size is not None:
            if is_multi_timeframe:
                yield from self._multi_timeframe_batch_iter(data, gt, batch_size, shuffle, infinite, time_window_size)
            else:
                yield from self._single_timeframe_batch_iter(data, gt, batch_size, shuffle, infinite, time_window_size)

    def _single_timeframe_batch_iter(self, data, gt, batch_size, shuffle, infinite, time_window_size):
        def get_indices():
            return torch.randperm(len(data), device=self._device) if shuffle else torch.arange(len(data), device=self._device)

        indices = get_indices()
        idx = 0
        while True:
            next_idx = idx + batch_size
            if next_idx > len(data):
                batch_indices = indices[idx:] # 如果超出数据长度，只取最后一部分作为一个不满的 batch
            else:
                batch_indices = indices[idx:next_idx]

            if time_window_size:
                batch_indices = self._expand_time_window(batch_indices, len(data), time_window_size)

            if gt is not None:
                if gt._device == 'cpu':
                    batch_indices = batch_indices.cpu()
                
            yield data[batch_indices], gt[batch_indices].to(self._device) if gt is not None else None, batch_indices

            idx = next_idx

            if idx >= len(data):
                if not infinite:
                    break
                # 重置索引与计数器
                indices = get_indices()
                idx = 0

    def _multi_timeframe_batch_iter(self, data, gt, batch_size, shuffle, infinite, time_window_size, agumentation_sample=False):
        camera_frame_num, time_frame_num = data.times.shape[:2]
        total = camera_frame_num * time_frame_num

        def get_indices():
            return torch.randperm(total, device=self._device) if shuffle else torch.arange(total, device=self._device)

        indices = get_indices()
        idx = 0
        while True:
            next_idx = idx + batch_size

            if next_idx > total:
                batch_indices = indices[idx:]
            else:
                batch_indices = indices[idx:next_idx]

            camera_frame_idx = batch_indices // time_frame_num
            time_frame_idx = batch_indices % time_frame_num

            if agumentation_sample:
                aug_time_frames = self._expand_agumentation_time_frames()
                if len(aug_time_frames) > 0:
                    # 默认增强帧来自所有相机（broadcast），也可以扩展为从随机相机采样
                    aug_camera_idx = torch.randint(0, camera_frame_num, size=(len(aug_time_frames),), device=self._device)

                    camera_frame_idx = torch.cat([camera_frame_idx, aug_camera_idx], dim=0)
                    time_frame_idx = torch.cat([time_frame_idx, aug_time_frames], dim=0)

            if time_window_size:
                time_frame_idx = self._expand_time_window(time_frame_idx, time_frame_num, time_window_size)
                camera_frame_idx = camera_frame_idx.repeat_interleave(time_window_size)

            sampled_data: DS_Cameras = data[camera_frame_idx]
            times = sampled_data.times[0, time_frame_idx].unsqueeze(-1)
            dts = sampled_data.dts[0, time_frame_idx].unsqueeze(-1)
            sampled_data = sampled_data.set_times(times, dts)

            if gt is not None:
                if gt._device == 'cpu':
                    batch_indices = batch_indices.cpu()

            yield sampled_data, gt[batch_indices].to(self._device) if gt is not None else None, time_frame_idx

            idx = next_idx
            if idx >= total:
                if not infinite:
                    break
                indices = get_indices()
                idx = 0

    def _expand_time_window(self, indices, max_len, window_size):
        expanded = []
        for i in indices:
            window = torch.arange(i, i + window_size, device=self._device)
            window = window[window < max_len]
            expanded.append(window)
        return torch.unique(torch.cat(expanded)).sort().values

    def _expand_agumentation_time_frames(self, num_frames: int = 4) -> torch.Tensor:
        if not self.agumentation_sample_frames:
            return torch.tensor([], dtype=torch.long, device=self._device)

        all_candidates = []
        for start, end in self.agumentation_sample_frames:
            all_candidates.extend(range(start, end + 1))

        if not all_candidates:
            return torch.tensor([], dtype=torch.long, device=self._device)

        candidates = torch.tensor(all_candidates, device=self._device)
        if len(candidates) <= num_frames:
            return candidates  # 不足4帧时全部返回
        else:
            indices = torch.randperm(len(candidates), device=self._device)[:num_frames]
            return candidates[indices]

    def get_train_iter(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        infinite: bool = True,
        frame_batch_size: Optional[int] = None,
        camera_batch_size: Optional[int] = None,
        time_window_size: Optional[int] = None,
    ) -> Generator[Tuple[DT, GT, IntArrayLike], None, None]:
        return self.get_iter('train', batch_size=batch_size, shuffle=shuffle, infinite=infinite, frame_batch_size=frame_batch_size, camera_batch_size=camera_batch_size, time_window_size=time_window_size)

    def get_val_iter(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        infinite: bool = True,
    ) -> Generator[Tuple[DT, GT, IntArrayLike], None, None]:
        return self.get_iter('val', batch_size=batch_size, shuffle=shuffle, infinite=infinite)

    def get_test_iter(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        infinite: bool = False,
    ) -> Generator[Tuple[DT, GT, IntArrayLike], None, None]:
        return self.get_iter('test', batch_size=batch_size, shuffle=shuffle, infinite=infinite)

    def get_orbit_vis_iter(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        infinite: bool = False,
    ) -> Generator[Tuple[DT, GT, IntArrayLike], None, None]:
        return self.get_iter('orbit_vis', batch_size=batch_size, shuffle=shuffle, infinite=infinite)

    def get_fix_vis_iter(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        infinite: bool = False,
    ) -> Generator[Tuple[DT, GT, IntArrayLike], None, None]:
        return self.get_iter('fix_vis', batch_size=batch_size, shuffle=shuffle, infinite=infinite)
    
    def dump(
        self,
        path: Path,
        *,
        exist_ok: bool = False,
        dataparser: Optional[Type[DS_BaseDataparser[DT, MT, GT]]] = None,
    ) -> None:
        path.mkdir(exist_ok=exist_ok, parents=True)
        if dataparser is None:
            dataparser = self.dataparser.__class__
        for split in ['train', 'test', 'val']:
            dataparser.dump(
                inputs=self.get_inputs(split=split),
                gt_outputs=self.get_gt_outputs(split=split),
                meta=self.get_meta(split=split),
                path=path,
                split=split,
            )

    def export_dataset_attributes(self) -> Dict[str, Any]:
        train_data = self.get_inputs(split='train')
        is_multi_timeframe = isinstance(train_data, DS_Cameras) and train_data.times.shape[1] > 1
        
        if is_multi_timeframe:
            time_frame_num = train_data.times.shape[1]
        else:
            time_frame_num = len(train_data)

        return { 'time_resolution': time_frame_num}

@dataclass
class SyntheticDynamicMonocularBlenderDepthDataset(DS_BaseDataset[DS_Cameras, DepthImages, List[DS_TriangleMesh]]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[DS_BaseDataparser[Any, Any, Any]]]:
        return [
            SyntheticDynamicMonocularBlenderDepthDataparser,
        ]


@dataclass
class SyntheticDynamicMonocularCostumeDepthDataset(DS_BaseDataset[DS_Cameras, DepthImages, List[DS_TriangleMesh]]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[DS_BaseDataparser[Any, Any, Any]]]:
        return [
            SyntheticDynamicMonocularCostumeDepthDataparser,
        ]


@dataclass
class SyntheticDynamicMultiViewBlenderDepthDataset(DS_BaseDataset[DS_Cameras, RGBImages, Any]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[DS_BaseDataparser[DS_Cameras, RGBImages, Any]]]:
        return [
            SyntheticDynamicMultiViewBlenderDepthDataparser, # parse camera, image, none
        ]



@dataclass
class SyntheticDynamicMultiViewCostumeDepthDataset(DS_BaseDataset[DS_Cameras, RGBImages, Any]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[DS_BaseDataparser[DS_Cameras, RGBImages, Any]]]:
        return [
            SyntheticDynamicMultiViewCostumeDepthDataparser, # parse camera, image, none
        ]




@dataclass
class SyntheticDynamicMonocularBlenderRGBDataset(DS_BaseDataset[DS_Cameras, RGBImages, Any]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[DS_BaseDataparser[DS_Cameras, RGBImages, Any]]]:
        return [
            SyntheticDynamicMonocularBlenderRGBDataparser, # parse camera, image, none (no mask, no depth, no mesh), usually real world dataset
        ]


@dataclass
class SyntheticDynamicMonocularBlenderRGBADataset(DS_BaseDataset[DS_Cameras, RGBImages, Any]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[DS_BaseDataparser[DS_Cameras, RGBImages, Any]]]:
        return [
            DNerfSyntheticDynamicMonocularBlenderRGBADataparser, # parse camera, image(with alpha), (mesh, depth). for example: dg-mesh dataset
            DGMeshSyntheticDynamicMonocularBlenderRGBADataparser,
            ObjSelSyntheticDynamicMonocularBlenderRGBADataparser,
        ]


@dataclass
class SyntheticDynamicMultiViewBlenderRGBDataset(DS_BaseDataset[DS_Cameras, RGBImages, Any]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[DS_BaseDataparser[DS_Cameras, RGBImages, Any]]]:
        return [
            SyntheticDynamicMultiviewBlenderRGBDataparser,
        ]


@dataclass
class SyntheticDynamicMultiViewBlenderRGBADataset(DS_BaseDataset[DS_Cameras, RGBImages, Any]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[DS_BaseDataparser[DS_Cameras, RGBImages, Any]]]:
        return [
            SyntheticDynamicMultiviewBlenderRGBADataparser,
        ]

@dataclass
class SyntheticTemporalDynamicMultiviewBlenderRGBADataset(DS_BaseDataset[DS_Cameras, RGBImages, Any]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[DS_BaseDataparser[DS_Cameras, RGBImages, Any]]]:
        return [
            SyntheticTemporalDynamicMultiviewBlenderRGBADataparser,
        ]

@dataclass
class RealDynamicMultiviewObjectRGBADataset(DS_BaseDataset[DS_Cameras, RGBImages, Any]):
    
    @classmethod
    def get_dataparser_list(cls) -> List[Type[DS_BaseDataparser[DS_Cameras, RGBImages, Any]]]:
        return [
            RealDynamicMultiviewObjectRGBADataparser,
        ]

@dataclass
class CMUPanonicRGBADataset(DS_BaseDataset[DS_Cameras, RGBImages, Any]):
    
    @classmethod
    def get_dataparser_list(cls) -> List[Type[DS_BaseDataparser[DS_Cameras, RGBImages, Any]]]:
        return [
            CMUPanonicRGBADataparser,
            SDFFlowRGBADataparser,
            ZJUMOCAPRGBADataparser
        ]
