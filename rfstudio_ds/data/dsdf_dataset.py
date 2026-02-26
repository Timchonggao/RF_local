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

# import rfstudio_ds modules
from .dataparser import (
    DSDF_BaseDataparser,
    DynamicMeshSDFDataparser
)


@dataclass
class DSDF_BaseDataset(ABC, Generic[DT, GT, MT]):
    """Base class for managing datasets with train/test/all splits.

    Attributes:
        path: Path to the dataset directory or file.
        dataparser: Optional parser for loading and processing dataset. If None, automatically selected.
    """

    path: Path
    """
    path to data
    """

    dataparser: Optional[DSDF_BaseDataparser[DT, GT, MT]] = None


    @classmethod
    @abstractmethod
    def get_dataparser_list(cls) -> List[Type[DSDF_BaseDataparser[DT, GT, MT]]]:
        ...

    def __setup__(self) -> None:
        if self.dataparser is None:
            for dataparser_class in self.__class__.get_dataparser_list():
                if dataparser_class.recognize(self.path):
                    self.dataparser = dataparser_class()
                    break
            else:
                raise RuntimeError("Cannot decide dataparser automatically.")
        self._pairs: Dict[Literal['train', 'test', 'all'], Tuple[Indexable[DT], Indexable[GT], MT]] # 字典，用于存储不同划分（训练集、测试集、验证集）的数据及其对应的元数据; 键为划分名称，值为元组（数据、标签、元数据）

        self._pairs = {}
        self._device = None

    def to(self, device: torch.device) -> None:
        self._device = device
    
    def load(self, *, split: Literal['train', 'test', 'all']) -> None:
        if split not in self._pairs:
            assert self.dataparser is not None
            self._pairs[split] = self.dataparser.parse(self.path, split=split, device=self._device)
            
    def get_inputs(self, *, split: Literal['train', 'test', 'all']) -> Indexable[DT]:
        self.load(split=split)
        dt = self._pairs[split][0]
        return dt
    
    def get_gt_outputs(self, *, split: Literal['train', 'test', 'all']) -> Indexable[GT]:
        self.load(split=split)
        gt = self._pairs[split][1]
        return gt

    def get_meta(self, *, split: Literal['train', 'test', 'all']) -> MT:
        self.load(split=split)
        return self._pairs[split][2]

    def get_size(self, *, split: Literal['train', 'test', 'all']) -> int:
        return len(self.get_inputs(split=split))

    def get_iter(
        self,
        split: Literal['train', 'test', 'all'],
        *,
        batch_size: int,
        shuffle: bool,
        infinite: bool,
    ) -> Generator[Tuple[DT, GT, IntArrayLike], None, None]:
        """Generate batches of data, ground truth, and indices for a given split.

        Args:
            split: Dataset split to iterate over ('train', 'test', or 'all').
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the data indices.
            infinite: Whether to loop indefinitely.

        Yields:
            Tuple containing (data, ground truth, indices) for each batch.

        Raises:
            ValueError: If batch_size is invalid or split is not loaded.
        """
        
        self.load(split=split)
        data = self.get_inputs(split=split)
        gt = self.get_gt_outputs(split=split)
        
        def get_indices():
            return torch.randperm(len(data), device=self._device) if shuffle else torch.arange(len(data), device=self._device)

        indices = get_indices()
        idx = 0
        while True:
            next_idx = idx + batch_size
            if next_idx > len(data):
                # 如果超出数据长度，只取最后一部分作为一个不满的 batch
                sampled_indices = indices[idx:]
            else:
                sampled_indices = indices[idx:next_idx]

            yield data[sampled_indices], gt[sampled_indices], sampled_indices
            
            idx = next_idx

            if idx >= len(data):
                if not infinite:
                    break
                # 重置索引与计数器
                indices = get_indices()
                idx = 0

    def get_train_iter(
        self,
        batch_size: int,
        *,
        shuffle: bool = True,
        infinite: bool = True,
    ) -> Generator[Tuple[DT, GT, IntArrayLike], None, None]:
        return self.get_iter('train', batch_size=batch_size, shuffle=shuffle, infinite=infinite)

    def get_test_iter(
        self,
        *,
        shuffle: bool = False,
        infinite: bool = False,
    ) -> Generator[Tuple[DT, GT, IntArrayLike], None, None]:
        return self.get_iter('test', batch_size=self.get_size(split='test'), shuffle=shuffle, infinite=infinite)

    def get_all_iter(
        self,
        *,
        shuffle: bool = False,
        infinite: bool = True,
    ) -> Generator[Tuple[DT, GT, IntArrayLike], None, None]:
        return self.get_iter('all', batch_size=self.get_size(split='all'), shuffle=shuffle, infinite=infinite)


@dataclass
class DynamicSDFDataset(DSDF_BaseDataset[Any, Any, Any]):

    @classmethod
    def get_dataparser_list(cls) -> List[Type[DSDF_BaseDataparser[Any, Any, Any]]]:
        return [
            DynamicMeshSDFDataparser,
        ]
