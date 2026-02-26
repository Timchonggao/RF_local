from __future__ import annotations

# import modules
import pathlib
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Literal, Tuple, Generic, Optional, List

import torch

# import rfstudio modules
from rfstudio.utils.typing import Indexable
from rfstudio.data.dataparser.base_dataparser import DT, GT, MT


@dataclass
class DS_BaseDataparser(ABC, Generic[DT, GT, MT]):
    '''
    TODO
    '''

    @abstractmethod
    def parse(
        self,
        path: pathlib.Path,
        *,
        split: Literal['train', 'test', 'val', 'orbit_vis', 'fix_vis'],
        parse_meshes: bool = False,
        costume_padding_size: Optional[int] = None,
        costume_sample_frames: Optional[List[int]] = None,
        device: torch.device,
        mode: Literal['uniform', 'random'] = 'uniform',
        test_ratio: Optional[float] = None
    ) -> Tuple[Indexable[DT], Indexable[GT], MT]:
        ...
    
    @staticmethod
    @abstractmethod
    def recognize(path: pathlib.Path) -> bool:
        ...

    @staticmethod
    def dump(
        inputs: DT,
        gt_outputs: GT,
        meta: MT,
        *,
        path: pathlib.Path,
        split: Literal['train', 'test', 'val', 'fix_vis'],
    ) -> None:
        raise NotImplementedError
