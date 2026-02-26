from __future__ import annotations

# import modules
import pathlib
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Literal, Tuple, Generic

import torch

# import rfstudio modules
from rfstudio.utils.typing import Indexable
from rfstudio.data.dataparser.base_dataparser import DT, GT, MT


@dataclass
class DSDF_BaseDataparser(ABC, Generic[DT, GT, MT]):
    '''
    Inherit from BaseDataparser
    '''

    @abstractmethod
    def parse(
        self,
        path: pathlib.Path,
        *,
        split: Literal['train', 'test', 'all'],
        parse_meshes: bool = False,
        device: torch.device,
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
        split: Literal['train', 'test', 'all'],
    ) -> None:
        raise NotImplementedError
