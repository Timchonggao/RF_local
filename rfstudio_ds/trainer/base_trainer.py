from __future__ import annotations

# import modules
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Literal

from torch import Tensor

# import rfstudio modules
from rfstudio.nn import Module

# import rfstudio_ds modules
from rfstudio_ds.data import DS_BaseDataset
from rfstudio_ds.optim import DS_ModuleOptimizers


@dataclass
class DS_BaseTrainer(ABC):
    '''
    Inherit from rfstudio BaseTrainer
        1. rewrite the abstract 'step' function
        2. rewrite some abstract function's defination, like setup's dataset type(DS_BaseDataset)
    '''

    num_steps: int = ...

    batch_size: Optional[int] = None

    frame_batch_size: Optional[int] = None

    camera_batch_size: Optional[int] = None

    time_window_size: Optional[int] = None

    num_steps_per_vis: Optional[int] = None

    num_steps_per_val: Optional[int] = None

    num_steps_per_val_pbr_attr: Optional[int] = None

    num_steps_per_fix_vis: Optional[int] = None

    num_steps_per_orbit_vis: Optional[int] = None

    num_steps_per_analyze_cube_curve: Optional[int] = None

    num_steps_per_save: Optional[int] = None

    full_test_after_train: bool = True

    full_fix_vis_after_train: bool = False

    full_orbit_vis_after_train: bool = False

    hold_after_train: bool = True

    mixed_precision: bool = False

    detect_anomaly: bool = False

    num_accums_per_batch: int = 1

    @abstractmethod
    def setup(self, model: Module, dataset: DS_BaseDataset[Any, Any, Any]) -> DS_ModuleOptimizers:
        ...

    @abstractmethod
    def step(
        self,
        model: Module,
        inputs: Any,
        gt_outputs: Any,
        *,
        indices: Optional[Tensor],
        mode: Literal['train', 'val', 'test'] = 'train',
        visual: bool = False,

        val_pbr_attr: bool = False,
        vis_downsample_factor: Optional[int] = None,
        analyse_curve_save_path: Optional[str] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor], Optional[Tensor]]:
        ...

    def before_update(
        self,
        model: Module,
        optimizers: DS_ModuleOptimizers,
        *,
        curr_step: int,
    ) -> None:
        pass

    def after_update(
        self,
        model: Module,
        optimizers: DS_ModuleOptimizers,
        *,
        curr_step: int,
    ) -> None:
        pass

    def after_backward(self, model: Module) -> None:
        pass

    def after_train(self, model: Module, dataset: DS_BaseDataset[Any, Any, Any]) -> None:
        pass
