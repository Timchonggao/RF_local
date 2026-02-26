from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Dict, Any

from torch import Tensor

from rfstudio.nn import Module
from rfstudio_ds.data import DSDF_BaseDataset
from rfstudio_ds.optim import DS_ModuleOptimizers
from rfstudio_ds.engine.experiment import DS_Experiment


@dataclass
class DSDF_BaseTrainer(ABC):
    """Abstract base class for training dynamic SDF models.

    Defines the interface for training, validation, and testing, with configurable parameters
    for batch size, steps, and visualization. Subclasses must implement `setup` and `step`.

    Attributes:
        batch_size: Number of samples per batch.
        num_steps: Total training steps.
        num_steps_per_vis: Steps between visualizations (None to disable).
        ...
    """

    batch_size: int = ...

    num_steps: int = ...

    num_steps_per_vis: Optional[int] = None

    num_steps_per_val: Optional[int] = None

    num_steps_per_orbit_vis: Optional[int] = None

    num_steps_per_save: Optional[int] = None

    full_test_after_train: bool = True

    full_fix_vis_after_train: bool = False

    hold_after_train: bool = True

    mixed_precision: bool = False

    detect_anomaly: bool = False

    num_accums_per_batch: int = 1

    @abstractmethod
    def setup(self, model: Module, dataset: DSDF_BaseDataset[Any, Any, Any]) -> DS_ModuleOptimizers:
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
        visual: Literal['pred', 'gt', 'all', 'none'] = 'none',
        experiment: Optional[DS_Experiment] = None,
        curr_step: Optional[int] = None,
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

    def after_train(self, model: Module, dataset: DSDF_BaseDataset[Any, Any, Any]) -> None:
        pass
