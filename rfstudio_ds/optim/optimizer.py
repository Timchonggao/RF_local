from __future__ import annotations

# import modules
import torch
from typing import Dict, Optional

# import rfstudio modules
from rfstudio.optim.optimizer import Optimizer

# import project inherite class
from rfstudio.optim.optimizer import ModuleOptimizers


class DS_ModuleOptimizers(ModuleOptimizers):
    ''''
    Inherit from ModuleOptimizers:
        1.rewrite __init__
        2.change step logic to support dynamic start learning rate decay.
    '''

    def __init__(
        self,
        mixed_precision: bool,
        optim_dict: Dict[str, Optimizer],
        start_steps: Optional[Dict[str, Optional[int]]] = None,
    ) -> None:
        super().__init__(mixed_precision=mixed_precision, optim_dict=optim_dict)
        self.start_steps = start_steps

    def step(self, curr_step: int) -> None:
        assert self.optimizers is not None
        for key, optimizer in self.optimizers.items():
            max_norm = self.optim_dict[key].max_norm
            if max_norm is not None:
                self.grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters[key], max_norm, error_if_nonfinite=True)
            self.grad_scaler.step(optimizer)
        self.grad_scaler.update()
        if self.start_steps is None:
            for scheduler in self.schedulers.values():
                    scheduler.step()
        else:
            for key, scheduler in self.schedulers.items():
                if self.start_steps[key] is not None:
                    if self.start_steps[key] >= 0 and curr_step >= self.start_steps[key]:
                        scheduler.step()
                else:
                    scheduler.step()
