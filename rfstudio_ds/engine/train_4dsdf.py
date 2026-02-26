from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, Type, Optional
import gc

import torch

from rfstudio.engine.task import Task
from rfstudio.nn import Module
from rfstudio.ui import console
from rfstudio.utils.pretty import P

from rfstudio_ds.engine.experiment import DS_Experiment
from rfstudio_ds.trainer import DSDF_BaseTrainer
from rfstudio_ds.data import DSDF_BaseDataset


T = TypeVar('T', bound='DSDF_TrainTask')


@dataclass
class DSDF_TrainTask(Task):
    """
    执行动态 SDF 模型的训练、验证和测试任务，支持检查点保存和可视化。
    """

    dataset: DSDF_BaseDataset = ...

    experiment: DS_Experiment = ...

    model: Module = ...

    trainer: DSDF_BaseTrainer = ...


    def __setup__(self) -> None:
        super().__setup__()
        self.model.to(self.device)
        self.dataset.to(self.device)

    @property
    def ckpt_path(self) -> Path:
        return self.experiment.base_path / "ckpts"

    @property
    def script_path(self) -> Path:
        return self.experiment.base_path / "task.py"

    def run(self) -> None:
        """Execute the training, validation, and testing pipeline.

        Loads data, trains the model, performs periodic validation, runs tests if enabled,
        and logs metrics and visualizations.
        """

        assert self.trainer.batch_size % self.trainer.num_accums_per_batch == 0
        self.save_as_script(self.script_path)

        """加载数据"""
        with console.status(P@'Loading dataset from {self.dataset.path} ...'):
            loader_train = self.dataset.get_train_iter(batch_size=self.trainer.batch_size, shuffle=True, infinite=True)
            if self.trainer.num_steps_per_val:
                loader_vis = self.dataset.get_all_iter(infinite=True)
                loader_val = self.dataset.get_test_iter(infinite=True)                
        
        with console.screen() as handle:

            """设置控制台布局，包括损失图和指标表。"""
            handle.set_layout(
                handle.cols[3, 1](
                    handle.plot['train-loss'],
                    (
                        handle.rows(handle.table['train-metrics'], handle.table['val-metrics'])
                        if self.trainer.num_steps_per_val
                        else handle.table['train-metrics']
                    )
                )
            )
            handle.progress['training'].update(curr=0, total=self.trainer.num_steps)
            handle.sync()

            optimizers = self.trainer.setup(self.model, self.dataset)
            if self.trainer.detect_anomaly:
                torch.set_anomaly_enabled(True)

            """Training Loop"""
            for step in range(1, self.trainer.num_steps + 1):
            
                """before train"""
                optimizers.zero_grad()
                self.model.train(True)
                self.trainer.before_update(self.model, optimizers=optimizers, curr_step=step)

                """train step"""
                with torch.autocast(device_type=self.device_type, enabled=self.trainer.mixed_precision):
                    inputs, gt_outputs, indices = next(loader_train)
                    train_loss, train_metrics, _ = self.trainer.step(
                        self.model,
                        inputs,
                        gt_outputs,
                        indices=indices,
                        mode='train',
                        visual='none',
                    )
                    self.experiment.log(P@'Step {step} Train Metrics: {train_metrics} ')
                    if self.trainer.detect_anomaly:
                        assert train_loss.isfinite()
                optimizers.backward(train_loss)

                """控制台指标更新"""
                handle.plot['train-loss'].update(
                    x=step,
                    y=train_loss.item(),
                )
                main_train_metrics = dict(list(train_metrics.items())[:3]) # 取前3个指标，用于控制台可视化
                handle.table['train-metrics'].update(**main_train_metrics)

                """after train"""
                self.trainer.after_backward(self.model)
                optimizers.step(curr_step=step)
                self.trainer.after_update(self.model, optimizers=optimizers, curr_step=step)
                if self.trainer.num_steps_per_save and step % self.trainer.num_steps_per_save == 0:
                    self.save_checkpoint(step)

                """val step"""
                if self.trainer.num_steps_per_val and step % self.trainer.num_steps_per_val == 0:
                    self.model.train(False)
                    
                    with torch.no_grad():
                        # load all for visualization
                        torch.cuda.empty_cache()
                        inputs, gt_outputs, indices = next(loader_vis)
                        self.trainer.step(
                            self.model,
                            inputs=inputs,
                            gt_outputs=gt_outputs,
                            indices=None,
                            mode='val',
                            visual='all',
                            experiment=self.experiment,
                            curr_step=step,
                        )                       

                    with torch.no_grad():
                        torch.cuda.empty_cache()
                        # load for val metrics
                        inputs, gt_outputs, indices = next(loader_val)
                        _, val_metrics, _ = self.trainer.step(
                            self.model,
                            inputs,
                            gt_outputs,
                            indices=indices,
                            mode='test',
                            visual='none',
                        )
                        self.experiment.log(P@'Step {step} Val Metrics: {val_metrics} ')
                        # 控制台指标更新
                        main_val_metrics = dict(list(val_metrics.items())[:3])
                        handle.table['val-metrics'].update(**main_val_metrics)
                        torch.cuda.empty_cache()

                """控制台输出更新"""
                handle.progress['training'].update(curr=step, total=self.trainer.num_steps)
                handle.sync()

            """finish train"""
            self.save_checkpoint(step) # 训练完成后，保存最后的检查点
            handle.sync(force=True) # 刷新控制台输出

            if self.trainer.after_train:
                self.trainer.after_train(self.model, self.dataset)

            "绘制trian loss曲线"
            self.experiment.parse_log_auto(self.experiment.log_path)

            if self.trainer.hold_after_train:
                handle.hold('Finished. Ctrl+C to exit.')

    def save_checkpoint(self, step: int) -> None:
        self.ckpt_path.mkdir(parents=True, exist_ok=True)
        ckpt_file_path = self.ckpt_path / f'{step:010d}.ckpt'
        torch.save(self.model.state_dict(), ckpt_file_path)

    def load_checkpoint(self, *, step: Optional[int] = None) -> Optional[int]:
        if not self.ckpt_path.exists():
            return
        if step is None:
            for filename in self.ckpt_path.glob('*.ckpt'):
                step_from_filename = int(filename.stem.rsplit('.', 1)[0])
                step = step_from_filename if step is None else max(step, step_from_filename)
            if step is None:
                return
        ckpt_file_path = self.ckpt_path / f'{step:010d}.ckpt'
        if not ckpt_file_path.exists():
            return
        self.model.load_state_dict(torch.load(ckpt_file_path, map_location='cpu'))
        return step

    @classmethod
    @torch.no_grad()
    def load_from_script(cls: Type[T], script_path: Path, *, step: Optional[int] = None) -> T:
        train_task = super().load_from_script(script_path)
        assert train_task.load_checkpoint(step=step) is not None
        return train_task
