from __future__ import annotations

# import modules
import gc
from itertools import chain
from pathlib import Path
from dataclasses import dataclass
from typing import TypeVar, Type, Optional

import torch

# import rfstudio modules
from rfstudio.engine.task import Task
from rfstudio.nn import Module
from rfstudio.ui import console
from rfstudio.utils.pretty import P
from rfstudio.visualization._optimization_visualizer import OptimizationVisualizer

# import rfstudio_ds modules
from rfstudio_ds.engine.experiment import DS_Experiment
from rfstudio_ds.trainer import DS_BaseTrainer
from rfstudio_ds.data import DS_BaseDataset
from rfstudio_ds.graphics import DS_Cameras


T = TypeVar('T', bound='DS_TrainTask')


@dataclass
class DS_TrainTask(Task):
    '''
    TODO
    '''

    dataset: DS_BaseDataset = ...

    experiment: DS_Experiment = ...

    model: Module = ...

    trainer: DS_BaseTrainer = ...

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

        self.save_as_script(self.script_path)

        """加载数据"""
        with console.status(f'Loading dataset from {self.dataset.path} ...'):
            # Training data iterator
            train_data_iterator = self.dataset.get_train_iter(
                batch_size=self.trainer.batch_size,
                frame_batch_size=self.trainer.frame_batch_size, 
                camera_batch_size=self.trainer.camera_batch_size, 
                time_window_size=self.trainer.time_window_size
            )

            # Validation-related iterators
            if self.trainer.num_steps_per_val:
                # For metric computation
                validation_data_iterator = self.dataset.get_val_iter(batch_size=1)

                # For visualization 
                if self.trainer.num_steps_per_orbit_vis:
                    validation_vis_batch_size = self.dataset.get_size(split='orbit_vis')
                    validation_orbit_vis_iterator = self.dataset.get_orbit_vis_iter(
                        batch_size=validation_vis_batch_size,
                        shuffle=False,
                        infinite=True,
                    )
                if self.trainer.num_steps_per_fix_vis:
                    validation_vis_batch_size = self.dataset.get_size(split='fix_vis')
                    validation_fix_vis_iterator = self.dataset.get_fix_vis_iter(
                        batch_size=validation_vis_batch_size,
                        shuffle=False,
                        infinite=True,
                    )

            # Test-related iterators
            if self.trainer.full_test_after_train:
                test_size = self.dataset.get_size(split='test')
                split_size = test_size // 25
                if split_size == 0:
                    split_size = 1
                test_batch_size = int(self.dataset.get_size(split='test') / split_size) # test camera is 6multi-view, downsample size is 4 * 6
                # For test metrics
                test_data_iterator = self.dataset.get_test_iter(
                    batch_size=test_batch_size,
                    shuffle=False,
                    infinite=False
                )

            # For visualization 
            if self.trainer.full_orbit_vis_after_train:
                test_vis_batch_size = int(self.dataset.get_size(split='orbit_vis') / 4) # oribit vis camera is monocular, downsample size is 4
                test_orbit_vis_iterator = self.dataset.get_orbit_vis_iter(
                    batch_size=test_vis_batch_size,
                    shuffle=False,
                    infinite=False,
                )
            if self.trainer.full_fix_vis_after_train:
                test_vis_batch_size = int(self.dataset.get_size(split='fix_vis') / 4) # fix vis camera is monocular, downsample size is 4
                test_fix_vis_iterator = self.dataset.get_fix_vis_iter(
                    batch_size=test_vis_batch_size,
                    shuffle=False,
                    infinite=False,
                )
                
            if self.trainer.num_steps_per_analyze_cube_curve is not None:
                analyse_curve_data_iterator = self.dataset.get_test_iter(
                    batch_size=int(self.dataset.get_size(split='test')),
                    shuffle=False,
                    infinite=True,
                )
        
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
                    inputs, gt_outputs, indices = next(train_data_iterator)
                    train_loss, train_metrics, _, _ = self.trainer.step(
                        self.model,
                        inputs,
                        gt_outputs,
                        indices=indices,
                        mode='train',
                        visual=False,
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
                main_train_metrics = dict(list(train_metrics.items())[-3:]) # 取最后3个指标，用于控制台可视化
                handle.table['train-metrics'].update(**main_train_metrics)

                """after train"""
                self.trainer.after_backward(self.model)
                # # 检查关键参数的梯度 L2 范数
                # def grad_norm(param):
                #     return param.grad.norm().item() if param.grad is not None else 0.0
                # print("Gradients:")
                # print(f"  static_sdf_params:        {grad_norm(self.model.static_sdf_params):.2e}")
                # print(f"  sdf_curve_poly:           {grad_norm(self.model.sdf_curve_poly_coefficient):.2e}")
                # print(f"  low_freq_fourier:         {grad_norm(self.model.sdf_curve_low_freq_fourier_coefficient):.2e}")
                # print(f"  mid_freq_fourier:         {grad_norm(self.model.sdf_curve_mid_freq_fourier_coefficient):.2e}")
                # print(f"  high_freq_fourier:        {grad_norm(self.model.sdf_curve_high_freq_fourier_coefficient):.2e}")
                # if step == 1:
                #     initial_params = {
                #         'static_sdf': self.model.static_sdf_params.data.clone().detach(),
                #         'poly': self.model.sdf_curve_poly_coefficient.data.clone().detach(),
                #         'low': self.model.sdf_curve_low_freq_fourier_coefficient.data.clone().detach(),
                #         'mid': self.model.sdf_curve_mid_freq_fourier_coefficient.data.clone().detach(),
                #         'high': self.model.sdf_curve_high_freq_fourier_coefficient.data.clone().detach(),
                #     }
                # def param_change(initial, current):
                #     return (initial - current).norm().item()
                # current_change = param_change(initial_params['static_sdf'], self.model.static_sdf_params.data)
                # print(f"static_sdf_params 已更新幅度: {current_change:.2e}")
                # diff = torch.abs(self.model.static_sdf_params.data - initial_params['static_sdf'])
                # max_diff = diff.max().item()
                # print("最大差异：", max_diff)
                optimizers.step(curr_step=step)
                self.trainer.after_update(self.model, optimizers=optimizers, curr_step=step)
                if self.trainer.num_steps_per_save and step % self.trainer.num_steps_per_save == 0:
                    self.save_checkpoint(step)

                """val step"""
                if self.trainer.num_steps_per_val and step % self.trainer.num_steps_per_val == 0:
                    self.model.train(False)
                    val_pbr_attr = self.trainer.num_steps_per_val_pbr_attr and step % self.trainer.num_steps_per_val_pbr_attr == 0
                    with torch.no_grad():
                        inputs, gt_outputs, indices = next(validation_data_iterator)
                        _, val_metrics, visualization, _ = self.trainer.step(
                            self.model,
                            inputs,
                            gt_outputs,
                            indices=indices,
                            mode='val',
                            visual=True,
                            val_pbr_attr=val_pbr_attr,
                        )
                        self.experiment.dump_image('val', index=step, image=visualization[0])
                        self.experiment.log(P@'Step {step} Val Metrics: {val_metrics}')
                        main_val_metrics = dict(list(val_metrics.items())[-3:])
                        handle.table['val-metrics'].update(**main_val_metrics)

                """val vis step"""
                if self.trainer.num_steps_per_orbit_vis and step % self.trainer.num_steps_per_orbit_vis == 0:
                    self.model.train(False)
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        inputs, gt_outputs, indices = next(validation_orbit_vis_iterator)
                        _, _, visualization, _ = self.trainer.step(
                            self.model,
                            inputs,
                            gt_outputs,
                            indices=indices,
                            mode='orbit_vis',
                            visual=True,
                            vis_downsample_factor=4,
                            val_pbr_attr=True if self.model.shader_type=="split_sum_pbr" else False,
                        )
                        self.experiment.dump_images2video('val_vis', name='orbit_camera', index=step, images=visualization, downsample=1, target_mb=0.5, duration=2.5)
                    torch.cuda.empty_cache()
                if self.trainer.num_steps_per_fix_vis and step % self.trainer.num_steps_per_fix_vis == 0:
                    self.model.train(False)
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        inputs, gt_outputs, indices = next(validation_fix_vis_iterator)
                        _, _, visualization, _ = self.trainer.step(
                            self.model,
                            inputs,
                            gt_outputs,
                            indices=indices,
                            mode='fix_vis',
                            visual=True,
                            vis_downsample_factor=4,
                            val_pbr_attr=True if self.model.shader_type=="split_sum_pbr" else False,
                        )
                        self.experiment.dump_images2video('val_vis', name='fix_camera', index=step, images=visualization, downsample=1, target_mb=0.5,duration=2.5)
                    torch.cuda.empty_cache()

                """val cube curve step"""
                if self.trainer.num_steps_per_analyze_cube_curve is not None and step % self.trainer.num_steps_per_analyze_cube_curve == 0:
                    self.model.train(False)
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        inputs, gt_outputs, indices = next(analyse_curve_data_iterator)
                        _, _, _, _ = self.trainer.step(
                            self.model,
                            inputs,
                            gt_outputs,
                            indices=indices,
                            mode='analyse_curve',
                            visual=True,
                            analyse_curve_save_path = self.experiment.dump_file_path(subfolder=f'eval_fix_points_cubes_curve/step_{step}'),
                        )
                    torch.cuda.empty_cache()

                handle.progress['training'].update(curr=step, total=self.trainer.num_steps)
                handle.sync()
                
                if step in [10, 100, 500, 1000]:
                    torch.cuda.empty_cache()
        
            """finish train"""
            del inputs, gt_outputs, indices, train_data_iterator
            if self.trainer.num_steps_per_val:
                del visualization, validation_data_iterator
                if self.trainer.num_steps_per_orbit_vis:
                    del validation_orbit_vis_iterator
                if self.trainer.num_steps_per_fix_vis:
                    del validation_fix_vis_iterator
            if self.trainer.num_steps_per_analyze_cube_curve is not None:
                del analyse_curve_data_iterator
            torch.cuda.empty_cache()
            gc.collect()
            self.save_checkpoint(step)
            handle.sync(force=True)
            if self.trainer.after_train:
                self.trainer.after_train(self.model, self.dataset)

            """test step compute metrics and visualization"""
            if self.trainer.full_test_after_train:
                self.model.train(False)
                #  compute test metrics step
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    test_metrics = {}
                    count = 0
                    for inputs, gt_outputs, indices in test_data_iterator:
                        _, metrics, visualization, _ = self.trainer.step(
                            self.model,
                            inputs,
                            gt_outputs,
                            indices=indices,
                            mode='test',
                            visual=True,
                            val_pbr_attr=True if self.model.shader_type=="split_sum_pbr" else False,
                        )
                        for i, image in enumerate(visualization):
                            self.experiment.dump_image('test', index=i+count*test_batch_size, image=image)
                        for key, val in metrics.items():
                            test_metrics.setdefault(key, []).append(val)
                        count += 1
                    test_metrics = { key: sum(val) / len(val) for key, val in test_metrics.items() }
                    self.experiment.log(P@'Test Metrics: {test_metrics}')
                    del inputs, gt_outputs, indices, visualization, test_data_iterator
                    torch.cuda.empty_cache()

            if self.trainer.full_orbit_vis_after_train:
                self.model.train(False)
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    orbit_vis = []  
                    for inputs, gt_outputs, indices in test_orbit_vis_iterator:
                        _, _, visualization, _ = self.trainer.step(
                            self.model,
                            inputs,
                            gt_outputs,
                            indices=indices,
                            mode='orbit_vis',
                            visual=True,
                            val_pbr_attr=True if self.model.shader_type=="split_sum_pbr" else False,
                        )
                        orbit_vis.append(visualization)
                    self.experiment.dump_images2video('test_vis', name='orbit_camera', images=list(chain(*orbit_vis)), downsample=1, fps = 48,duration=5)
                    del inputs, gt_outputs, indices, visualization, test_orbit_vis_iterator, orbit_vis
                    torch.cuda.empty_cache()
            if self.trainer.full_fix_vis_after_train:
                self.model.train(False)
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    fix_vis = []  
                    for inputs, gt_outputs, indices in test_fix_vis_iterator:
                        _, _, visualization, _ = self.trainer.step(
                            self.model,
                            inputs,
                            gt_outputs,
                            indices=indices,
                            mode='fix_vis',
                            visual=True,
                            val_pbr_attr=True if self.model.shader_type=="split_sum_pbr" else False,
                        )
                        fix_vis.append(visualization)
                    self.experiment.dump_images2video('test_vis', name='fix_camera', images=list(chain(*fix_vis)), downsample=1,fps=48,duration=5)
                    del inputs, gt_outputs, indices, visualization, test_fix_vis_iterator, fix_vis
                    torch.cuda.empty_cache()

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
    def load_from_script(cls: Type[T], script_path: Path, *, step: Optional[int] = None, load_checkpoint: bool = True) -> T:
        train_task = super().load_from_script(script_path)
        if load_checkpoint:
            assert train_task.load_checkpoint(step=step) is not None
        return train_task
