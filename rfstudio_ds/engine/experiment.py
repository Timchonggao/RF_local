from __future__ import annotations

# import modules
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, List, Union, Tuple
import re
import matplotlib.pyplot as plt
import pathlib
from collections import defaultdict

# import rfstudio modules
from rfstudio.io import open_video_renderer, dump_float32_image

# import rfstudio classes to inherite
from rfstudio.engine.experiment import Experiment


@dataclass
class DS_Experiment(Experiment):

    """
    Inherits from Experiment
        1. adds methods for dumping images to video.
    """

    def dump_file_path(self, subfolder: str, *, file_name: Optional[str] = None, mkdir: bool = True) -> None:
        """
        Return the path to the dumped file.

        Parameters
        ----------
        subfolder : str
            Subfolder name under :attr:`dump_path`.
        index : int
            Index used for generating filename: ``{index:04d}.png``.
        file : str or Path
            File path to be dumped.
        mkdir : bool
            If True, create the directory if it does not exist.

        """
        path = self.dump_path / subfolder
        if mkdir:
            path.mkdir(exist_ok=True, parents=True)
        assert path.exists()
        file_path = path / file_name if file_name else path
        return file_path

    def dump_image(self, subfolder: str, *, image: Optional[Tensor] = None, mkdir: bool = True, index: Optional[int] = None, name: Optional[str] = None,) -> None:
        """
        Dump image to a file.

        Parameters
        ----------
        subfolder : str
            Subfolder name under :attr:`dump_path`.
        index : int
            Index used for generating filename: ``{index:04d}.png``.
        image : Tensor
            Image tensor with shape (3, H, W). The image values are expected to be in [0, 1].
        mkdir : bool
            If True, create the directory if it does not exist.

        Notes
        -----
        The image is dumped to ``{dump_path}/{subfolder}/{index:04d}.png``.
        """
        path = self.dump_path / subfolder
        if mkdir:
            path.mkdir(exist_ok=True, parents=True)
        assert path.exists()
        if index is None and name is None:
            raise ValueError("Either index or name should be specified.")
        if index is not None and name is not None:
            filename = path / f'{index:04d}_{name}.png'
        elif index is not None:
            filename = path / f'{index:04d}.png'
        else:
            filename = path / f'{name}.png'

        if image is not None:        
            assert image.min().item() >= 0 and image.max().item() <= 1
            dump_float32_image(filename, image)
        else:
            return filename

    def dump_images2video(self, subfolder: str, *, mkdir: bool = True, images: List[Tensor], 
                          target_mb = None, downsample: int = None, fps: float = 24, duration: float = None,
                          index: Optional[int] = None, name: Optional[str] = None) -> None:
        """
        Dump images to a video file.

        Parameters
        ----------
        subfolder : str
            Subfolder name under :attr:`dump_path`.
        index : int
            Index used for generating filename: ``{index:04d}.png``.
        images : List[Tensor]
            Images tensor with shape (B, H, W, 3). The image values are expected to be in [0, 1].
        mkdir : bool
            If True, create the directory if it does not exist.

        Notes
        -----
        The image is dumped to ``{dump_path}/{subfolder}/{index:04d}.png``.
        """
        path = self.dump_path / subfolder
        if mkdir:
            path.mkdir(exist_ok=True, parents=True)
        assert path.exists()
        if index is None and name is None:
            raise ValueError("Either index or name should be specified.")
        if index is not None and name is not None:
            filename = path / f'{index:04d}_{name}.mp4'
        elif index is not None:
            filename = path / f'{index:04d}.mp4'
        else:
            filename = path / f'{name}.mp4'

        if duration is not None:
            fps = max(1, int(len(images) / duration))  # 最低 1 FPS
            
        with open_video_renderer(
            filename,
            fps=fps,
            target_mb=target_mb,
            downsample=downsample
        ) as renderer:
            for i in range(len(images)):
                image = images[i]
                assert image.min().item() >= 0 and image.max().item() <= 1
                renderer.write(image)

    def parse_log_auto(self, log_file: Union[str, pathlib.Path]) -> None:
        log_file = pathlib.Path(log_file)
        if not log_file.exists():
            print(f"[ERROR] File not found: {log_file}")
            return

        content = log_file.read_text()
        metric_blocks = re.findall(
            r'\[.*?\]\s*(Step\s+(\d+))?\s*(Train|Val|Test) Metrics:(.*?)((?=\[)|$)',
            content,
            re.DOTALL
        )

        # 结构：{metric_name -> {mode -> {step, value}}}
        metrics_by_name = defaultdict(lambda: defaultdict(lambda: {'step': [], 'value': []}))

        for _, step, mode, metrics_str, _ in metric_blocks:
            if mode == "Test":
                step = 0  # 固定 Test Metrics 出现在 step=0
            else:
                step = int(step) if step else 0  # 使用 0 或日志中提供的 step

            for match in re.findall(r'(\w[\w\-]*)\s*=\s*([0-9.eE+-]+)', metrics_str):
                name, val = match
                metrics_by_name[name][mode]['step'].append(step)
                metrics_by_name[name][mode]['value'].append(float(val))

        # base_path = log_file.with_suffix('')
        base_path = log_file.parent / 'dump' / 'log_parsed'
        base_path.mkdir(exist_ok=True, parents=True)

        # 绘图
        for metric_name, mode_data in metrics_by_name.items():
            plt.figure(figsize=(10, 4))
            
            for mode in ['Train', 'Val']:
                if mode in mode_data:
                    steps = mode_data[mode]['step']
                    values = mode_data[mode]['value']
                    plt.plot(steps, values, marker='o', label=mode)
            
            # 特殊处理 Test：只画点，不连线
            if 'Test' in mode_data:
                test_steps = mode_data['Test']['step']
                test_values = mode_data['Test']['value']
                plt.scatter(test_steps, test_values, color='black', marker='x', s=80, label='Test')  # 黑色大叉

            plt.title(f"{metric_name}")
            plt.xlabel("Step")
            plt.ylabel(metric_name)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            # filename = f"{base_path}_{metric_name.lower().replace(' ', '_')}.png"
            filename = f"{base_path}/{metric_name.lower().replace(' ', '_')}.png"
            plt.savefig(filename)
            print(f"[INFO] Saved: {filename}")
            plt.close()
