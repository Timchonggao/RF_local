from __future__ import annotations

import contextlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator, Optional, Tuple

import cv2
import ffmpegcv
import numpy as np
import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.utils.process import run_command


@dataclass
class _VideoRenderer:

    filename: Path

    fps: float

    target_mb: Optional[float] = None

    downsample: Optional[float] = None

    def __post_init__(self) -> None:
        self._shape = None
        self._realshape = None
        self._frames = []

    def write(self, image: torch.Tensor) -> None:
        assert image.ndim == 3 and image.shape[-1] == 3
        image = (image.detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
        if self._shape is None:
            self._shape = image.shape[:2]
            self._realshape = (
                (self._shape[1], self._shape[0])
                if self.downsample is None else
                (int(self._shape[1] / self.downsample), int(self._shape[0] / self.downsample))
            )
        else:
            assert image.shape[:2] == self._shape
        if self.downsample:
            image = cv2.resize(image, self._realshape)
        self._frames.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._shape is not None:
            bitrate = None
            if self.target_mb is not None:
                seconds = len(self._frames) / self.fps
                bitrate_value = self.target_mb * 8 / seconds
                bitrate = f'{bitrate_value:.2f}M'
            writer = ffmpegcv.VideoWriter(str(self.filename), 'h264', self.fps, bitrate=bitrate)
            for frame in self._frames:
                writer.write(frame)
            writer.release()


@contextlib.contextmanager
def open_video_renderer(
    filename: Path,
    fps: float,
    target_mb: Optional[float] = None,
    downsample: Optional[float] = None,
) -> Iterator[_VideoRenderer]:
    tmpdir = None
    renderer = None
    assert filename.suffix in ['.mp4', '.gif']
    try:
        tmpdir = TemporaryDirectory()
        temp_filename = Path(tmpdir.__enter__()) / 'output.mp4'
        renderer = _VideoRenderer(filename=temp_filename, fps=fps, downsample=downsample, target_mb=target_mb)
        yield renderer
        renderer.close()
        renderer = None
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.unlink(missing_ok=True)
        if filename.suffix == '.mp4':
            shutil.move(temp_filename, filename)
        else:
            ffmpeg_cmd = [
                "ffmpeg",
                f"-i {temp_filename}",
                f"{filename}",
            ]
            ffmpeg_cmd = " ".join(ffmpeg_cmd)
            run_command(ffmpeg_cmd)
    finally:
        if renderer is not None:
            renderer.close()
        if tmpdir is not None:
            tmpdir.__exit__(None, None, None)


def get_video_frame_shape(filename: Path) -> Tuple[int, int, int]:
    assert filename.exists()
    info = ffmpegcv.video_info.get_info(str(filename))
    return info.count, info.height, info.width


def load_float32_video_frames(
    filename: Path,
    *,
    target_samples: Optional[int] = None,
    target_duration: Optional[float] = None,
    scale_factor: Optional[float] = None,
) -> Float32[Tensor, "N H W 3"]:
    assert filename.exists()
    assert target_duration is None or target_samples is None
    info = ffmpegcv.video_info.get_info(str(filename))
    if scale_factor is not None:
        W = int(info.width * scale_factor)
        H = int(info.height * scale_factor)
        reader = ffmpegcv.VideoReader(str(filename), pix_fmt='rgb24', resize=(W, H))
    else:
        reader = ffmpegcv.VideoReader(str(filename), pix_fmt='rgb24')
    if target_duration is not None:
        target_samples = round(info.fps * target_duration)
    assert target_samples <= info.count
    if target_samples is not None:
        sample_indices = torch.linspace(0, info.count - 1, target_samples).round().long()
    else:
        sample_indices = torch.arange(info.count)
    frames = []
    curr_idx = 0
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        if curr_idx == sample_indices[len(frames)]:
            frames.append(frame)
        curr_idx += 1
    frames = torch.from_numpy(np.stack(frames) / 255).float()
    if target_samples is not None:
        assert frames.shape[0] == target_samples
    assert frames.shape[-1] == 3
    return frames
