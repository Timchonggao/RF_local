from __future__ import annotations

import atexit
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from torch import Tensor

from rfstudio.io import dump_float32_image
from rfstudio.utils.pretty import depretty


@dataclass
class Experiment:

    """
    TODO
    """

    name: str = ...
    """
    related to path for dumping experiment files: ${output_dir}/${name}/${timestamp}
    """

    output_dir: Path = Path('outputs')
    """
    related to path for dumping experiment files: ${output_dir}/${name}/${timestamp}
    """

    timestamp: str = None
    """
    related to path for dumping experiment files: ${output_dir}/${name}/${timestamp}.
    Use current timestamp when not specified.
    """

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        self._logger = None

    @property
    def base_path(self) -> Path:
        return self.output_dir / self.name / self.timestamp

    @property
    def log_path(self) -> Path:
        return self.base_path / "log.txt"

    @property
    def dump_path(self) -> Path:
        return self.base_path / "dump"

    def log(self, text: str, new_logfile: Optional[str] = None) -> None:
        if self._logger is None or new_logfile is not None:
            if new_logfile is not None:
                log_path = self.base_path / new_logfile
                log_path.parent.mkdir(parents=True, exist_ok=True)
                file = open(log_path, "a")
            else:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                file = open(self.log_path, "a")
            atexit.register(lambda: file.close())
            self._logger = file
        lines = depretty(text).splitlines()
        time_str = time.strftime("[%Y-%m-%d %H:%M:%S] ")
        space_str = ' ' * len(time_str)
        self._logger.write(time_str + lines[0].rstrip() + '\n')
        for i in range(1, len(lines)):
            self._logger.write(space_str + lines[i].rstrip() + '\n')
        self._logger.flush()

    def dump_image(self, subfolder: str, *, index: int, image: Tensor, mkdir: bool = True) -> None:
        assert image.min().item() >= 0 and image.max().item() <= 1
        path = self.dump_path / subfolder
        if mkdir:
            path.mkdir(exist_ok=True, parents=True)
        assert path.exists()
        filename = path / f'{index:04d}.png'
        dump_float32_image(filename, image)

    def get_dumped_images(self, subfolder: str) -> List[Path]:
        image_list = list((self.dump_path / subfolder).glob("*.png"))
        image_list.sort(key=lambda p: p.name)
        return image_list
