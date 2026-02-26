from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rfstudio.engine.task import Task
from rfstudio.graphics import RGBImages
from rfstudio.io import load_float32_image
from rfstudio.nn.preprocess import DINOv2


@dataclass
class Script(Task):

    input: Path = ...

    def run(self) -> None:
        rgb = RGBImages(load_float32_image(self.input)).resize_to(518, 518)
        dinov2 = DINOv2()
        dinov2.__setup__()
        dinov2.to(self.device)
        assert dinov2(rgb.to(self.device)).get_num_channels() == 768

if __name__ == '__main__':
    Script(cuda=0).run()
