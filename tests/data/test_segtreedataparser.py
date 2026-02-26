from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rfstudio.data import SegTreeDataset
from rfstudio.engine.task import Task
from rfstudio.io import dump_float32_image


@dataclass
class Script(Task):

    dataset: SegTreeDataset = SegTreeDataset(path=Path('data') / 'artgs' / 'usb')
    view: int = 0

    def run(self) -> None:
        self.dataset.to(self.device)
        segtree = [x for x in self.dataset.get_gt_outputs(split='test')][self.view]
        dump_float32_image(Path('temp.png'), segtree.visualize_masks())
        dump_float32_image(Path('temp2.png'), segtree.visualize_correlation())

if __name__ == '__main__':
    Script(cuda=0).run()
