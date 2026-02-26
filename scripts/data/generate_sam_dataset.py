from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch

from rfstudio.data import DynamicDataset, MeshViewSynthesisDataset
from rfstudio.data.dataparser import ArticulationDataparser, RFSegTreeDataparser
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.graphics import Cameras, SegTree
from rfstudio.nn.preprocess import SamAutoMasker


@dataclass
class MVS2SegTree(Task):

    input: Path = ...

    output: Path = ...

    sam: SamAutoMasker = SamAutoMasker(points_per_side=64)

    vis: bool = False

    @torch.no_grad()
    def run(self) -> None:
        dataset = MeshViewSynthesisDataset(path=self.input)
        dataset.__setup__()
        dataset.to(self.device)
        self.sam.to(self.device)

        cameras = []
        segtree = []
        for split in ['train', 'val', 'test']:
            cameras.append(dataset.get_inputs(split=split)[...])
            segtree += self.sam.hierarchical_segment(
                dataset.get_gt_outputs(split=split)[...],
                progress=f'Segment {split} set',
            )
        self.output.mkdir(parents=True, exist_ok=True)
        RFSegTreeDataparser.dump(
            Cameras.cat(cameras, dim=0),
            segtree,
            None,
            path=self.output,
            split='all',
            vis=self.vis,
            progress='Visualizing' if self.vis else None,
        )

@dataclass
class Art2SegTree(Task):

    input: Path = ...

    output: Path = ...

    sam: SamAutoMasker = SamAutoMasker(points_per_side=64)

    state: Literal['start', 'end'] = 'start'

    vis: bool = False

    @torch.no_grad()
    def run(self) -> None:
        dataset = DynamicDataset(path=self.input, dataparser=ArticulationDataparser())
        dataset.__setup__()
        dataset.to(self.device)
        self.sam.to(self.device)

        cameras = []
        segtree: List[SegTree] = []
        for split in ['train', 'val', 'test']:
            timestamps = dataset.get_meta(split=split)
            mask = (timestamps >= 0.5) if self.state == 'end' else (timestamps < 0.5)
            indices = mask.nonzero().flatten().tolist()
            cameras.append(dataset.get_inputs(split=split)[indices].cpu())
            segtree += [x.cpu() for x in self.sam.hierarchical_segment(
                dataset.get_gt_outputs(split=split)[indices],
                progress=f'Segment {split} set',
            )]
        self.output.mkdir(parents=True, exist_ok=True)
        RFSegTreeDataparser.dump(
            Cameras.cat(cameras, dim=0),
            segtree,
            None,
            path=self.output,
            split='all',
            vis=self.vis,
            progress='Visualizing' if self.vis else None,
        )

if __name__ == '__main__':
    TaskGroup(
        mvs2segtree=MVS2SegTree(cuda=0),
        art2segtree=Art2SegTree(cuda=0),
    ).run()
