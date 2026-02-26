from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.engine.task import Task, TaskGroup
from rfstudio.graphics import RGBAImages, SegTree
from rfstudio.io import dump_float32_image, load_float32_masked_image
from rfstudio.nn.preprocess import SamAutoMasker


@dataclass
class TestSample(Task):

    output: Path = Path('temp.png')
    N: int = 6

    def run(self) -> None:
        num_clusters: int = 4
        num_masks: int = 4
        image_width: int = self.N
        image_height: int = self.N

        cluster_correlation = torch.zeros(num_clusters, num_clusters)
        pixel2cluster = torch.zeros(image_height, image_width, dtype=torch.long)
        cluster2mask = torch.zeros(num_clusters, num_masks, dtype=torch.long)
        masks = torch.zeros(image_height, image_width, num_masks, dtype=torch.bool)
        image = torch.randint(0, 2, (image_height, image_width, 4)).float()
        segtree = SegTree(
            cluster_correlation=cluster_correlation,
            pixel2cluster=pixel2cluster,
            cluster2mask=cluster2mask,
            masks=masks,
            image=image,
        ).to(self.device)
        pixels = segtree.sample_from_patches(approximate_num_patches=self.N)

        vis_input = segtree.image[..., 3:].repeat(1, 1, 3)
        vis_output = torch.zeros_like(vis_input)
        vis_output[pixels[..., 1], pixels[..., 0], :] = 1
        dump_float32_image(self.output, torch.cat((vis_input, vis_output), dim=1))

@dataclass
class TestVis(Task):

    input: Path = ...
    output: Path = Path('temp.png')
    sam: SamAutoMasker = SamAutoMasker(points_per_side=64)

    def run(self) -> None:
        self.sam.to(self.device)
        img = RGBAImages(load_float32_masked_image(self.input)).to(self.device)
        segtree = self.sam.hierarchical_segment(img)[0]
        row1 = segtree.visualize_masks(num_cols=4)
        row2 = segtree.visualize_correlation(num_cols=4)
        row3 = torch.cat([segtree.merge().visualize().item()] * 4, dim=1)
        dump_float32_image(self.output, torch.cat((row1, row2, row3)))

if __name__ == '__main__':
    TaskGroup(
        sample=TestSample(cuda=0),
        vis=TestVis(cuda=0),
    ).run()
