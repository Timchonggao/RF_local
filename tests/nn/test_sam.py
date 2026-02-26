from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import RGBAImages, RGBImages
from rfstudio.io import dump_float32_image, load_float32_image, load_float32_masked_image
from rfstudio.nn.preprocess import SamAutoMasker
from rfstudio.utils.download import download_model_weights


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def official_impl(input: Path, output: Path) -> None:

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    image = cv2.imread(str(input))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam_checkpoint = download_model_weights(
        'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
        check_hash=True,
        verbose=True,
    )
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(str(output))

@dataclass
class Script(Task):

    input: Path = ...
    output: Path = Path('temp.png')
    gt: bool = False
    sam: SamAutoMasker = SamAutoMasker(points_per_side=64)
    hierarchical: bool = False

    def run(self) -> None:
        if self.gt:
            assert not self.hierarchical
            return official_impl(self.input, self.output)
        self.sam.to(self.device)
        if self.hierarchical:
            img = RGBAImages(load_float32_masked_image(self.input)).to(self.device)
            segtree = self.sam.hierarchical_segment(img)[0]
            print(f'#Masks:    {segtree.num_masks}')
            print(f'#Clusters: {segtree.num_clusters}')
            dump_float32_image(
                self.output,
                torch.cat((segtree.merge().visualize().item(), img.item()[..., 3:]), dim=-1),
            )
        else:
            img = RGBImages(load_float32_image(self.input, alpha_color=(1, 1, 1))).to(self.device)
            vis = self.sam(img).visualize().item()
            vis = torch.cat((
                img.resize_to(vis.shape[1], vis.shape[0]).item(),
                vis,
            ), dim=1)
            dump_float32_image(self.output, vis)


if __name__ == '__main__':
    Script(cuda=0).run()
