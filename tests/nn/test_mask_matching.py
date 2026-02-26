from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from rfstudio.engine.task import Task, TaskGroup
from rfstudio.graphics import RGBImages
from rfstudio.io import dump_float32_image, load_float32_image
from rfstudio.nn.preprocess import DINOv2, SamAutoMasker
from rfstudio.utils.colormap import IntensityColorMap


@dataclass
class Vis(Task):

    a: Path = ...
    b: Path = ...
    output: Path = Path('temp.png')
    sam: SamAutoMasker = SamAutoMasker(points_per_side=64)
    dinov2: DINOv2 = DINOv2()
    x: int = 18 # [0, 36]
    y: int = 15 # [0, 36]

    def run(self) -> None:
        self.sam.to(self.device)
        self.dinov2.to(self.device)
        img_a = RGBImages(load_float32_image(self.a, alpha_color=(1, 1, 1))).to(self.device)
        img_b = RGBImages(load_float32_image(self.b, alpha_color=(1, 1, 1))).to(self.device)
        seg_a = self.sam(img_a).item() # [H, W, 1]
        H, W = seg_a.shape[:2]
        mask = seg_a == seg_a[
            int((self.y + 0.5) / 37 * H),
            int((self.x + 0.5) / 37 * W),
            :,
        ] # [H, W, 1]
        mark = torch.zeros(37, 37, 3, device=self.device)
        mark[self.y, self.x, 1] = 1
        mark = RGBImages(mark).resize_to(H, W).item()
        dino_a = self.dinov2(img_a.resize_to(518, 518)).resize_to(W, H).item() # [H, W, 768]
        dino_a_mean = (dino_a * mask.float()).sum(dim=(0, 1)) / mask.sum().clamp_min(1) # [768]
        dino_b = self.dinov2(img_b.resize_to(518, 518)).resize_to(W, H).item() # [H, W, 768]
        # dino_b = self.dinov2(img_b.resize_to(518, 518)).item() # [37, 37, 768]
        # similarity = torch.zeros(37 * 37, 3, device=self.device)
        # similarity[F.cosine_similarity(dino_a_mean, dino_b, dim=-1).flatten().argmax(), :] = 1
        # similarity = RGBImages(similarity.view(37, 37, 3)).resize_to(H, W).item()[..., :1]
        similarity = F.cosine_similarity(dino_a_mean, dino_b, dim=-1).unsqueeze(-1) # [H, W, 1]
        # similarity = similarity.flatten().softmax(0).view_as(similarity)
        similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min()).clamp_min(1e-6)
        vis = IntensityColorMap().from_scaled(similarity) * 0.7 + img_b.item() * 0.3
        dump_float32_image(self.output, torch.cat((
            img_a.item() * (1 - mark.sum(-1, keepdim=True)) + mark,
            img_b.item(),
            vis,
        ), dim=1).clamp(0, 1))


if __name__ == '__main__':
    TaskGroup(
        vis=Vis(cuda=0),
        match=Vis(cuda=0),
    ).run()

