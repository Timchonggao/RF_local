from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch

from rfstudio.engine.task import Task
from rfstudio.nn.preprocess._sam import _mask_to_rle_pytorch
from rfstudio.utils.context import create_profiler


def _raw_mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape # [B, H, W]
    tensor = tensor.permute(0, 2, 1).flatten(1) # [B, W, H] -> [B, W*H]

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat((
            torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
            cur_idxs + 1,
            torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
        ))
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        if not (tensor[i, 0] == 0):
            btw_idxs = torch.cat((torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), btw_idxs))
        out.append({"size": [h, w], "counts": btw_idxs.cpu().numpy()})
    return out

@dataclass
class Script(Task):

    def run(self) -> None:
        masks = torch.rand(2048, 128, 128, device=self.device) > 0.5
        _raw_mask_to_rle_pytorch(masks)
        with create_profiler() as profiler:
            gt_outputs = _raw_mask_to_rle_pytorch(masks)
        print(f'original: {profiler.duration:.3f}s')
        _mask_to_rle_pytorch(masks)
        with create_profiler() as profiler:
            our_outputs = _mask_to_rle_pytorch(masks)
        print(f'ours: {profiler.duration:.3f}s')
        assert len(gt_outputs) == len(our_outputs["starts"])
        for (i, a) in enumerate(gt_outputs):
            b = our_outputs["counts"][our_outputs["starts"][i]:our_outputs["ends"][i]].cpu().numpy()
            assert a["counts"].shape == b.shape, (i, a["counts"], b)
            assert np.all(a["counts"] == b), (i, a["counts"], b)

if __name__ == '__main__':
    Script(cuda=0, seed=123).run()


