from __future__ import annotations

from typing import Iterator, Literal, Optional, Sequence, Tuple, TypeVar, Union

import torch

from rfstudio.utils.tensor_dataclass import TensorLike

T = TypeVar('T', bound='TensorLike')

def batchify(
    seq: Sequence[T],
    *,
    num_batches: int,
    cuda_devices: Union[
        Tuple[Union[Optional[torch.device], int], ...],
        Union[Optional[torch.device], int],
        Literal['all', 'current'],
    ] = 'current',
) -> Iterator[T]:
    original_cuda_idx = torch.cuda.current_device()
    if isinstance(cuda_devices, str):
        if cuda_devices == 'current':
            cuda_indices = [original_cuda_idx]
        else:
            cuda_indices = list(range(torch.cuda.device_count()))
    elif cuda_devices is None:
        cuda_indices = [original_cuda_idx]
    elif isinstance(cuda_devices, torch.device):
        cuda_indices = [cuda_devices.index]
    else:
        cuda_indices = set()
        for item in cuda_devices:
            if item is None:
                cuda_indices.add(original_cuda_idx)
            elif isinstance(item, torch.device):
                cuda_indices.add(int(item.index))
            else:
                cuda_indices.add(int(item))
        cuda_indices = list(cuda_indices)

    N = len(cuda_indices)
    stream_indices = torch.linspace(0, num_batches, len(seq)).floor().int().clamp_max(num_batches) # [L] \in B
    selected_cuda_indices = torch.linspace(0, N, num_batches).floor().int().clamp_max(N - 1) # [B] \in N
    try:
        last_cuda_idx = original_cuda_idx
        for stream_i in range(num_batches):
            cuda_idx = selected_cuda_indices[stream_i].item()
            if last_cuda_idx is None or last_cuda_idx != cuda_idx:
                torch.cuda.set_device(cuda_idx)
            last_cuda_idx = cuda_idx
            with torch.cuda.stream(torch.cuda.Stream(cuda_idx)):
                for seq_i in (stream_indices == stream_i).nonzero().flatten():
                    yield seq[seq_i].to(torch.device(f'cuda:{cuda_idx}'))
    finally:
        torch.cuda.set_device(original_cuda_idx)
        for cuda_idx in cuda_indices:
            torch.cuda.synchronize(cuda_idx)
