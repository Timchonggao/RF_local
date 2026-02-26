from __future__ import annotations

from dataclasses import dataclass

import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import IsoCubes
from rfstudio.utils.context import create_profiler
from rfstudio.utils.parallel import batchify


@dataclass
class TestBatchify(Task):

    @torch.no_grad()
    def run(self) -> None:
        cubes = [IsoCubes.from_resolution(128, device=self.device) for _ in range(32)]
        with create_profiler() as p:
            [cube.marching_cubes() for cube in cubes]
        print(f'Sequential: {p.duration:.3f}')
        with create_profiler() as p:
            [cube.marching_cubes() for cube in batchify(cubes, num_batches=4)]
        print(f'Single CUDA Parallel: {p.duration:.3f}')
        with create_profiler() as p:
            [cube.marching_cubes() for cube in batchify(cubes, num_batches=4, cuda_devices=[0, 1])]
        print(f'Multiple CUDA Parallel: {p.duration:.3f}')

if __name__ == '__main__':
    TestBatchify(cuda=0).run()
