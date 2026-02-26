from pathlib import Path

import torch

from rfstudio.data import MultiViewDataset
from rfstudio.graphics import Cameras

from rfstudio_ds.visualization import Visualizer
from rfstudio_ds.graphics import DS_Cameras


def test_sample_sequentially():
    dataset = MultiViewDataset(path=Path('data') / 'blender' / 'lego')
    dataset.__setup__()
    cameras = dataset.get_inputs(split='test')
    paths = cameras.sample_sequentially(num_samples=1000, uniform_by='distance')
    Visualizer(port=6789).show(paths=paths)

def test_hemisphere():
    cameras111 = Cameras.from_hemisphere(
        center=(0, 1, 0),
        up=(1, 1, 1),
        radius=0.5,
        num_samples=128
    )
    cameras001 = Cameras.from_hemisphere(
        center=(0, 0, 0),
        up=(0, 0, 1),
        radius=1,
        num_samples=128,
        device=torch.device('cuda:0')
    )
    Visualizer(port=6789).show(cam111=cameras111, cam001=cameras001)

def test_sphere():
    sampling_heuristic = DS_Cameras.from_sphere
    total_frames = 400
    cameras = sampling_heuristic(
        center=(0, 0, 0),
        up=(0, 1, 0),
        radius=1.5,
        num_samples=total_frames, # for train, val and test view
        near=1e-2,
        far=1e2,
        hfov_degree=60,
    )
    dt = 1 / total_frames - 1
    camera_dts = torch.full((total_frames,), dt)
    camera_times = torch.linspace(0, 1, total_frames)
    cameras = cameras.set_times(camera_times, camera_dts)
    indices = torch.arange(total_frames)
    train_indices = indices[::2]
    val_indices = indices[1::2]
    # train_indices = indices[:200]
    # val_indices = indices[200:]

    cameras_train = cameras[train_indices]
    cameras_val = cameras[val_indices]
    Visualizer(port=6789).show(cam111=cameras_train, cam001=cameras_val)

if __name__ == '__main__':
    # test_sample_sequentially()
    # test_hemisphere()
    test_sphere()
