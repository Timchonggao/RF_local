from pathlib import Path

import torch

from rfstudio_ds.data import CMUPanonicRGBADataset
from rfstudio.graphics import Cameras

from rfstudio_ds.visualization import Visualizer
from rfstudio_ds.graphics import DS_Cameras




def test_vis_cmucamera():
    dataset=CMUPanonicRGBADataset(
        path=Path("data/sdfflow/cello1/"),
    )
    dataset.__setup__()
    camera = dataset.get_inputs(split='train')
    camera_sphere = camera.transform_to_fit_sphere(radius=3)
    Visualizer(port=6789).show(cam=camera, cam_sphere=camera_sphere)




if __name__ == '__main__':
    # test_sample_sequentially()
    # test_hemisphere()
    test_vis_cmucamera()
