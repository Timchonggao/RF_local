import types
from typing import TYPE_CHECKING

from rfstudio.utils.lazy_wrapper import _LazyWrapper, lazy_wrapper


def _mat_o3d() -> types.ModuleType:
    import open3d
    return open3d

def _mat_dr() -> types.ModuleType:
    import nvdiffrast.torch as dr
    return dr

def _mat_tcnn() -> types.ModuleType:
    import tinycudann as tcnn
    return tcnn

def _mat_test() -> types.ModuleType:
    import module_name_that_is_impossible_to_exist
    return module_name_that_is_impossible_to_exist

def _mat_sam() -> types.ModuleType:
    import segment_anything
    return segment_anything

def _mat_torchvision() -> types.ModuleType:
    import torchvision
    return torchvision

def _mat_trimesh() -> types.ModuleType:
    import trimesh
    return trimesh

def _mat_tetgen() -> types.ModuleType:
    import tetgen
    return tetgen

def _mat_rfviser() -> types.ModuleType:
    import rfviser
    return rfviser

def _mat_rfviser_tf() -> types.ModuleType:
    from rfviser import transforms as tf
    return tf

def _mat_torchmetrics_F() -> types.ModuleType:
    from torchmetrics.functional import image as torchmetrics_F
    return torchmetrics_F

if TYPE_CHECKING:
    import module_name_that_is_impossible_to_exist as _mat_test
    import nvdiffrast.torch as dr
    import open3d as o3d
    import rfviser
    import segment_anything as sam
    import tetgen
    import tinycudann as tcnn
    import torchvision
    import trimesh
    from rfviser import transforms as rfviser_tf
    from torchmetrics.functional import image as torchmetrics_F
else:
    o3d = lazy_wrapper(_mat_o3d)
    dr = lazy_wrapper(_mat_dr, "pip install git+https://github.com/NVlabs/nvdiffrast/")
    _test = lazy_wrapper(_mat_test, "only for test")
    tcnn = lazy_wrapper(_mat_tcnn, "pip install --global-option=\"--no-networks\" git+https://github.com/NVlabs/tiny-cuda-nn#subdirectory=bindings/torch")
    sam = lazy_wrapper(_mat_sam, "pip install git+https://github.com/facebookresearch/segment-anything.git")
    torchvision = lazy_wrapper(_mat_torchvision)
    trimesh = lazy_wrapper(_mat_trimesh)
    tetgen = lazy_wrapper(_mat_tetgen)
    rfviser = lazy_wrapper(_mat_rfviser)
    rfviser_tf = lazy_wrapper(_mat_rfviser_tf)
    torchmetrics_F = lazy_wrapper(_mat_torchmetrics_F)


def is_materialized(module: types.ModuleType) -> bool:
    assert isinstance(module, _LazyWrapper)
    return module._instance is not None
