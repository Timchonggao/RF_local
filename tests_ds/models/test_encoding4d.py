from rfstudio_ds.model.density_field.components.encoding_4d import Grid4d_HashEncoding, KplaneEncoding
import torch
from torch import nn

def test_3d_sum():
    encoder = KplaneEncoding(
        input_dim=3,
        num_components=16,
        spatial_resolution=32,
        reduce="sum",
    )
    encoder.__setup__()

    pts = torch.rand(200, 3) * 2 - 1
    out = encoder(pts)

    assert out.shape == (200, 16)
    print("✔ test_3d_sum passed")

def test_3d_concat():
    encoder = KplaneEncoding(
        input_dim=3,
        num_components=8,
        spatial_resolution=16,
        reduce="concat",
    )
    encoder.__setup__()

    pts = torch.rand(10, 3) * 2 - 1
    out = encoder(pts)

    # concat 3 planes → feature_dim = 3 * 8
    assert out.shape == (10, 24)
    print("✔ test_3d_concat passed")

def test_4d_product():
    encoder = KplaneEncoding(
        input_dim=4,
        num_components=12,
        spatial_resolution=16,
        time_resolution=8,
        reduce="product",
    )
    encoder.__setup__()

    pts = torch.rand(50, 4) * 2 - 1
    out = encoder(pts)

    # product → feature dim = num_components
    assert out.shape == (50, 12)
    # ensure 6 planes created for 4D
    assert len(encoder.plane_coefs) == 6
    print("✔ test_4d_product passed")

def test_batch_shape():
    encoder = KplaneEncoding(
        input_dim=3,
        num_components=4,
        spatial_resolution=8,
        reduce="sum",
    )
    encoder.__setup__()

    pts = torch.rand(2, 5, 3) * 2 - 1   # (B=2, N=5)
    out = encoder(pts)

    assert out.shape == (2, 5, 4)
    print("✔ test_batch_shape passed")

def test_bbox_norm():
    bbox_min = torch.tensor([0.0, 0.0, 0.0])
    bbox_max = torch.tensor([1.0, 1.0, 1.0])

    encoder = KplaneEncoding(
        input_dim=3,
        num_components=6,
        spatial_resolution=32,
        reduce="sum",
        bbox=(bbox_min, bbox_max),
    )
    encoder.__setup__()

    pts = torch.rand(100, 3)  # [0,1]
    out = encoder(pts)

    assert out.shape == (100, 6)
    print("✔ test_bbox_norm passed")

class SimpleMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

    def forward(self, x):
        return self.net(x)

def test_decoder():
    encoder = KplaneEncoding(
        input_dim=3,
        num_components=8,
        spatial_resolution=32,
        reduce="concat",  # output dim = 3 * 8 = 24
        decoder=SimpleMLP(24, 10)
    )
    encoder.__setup__()

    pts = torch.rand(20, 3) * 2 - 1
    out = encoder(pts)

    assert out.shape == (20, 10)
    print("✔ test_decoder passed")

if __name__ == "__main__":
    # test_3d_sum()
    # test_3d_concat()
    # test_4d_product()
    # test_batch_shape()
    # test_bbox_norm()
    test_decoder()

    print("\nAll tests passed ✓")

