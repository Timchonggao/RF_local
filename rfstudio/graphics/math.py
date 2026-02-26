from __future__ import annotations

from typing import Literal, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int64
from torch import Tensor, pi as PI


def rgb2sh(x: Float[Tensor, "*bs C"]) -> Float[Tensor, "*bs C"]:
    return (x - 0.5) / 0.28209479177387814


def sh2rgb(x: Float[Tensor, "*bs C"]) -> Float[Tensor, "*bs C"]:
    return x * 0.28209479177387814 + 0.5


def sh_deg2dim(sh_degree: int) -> Literal[1, 4, 9, 16, 25]:
    if sh_degree == 0:
        return 1
    if sh_degree == 1:
        return 4
    if sh_degree == 2:
        return 9
    if sh_degree == 3:
        return 16
    if sh_degree == 4:
        return 25
    raise ValueError(sh_degree)


def sh_dim2deg(sh_dim: int) -> Literal[0, 1, 2, 3, 4]:
    if sh_dim == 1:
        return 0
    if sh_dim == 4:
        return 1
    if sh_dim == 9:
        return 2
    if sh_dim == 16:
        return 3
    if sh_dim == 25:
        return 4
    raise ValueError(sh_dim)

def get_angle_from_positions(
    edge_point_0: Float[Tensor, "*bs 3"],
    angular_point: Float[Tensor, "*bs 3"],
    edge_point_1: Float[Tensor, "*bs 3"],
    *,
    eps: float = 1e-6
) -> Float[Tensor, "*bs 1"]:
    e0 = safe_normalize(angular_point - edge_point_0)
    e1 = safe_normalize(angular_point - edge_point_1)
    non_saturated = 1 - eps
    return (e0 * e1).sum(-1, keepdim=True).clamp(-non_saturated, non_saturated).acos()

def get_skew_from_cross(cross: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs 3 3"]:
    # [0, -v[2], v[1]]
    # [v[2], 0, -v[0]]
    # [-v[1], v[0], 0]

    zeros = torch.zeros_like(cross[..., 0])    # [...]
    return torch.stack((
        zeros, -cross[..., 2], cross[..., 1],
        cross[..., 2], zeros, -cross[..., 0],
        -cross[..., 1], cross[..., 0], zeros
    ), dim=-1).view(*cross.shape, 3)           # [..., 3, 3]


def get_random_quaternion(
    size: Union[int, Tuple[int, ...], torch.Size],
    *,
    device: Optional[torch.device] = None,
) -> Float[Tensor, "*bs 4"]:
    u = torch.rand(size, device=device)
    v = torch.rand(size, device=device) * (2 * PI)
    w = torch.rand(size, device=device) * (2 * PI)
    return torch.stack([
        torch.sqrt(1 - u) * torch.sin(v),
        torch.sqrt(1 - u) * torch.cos(v),
        torch.sqrt(u) * torch.sin(w),
        torch.sqrt(u) * torch.cos(w),
    ], dim=-1)


def get_random_normal_from_sphere(num: int, *, device: Optional[torch.device] = None) -> Float[Tensor, "N 3"]:
    z = 2.0 * torch.rand(num, device=device) - 1.0     # Uniformly sampled z-coordinates
    z_ = (1 - z ** 2).sqrt()
    theta = 2 * torch.pi * torch.rand(num, device=device)  # Uniformly sampled azimuthal angles
    return torch.stack((theta.cos() * z_, theta.sin() * z_, z), dim=-1)


def get_uniform_normal_from_sphere(num: int, *, device: Optional[torch.device] = None) -> Float[Tensor, "N 3"]:
    phi = torch.linspace(1 / num - 1, 1 - 1 / num, num, device=device).acos()
    theta = ((num * torch.pi) ** 0.5) * phi
    return torch.stack((theta.cos() * phi.sin(), theta.sin() * phi.sin(), phi.cos()), dim=-1)


def get_random_normal_from_hemisphere(
    num: int,
    *,
    device: Optional[torch.device] = None,
    direction: Union[Tuple[float, float, float], Float[Tensor, "3"]] = (0, 0, 1),
) -> Float[Tensor, "N 3"]:
    if isinstance(direction, tuple):
        direction = torch.tensor(direction, device=device, dtype=torch.float32)
    else:
        device = direction.device
    normals = get_random_normal_from_sphere(num, device=device)
    return torch.where((normals * direction).sum(-1, keepdim=True) > 0, normals, -normals)


def get_uniform_normal_from_hemisphere(
    num: int,
    *,
    device: Optional[torch.device] = None,
    direction: Union[Tuple[float, float, float], Float[Tensor, "3"]] = (0, 0, 1),
) -> Float[Tensor, "N 3"]:
    if isinstance(direction, tuple):
        direction = torch.tensor(direction, device=device, dtype=torch.float32)
    else:
        device = direction.device
    phi = torch.linspace(1 / num - 1, 1 - 1 / num, num, device=device).abs().acos()
    theta = ((num * torch.pi) ** 0.5) * phi
    normals = torch.stack((theta.cos() * phi.sin(), theta.sin() * phi.sin(), phi.cos()), dim=-1)
    rot = get_rotation_from_relative_vectors(torch.tensor([0., 0., 1.]).to(direction).expand_as(direction), direction)
    return (rot @ normals.unsqueeze(-1)).squeeze(-1)


def safe_normalize(vectors: Float[Tensor, "*bs C"]) -> Float[Tensor, "*bs C"]:
    lengths = vectors.norm(dim=-1, keepdim=True) # [..., 1]
    results = torch.where(
        lengths < 1e-6,
        torch.cat((torch.zeros(vectors.shape[-1] - 1), torch.ones(1))).to(vectors),
        vectors / lengths.clamp_min(1e-6),
    )
    if torch.is_anomaly_enabled():
        assert (results.detach().norm(dim=-1) - 1).abs().max() < 1e-6
    return results


def get_arbitrary_tangents_from_normals(normals: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs 3"]:

    # (X,Y,Z) -> (0,-Z,Y) & (-Y,X,0)
    tangents = normals[..., [2, 2, 1, 1, 0, 0]]                                     # [..., 6]
    base = torch.tensor([0, -1, 1, -1, 1, 0]).to(normals)
    tangents = (tangents * base).view(*normals.shape[:-1], 2, 3)                         # [..., 2, 3]
    selection = tangents[..., 0, :].norm(dim=-1) < tangents[..., 1, :].norm(dim=-1) # [...]
    tangents = torch.where(
        selection[..., None],
        tangents[..., 1, :],
        tangents[..., 0, :],
    )                                                                               # [..., 3]
    return tangents / tangents.norm(dim=-1, keepdim=True)


def get_rotation_from_axis_angle(
    *,
    axis: Float[Tensor, "*bs 3"],
    angle: Float[Tensor, "*bs 2"],
) -> Float[Tensor, "*bs 3 3"]:
    axis = axis / axis.norm(dim=-1, keepdim=True)
    cos = angle[..., None, None, 0]            # [..., 1, 1]
    sin = angle[..., None, None, 1]            # [..., 1, 1]
    skew_mat = get_skew_from_cross(axis)       # [..., 3, 3]
    R = cos * torch.eye(3, device=axis.device) + (1 - cos) * (axis[..., None] @ axis[..., None, :]) + sin * skew_mat
    return R


def get_rotation_from_relative_vectors(
    a: Float[Tensor, "*bs 3"],
    b: Float[Tensor, "*bs 3"],
    *,
    eps: float = 1e-6,
) -> Float[Tensor, "*bs 3 3"]:
    """
    Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / a.norm(dim=-1, keepdim=True)       # [..., 3]
    b = b / b.norm(dim=-1, keepdim=True)       # [..., 3]
    c = (a * b).sum(-1)                        # [...]
    invalid_mask = c < -1 + eps                # [...]

    # If vectors are exactly opposite, we add a little noise to one of them
    if invalid_mask.any():
        offset = torch.where(invalid_mask[..., None], (torch.rand(a.shape, device=a.device) - 0.5) * 0.01, 0)
        return get_rotation_from_relative_vectors(a + offset, b)

    v = torch.cross(a.expand(*c.shape, 3), b.expand(*c.shape, 3), dim=-1) # [..., 3]
    s = v.norm(dim=-1)                                                    # [...]
    skew_sym_mat = get_skew_from_cross(v)                                 # [..., 3, 3]
    factor = ((1 - c) / (s ** 2 + eps))                                   # [...]
    return torch.eye(3, device=a.device) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * factor[..., None, None]


def least_squares(A: Float[Tensor, "M N"], b: Optional[Float[Tensor, "M K"]] = None) -> Float[Tensor, "N K"]:
    assert A.ndim == 2
    assert A.shape[0] >= A.shape[1], "A.shape[0] must be larger than A.shape[1]."
    if b is None:
        _, _, Vt = torch.linalg.svd(A)
        return Vt[-1].unsqueeze(-1)
    return torch.linalg.lstsq(A, b).solution


def estimate_surface_normal_from_samples(samples: Float[Tensor, "*bs 3"]) -> Float[Tensor, "3"]:
    samples = samples.view(-1, 3) # [N, 3]
    assert samples.shape[0] > 2, "Insufficient points to estimate a plane."
    translation_diff = samples - samples.mean(0)
    _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
    assert eigvec.dtype == samples.dtype
    return eigvec[:, -1]


def get_projection(
    positions: Float[Tensor, "*bs 3"],
    *,
    plane: Literal['xy', 'yz', 'xz', 'pca'],
) -> Tuple[Float[Tensor, "*bs 2"], Float[Tensor, "2 3"]]:
    positions = positions.view(-1, 3)  # [N, 3]
    if plane == 'pca':
        assert positions.shape[0] > 2, "Insufficient points to estimate a plane."
        translation_diff = positions - positions.mean(0)
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        assert eigvec.dtype == positions.dtype
        eigvec = torch.flip(eigvec, dims=(-1, ))
        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]
        proj_matrix = eigvec[:2, :]
    elif plane == 'xy':
        proj_matrix = torch.tensor([[1, 0, 0], [0, 1, 0]]).to(positions)
    elif plane == 'yz':
        proj_matrix = torch.tensor([[0, 1, 0], [0, 0, 1]]).to(positions)
    elif plane == 'xz':
        proj_matrix = torch.tensor([[1, 0, 0], [0, 0, 1]]).to(positions)
    else:
        raise NotImplementedError
    return (proj_matrix @ positions.unsqueeze(-1)).squeeze(-1), proj_matrix


def get_polar_from_rect_2d(positions: Float[Tensor, "*bs 2"]) -> Tuple[Float[Tensor, "*bs"], Float[Tensor, "*bs"]]:
    r = positions.norm(dim=-1)
    theta = torch.arctan2(positions[..., 1], positions[..., 0])        # [...]
    return r, theta


def get_radian_distance(lhs: Tensor, rhs: Tensor) -> Tensor:
    diff = (lhs - rhs).remainder(PI * 2)
    return torch.minimum(diff, 2 * PI - diff)


def rot2quat(rots: Float[Tensor, "*bs 3 3"]) -> Float[Tensor, "*bs 4"]:
    batch_dim = rots.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(rots.reshape(*batch_dim, 9), dim=-1)

    q = torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1)

    q_abs = torch.zeros_like(q)
    positive_mask = q > 0
    q_abs[positive_mask] = torch.sqrt(q[positive_mask])

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(*batch_dim, 4)


def quat2rot(quats: Float[Tensor, "*bs 4"]) -> Float[Tensor, "*bs 3 3"]:
    r, i, j, k = torch.unbind(quats, -1)
    two_s = 2.0 / (quats * quats).sum(-1)
    o = torch.stack((
        1 - two_s * (j * j + k * k),
        two_s * (i * j - k * r),
        two_s * (i * k + j * r),
        two_s * (i * j + k * r),
        1 - two_s * (i * i + k * k),
        two_s * (j * k - i * r),
        two_s * (i * k - j * r),
        two_s * (j * k + i * r),
        1 - two_s * (i * i + j * j),
    ), dim=-1)
    return o.reshape(quats.shape[:-1] + (3, 3))


def slerp(
    rots_a: Float[Tensor, "*bs 3 3"],
    rots_b: Float[Tensor, "*bs 3 3"],
    weights: Float[Tensor, "*bs"],
) -> Float[Tensor, "*bs 3 3"]:
    quats_a = rot2quat(rots_a)                                          # [..., 4]
    quats_b = rot2quat(rots_b)                                          # [..., 4]
    cos = (quats_a * quats_b).sum(-1)                                   # [...]
    neg_mask = cos < 0
    cos = torch.where(neg_mask, -cos, cos)
    quats_a = torch.where(neg_mask[..., None], -quats_a, quats_a)
    angle = torch.acos(cos).clamp(min=1e-8)
    isin = 1.0 / torch.sin(angle)
    quats = torch.add(
        quats_a * (torch.sin((1 - weights) * angle) * isin)[..., None],
        quats_b * (torch.sin(weights * angle) * isin)[..., None]
    )
    return quat2rot(quats)


def get_bounding_box(mask: Bool[Tensor, "... H W"], *, allow_empty: bool = True) -> Int64[Tensor, "... 4"]:
    assert mask.dtype == torch.bool
    shape = mask.shape[:-2]
    H, W = mask.shape[-2:]
    mask = mask.view(-1, H, W) # [B, H, W]
    nonempty = mask.view(-1, H * W).any(1) # [B]
    assert allow_empty or nonempty.all(), 'Inputs must be non-empty.'
    row_projection = mask.any(1).int() # [B, W]
    col_projection = mask.any(2).int() # [B, H]
    xmin = row_projection.argmax(1) # [B]
    xmax = W - row_projection.flip(1).argmax(1) # [B]
    ymin = col_projection.argmax(1) # [B]
    ymax = H - col_projection.flip(1).argmax(1) # [B]
    bbox = torch.stack((xmin, ymin, xmax, ymax), dim=-1) # [B, 4]
    if allow_empty:
        bbox = torch.where(nonempty.unsqueeze(-1), bbox, -1)
    return bbox.view(*shape, 4)


def get_connected_components(connectivity: Bool[Tensor, "N N"]) -> Bool[Tensor, "K N"]:

    from scipy.sparse import csgraph

    K, labels = csgraph.connected_components(connectivity.detach().int().cpu().numpy())
    results = (
        torch.arange(K, device=connectivity.device).unsqueeze(-1) ==
        torch.from_numpy(labels).to(connectivity.device)
    ) # [K, N]
    # assert (connectivity == connectivity.T).all()
    # eigenvalues, eigenvectors = torch.linalg.eigh(connectivity.float())
    # results: Tensor = (eigenvectors.T[eigenvalues.abs() > 1e-3].abs() > 1e-3).unique(dim=0) # [K, N]
    assert results.any(0).all() # check completeness
    cross_overlap = (results.unsqueeze(1) & results).any(-1) # [K, K]
    assert (
        ~cross_overlap |
        torch.eye(cross_overlap.shape[0], dtype=torch.bool, device=cross_overlap.device)
    ).all() # check no cross overlap
    return results


@torch.no_grad()
def principal_component_analysis(
    x: Float[Tensor, "... C ..."],
    *,
    dim: int,
    num_components: int,
) -> Float[Tensor, "... num_components ..."]:
    if dim != -1:
        x = x.transpose(dim, -1)  # [..., C]
    C = x.shape[-1]
    assert C > num_components
    x_flat = x.reshape(-1, C) # [N, C]
    x_centered = x_flat - x_flat.mean(dim=0) # [N, C]
    covariance = x_centered.T @ x_centered / (x_flat.shape[0] - 1) # [C, C]
    eigvals, eigvecs = torch.linalg.eigh(covariance)  # [C], [C, C], ascending order
    topk_eigvecs: Tensor = eigvecs[:, -num_components:] # [C, K], select top k components
    x_pca = x_centered @ topk_eigvecs  # [N, K], project data
    x_pca = x_pca.view(*x.shape[:-1], num_components)
    if dim != -1:
        x_pca = x_pca.transpose(dim, -1)
    return x_pca


class _cluster(NamedTuple):
    indices: Tensor
    values: Optional[Tensor]
    centers: Optional[Tensor]

@torch.no_grad()
def kmeans(
    x: Float[Tensor, "... C ..."],
    *,
    dim: int,
    num_clusters: int,
    return_values: bool = False,
    return_centers: bool = False,
) -> _cluster: # values [... C ...], indices [... 1 ...]

    from sklearn.cluster import KMeans

    if dim != -1:
        x = x.transpose(dim, -1)  # [..., C]
    shape = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1]) # [N, C]
    kmeans = KMeans(
        n_clusters=num_clusters,
        random_state=0,
    )
    labels = kmeans.fit_predict(x.cpu().numpy())
    indices = torch.from_numpy(labels).long().to(x.device).view(-1, 1) # [N, 1]
    if return_values or return_centers:
        centers = x.new_zeros(num_clusters, x.shape[-1]) # [Nc, C]
        centers.scatter_reduce_(dim=0, index=indices.expand_as(x), src=x, reduce='mean', include_self=False)
    if return_values:
        values = centers[indices.squeeze(-1)].reshape(*shape, -1) # [..., C]
        if dim != -1:
            values = values.transpose(dim, -1) # [..., C, ...]
    return _cluster(
        indices.view(*shape).unsqueeze(dim),
        values if return_values else None,
        centers if return_centers else None,
    )

@torch.no_grad()
def spectral_clustering(
    x: Float[Tensor, "... C ..."],
    *,
    dim: int,
    downsample_to: Optional[int] = None,
    num_clusters: int,
    affinity: Literal['rbf', 'nearest', 'cosine'] = 'rbf',
    return_values: bool = False,
    return_centers: bool = False,
) -> _cluster: # values [... C ...], indices [... 1 ...]

    from sklearn.cluster import SpectralClustering

    if dim != -1:
        x = x.transpose(dim, -1)  # [..., C]
    shape = x.shape[:-1]
    raw_x = x.reshape(-1, x.shape[-1]) # [N, C]
    if downsample_to and raw_x.shape[0] > downsample_to:
        x = x[torch.linspace(0, raw_x.shape[0] - 1, downsample_to, device=x.device).round().long(), :]
    else:
        x = raw_x
    spectral_clustering = SpectralClustering(
        n_clusters=num_clusters,
        assign_labels='discretize',
        random_state=0,
        affinity={
            'rbf': 'rbf',
            'nearest': 'nearest_neighbors',
            'cosine': 'precomputed',
        }[affinity],
    )
    if affinity == 'cosine':
        similarity = (F.cosine_similarity(x, x.unsqueeze(1), dim=-1) * 0.5 + 0.5) # [N, N]
        labels = spectral_clustering.fit_predict(similarity.cpu().numpy())
    else:
        labels = spectral_clustering.fit_predict(x.cpu().numpy())
    indices = torch.from_numpy(labels).long().to(x.device).view(-1, 1) # [N, 1]
    if return_values or return_centers or (downsample_to and raw_x.shape[0] > downsample_to):
        centers = x.new_zeros(num_clusters, x.shape[-1]) # [Nc, C]
        centers.scatter_reduce_(dim=0, index=indices.expand_as(x), src=x, reduce='mean', include_self=False)
    if downsample_to and raw_x.shape[0] > downsample_to:
        indices = F.cosine_similarity(raw_x.unsqueeze(1), centers, dim=-1).argmax(1, keepdim=True) # [N, 1]
    if return_values:
        values = centers[indices.squeeze(-1)].reshape(*shape, -1) # [..., C]
        if dim != -1:
            values = values.transpose(dim, -1) # [..., C, ...]
    return _cluster(
        indices.view(*shape).unsqueeze(dim),
        values if return_values else None,
        centers if return_centers else None,
    )
