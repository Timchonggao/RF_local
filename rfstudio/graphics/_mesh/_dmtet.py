from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, Int64
from torch import Tensor

from rfstudio.utils.colormap import IntensityColorMap
from rfstudio.utils.lazy_module import tetgen
from rfstudio.utils.tensor_dataclass import Float, Long, Size, TensorDataclass

from .._cameras import Cameras
from .._images import RGBAImages
from .._points import Points
from ..math import safe_normalize
from ..shaders import MCShader, _get_daylight_cubemap
from ._texture import Texture2D
from ._triangle_mesh import TriangleMesh

_ASSETS_DIR: Path = files('rfstudio') / 'assets' / 'geometry' / 'dmtet'


@lru_cache(maxsize=64)
def _get_base_tet_edges(device: torch.device) -> Int64[Tensor, "2 6"]:
    return torch.tensor([
        [0, 0, 0, 1, 1, 2],
        [1, 2, 3, 2, 3, 3],
    ], dtype=torch.long).to(device)


@lru_cache(maxsize=64)
def _get_triangle_table(device: torch.device) -> Int64[Tensor, "16 6"]:
    _ = -1
    return torch.tensor([
        [_, _, _, _, _, _],
        [1, 0, 2, _, _, _],
        [4, 0, 3, _, _, _],
        [1, 4, 2, 1, 3, 4],
        [3, 1, 5, _, _, _],
        [2, 3, 0, 2, 5, 3],
        [1, 4, 0, 1, 5, 4],
        [4, 2, 5, _, _, _],
        [4, 5, 2, _, _, _],
        [4, 1, 0, 4, 5, 1],
        [3, 2, 0, 3, 5, 2],
        [1, 3, 5, _, _, _],
        [4, 1, 2, 4, 3, 1],
        [3, 0, 4, _, _, _],
        [2, 0, 1, _, _, _],
        [_, _, _, _, _, _],
    ], dtype=torch.long).to(device)


@lru_cache(maxsize=64)
def _get_num_triangles_table(device: torch.device) -> Int64[Tensor, "16"]:
    return torch.tensor([
        [0, 1, 1, 2],
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        [2, 1, 1, 0],
    ], dtype=torch.long).flatten().to(device)


@lru_cache(maxsize=4)
def _get_uvs(num_tets: int, device: torch.device) -> Float32[Tensor, "T*4 2"]:
    N = int(np.ceil(np.sqrt(num_tets)))
    padding = 0.9 / N
    tex_y, tex_x = torch.meshgrid(
        torch.linspace(0, 1 - (1 / N), N, device=device),
        torch.linspace(0, 1 - (1 / N), N, device=device),
        indexing='ij'
    ) # [N, N], [N, N]
    uvs = torch.stack([
        tex_x,
        tex_y,
        tex_x + padding,
        tex_y,
        tex_x + padding,
        tex_y + padding,
        tex_x,
        tex_y + padding,
    ], dim=-1).view(-1, 2)      # [N*N*4, 2]
    return uvs

def _compute_circumcenter(vertices, tets):
    """
    Compute the circumcenters of a set of tetrahedra.

    Args:
    - vertices: Tensor of shape (V, 3), where V is the number of vertices.
    - tets: Tensor of shape (F, 4), where F is the number of tetrahedra.

    Returns:
    - circumcenters: Tensor of shape (F, 3) representing the circumcenters.
    - circumradii: Tensor of shape (F,) representing the radii of the circumspheres.

    Source: https://mathworld.wolfram.com/Circumsphere.html or https://rodolphe-vaillant.fr/entry/127/find-a-tetrahedron-circumcenter
    """
    tet_vertices = vertices[tets]
    v0 = tet_vertices[:, 0]
    v1 = tet_vertices[:, 1]
    v2 = tet_vertices[:, 2]
    v3 = tet_vertices[:, 3]
    a = v1 - v0
    b = v2 - v0
    c = v3 - v0
    d = 2 * torch.det(torch.stack([a, b, c], dim=-1))  # Shape (F,)
    A = torch.linalg.norm(a, dim=-1) ** 2  # Shape (F,)
    B = torch.linalg.norm(b, dim=-1) ** 2  # Shape (F,)
    C = torch.linalg.norm(c, dim=-1) ** 2  # Shape (F,)
    circumcenters = v0 + (A.unsqueeze(-1) * torch.cross(b, c, dim=-1) +
                          B.unsqueeze(-1) * torch.cross(c, a, dim=-1) +
                          C.unsqueeze(-1) * torch.cross(a, b, dim=-1)) / d.unsqueeze(-1)
    circumradii = (v0 - circumcenters).norm(dim=-1)
    return circumcenters, circumradii


def _compute_inertia_tensor(vertices, tets, circumcenters, volumes):
    """
    Compute the sum of principal moments of inertia (trace of inertia tensor) for tetrahedra using the formula in the referenced paper.

    Args:
    - vertices: Tensor of shape (V, 3), where V is the number of vertices.
    - tets: Tensor of shape (F, 4), where F is the number of tetrahedra.
    - circumcenters: Tensor of shape (F, 3), circumcenters for each tetrahedron.
    - volumes: Tensor of shape (F,), volumes of the tetrahedra.

    Returns:
    - M_T: Tensor of shape (F,) representing the sum of the principal moments of inertia for each tetrahedron.
    """

    tet_vertices = vertices[tets]  # Shape: (F, 4, 3)
    rel_vertices = tet_vertices - circumcenters.unsqueeze(1)  # Shape: (F, 4, 3)
    x = rel_vertices[..., 0]  # x-coordinates (F, 4)
    y = rel_vertices[..., 1]  # y-coordinates (F, 4)
    z = rel_vertices[..., 2]  # z-coordinates (F, 4)

    x_sum = x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1] + x[..., 2] * x[..., 2] + x[..., 3] * x[..., 3] \
            + x[..., 0] * x[..., 1] + x[..., 0] * x[..., 2] + x[..., 0] * x[..., 3] \
            + x[..., 1] * x[..., 2] + x[..., 1] * x[..., 3] + x[..., 2] * x[..., 3]

    y_sum = y[..., 0] * y[..., 0] + y[..., 1] * y[..., 1] + y[..., 2] * y[..., 2] + y[..., 3] * y[..., 3] \
            + y[..., 0] * y[..., 1] + y[..., 0] * y[..., 2] + y[..., 0] * y[..., 3] \
            + y[..., 1] * y[..., 2] + y[..., 1] * y[..., 3] + y[..., 2] * y[..., 3]

    z_sum = z[..., 0] * z[..., 0] + z[..., 1] * z[..., 1] + z[..., 2] * z[..., 2] + z[..., 3] * z[..., 3] \
            + z[..., 0] * z[..., 1] + z[..., 0] * z[..., 2] + z[..., 0] * z[..., 3] \
            + z[..., 1] * z[..., 2] + z[..., 1] * z[..., 3] + z[..., 2] * z[..., 3]

    Ix = 6.0 * volumes * (y_sum + z_sum) / 60.0
    Iy = 6.0 * volumes * (x_sum + z_sum) / 60.0
    Iz = 6.0 * volumes * (x_sum + y_sum) / 60.0

    M_T = Ix + Iy + Iz
    return M_T

def _compute_shell_moment(circumradii, volumes):
    """
    Compute the moment of inertia for a spherical shell with equivalent mass.

    Args:
    - circumradii: Tensor of shape (F,) representing the circumradii.
    - volumes: Tensor of shape (F,) representing the volumes of the tetrahedra.

    Returns:
    - M_S: Moment of inertia for each tetrahedron's circumshell.
    """
    M_S =  2.0/5.0*volumes * (circumradii ** 2)
    return M_S

def _E_ODT(vertices, tets):
    """
    Compute the optimal Delaunay triangulation energy for a tetrahedral mesh.

    Args:
    - vertices: Tensor of shape (V, 3), where V is the number of vertices.
    - tets: Tensor of shape (F, 4), where F is the number of tetrahedra.

    Returns:
    - energy: A scalar value representing the ODT energy of the whole mesh.
    """
    circumcenters, circumradii = _compute_circumcenter(vertices, tets)
    tet_vertices = vertices[tets]

    v0, v1, v2, v3 = tet_vertices[:, 0], tet_vertices[:, 1], tet_vertices[:, 2], tet_vertices[:, 3]
    volumes = torch.abs(torch.det(torch.stack([v1 - v0, v2 - v0, v3 - v0], dim=-1))) / 6.0

    M_T = _compute_inertia_tensor(vertices, tets, circumcenters, volumes)
    M_S = _compute_shell_moment(circumradii, volumes)
    energy = torch.mean(torch.abs(M_T - M_S))

    return energy

@dataclass
class DMTet(TensorDataclass):

    num_vertices: int = Size.Dynamic
    num_tets: int = Size.Dynamic

    vertices: Tensor = Float[num_vertices, 3]
    sdf_values: Tensor = Float[num_vertices, 1]
    indices: Tensor = Long[num_tets, 4]

    @classmethod
    def from_ball_sampling_delaunay(
        cls,
        num_samples: int,
        *,
        scale: float = 1.0,
        random_sdf: bool = True,
        device: Optional[torch.device] = None
    ) -> DMTet:
        points = torch.randn(num_samples, 3, device=device)
        radii = torch.rand(num_samples, 1, device=device) ** (1/3) * scale
        points = (points / points.norm(dim=-1, keepdim=True).clamp_min(1e-6)) * radii
        return cls.from_delaunay(Points(positions=points), random_sdf=random_sdf)

    @classmethod
    def from_delaunay(
        cls,
        pts: Points,
        *,
        random_sdf: bool = True,
        perturb: bool = True,
    ) -> DMTet:
        if perturb:
            assert pts.positions.flatten().grad_fn is None
            vertices = Points(pts.positions).flatten().perturb_to_avoid_duplication_().positions
        else:
            vertices = pts.positions

        with torch.no_grad():
            tets = tetgen.TetGen(vertices.cpu().numpy(), np.array([], dtype=int)).tetrahedralize(switches="Q")[1]
            tets = torch.from_numpy(tets).long().to(vertices.device)

            faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=torch.long, device=pts.device)
            tets_combinations = tets[:, faces]
            normals_per_tets = torch.cross(
                vertices[tets_combinations[:, :, 1]] - vertices[tets_combinations[:, :, 0]],
                vertices[tets_combinations[:, :, 2]] - vertices[tets_combinations[:, :, 0]],
            )
            tet_barycenter = vertices[tets].mean(dim=1)
            faces_barycenter = vertices[tets_combinations].mean(dim=2)
            faces_b_to_tet_b = faces_barycenter - tet_barycenter.unsqueeze(1)
            dot = torch.sum(normals_per_tets * faces_b_to_tet_b, dim=2)
            inside_oriented_triangles = dot < 0
            to_flip = inside_oriented_triangles.all(dim=1)
            tets[to_flip] = tets[to_flip][:, [0, 2, 1, 3]]
            sdfs = (
                (torch.rand_like(vertices[..., 0:1]) - 0.1)
                if random_sdf
                else torch.zeros_like(vertices[..., 0:1])
            )
        return DMTet(vertices=vertices, indices=tets, sdf_values=sdfs)

    @classmethod
    def from_predefined(
        cls,
        *,
        resolution: Literal[32, 64, 128],
        scale: float = 1.0,
        random_sdf: bool = True,
        device: Optional[torch.device] = None
    ) -> DMTet:
        predefined_file = _ASSETS_DIR / f'{resolution}_tets.npz'
        assert predefined_file.exists()
        predefined = np.load(predefined_file)
        vertices = torch.tensor(predefined['vertices'], dtype=torch.float32, device=device) * (2 * scale)
        indices = torch.tensor(predefined['indices'], dtype=torch.long, device=device)
        sdfs = (
            (torch.rand_like(vertices[..., 0:1]) - 0.1)
            if random_sdf
            else torch.zeros_like(vertices[..., 0:1])
        )
        return DMTet(
            vertices=vertices,
            indices=indices,
            sdf_values=sdfs,
        )

    @torch.no_grad()
    def _get_interp_edges(self) -> Tuple[
        Int64[Tensor, "num_tets"],
        Int64[Tensor, "num_tets"],
        Int64[Tensor, "num_edges 2"],
        Int64[Tensor, "num_tets 6"],
    ]:
        base_tet_edges = _get_base_tet_edges(self.device)                             # [2, 6]
        T = self.num_tets
        occupancy = (self.sdf_values > 0).squeeze(-1)                                 # [V]
        vertex_occupancy = occupancy.unsqueeze(-1).gather(
            dim=-2,
            index=self.indices.view(T * 4, 1)                                         # [4T, 1]
        ).view(T, 4)                                                                  # [T, 4]
        valid_tets = (vertex_occupancy.any(-1)) & ~(vertex_occupancy.all(-1))         # [T]
        valid_indices = self.indices[valid_tets.view(-1, 1).expand(T, 4)].view(-1, 4) # [T', 4]
        tet_codes = torch.mul(
            vertex_occupancy[valid_tets, :].long(),
            torch.pow(2, torch.arange(4, device=valid_tets.device)),
        ).sum(-1)                                                                     # [T']
        tet_global_indices = torch.arange(
            self.num_tets,
            dtype=torch.long,
            device=self.device,
        )[valid_tets]                                                                 # [T]

        # find all vertices
        endpoint_a = valid_indices[..., base_tet_edges[0]]                      # [T', 6]
        endpoint_b = valid_indices[..., base_tet_edges[1]]                      # [T', 6]
        idx_map = -torch.ones_like(endpoint_a, dtype=torch.long)                # [T', 6]
        edge_mask = occupancy[endpoint_a] != occupancy[endpoint_b]              # [T', 6]
        valid_a = endpoint_a[edge_mask]                                         # [E]
        valid_b = endpoint_b[edge_mask]                                         # [E]
        valid_edges = torch.stack((
            torch.minimum(valid_a, valid_b),
            torch.maximum(valid_a, valid_b),
        ), dim=-1)                                                              # [E, 2]
        unique_edges, inv_inds = valid_edges.unique(dim=0, return_inverse=True) # [E', 2], Map[E -> E']
        idx_map[edge_mask] = torch.arange(
            valid_a.shape[0],
            device=valid_a.device,
        )[inv_inds]                                                             # [E]
        return tet_global_indices, tet_codes, unique_edges, idx_map

    def _get_interp_vertices(
        self,
        edges: Int64[Tensor, "E 2"],
        *,
        sdf_eps: Optional[float],
    ) -> Float32[Tensor, "E 3"]:
        v_a = self.vertices[edges[:, 0], :]                # [E, 3]
        v_b = self.vertices[edges[:, 1], :]                # [E, 3]
        sdf_a = self.sdf_values[edges[:, 0], :]            # [E, 1]
        sdf_b = self.sdf_values[edges[:, 1], :]            # [E, 1]
        w_b = sdf_a / (sdf_a - sdf_b)                      # [E, 1]
        if sdf_eps is not None:
            w_b = (1 - sdf_eps) * w_b + (sdf_eps / 2)      # [E, 1]
        return v_b * w_b + v_a * (1 - w_b)                 # [E, 3]

    def marching_tets(
        self,
        *,
        map_uv: bool = False,
        sdf_eps: Optional[float] = None,
    ) -> TriangleMesh:
        return self.marching_tets_with_edges(map_uv=map_uv, sdf_eps=sdf_eps)[0]

    def marching_tets_with_edges(
        self,
        *,
        map_uv: bool = False,
        sdf_eps: Optional[float] = None,
    ) -> Tuple[TriangleMesh, Int64[Tensor, "E 2"]]:

        triangle_table = _get_triangle_table(self.device)              # [16, 6]
        num_triangles_table = _get_num_triangles_table(self.device)    # [16]
        [
            tet_indices,                                               # [T']
            tet_codes,                                                 # [T']
            edges,                                                     # [E, 2]
            idx_map                                                    # Map[[T', 6] -> E]
        ] = self._get_interp_edges()
        vertices = self._get_interp_vertices(edges, sdf_eps=sdf_eps)           # [E, 3]

        num_triangles = num_triangles_table[tet_codes]     # [T']
        one_tri_mask = num_triangles == 1                  # [T']
        two_tri_mask = num_triangles == 2                  # [T']

        # Generate triangle indices
        indices = torch.cat((
            torch.gather(
                input=idx_map[one_tri_mask, :],                    # Map[[T'', 6] -> E]
                dim=1,
                index=triangle_table[tet_codes[one_tri_mask], :3]  # [T'', 3]
            ).reshape(-1, 3),
            torch.gather(
                input=idx_map[two_tri_mask, :],                    # Map[[T'', 6] -> E]
                dim=1,
                index=triangle_table[tet_codes[two_tri_mask], :6]  # [T'', 6]
            ).reshape(-1, 3),
        ), dim=0)                                                  # [F, 3]

        assert indices.min().item() == 0 and indices.max().item() + 1 == vertices.shape[0]

        if map_uv:
            # Generate triangle uvs
            face_global_indices = torch.cat((
                tet_indices[one_tri_mask] * 2,
                torch.stack((
                    tet_indices[two_tri_mask] * 2,
                    tet_indices[two_tri_mask] * 2 + 1
                ), dim=-1).view(-1)
            ), dim=0)                                      # [F] \in [0, 2T-1]

            tet_idx = (face_global_indices // 2) * 4       # [F] \in [0, 4T-4]
            tri_idx = face_global_indices % 2              # [F] \in [0, 1]

            uv_idx = torch.stack((
                tet_idx,
                tet_idx + tri_idx + 1,
                tet_idx + tri_idx + 2
            ), dim=-1)                                     # [F, 3] \in [0, 4T-1]

            uvs = _get_uvs(self.num_tets, self.device)[uv_idx.view(-1), :].view(-1, 3, 2)
        else:
            uvs = None

        if torch.is_anomaly_enabled():
            assert vertices.isfinite().all()
        return TriangleMesh(vertices=vertices, indices=indices, uvs=uvs), edges

    def as_mesh(self) -> TriangleMesh:
        vertices = self.vertices
        tet_face_indices = torch.tensor([
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ], dtype=torch.long, device=self.device).flatten()
        indices = self.indices[..., tet_face_indices].view(*self.shape, -1, 3) # [..., 4T, 3]
        return TriangleMesh(vertices=vertices, indices=indices)

    def get_active_tet_mask(self) -> Bool[Tensor, "T"]:
        T = self.num_tets
        occupancy = (self.sdf_values > 0).squeeze(-1)                       # [V]
        vertex_occupancy = occupancy.unsqueeze(-1).gather(
            dim=-2,
            index=self.indices.view(T * 4, 1)                               # [4T, 1]
        ).view(T, 4)                                                        # [T, 4]
        return (vertex_occupancy.any(-1)) & ~(vertex_occupancy.all(-1))     # [T]

    def get_active_vertex_indices(self) -> Int64[Tensor, "V"]:
        return self.indices[self.get_active_tet_mask()].flatten().unique()

    def get_neighbor_active_vertex_indices(self) -> Int64[Tensor, "V"]:
        mask = torch.isin(self.indices, self.get_active_vertex_indices()).any(-1)
        return self.indices[mask].flatten().unique()

    def get_barycentric_coords(
        self,
        positions: Float32[Tensor, "... 3"],
        *,
        chunk_size: int = 128,
    ) -> Tuple[Int64[Tensor, "... 1"], Float32[Tensor, "... 4"]]:
        """
        Check if points are inside a set of tetrahedra using barycentric coordinates,
        processing in chunks for large datasets.

        Args:
        - points: Tensor of shape (P, 3), where P is the number of points.
        - chunk_size: Number of points to process in each chunk.

        Returns:
        - tet_indices: Tensor of shape (P,) with the index of the first tetrahedron that contains each point,
          or -1 if no tetrahedron contains the point.
        - barycentric_coords: Tensor of shape (P, 4) with the barycentric coordinates of each point in the first
          tetrahedron it is inside, or zeros if none found.
        """
        shape = positions.shape[:-1]
        positions = positions.view(-1, 3)
        tet_indices = []
        barycentric_coords = []
        tets_vertices = self.vertices[self.indices.flatten(), :].view(-1, 4, 3) # [T, 4, 3]
        mat = (tets_vertices[:, :3, :] - tets_vertices[:, 3:, :]).permute(0, 2, 1).contiguous() # [T, 3, 3]

        for i in range(0, positions.shape[0], chunk_size):
            d = (positions[i:i+chunk_size, :].unsqueeze(1) - tets_vertices[:, 3, :]).permute(1, 2, 0) # [T, 3, N]
            chunk_bary_coords: Tensor = torch.linalg.solve(mat, d).permute(2, 0, 1) # [T, 3, N] -> [N, T, 3]
            chunk_bary_coords = torch.cat((
                chunk_bary_coords,
                1 - chunk_bary_coords.sum(-1, keepdim=True),
            ), dim=-1) # [N, T, 4]
            inside = (chunk_bary_coords >= 0).all(-1) # [N, T]
            valid_tet_exists = inside.any(dim=-1) # [N]
            chunk_inside_tet = torch.where(valid_tet_exists, inside.int().argmax(dim=1), -1) # [N]
            chunk_bary_coords = chunk_bary_coords.gather(
                dim=1,
                index=chunk_inside_tet.clamp_min(0).view(-1, 1, 1).expand(-1, 1, 4)
            ).squeeze(1) # [N, 4]
            tet_indices.append(chunk_inside_tet)
            barycentric_coords.append(chunk_bary_coords)

        return torch.cat(tet_indices).view(*shape, 1), torch.cat(barycentric_coords).view(*shape, 4)

    def interp_sdf_values(self, positions: Float32[Tensor, "... 3"]) -> Float32[Tensor, "... 1"]:
        shape = positions.shape[:-1]
        positions = positions.view(-1, 3) # [N, 3]
        tet_indices, barycentric_coords = self.get_barycentric_coords(positions) # [N, 1], [N, 4]
        tet_sdf_values = self.sdf_values[self.indices.flatten(), :].view(-1, 4) # [T, 4]
        return torch.where(
            tet_indices >= 0,
            (tet_sdf_values[tet_indices.flatten(), :] * barycentric_coords).sum(-1, keepdim=True), # [N, 1]
            1.0,
        ).view(*shape, 1)

    def compute_entropy(self) -> Float32[Tensor, "1"]:
        edges = self._get_interp_edges()[2]                                 # [E, 2]
        sdf_a = self.sdf_values[edges[:, 0]]                                # [E]
        sdf_b = self.sdf_values[edges[:, 1]]                                # [E]
        return torch.add(
            F.binary_cross_entropy_with_logits(sdf_a, (sdf_b > 0).float()),
            F.binary_cross_entropy_with_logits(sdf_b, (sdf_a > 0).float())
        )

    def compute_fairness(self) -> Float32[Tensor, "1"]:
        edges = self._get_interp_edges()[2]                     # [E, 2]
        sdf_a_raw = self.sdf_values[edges[:, 0]]                # [E]
        sdf_b_raw = self.sdf_values[edges[:, 1]]                # [E]
        # sdf_a = torch.where(sdf_a_raw > 0, sdf_a_raw, sdf_b_raw)
        # sdf_b = torch.where(sdf_a_raw > 0, sdf_b_raw, sdf_a_raw)
        # return 2 * (sdf_a - sdf_b).log().mean() - sdf_a.log().mean() - (-sdf_b).log().mean()
        return (sdf_a_raw + sdf_b_raw).abs().mean()

    def compute_uncertainty(
        self,
        mesh_v_grad: Float32[Tensor, "V' 3"],
        *,
        sdf_eps: Optional[float] = None,
        edges: Optional[Tensor] = None,
    ) -> Float32[Tensor, "V 1"]:
        if edges is None:
            [
                tet_indices,                                               # [T']
                tet_codes,                                                 # [T']
                edges,                                                     # [E, 2]
                idx_map                                                    # Map[[T', 6] -> E]
            ] = self._get_interp_edges()
        v_a = self.vertices[edges[:, 0], :]                # [E, 3]
        v_b = self.vertices[edges[:, 1], :]                # [E, 3]
        sdf_a = self.sdf_values[edges[:, 0], :]            # [E, 1]
        sdf_b = self.sdf_values[edges[:, 1], :]            # [E, 1]
        w_b = sdf_a / (sdf_a - sdf_b)                      # [E, 1]
        if sdf_eps is not None:
            w_b = (1 - sdf_eps) * w_b + (sdf_eps / 2)      # [E, 1]
        v_grad_norm = (mesh_v_grad * safe_normalize(v_b - v_a)).sum(-1).abs() # [E]
        uncertainty = torch.zeros_like(self.sdf_values[..., 0])    # [V]
        uncertainty.scatter_add_(dim=0, index=edges[:, 1], src=v_grad_norm * w_b.squeeze(-1))
        uncertainty.scatter_add_(dim=0, index=edges[:, 0], src=v_grad_norm * (1 - w_b.squeeze(-1)))
        return uncertainty.unsqueeze(-1)

    def compute_delaunay_energy(self) -> Float32[Tensor, "1"]:
        return _E_ODT(self.vertices, self.indices)

    @torch.no_grad()
    def render_pretty(
        self,
        cameras: Cameras,
        *,
        uncertainty: Optional[Tensor] = None,
        point_shape: Literal['square', 'circle'] = 'circle',
        point_size: float = 0.02,
        z_up: bool = False,
    ) -> RGBAImages:
        mesh = self.marching_tets()
        mesh.replace_(
            kd=Texture2D.from_constants((119/255, 150/255, 170/255), device=mesh.device),
            ks=Texture2D.from_constants((1.0, 0.25, 0.05), device=mesh.device),
            uvs=mesh.vertices.new_zeros(mesh.num_faces, 3, 2)
        )
        if point_shape == 'square':
            ref_mesh = TriangleMesh.create_cube(size=point_size).to(mesh.device)
        elif point_shape == 'circle':
            ref_mesh = TriangleMesh.create_sphere(radius=point_size, resolution=16).to(mesh.device)
        else:
            raise ValueError(point_shape)

        point_mesh = TriangleMesh(
            vertices=(ref_mesh.vertices + self.vertices.view(-1, 1, 3)).view(-1, 3),
            indices=(
                ref_mesh.indices +
                torch.arange(self.num_vertices).to(ref_mesh.indices).view(-1, 1, 1) * ref_mesh.num_vertices
            ).view(-1, 3),
        )
        if uncertainty is None:
            point_mesh.replace_(
                uvs=point_mesh.vertices.new_zeros(point_mesh.num_faces, 3, 2),
                kd=Texture2D.from_constants((0.659, 0.212, 0.013), device=point_mesh.device),
                ks=Texture2D.from_constants((1.0, 1.0, 0.0), device=mesh.device),
            )
        else:
            point_mesh.replace_(
                ks=Texture2D.from_constants((1.0, 1.0, 0.0), device=mesh.device),
            ).build_texture_from_tensors_(
                IntensityColorMap(discretization=64)(uncertainty)
                    .view(-1, 1, 3)
                    .repeat(1, ref_mesh.num_faces, 1)
                    .view(-1, 3),
                attrs='flat',
                target='kd',
            )

        final_mesh = TriangleMesh.merge(mesh, point_mesh)
        envmap = _get_daylight_cubemap(mesh.device, z_up=z_up).as_latlng()
        return final_mesh.render(cameras, shader=MCShader(envmap=envmap, normal_type='flat')).rgb2srgb()
