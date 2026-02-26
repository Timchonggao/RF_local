from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Union, Dict, Literal

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float32, Int64
from torch import Tensor
import pywt
import numpy as np

from diso import DiffDMC

from rfstudio.utils.tensor_dataclass import Float, Long, Size, TensorDataclass
from rfstudio.graphics._mesh._flexicubes import _get_cube_edges, _get_check_table, _get_dmc_table,_get_num_vd_table

from ._triangle_mesh import DS_TriangleMesh


def generate_wavelet_basis(time_resolution, wavelet_name='db2', level=3, min_step=1):
    """
    在输入时间轴 t 上生成多个小波基函数
    t 是时间轴，等间距采样点，如 np.linspace(0,1,N)
    wavelet_name='db2' 指定使用哪种小波（这里是 Daubechies 2 阶）
    level=3 小波的“缩放层数” —— 控制它变宽的程度，层数越大，小波越宽（频率越低）
    返回：list of basis functions，每个 shape 为 (len(t),)
    """
    wavelet = pywt.Wavelet(wavelet_name)
    n = time_resolution
    basis_functions = []
    max_level = level

    # 对于每一级的小波函数 (detail coefficients)
    for j in range(1, max_level + 1): # # 小波的“层级”（频率/宽度）
        # step = 2 ** j # 每一层小波的支持宽度翻倍
        step = 2 ** (j + min_step -2)
        for k in range(0, n, step): # 遍历整个时间轴，每隔 step 个点生成一个新的小波位置.实现“平移”小波的效果
            psi = np.zeros(n)
            idx_range = np.arange(k, min(k + step, n))
            width = len(idx_range)
            _, phi, psi_wave = wavelet.wavefun(level=j) # 从 wavefun() 得到一段“标准小波”
            psi_interp = np.interp(np.linspace(0, 1, width), np.linspace(0, 1, len(psi_wave)), psi_wave) # 把它插值成适合我们这个窗口的长度
            psi[idx_range] = psi_interp # 塞进零数组里，只在 idx_range 的那一段非零
            basis_functions.append(psi) # 最后 append 到 basis_functions

    return np.stack(basis_functions)  # shape: (num_basis, len(t)). 每一行是一个“小波基函数”,看上去像一个局部波动的小鼓包,在某段时间内激活,可以组合起来拟合复杂信号（特别是局部高频、突变、异常点）


@dataclass
class DualDomain4DFlexiCubes(TensorDataclass):
    """
    A class representing a dual-domain 4D isocubes mesh.
    将一个四维 SDF 曲线场嵌套在一个三维体素网格中
    """

    # static 3d geometry representation at any timestamp
    num_vertices: int = Size.Dynamic
    num_cubes: int = Size.Dynamic

    vertices: Tensor = Float[num_vertices, 3]
    sdf_values: Tensor = Float[num_vertices, 1]
    sdf_flow_values: Tensor = Float[num_vertices, 1]
    indices: Tensor = Long[num_cubes, 8]
    resolution: Tensor = Long[3]

    alpha: Optional[Tensor] = Float[num_cubes, 8]
    beta: Optional[Tensor] = Float[num_cubes, 12]
    gamma: Optional[Tensor] = Float[num_cubes, 1]

    # dynamic simulator
    static_sdf_values: Tensor = Float[num_vertices, 1]

    sdf_curve_poly_degree: int = Size.Dynamic
    sdf_curve_poly_coefficient: Tensor = Float[num_vertices, sdf_curve_poly_degree]

    sdf_curve_low_frequency_fourier_degree: int = Size.Dynamic
    sdf_curve_low_freq_fourier_omega: Tensor = Float[sdf_curve_low_frequency_fourier_degree]
    sdf_curve_low_frequency_fourier_coefficient_dim: int = Size.Dynamic
    sdf_curve_low_freq_fourier_coefficient: Tensor = Float[num_vertices, sdf_curve_low_frequency_fourier_coefficient_dim]
    
    sdf_curve_mid_frequency_fourier_degree: int = Size.Dynamic
    sdf_curve_mid_freq_fourier_omega: Tensor = Float[sdf_curve_mid_frequency_fourier_degree]
    sdf_curve_mid_frequency_fourier_coefficient_dim: int = Size.Dynamic
    sdf_curve_mid_freq_fourier_coefficient: Tensor = Float[num_vertices, sdf_curve_mid_frequency_fourier_coefficient_dim]
    
    sdf_curve_high_frequency_fourier_degree: int = Size.Dynamic
    sdf_curve_high_freq_fourier_omega: Tensor = Float[sdf_curve_high_frequency_fourier_degree]
    sdf_curve_high_frequency_fourier_coefficient_dim: int = Size.Dynamic
    sdf_curve_high_freq_fourier_coefficient: Tensor = Float[num_vertices, sdf_curve_high_frequency_fourier_coefficient_dim]
    
    sdf_curve_wavelet_num_basis: int = Size.Dynamic
    time_resolution: int = Size.Dynamic
    time_frames: Tensor = Float[time_resolution]
    sdf_curve_wavelet_basis: Tensor = Float[sdf_curve_wavelet_num_basis, time_resolution]
    sdf_curve_wavelet_coefficient: Tensor = Float[num_vertices, sdf_curve_wavelet_num_basis]

    @classmethod
    def from_resolution(
        cls,
        *resolution: int,
        scale: float = 1.0,
        poly_degree: int = 3,
        low_freq_fourier_bands: List[int] = field(default_factory=lambda: [1, 3]),
        mid_freq_fourier_bands: List[int] = field(default_factory=lambda: [4, 9]),
        high_freq_fourier_bands: List[int] = field(default_factory=lambda: [10, 18]),
        wavelet_name: Literal['haar', 'db2', 'db4',] = 'db2',
        wavelet_level: int = 3,
        wavelet_min_step: int = 2,
        time_frames: Optional[Tensor] = None,
        random_init_constant_sdf: bool = True,
        device: Optional[torch.device] = None,
    ) -> DualDomain4DFlexiCubes: 
        # === Grid Construction ===
        assert len(resolution) in [1, 3]
        if len(resolution) == 1:
            resolution = (resolution[0], resolution[0], resolution[0])
        voxel_grid_template = torch.ones(
            resolution[0] + 1,
            resolution[1] + 1,
            resolution[2] + 1,
            device=device,
        ) # [R + 1, R + 1, R + 1]

        cube_corners = torch.tensor([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
            [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=torch.long, device=device) # [8, 3]

        res = torch.tensor(resolution, dtype=torch.long, device=device) # [3]
        coords = torch.nonzero(voxel_grid_template).float() # [R+ * R+ * R+, 3]
        verts = coords.reshape(-1, 3) / res # [R+ * R+ * R+, 3] 这个verts是 z first 增加，所以 verts flatten indice 是： z offset(最小单位) + y offset * res[1] + x offset * res[1] * res[2]

        # === Cube Indices ===
        cubes = torch.arange(resolution[0] * resolution[1] * resolution[2], device=device) # [R*R*R]
        cubes = torch.stack((
            cubes % resolution[0],
            (cubes // resolution[0]) % resolution[1],
            cubes // (resolution[1] * resolution[0]),
        ), dim=-1)[:, None, :] + cube_corners # [R*R*R, 8, 3] 这里的cubes是x first增加
        cubes = (cubes[..., 2] * (1 + resolution[1]) + cubes[..., 1]) * (1 + resolution[0]) + cubes[..., 0] # [R * R * R, 8] 
        # 注意这里计算 cube 存储的coord flatten indice的算法是： i offset + j offset * (res[1] + 1) + k offset * (res[1] + 1) * (res[2] + 1)
        # cube 通过 自身的 flatten indice 存储着 cube上 8个 coord 的flatten indice（公式之间的联系）
        # 为了得到 cube 的 8 个 coord 的flatten indice，首先要的到 cube 的 flatten indice，这里就要根据 “cube上 8个 coord 的flatten indice（公式之间的联系）”，即 z_offset = i, y_offset = j, x_offset = k，确保二者是统一的offest，从而公式是成立的，cube正确地存储了 coord 的 flatten indice，也能通过 coord 的 offest 得到 cube 的 flatten indice。
        # 到此，就梳理清楚了 cube flatten indice 和 cube flatten indice 的关系。其实就是数组本身的indice和数组本身数据之间的规律关系，数组本身的indice递增始终是要考虑哪个轴优先递增，如果这个时候数组本身数据记录了这种递增规律，就可以用来计算出正确的indice。

        # === SDF Initialization ===
        static_sdf_values = (
            (torch.rand_like(verts[..., 0:1]) - 0.1) # [-0.1, 0.9]
            if random_init_constant_sdf
            else torch.zeros_like(verts[..., 0:1])
        )
        
        # === Curve Coefficients ===
        def __init_fourier_coefficients(start: int, end: int, num_verts: int):
            if end <= start or start < 0:
                return torch.empty(0, device=device), torch.empty(num_verts, 0, device=device)
            omega = (
                2 * torch.pi * torch.arange(start, end + 1, device=device)
            )
            coeff = torch.zeros(num_verts, (end - start + 1) * 2, device=device)
            return omega, coeff

        num_vertices = verts.shape[0]
        # Initialize polynomial coefficients
        poly_coefficient = torch.zeros(num_vertices, poly_degree, device=device)

        # Initialize Fourier coefficients
        omega_low, coeff_low = __init_fourier_coefficients(low_freq_fourier_bands[0], low_freq_fourier_bands[1], num_vertices)
        omega_mid, coeff_mid = __init_fourier_coefficients(mid_freq_fourier_bands[0], mid_freq_fourier_bands[1], num_vertices)
        omega_high, coeff_high = __init_fourier_coefficients(high_freq_fourier_bands[0], high_freq_fourier_bands[1], num_vertices)

        # Initialize wavelet basis
        if time_frames is None:
            time_frames = torch.linspace(0, 1, 10, device=device) # create a default time frame
        wavelet_basis_np = generate_wavelet_basis(time_resolution=time_frames.shape[0], wavelet_name=wavelet_name, level=wavelet_level, min_step=wavelet_min_step)
        wavelet_basis = torch.FloatTensor(wavelet_basis_np).to(device)
        wavelet_coefficient = torch.zeros(num_vertices, wavelet_basis.shape[0], device=device)
        
        return DualDomain4DFlexiCubes(
            num_vertices=num_vertices,
            num_cubes=cubes.shape[0],

            vertices=(2 * verts - 1) * scale,
            indices=cubes,
            sdf_values=torch.zeros_like(static_sdf_values),
            sdf_flow_values=torch.zeros_like(static_sdf_values),
            resolution=res,
            time_frames=time_frames,

            static_sdf_values=static_sdf_values,
            sdf_curve_poly_coefficient=poly_coefficient,
            sdf_curve_low_freq_fourier_omega=omega_low,
            sdf_curve_low_freq_fourier_coefficient=coeff_low,
            sdf_curve_mid_freq_fourier_omega=omega_mid,
            sdf_curve_mid_freq_fourier_coefficient=coeff_mid,
            sdf_curve_high_freq_fourier_omega=omega_high,
            sdf_curve_high_freq_fourier_coefficient=coeff_high,
            sdf_curve_wavelet_basis=wavelet_basis,
            sdf_curve_wavelet_coefficient=wavelet_coefficient,
        )
    
    @torch.no_grad()
    def _get_case_id(
        self,
        occupancy: Bool[Tensor, "F 8"],
        surf_cubes: Bool[Tensor, "F"],
    ) -> Int64[Tensor, "N"]:
        """
        Obtains the ID of topology cases based on cell corner occupancy. This function resolves the 
        ambiguity in the Dual Marching Cubes (DMC) configurations as described in Section 1.3 of the 
        supplementary material. It should be noted that this function assumes a regular grid.
        """
        res = self.resolution.tolist()
        cube_corners_idx = torch.pow(2, torch.arange(8, device=occupancy.device)) # [8]
        case_ids = (occupancy[surf_cubes, :] * cube_corners_idx).sum(-1) # [N]

        problem_config = _get_check_table(occupancy.device)[case_ids, :] # [N, 5]
        to_check = problem_config[..., 0] == 1                           # [N]
        problem_config = problem_config[to_check, :]                     # [P, 5]

        # The 'problematic_configs' only contain configurations for surface cubes. Next, we construct a 3D array,
        # 'problem_config_full', to store configurations for all cubes (with default config for non-surface cubes).
        # This allows efficient checking on adjacent cubes.
        problem_config_full = problem_config.new_zeros(res + [5]) # [R, R, R, 5]
        vol_idx = torch.nonzero(problem_config_full[..., 0] == 0) # [R * R * R, 3] = [F, 3]
        vol_idx_problem = vol_idx[surf_cubes, :][to_check, :] # [P, 3]
        problem_config_full[
            vol_idx_problem[..., 0],
            vol_idx_problem[..., 1],
            vol_idx_problem[..., 2],
        ] = problem_config
        vol_idx_problem_adj = vol_idx_problem + problem_config[..., 1:4] # [P, 3]

        within_range = (vol_idx_problem_adj >= 0).all(-1) & (vol_idx_problem_adj < self.resolution).all(-1) # [P]

        vol_idx_problem = vol_idx_problem[within_range]
        vol_idx_problem_adj = vol_idx_problem_adj[within_range]
        problem_config = problem_config[within_range]
        problem_config_adj = problem_config_full[
            vol_idx_problem_adj[..., 0],
            vol_idx_problem_adj[..., 1],
            vol_idx_problem_adj[..., 2]
        ]
        # If two cubes with cases C16 and C19 share an ambiguous face, both cases are inverted.
        to_invert = (problem_config_adj[..., 0] == 1)
        idx = torch.arange(case_ids.shape[0], device=self.device)[to_check][within_range][to_invert]
        case_ids.index_put_((idx,), problem_config[to_invert][..., -1])
        return case_ids

    @torch.no_grad()
    def _identify_surf_edges(self, surf_cubes: Float32[Tensor, "F"]) -> Tuple[
        Float32[Tensor, "E 2"],
        Int64[Tensor, "F*12"],
        Int64[Tensor, "F*12"],
        Bool[Tensor, "F*12"],
    ]:
        """
        Identifies grid edges that intersect with the underlying surface by checking for opposite signs. As each edge 
        can be shared by multiple cubes, this function also assigns a unique index to each surface-intersecting edge 
        and marks the cube edges with this index.
        """
        occupancy = self.sdf_values < 0 # [V, 1]
        all_edges = self.indices[surf_cubes][:, _get_cube_edges(surf_cubes.device)].view(-1, 2) # [F * 12, 2]
        (
            unique_edges, # [U, 2]
            _idx_map,     # [F * 12]
            counts,       # [F * 12]
        ) = all_edges.unique(dim=0, return_inverse=True, return_counts=True)

        mask_edges = occupancy[unique_edges.flatten()].view(-1, 2).sum(-1) == 1 # [E]

        surf_edges_mask = mask_edges[_idx_map] # [F * 12]
        counts = counts[_idx_map]              # [F * 12]

        mapping = -self.indices.new_ones(unique_edges.shape[0])
        mapping[mask_edges] = torch.arange(mask_edges.sum(), device=mapping.device)
        # Shaped as [number of cubes x 12 edges per cube]. This is later used to map a cube edge to the unique index
        # for a surface-intersecting edge. Non-surface-intersecting edges are marked with -1.
        idx_map = mapping[_idx_map]              # [F * 12]
        surf_edges = unique_edges[mask_edges, :] # [E, 2]
        return surf_edges, idx_map, counts, surf_edges_mask
    
    def _linear_interp(
        self,
        edges_weight: Float32[Tensor, "*bs 2 1"],            # sdf 值 (a, b)
        edges_x: Float32[Tensor, "*bs 2 3"],                 # 顶点坐标 (a, b)
        *,
        sdf_eps: Optional[float] = None,
    ) -> Tuple[Float32[Tensor, "*bs 3"], Optional[Float32[Tensor, "*bs 1"]]]:
        """
        Computes the location of zero-crossings on 'edges_x' using linear interpolation with 'edges_weight'.
        """
        sdf_a = edges_weight[..., 0, :]  # [..., 1]
        sdf_b = edges_weight[..., 1, :]  # [..., 1]
        w_b = sdf_a / (sdf_a - sdf_b)    # [..., 1]
        if sdf_eps is not None:
            w_b = (1 - sdf_eps) * w_b + (sdf_eps / 2)

        # === 插值顶点坐标 ===
        v_a = edges_x[..., 0, :]         # [..., 3]
        v_b = edges_x[..., 1, :]         # [..., 3]
        vertices = v_b * w_b + v_a * (1 - w_b)  # [..., 3]

        return vertices

    def dual_marching_cubes(
        self,
        *,
        grad_func: Optional[Callable[[Tensor], Tensor]] = None,
        sdf_eps: Optional[float] = None,
        weight_scale: float = 0.99,
        get_mesh_sdf_flow: Optional[bool] = False,
    ) -> Tuple[DS_TriangleMesh, Float32[Tensor, "K"]]:

        cube_occupancy = (self.sdf_values < 0)[self.indices.flatten()].view(-1, 8) # [N, 8]
        cube_occ_sum = cube_occupancy.sum(-1)                                      # [N]
        surf_cubes = (cube_occ_sum > 0) & (cube_occ_sum < 8)                       # [N]
        N = surf_cubes.sum().item()
        assert N > 0

        beta = (
            torch.ones(N, 12, device=surf_cubes.device)
            if self.beta is None
            else (self.beta[surf_cubes].tanh() * weight_scale + 1)
        ) # [N, 12]
        alpha = (
            None
            if self.alpha is None
            else (self.alpha[surf_cubes].tanh() * weight_scale + 1)
        ) # [N, 8]
        gamma = (
            torch.ones(N, 1, device=surf_cubes.device)
            if self.gamma is None
            else (self.gamma[surf_cubes].sigmoid() * weight_scale + (1 - weight_scale) / 2)
        ) # [N, 1]
        case_ids = self._get_case_id(cube_occupancy, surf_cubes) # [N]
        (
            surf_edges,      # [E, 2]
            idx_map,         # [N * 12]
            edge_counts,     # [N * 12]
            surf_edges_mask  # [N * 12]
        ) = self._identify_surf_edges(surf_cubes)

        if alpha is not None:
            alpha = alpha.index_select(index=_get_cube_edges(surf_edges.device), dim=1).view(-1, 2) # [N * 12, 2]
        surf_edges_x = self.vertices.index_select(index=surf_edges.view(-1), dim=0).view(-1, 2, 3) # [E, 2, 3]
        surf_edges_s = self.sdf_values.index_select(index=surf_edges.view(-1), dim=0).view(-1, 2, 1) # [E, 2, 1]
        zero_crossing = self._linear_interp(
            surf_edges_s,            # sdf
            surf_edges_x,            # 顶点坐标
            sdf_eps=sdf_eps,
        )
        idx_map = idx_map.view(-1, 12) # [N, 12]
        num_vd = _get_num_vd_table(idx_map.device).index_select(index=case_ids, dim=0) # [N]
        edge_group, edge_group_to_vd, edge_group_to_cube, vd_num_edges, vd_gamma = [], [], [], [], []

        total_num_vd = 0
        vd_idx_map = idx_map.new_zeros(case_ids.shape[0], 12) # [N, 12]
        if grad_func is not None:
            raise NotImplementedError
        dmc_table = _get_dmc_table(idx_map.device) # [256, 4, 7]
        for num in range(5):
            cur_cubes: Tensor = (num_vd == num) # [N], consider cubes with the same numbers of vd emitted (for batching)
            curr_num_vd: int = cur_cubes.sum().item() * num
            if curr_num_vd == 0:
                continue
            curr_edge_group = dmc_table[case_ids[cur_cubes], :num, :].view(-1, num * 7) # [G, num * 7]
            curr_edge_group_to_vd = torch.arange(curr_num_vd, device=self.device) + total_num_vd # [G * num]
            curr_edge_group_to_vd = curr_edge_group_to_vd[:, None].repeat(1, 7).view_as(curr_edge_group) # [G, num * 7]
            total_num_vd += curr_num_vd
            curr_edge_group_to_cube = torch.arange(idx_map.shape[0], device=self.device)[cur_cubes] # [G]
            curr_edge_group_to_cube = curr_edge_group_to_cube.unsqueeze(-1).expand_as(curr_edge_group) # [G, num * 7]

            curr_mask = (curr_edge_group != -1) # [G, num * 7]
            edge_group.append(torch.masked_select(curr_edge_group, curr_mask)) # [G']
            edge_group_to_vd.append(torch.masked_select(curr_edge_group_to_vd, curr_mask)) # [G']
            edge_group_to_cube.append(torch.masked_select(curr_edge_group_to_cube, curr_mask)) # [G']
            vd_num_edges.append(curr_mask.reshape(-1, 7).sum(-1, keepdims=True)) # [G * num, 1]
            vd_gamma.append(gamma[cur_cubes, :].repeat(1, num).reshape(-1)) # [G * num]

        edge_group = torch.cat(edge_group)                            # [K]
        edge_group_to_vd = torch.cat(edge_group_to_vd)                # [K]
        edge_group_to_cube = torch.cat(edge_group_to_cube)            # [K]
        vd_num_edges = torch.cat(vd_num_edges, dim=0)                 # [Q, 1]
        vd_gamma = torch.cat(vd_gamma, dim=0)                         # [Q, 1]
        edge_group_idx = edge_group_to_cube * 12 + edge_group         # [K]

        vd = torch.zeros((total_num_vd, 3), device=self.device)       # [Q, 3]
        beta_sum = torch.zeros((total_num_vd, 1), device=self.device) # [Q, 1]

        idx_group = idx_map.flatten()[edge_group_idx]       # [K]
        x_group = surf_edges_x[idx_group, ...]              # [K, 2, 3]
        s_group = surf_edges_s[idx_group, ...]              # [K, 2, 1]
        zero_crossing_group = zero_crossing[idx_group, ...] # [K, 3]

        if alpha is not None:
            alpha_group = alpha.index_select(dim=0, index=edge_group_idx).view(-1, 2, 1)
            ue_group = self._linear_interp(s_group * alpha_group,  x_group, sdf_eps=sdf_eps)  # [K, 3], [K, 1]
        else:
            ue_group = self._linear_interp(s_group, x_group, sdf_eps=sdf_eps)  # [K, 3], [K, 1]

        beta_group = beta.flatten()[edge_group_idx].unsqueeze(-1)                               # [K, 1]
        beta_sum = beta_sum.index_add_(0, index=edge_group_to_vd, source=beta_group)            # [Q, 1]
        vd = vd.index_add_(0, index=edge_group_to_vd, source=ue_group * beta_group) / beta_sum  # [Q, 3]
        L_dev = self._compute_reg_loss(vd, zero_crossing_group, edge_group_to_vd, vd_num_edges) # [K]

        v_idx = torch.arange(vd.shape[0], device=self.device) # [N]

        vd_idx_map = vd_idx_map.flatten().scatter(
            dim=0,
            index=edge_group_idx,                     # [K]
            src=v_idx[edge_group_to_vd],              # [K]
        )                                             # [N * 12]

        vertices, faces, _, _ = self._triangulate(
            surf_edges,                        # [E, 2]
            vd,                                # [Q, 3]
            vd_gamma,                          # [Q]
            edge_counts,                       # [N * 12]
            idx_map,                           # [N, 12]
            vd_idx_map,                        # [N * 12]
            surf_edges_mask,                   # [N * 12]
            grad_func,
        )
        if torch.is_anomaly_enabled():
            assert vertices.isfinite().all()
        
        if get_mesh_sdf_flow:
            vertices_sdf_flow_values = self.get_pos_sdf_flow(positions=vertices)
            return DS_TriangleMesh(vertices=vertices, indices=faces, vertices_sdf_flow=vertices_sdf_flow_values), L_dev
        else:
            return DS_TriangleMesh(vertices=vertices, indices=faces), L_dev

    def _triangulate(
        self,
        surf_edges,
        vd,
        vd_gamma,
        edge_counts,
        idx_map,
        vd_idx_map,
        surf_edges_mask,
        grad_func,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Connects four neighboring dual vertices to form a quadrilateral. The quadrilaterals are then split into
        triangles based on the gamma parameter, as described in Section 4.3.
        """
        s_n = self.sdf_values
        with torch.no_grad():
            group_mask = (edge_counts == 4) & surf_edges_mask  # surface edges shared by 4 cubes.
            group = idx_map.reshape(-1)[group_mask]
            vd_idx = vd_idx_map[group_mask]
            edge_indices, indices = torch.sort(group, stable=True)
            quad_vd_idx = vd_idx[indices].reshape(-1, 4)

            # Ensure all face directions point towards the positive SDF to maintain consistent winding.
            s_edges = s_n[surf_edges[edge_indices.reshape(-1, 4)[:, 0]].reshape(-1)].reshape(-1, 2)
            flip_mask = s_edges[:, 0] > 0
            quad_vd_idx = torch.cat((quad_vd_idx[flip_mask][:, [0, 1, 3, 2]],
                                     quad_vd_idx[~flip_mask][:, [2, 3, 1, 0]]))
        if grad_func is not None:
            # when grad_func is given, split quadrilaterals along the diagonals with more consistent gradients.
            with torch.no_grad():
                vd_gamma = torch.nn.functional.normalize(grad_func(vd), dim=-1)
                quad_gamma = torch.index_select(input=vd_gamma, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4, 3)
                gamma_02 = (quad_gamma[:, 0] * quad_gamma[:, 2]).sum(-1, keepdims=True)
                gamma_13 = (quad_gamma[:, 1] * quad_gamma[:, 3]).sum(-1, keepdims=True)
        else:
            quad_gamma = torch.index_select(input=vd_gamma, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4)
            gamma_02 = (quad_gamma[:, 0] * quad_gamma[:, 2]).unsqueeze(-1)
            gamma_13 = (quad_gamma[:, 1] * quad_gamma[:, 3]).unsqueeze(-1)
        # if not s_n.requires_grad:
        #     mask = (gamma_02 > gamma_13).squeeze(1)
        #     faces = torch.zeros((quad_gamma.shape[0], 6), dtype=torch.long, device=quad_vd_idx.device)
        #     faces[mask] = quad_vd_idx[mask][:, [0, 1, 2, 0, 2, 3]]
        #     faces[~mask] = quad_vd_idx[~mask][:, [0, 1, 3, 3, 1, 2]]
        #     faces = faces.reshape(-1, 3)
        # else:
        if True:
            vd_quad = torch.index_select(input=vd, index=quad_vd_idx.reshape(-1), dim=0).reshape(-1, 4, 3)
            vd_02 = (torch.index_select(input=vd_quad, index=torch.tensor(0, device=self.device), dim=1) +
                     torch.index_select(input=vd_quad, index=torch.tensor(2, device=self.device), dim=1)) / 2
            vd_13 = (torch.index_select(input=vd_quad, index=torch.tensor(1, device=self.device), dim=1) +
                     torch.index_select(input=vd_quad, index=torch.tensor(3, device=self.device), dim=1)) / 2
            weight_sum = (gamma_02 + gamma_13) + 1e-8
            vd_center = ((vd_02 * gamma_02.unsqueeze(-1) + vd_13 * gamma_13.unsqueeze(-1)) /
                         weight_sum.unsqueeze(-1)).squeeze(1)
            vd_center_idx = torch.arange(vd_center.shape[0], device=self.device) + vd.shape[0]
            vd = torch.cat([vd, vd_center])
            faces = quad_vd_idx[:, [0, 1, 1, 2, 2, 3, 3, 0]].reshape(-1, 4, 2)
            faces = torch.cat([faces, vd_center_idx.reshape(-1, 1, 1).repeat(1, 4, 1)], -1).reshape(-1, 3)
        return vd, faces, s_edges, edge_indices
    
    def _compute_reg_loss(
        self,
        vd: Float32[Tensor, "Q 3"],
        ue: Float32[Tensor, "K 3"],
        edge_group_to_vd: Int64[Tensor, "K"],
        vd_num_edges: Int64[Tensor, "Q 1"],
    ) -> Float32[Tensor, "K"]:
        """
        Regularizer L_dev as in Equation 8
        计算每个零交叉点到对应双重顶点的 L2 距离
        计算每个双重顶点的平均距离，并基于绝对偏差（MAD）生成损失 L_dev，形状为 [K]
        """
        dist = (ue - torch.index_select(input=vd, index=edge_group_to_vd, dim=0)).norm(dim=-1)
        mean_l2 = torch.zeros_like(vd[:, 0])
        mean_l2 = mean_l2.index_add_(0, edge_group_to_vd, dist) / vd_num_edges.squeeze(1).float()
        mad = (dist - torch.index_select(input=mean_l2, index=edge_group_to_vd, dim=0)).abs()
        return mad

    def compute_sdf_entropy(self) -> Float32[Tensor, "1"]:
        """SDF 值的正负性一致性正则项，避免物体内部冗余 surface"""
        all_edges = self.indices[:, _get_cube_edges(self.device)].view(-1, 2)  # [F * 12, 2]
        unique_edges = all_edges.unique(dim=0)                                 # [E, 2]
        occupancy = (self.sdf_values < 0).squeeze(-1)                          # [V]
        valid = occupancy[unique_edges[:, 0]] != occupancy[unique_edges[:, 1]] # [E]
        sdf_a = self.sdf_values[unique_edges[valid, 0]]                        # [E]
        sdf_b = self.sdf_values[unique_edges[valid, 1]]                        # [E]
        return torch.add(
            F.binary_cross_entropy_with_logits(sdf_a, (sdf_b > 0).float()),
            F.binary_cross_entropy_with_logits(sdf_b, (sdf_a > 0).float())
        )

    def compute_sdf_eikonal_loss(self, scale: float = 1.0, norm_type: str = "L1") -> Float32[Tensor, "1"]:
        """基于梯度范数的正则，约束 SDF 满足 eikonal 方程"""
        R_X, R_Y, R_Z = self.resolution.tolist()
        delta_x = 2 * scale / R_X
        delta_y = 2 * scale / R_Y
        delta_z = 2 * scale / R_Z

        sdf_values = self.sdf_values.reshape(R_X+1, R_Y+1, R_Z+1) # grid resolution
        dx = (sdf_values[1:, :-1, :-1] - sdf_values[:-1, :-1, :-1]) / delta_x
        dy = (sdf_values[:-1, 1:, :-1] - sdf_values[:-1, :-1, :-1]) / delta_y
        dz = (sdf_values[:-1, :-1, 1:] - sdf_values[:-1, :-1, :-1]) / delta_z

        # 计算梯度范数
        grad_norm = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)  # 避免 sqrt(0)
        if norm_type == "L1":
            eikonal_loss = (grad_norm - 1.0).abs().mean()
        elif norm_type == "L2":
            eikonal_loss = ((grad_norm - 1.0) ** 2).mean()
        
        return eikonal_loss
    
    def compute_coeff_tv_loss(self, norm_type: str = "L1") -> Float32[Tensor, "1"]: # todo 思考如何使用 cuda 版本加速
        """
        多项式 & 傅里叶系数 & 小波函数系数的空间 TV 正则项
        """
        coeffs = torch.cat([
            # self.static_sdf_values,
            self.sdf_curve_poly_coefficient,
            self.sdf_curve_low_freq_fourier_coefficient,
            self.sdf_curve_mid_freq_fourier_coefficient,
            self.sdf_curve_high_freq_fourier_coefficient,
            # self.sdf_curve_wavelet_coefficient, # 系数矩阵太大了，如果对这个系数矩阵计算 TV loss，会造成显存爆炸
        ], dim=1).contiguous()  # [num_vertices, C_total]

        R_X, R_Y, R_Z = (self.resolution + 1).tolist() # grid resolution
        if coeffs.shape[0] != R_X * R_Y * R_Z:
            raise ValueError(f"Expected first dim to be {R_X * R_Y * R_Z}, got {coeffs.shape[0]}")

        coeff_reshaped = coeffs.view(R_X, R_Y, R_Z, -1)
        dx = coeff_reshaped[1:, :, :, :] - coeff_reshaped[:-1, :, :, :]
        dy = coeff_reshaped[:, 1:, :, :] - coeff_reshaped[:, :-1, :, :]
        dz = coeff_reshaped[:, :, 1:, :] - coeff_reshaped[:, :, :-1, :]

        if norm_type == "L1":
            return dx.abs().mean() + dy.abs().mean() + dz.abs().mean()
        elif norm_type == "L2":
            return (dx**2).mean() + (dy**2).mean() + (dz**2).mean()
        else:
            raise ValueError("Unsupported norm_type. Use 'L1' or 'L2'.")

    def compute_wavelet_sparse_loss(self, norm_type: str = "L1") -> Float32[Tensor, "1"]:
        """
        基于 wavelet 基函数的稀疏正则项
        """
        coeffs = self.sdf_curve_wavelet_coefficient.contiguous()  # [num_vertices, num_basis]
        R_X, R_Y, R_Z = (self.resolution + 1).tolist()
        if coeffs.shape[0] != R_X * R_Y * R_Z:
            raise ValueError(f"Expected first dim to be {R_X * R_Y * R_Z}, got {coeffs.shape[0]}")
        
        if norm_type == "L1":
            sparse_loss = coeffs.abs().mean()
        elif norm_type == "L2":
            sparse_loss = (coeffs ** 2).mean()
        else:
            raise ValueError("Unsupported norm_type. Use 'L1' or 'L2'.")
        return sparse_loss

    def _compute_polynomial(self, t: torch.Tensor, batch_size: int, derivative: bool = False) -> torch.Tensor:
        """Compute polynomial and its time derivative."""
        assert self.sdf_curve_poly_degree > 0

        y = 0.0
        for i in range(self.sdf_curve_poly_degree):
            coeff = self.sdf_curve_poly_coefficient[:, i].unsqueeze(0)  # [1, num_vertices]
            if derivative:  
                y += (i + 1) * coeff * (t ** i)  # derivative
            else:
                y += coeff * (t ** (i + 1))  # degree i

        return y
    
    def _compute_fourier(
        self,
        t: torch.Tensor,
        batch_size: int,
        omega: torch.Tensor,
        coeffs: torch.Tensor,
        degree: int,
        derivative: bool = False
    ) -> torch.Tensor: # todo 是否可以使用tachi 加速
        """Compute Fourier and its derivative."""
        assert degree > 0

        y = 0.0
        a_b = coeffs.reshape(-1, degree, 2)  # [num_vertices, degree, 2]
        omega_expanded = omega.unsqueeze(0).expand(batch_size, -1)  # [batch_size, degree]

        for i in range(degree):
            a_i = a_b[:, i, 0]  # [num_vertices]
            b_i = a_b[:, i, 1]
            omega_i = omega_expanded[:, i:i+1]  # [batch_size, 1]

            cos_term = torch.cos(omega_i * t)
            sin_term = torch.sin(omega_i * t)

            if derivative:
                y += -a_i.unsqueeze(0) * omega_i * sin_term + b_i.unsqueeze(0) * omega_i * cos_term
            else:
                y += a_i.unsqueeze(0) * cos_term + b_i.unsqueeze(0) * sin_term

        return y

    def _compute_wavelet(
        self,
        t: torch.Tensor,
        derivative: bool = False
    ) -> torch.Tensor:
        """
        计算小波基函数展开的 SDF 值
        basis: [num_basis, len(t)] 小波基函数
        coeffs: [num_vertices, num_basis]
        返回: [batch_size, num_vertices]
        """
        t = t.flatten()  # 从 [batch, 1] 转换为 [batch]
        diff = torch.abs(self.time_frames.unsqueeze(0) - t.unsqueeze(1)) # 计算 t 和 self.time_frames 之间的差值绝对值
        indices = torch.argmin(diff, dim=1) # 找到每个 t_i 在 self.time_frames 中最近邻的索引

        if not derivative:
            # 计算 SDF 值
            wavelet_basis_batch = self.sdf_curve_wavelet_basis[:, indices]  # [num_basis, batch_size]
            output = (self.sdf_curve_wavelet_coefficient @ wavelet_basis_batch).T  # [batch_size, num_vertices]
        else: # todo 可以通过预计算的小波基函数的导数矩阵来实现，找到查询时间点的导数再乘与系数矩阵就可以得到导数值
            # 计算时间导数：使用中心差分
            time_resolution = self.time_frames.shape[0]
            dt = (self.time_frames[-1] - self.time_frames[0]) / (time_resolution - 1)  # 时间步长

            # 计算前后索引
            indices_prev = (indices - 1).clamp(0, time_resolution - 1)  # [batch_size]
            indices_next = (indices + 1).clamp(0, time_resolution - 1)  # [batch_size]

            # 计算前后时间点的 wavelet_output
            basis_prev = self.sdf_curve_wavelet_basis[:, indices_prev]  # [num_basis, batch_size]
            basis_next = self.sdf_curve_wavelet_basis[:, indices_next]  # [num_basis, batch_size]
            wavelet_output_prev = (self.sdf_curve_wavelet_coefficient @ basis_prev).T  # [batch_size, num_vertices]
            wavelet_output_next = (self.sdf_curve_wavelet_coefficient @ basis_next).T  # [batch_size, num_vertices]

            # 中心差分：(f(t+dt) - f(t-dt)) / (2*dt)
            output = (wavelet_output_next - wavelet_output_prev) / (2 * dt)  # [batch_size, num_vertices]

            # 处理边界情况：对于 indices == 0 或 indices == time_resolution-1，使用单侧差分
            mask_prev = indices == 0
            mask_next = indices == time_resolution - 1
            if mask_prev.any():
                basis_curr = self.sdf_curve_wavelet_basis[:, indices[mask_prev]]  # [num_basis, batch_size_prev]
                wavelet_output_curr = (self.sdf_curve_wavelet_coefficient @ basis_curr).T  # [batch_size_prev, num_vertices]
                output[mask_prev] = (wavelet_output_next[mask_prev] - wavelet_output_curr) / dt  # 前向差分
            if mask_next.any():
                basis_curr = self.sdf_curve_wavelet_basis[:, indices[mask_next]]  # [num_basis, batch_size_next]
                wavelet_output_curr = (self.sdf_curve_wavelet_coefficient @ basis_curr).T  # [batch_size_next, num_vertices]
                output[mask_next] = (wavelet_output_curr - wavelet_output_prev[mask_next]) / dt  # 后向差分

        return output

    def _prune_wavelet_coefficients(self, threshold: float = 1e-4) -> None:
        """
        修剪小波系数（稀疏化）
        
        Args:
            threshold: 修剪阈值
        """
        # 计算系数的重要性
        importance = self.sdf_curve_wavelet_coefficient.abs().max(dim=0).values  # [num_basis]
        
        # 标记需要保留的系数
        keep_mask = importance > threshold
        
        # 修剪系数
        self.sdf_curve_wavelet_coefficient = self.sdf_curve_wavelet_coefficient[:, keep_mask]
        self.sdf_curve_wavelet_basis = self.sdf_curve_wavelet_basis[keep_mask, :]
        
        # 更新相关参数
        self.sdf_curve_wavelet_num_basis = keep_mask.sum().item()
        
        print(f"保留了 {self.sdf_curve_wavelet_num_basis} 个小波基函数（原始：{len(keep_mask)}）")

    def query_sdf_at_times(
        self,
        t: torch.Tensor,
        model_stage: dict = {},
        compute_sdf_flow: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor,]:
        assert model_stage is not None
        """在给定时间 t 查询所有点的 SDF 值和时间导数"""
        sdf_values = 0.0 
        sdf_flow_values = 0.0

        if t.device != self.vertices.device:
            t = t.to(self.vertices.device)
        batch_size = t.shape[0]
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [batch_size, 1]
        assert t.dim() == 2 and t.shape[0] == batch_size and t.shape[1] == 1

        sdf_values = 0.0
        sdf_flow_values = 0.0
        if model_stage['static_sdf_params']:
            sdf_values += self.static_sdf_values.T.expand(batch_size, -1)
        if self.sdf_curve_poly_degree > 0 and model_stage['sdf_curve_poly_coefficient']:
            sdf_values += self._compute_polynomial(t, batch_size)
            sdf_flow_values += self._compute_polynomial(t, batch_size, derivative=True) if compute_sdf_flow else 0.0
        if self.sdf_curve_low_frequency_fourier_degree > 0 and model_stage['sdf_curve_low_freq_fourier_coefficient']:
            sdf_values += self._compute_fourier(
                t,
                batch_size,
                self.sdf_curve_low_freq_fourier_omega,
                self.sdf_curve_low_freq_fourier_coefficient,
                self.sdf_curve_low_frequency_fourier_degree,
            )
            sdf_flow_values += self._compute_fourier(
                t,
                batch_size,
                self.sdf_curve_low_freq_fourier_omega,
                self.sdf_curve_low_freq_fourier_coefficient,
                self.sdf_curve_low_frequency_fourier_degree,
                derivative=True
            ) if compute_sdf_flow else 0.0
        if self.sdf_curve_mid_frequency_fourier_degree > 0 and model_stage['sdf_curve_mid_freq_fourier_coefficient']:
            sdf_values += self._compute_fourier(
                t,
                batch_size,
                self.sdf_curve_mid_freq_fourier_omega,
                self.sdf_curve_mid_freq_fourier_coefficient,
                self.sdf_curve_mid_frequency_fourier_degree,
            )
            sdf_flow_values += self._compute_fourier(
                t,
                batch_size,
                self.sdf_curve_mid_freq_fourier_omega,
                self.sdf_curve_mid_freq_fourier_coefficient,
                self.sdf_curve_mid_frequency_fourier_degree,
                derivative=True
            ) if compute_sdf_flow else 0.0
        if self.sdf_curve_high_frequency_fourier_degree > 0 and model_stage['sdf_curve_high_freq_fourier_coefficient']:
            sdf_values += self._compute_fourier(
                t,
                batch_size,
                self.sdf_curve_high_freq_fourier_omega,
                self.sdf_curve_high_freq_fourier_coefficient,
                self.sdf_curve_high_frequency_fourier_degree,
            )
            sdf_flow_values += self._compute_fourier(
                t,
                batch_size,
                self.sdf_curve_high_freq_fourier_omega,
                self.sdf_curve_high_freq_fourier_coefficient,
                self.sdf_curve_high_frequency_fourier_degree,
                derivative=True
            ) if compute_sdf_flow else 0.0
        if self.sdf_curve_wavelet_basis is not None and model_stage['sdf_curve_wavelet_coefficient']:
            sdf_values += self._compute_wavelet(t)
            sdf_flow_values += self._compute_wavelet(t, derivative=True) if compute_sdf_flow else 0.0

        return sdf_values.unsqueeze(-1), sdf_flow_values.unsqueeze(-1) if compute_sdf_flow else None

    def dual_marching_cubes_at_times(
        self,
        t: torch.Tensor,
        scale: float = 1.0,
        model_stage: dict = {},
        compute_sdf_flow: bool = False,
        compute_sdf_entropy: bool = False,
        compute_sdf_eikonal: bool = False,
        compute_scene_flow_smoothness: bool = False,
        compute_time_tv_loss: bool = False,
        compute_wavelet_sparse_loss: bool = False,
        compute_coeff_tv_loss: bool = False,
        sdf_eps: Optional[float] = None,
        implementation: Optional[Literal["diso", "codebase"]] = "codebase",
        compute_sdf_eikonal_type: Optional[Literal["L2", "L1"]] = "L2",
        compute_coeff_tv_type: Optional[Literal["L2", "L1"]] = "L1",
        return_sdf_values: bool = False,
        return_meshes: bool = True,
    ) -> Tuple[List[DS_TriangleMesh], List[Tensor], List[Tensor]]:
        """动态时间的整体前向流程：SDF 查询 + 网格重建 + 各类 Loss 计算"""
        # Query SDF values at time t
        dynamic_sdf_values, dynamic_sdf_flow_values = self.query_sdf_at_times(t, model_stage, compute_sdf_flow)
        
        meshes = []       
        L_dev_values = []
        sdf_entropy_values = []
        sdf_eikonal_values = []
        coeff_tv_loss = None
        time_tv_loss = None
        wavelet_sparse_loss = None
        
        for batch_idx in range(dynamic_sdf_values.shape[0]): # todo 分析如何并行处理：https://grok.com/share/bGVnYWN5_9bd33130-71ac-43ff-88c9-32980c3f8efd ; pytorch ddp
            # Extract the SDF values for the current batch
            self.replace_(sdf_values = dynamic_sdf_values[batch_idx])
            if compute_sdf_flow:
                self.replace_(sdf_flow_values = dynamic_sdf_flow_values[batch_idx]) 
                
            if compute_sdf_entropy:
                sdf_entropy_values.append(self.compute_sdf_entropy())
            if compute_sdf_eikonal:
                sdf_eikonal_values.append(self.compute_sdf_eikonal_loss(scale=scale, norm_type=compute_sdf_eikonal_type))
           
            if return_meshes:
                mesh, L_dev = self.dual_marching_cubes(sdf_eps = sdf_eps, get_mesh_sdf_flow=compute_scene_flow_smoothness)  # [num_vertices, 3]
                L_dev_values.append(L_dev.mean())
                meshes.append(mesh)
        
        if compute_coeff_tv_loss:
            coeff_tv_loss = self.compute_coeff_tv_loss(norm_type=compute_coeff_tv_type)
        if compute_time_tv_loss:
            time_tv_loss = dynamic_sdf_flow_values.abs().mean()
        if compute_wavelet_sparse_loss:
            wavelet_sparse_loss = self.compute_wavelet_sparse_loss()
        loss_dict = {
            "L_dev": torch.stack(L_dev_values).mean() if len(L_dev_values) > 0 else 0.0,
            "sdf_entropy_loss": torch.stack(sdf_entropy_values).mean() if len(sdf_entropy_values) > 0 else 0.0,
            "sdf_eikonal_loss": torch.stack(sdf_eikonal_values).mean() if len(sdf_eikonal_values) > 0 else 0.0,
            "coeff_tv_loss": coeff_tv_loss if coeff_tv_loss is not None else 0.0,
            "time_tv_loss": time_tv_loss if time_tv_loss is not None else 0.0,
            "wavelet_sparse_loss": wavelet_sparse_loss if wavelet_sparse_loss is not None else 0.0,
        }
        
        meshes_out = meshes if return_meshes else None
        sdf_out = dynamic_sdf_values if return_sdf_values else None
        return meshes_out, loss_dict, sdf_out

    def dual_marching_cubes_at_times_precomputesdfs(
        self,
        dynamic_sdf_values: Tensor = None,
        dynamic_sdf_flow_values: Tensor = None,
        compute_sdf_entropy: bool = False,
        compute_time_tv_loss: bool = False,
        compute_wavelet_sparse_loss: bool = False,
        sdf_eps: Optional[float] = None,
    ) -> Tuple[List[DS_TriangleMesh], List[Tensor], List[Tensor]]:
        meshes = []
        sdf_entropy_values = []
        L_dev_values = []
        time_tv_loss = None
        wavelet_sparse_loss = None
        for batch_idx in range(dynamic_sdf_values.shape[0]): # todo 分析如何并行处理：https://grok.com/share/bGVnYWN5_9bd33130-71ac-43ff-88c9-32980c3f8efd ; pytorch ddp
            # Extract the SDF values for the current batch
            self.replace_(sdf_values = dynamic_sdf_values[batch_idx])
            if dynamic_sdf_flow_values is not None:
                self.replace_(sdf_flow_values = dynamic_sdf_flow_values[batch_idx]) 

            # Extract the vertices and indices for the current batch
            mesh, L_dev = self.dual_marching_cubes(sdf_eps = sdf_eps)  # [num_vertices, 3]
            meshes.append(mesh)
            L_dev_values.append(L_dev.mean())
            if compute_sdf_entropy:
                sdf_entropy_values.append(self.compute_sdf_entropy())
        if compute_time_tv_loss and dynamic_sdf_flow_values is not None:
            time_tv_loss = dynamic_sdf_flow_values.abs().mean()
        if compute_wavelet_sparse_loss:
            wavelet_sparse_loss = self.compute_wavelet_sparse_loss()
        loss_dict = {
            "L_dev": torch.stack(L_dev_values).mean(),
            "sdf_entropy_loss": torch.stack(sdf_entropy_values).mean() if len(sdf_entropy_values) > 0 else 0.0,
            "time_tv_loss": time_tv_loss if time_tv_loss is not None else 0.0,
            "wavelet_sparse_loss": wavelet_sparse_loss if wavelet_sparse_loss is not None else 0.0,
        }
        return meshes, loss_dict

    def get_curve_coefficients_and_coords(
        self,
        indices: Union[Int64[Tensor, "N"], Int64[Tensor, "N 3"]]
    ) -> Dict[str, Tensor]:
        """
        Retrieve curve coefficients and vertex coordinates for given indices.

        Args:
            indices: Either global vertex indices with shape [N] (values in [0, num_vertices-1]),
                    or grid coordinates with shape [N, 3] (values in [0, resolution[i]]).

        Returns:
            Dict containing:
                - vertices: Vertex coordinates, shape [N, 3].
                - grid_indices: Global vertex indices, shape [N, 3].
                - sdf_curve_poly_coefficient: Polynomial coefficients, shape [N, sdf_curve_poly_degree].
                - sdf_curve_low_freq_fourier_coefficient: Low-frequency Fourier coefficients, shape [N, low_freq_coeff_dim].
                - sdf_curve_mid_freq_fourier_coefficient: Mid-frequency Fourier coefficients, shape [N, mid_freq_coeff_dim].
                - sdf_curve_high_freq_fourier_coefficient: High-frequency Fourier coefficients, shape [N, high_freq_coeff_dim].

        Raises:
            ValueError: If indices are invalid or out of bounds.
        """
        device = self.vertices.device
        indices = indices.to(device)

        # Determine if indices are global or grid coordinates
        if indices.ndim == 1:
            # Global indices: [N]
            if indices.max() >= self.num_vertices or indices.min() < 0:
                raise ValueError(f"Global indices must be in [0, {self.num_vertices-1}], got {indices.min()} to {indices.max()}.")
            global_indices = indices
            # Convert global indices to grid coordinates [x, y, z]
            res_x, res_y, res_z = self.resolution[0], self.resolution[1], self.resolution[2]
            x = global_indices % (res_x + 1)
            y = (global_indices // (res_x + 1)) % (res_y + 1)
            z = global_indices // ((res_y + 1) * (res_x + 1))
            grid_indices = torch.stack([x, y, z], dim=-1).long()  # [N, 3]
        elif indices.ndim == 2 and indices.shape[1] == 3:
            # Grid coordinates: [N, 3]
            if (indices < 0).any() or (indices >= self.resolution).any():
                raise ValueError(f"Grid indices must be in [0, {self.resolution.tolist()}], got invalid values.")
            grid_indices = indices
            # Convert grid coordinates [x, y, z] to global indices
            global_indices = (
                indices[:, 2] * (self.resolution[1] + 1) * (self.resolution[0] + 1) +
                indices[:, 1] * (self.resolution[0] + 1) +
                indices[:, 0]
            ).long()
            if global_indices.max() >= self.num_vertices:
                raise ValueError(f"Computed global indices out of bounds: max {global_indices.max()}, num_vertices {self.num_vertices}.")
        else:
            raise ValueError(f"Indices must be 1D (global indices) or 2D (grid coordinates [N, 3]), got shape {indices.shape}.")
        
        # Gather data
        result = {
            "vertices": self.vertices[global_indices],  # [N, 3]
            "grid_indices": grid_indices,  # [N, 3],
            'global_indices': global_indices,  # [N]
            "static_sdf_values": self.static_sdf_values[global_indices],  # [N, 1]
            "sdf_curve_poly_coefficient": self.sdf_curve_poly_coefficient[global_indices],  # [N, poly_degree]
            "sdf_curve_low_freq_fourier_coefficient": self.sdf_curve_low_freq_fourier_coefficient[global_indices],  # [N, low_freq_coeff_dim]
            "sdf_curve_mid_freq_fourier_coefficient": self.sdf_curve_mid_freq_fourier_coefficient[global_indices],  # [N, mid_freq_coeff_dim]
            "sdf_curve_high_freq_fourier_coefficient": self.sdf_curve_high_freq_fourier_coefficient[global_indices],  # [N, high_freq_coeff_dim]
        }
        return result

    def get_cube_indices_from_positions(
        self,
        positions: Tensor,
    ) -> Tensor:
        """
        Retrieve grid indices for given positions.

        Args:
            positions: Positions with shape [N, 3].

        Returns:
            Grid indices with shape [N, 3], i,j,k indices.
        """
        device = self.vertices.device
        positions = positions.to(device)

        max_bound = self.vertices.view(-1, 3).max(0).values
        min_bound = self.vertices.view(-1, 3).min(0).values
        grid_indices = (self.resolution * (positions - min_bound) / (max_bound - min_bound)).floor().long()

        # 因为 coord 的默认优先递增是在 z 轴上，但是 cube 的默认优先递增是在 i 轴上，为了统一 cube 中计算flatten indice的公式，所以需要把z 放在i的位置
        return grid_indices[:, [2, 1, 0]]

    def get_cube_curve_info(
        self,
        positions: Optional[Tensor] = None,
        indices: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Retrieve curve coefficients and vertex coordinates for given positions.

        Args:
            positions: Positions with shape [N, 3].
            indices: Either global cube indices with shape [N] or cube indices with shape [N, 3] (values in [0, resolution[i]]).

        Returns:
            Dict containing:
                - positions: Positions, shape [N, 3].
                - cube_indices: Cube indices, shape [N, 3].
                - cube_vertices: Cube vertices, shape [N, 8, 3].
                - static_sdf_values: Static SDF values, shape [N, 8, 1].
                - sdf_curve_poly_coefficient: Polynomial coefficients, shape [N, 8, poly_degree].
                - sdf_curve_low_freq_fourier_coefficient: Low-frequency Fourier coefficients, shape [N, 8, low_freq_coeff_dim].
                - sdf_curve_mid_freq_fourier_coefficient: Mid-frequency Fourier coefficients, shape [N, 8, mid_freq_coeff_dim].
                - sdf_curve_high_freq_fourier_coefficient: High-frequency Fourier coefficients, shape [N, 8, high_freq_coeff_dim].
        """ 
        assert positions is not None or indices is not None, "Either positions or indices must be provided."
        R = self.resolution
        if positions is not None:
            positions = positions.to(self.vertices.device)
            cube_indices = self.get_cube_indices_from_positions(positions) # [N, 3] cube indices: i,j,k
        if indices is not None:
            indices = indices.to(self.vertices.device)
            if indices.ndim == 1:
                x = indices % (R[0]) # 通常 flatten indice都是 x first的，所以x_offset的计算公式是这个
                y = (indices // (R[0])) % (R[1])
                z = indices // (R[1] * R[0])
                indices = torch.stack([z, y, x], dim=-1).long()  # [N, 3] 这是因为 cube 和ijk和 xyz offset的关系
            cube_indices = indices.to(self.vertices.device) 

        valid = (cube_indices >= 0).all(-1) & (cube_indices < R).all(-1) # [N] valid cube indices
        cube_indices = cube_indices[valid] # [N', 3]
        
        flatten_cube_indices = (cube_indices * (R ** torch.arange(3, device=R.device))).sum(-1) # [N'] i 轴 增长first
        indice = self.indices[flatten_cube_indices] # [N', 8] indices of 8 vertices for each cube
        static_sdf_values = self.static_sdf_values[indice] # [N', 8, 1] 8 static sdf values of each cube
        sdf_curve_poly_coefficient = self.sdf_curve_poly_coefficient[indice] # [N', 8, poly_degree] 8 polynomial coefficients of each cube
        sdf_curve_low_freq_fourier_coefficient = self.sdf_curve_low_freq_fourier_coefficient[indice] # [N', 8, low_freq_coeff_dim] 8 low-frequency Fourier coefficients of each cube
        sdf_curve_mid_freq_fourier_coefficient = self.sdf_curve_mid_freq_fourier_coefficient[indice] # [N', 8, mid_freq_coeff_dim] 8 mid-frequency Fourier coefficients of each cube
        sdf_curve_high_freq_fourier_coefficient = self.sdf_curve_high_freq_fourier_coefficient[indice] # [N', 8, high_freq_coeff_dim] 8 high-frequency Fourier coefficients of each cube

        cube_vertices = self.vertices[indice] # [N', 8, 3] 8 vertices cordinate of each cube
        mean_cube_vertices = cube_vertices.mean(1) # [N', 3] mean vertices cordinate of each cube

        # Gather data
        result = {
            "cube_positions": cube_vertices,  # [N', 8, 3]
            "mean_cube_positions": mean_cube_vertices,  # [N', 3]
            "cube_indices": cube_indices,  # [N', 3]
            "flatten_cube_indices": flatten_cube_indices,  # [N']
            "flatten_indices": indice,  # [N',8]
            "static_sdf_values": static_sdf_values,  # [N', 8, 1]
            "sdf_curve_poly_coefficient": sdf_curve_poly_coefficient,  # [N', 8, poly_degree]
            "sdf_curve_low_freq_fourier_coefficient": sdf_curve_low_freq_fourier_coefficient,  # [N', 8, low_freq_coeff_dim]
            "sdf_curve_mid_freq_fourier_coefficient": sdf_curve_mid_freq_fourier_coefficient,  # [N', 8, mid_freq_coeff_dim]
            "sdf_curve_high_freq_fourier_coefficient": sdf_curve_high_freq_fourier_coefficient,  # [N', 8, high_freq_coeff_dim]
        }

        return result

    def get_pos_sdf_flow(
        self,
        positions: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        R = self.resolution
        positions = positions.to(self.vertices.device)
        cube_indices = self.get_cube_indices_from_positions(positions) # [N, 3] cube indices: i,j,k
        
        valid = (cube_indices >= 0).all(-1) & (cube_indices < R).all(-1) # [N] valid cube indices
        cube_indices = cube_indices[valid] # [N', 3]
        
        flatten_cube_indices = (cube_indices * (R ** torch.arange(3, device=R.device))).sum(-1) # [N'] i 轴 增长first
        indice = self.indices[flatten_cube_indices] # [N', 8] indices of 8 vertices for each cube

        cube_sdf_flow_values = self.sdf_flow_values[indice] # [N', 8, 3] 8 sdf flow values of each cube    
        cube_vertices = self.vertices[indice] # [N', 8, 3] 8 vertices cordinate of each cube
        
        sdf_flow = self.trilinear_interpolation(positions, cube_vertices, cube_sdf_flow_values) # [N,1]

        return sdf_flow
    
    def trilinear_interpolation(self, positions, cube_vertices, cube_sdf_flow_values):
        # positions: [N, 3]
        # cube_vertices: [N, 8, 3]
        # cube_sdf_flow_values: [N, 8, 1]

        # 假设 cube_vertices 排列顺序是标准顺序：
        # 0: (0,0,0), 1: (0,0,1), 2: (0,1,0), 3: (0,1,1)
        # 4: (1,0,0), 5: (1,0,1), 6: (1,1,0), 7: (1,1,1)

        v000 = cube_vertices[:, 0, :]
        v001 = cube_vertices[:, 1, :]
        v010 = cube_vertices[:, 2, :]
        v011 = cube_vertices[:, 3, :]
        v100 = cube_vertices[:, 4, :]
        v101 = cube_vertices[:, 5, :]
        v110 = cube_vertices[:, 6, :]
        v111 = cube_vertices[:, 7, :]

        # positions 相对归一化坐标
        u = ((positions[:, 0] - v000[:, 0]) / (v100[:, 0] - v000[:, 0] + 1e-6)).unsqueeze(-1)
        v = ((positions[:, 1] - v000[:, 1]) / (v010[:, 1] - v000[:, 1] + 1e-6)).unsqueeze(-1)
        w = ((positions[:, 2] - v000[:, 2]) / (v001[:, 2] - v000[:, 2] + 1e-6)).unsqueeze(-1)

        f = (
            cube_sdf_flow_values[:, 0]*(1-u)*(1-v)*(1-w) +
            cube_sdf_flow_values[:, 4]*u*(1-v)*(1-w) +
            cube_sdf_flow_values[:, 2]*(1-u)*v*(1-w) +
            cube_sdf_flow_values[:, 1]*(1-u)*(1-v)*w +
            cube_sdf_flow_values[:, 5]*u*(1-v)*w +
            cube_sdf_flow_values[:, 6]*u*v*(1-w) +
            cube_sdf_flow_values[:, 3]*(1-u)*v*w +
            cube_sdf_flow_values[:, 7]*u*v*w
        )   
        return f  # [N,1]
