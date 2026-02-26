from dataclasses import dataclass
from pathlib import Path


from rfstudio.engine.task import Task
from rfstudio.graphics import IsoCubes

from pytorch3d.ops import knn_points
from torch_geometric.utils import geodesic_distance

from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from torch_geometric.utils import to_torch_csc_tensor  # PyG 2.0+ 支持稀疏tensor操作
from torch_geometric.typing import SparseTensor

import torch

from rfstudio_ds.graphics import DS_TriangleMesh

@dataclass
class Test(Task):

        
    @torch.no_grad()
    def _compute_euclidean_knn_pytorch3d(self, vertices, k, max_dist, device):
        # 确保 vertices 形状为 [1, V, 3]（batch 维度）
        vertices = vertices.unsqueeze(0)  # [1, V, 3]
        
        # 使用 knn_points 计算 k 个最近邻（包括自身）
        knn_result = knn_points(
            vertices, vertices, K=k, return_sorted=True, norm=2
        )
        indices = knn_result.idx[0, :, :].long()  # [V, k]
        distances = knn_result.dists[0, :, :].sqrt()  # [V, k];knn_points 返回平方距离, 所以这里需要开方
        
        # 应用 max_dist 过滤
        valid_mask = distances <= max_dist
        # distances[~valid_mask] = float('inf')
        distances = torch.where(valid_mask, distances, torch.full_like(distances, float(10)))
        
        return distances, indices  # 返回距离（非平方）和索引

    @torch.no_grad()
    def _compute_geodesic_knn_pytorch_geometry(self, vertices, face_indices, k, max_dist, device):
        # abandon too slow
        indices = face_indices.transpose(0, 1)
        distance = geodesic_distance(vertices, indices, norm=False, max_dist=0.2,num_workers=-1)
        return distance
    
    def face_to_edge_index(self, face_indices):
        face = face_indices.transpose(0, 1)
        edge_index = torch.cat([
            face[:2],
            face[1:],
            face[::2],
        ], dim=1)
        return edge_index
    
    def get_k_hop_neighbors(self, edge_index, num_nodes, K=10):
        # 转换为稀疏邻接矩阵
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes, num_nodes))
        # 计算 K-hop 邻接：adj_t^K
        adj_power = adj_t.clone()
        for _ in range(K-1):
            adj_power = adj_power @ adj_t
        # adj_2 = adj_power @ adj_power
        # adj_4 = adj_2 @ adj_2
        # adj_8 = adj_4 @ adj_4
        # adj_16 = adj_8 @ adj_8
        # adj_power[i, j] 非零表示 j 是 i 的 10-hop 邻居
        # 获取邻居索引
        row, col, _ = adj_power.coo()  # 非零元素的行列坐标
        # 这里 (row[i], col[i]) 就是邻居对
        
        # 排序 row，保证同一个点的邻居连续
        sorted_row, perm = torch.sort(row)
        sorted_col = col[perm]

        # 计算每个顶点邻居数（row是已排序的）
        counts = torch.bincount(sorted_row, minlength=num_nodes)

        # 按 counts 分割 sorted_col，即邻居列表的连续区间
        neighbors_split = torch.split(sorted_col, counts.tolist())

        neighbor_counts = torch.tensor([len(n) for n in neighbors_split])
        max_neighbors = neighbor_counts.max().item()
        min_neighbors = neighbor_counts.min().item()
        # 找到小于 200 的顶点的个数
        num_small_nodes = (counts < 200).sum().item()
        assert min_neighbors > 200, f"min_neighbors={min_neighbors} < 200"

        return neighbors_split

    def run(self) -> None:
        mesh = DS_TriangleMesh.from_file(Path('data/dg-mesh/horse/mesh_gt/Brown_Horse0.obj'), read_mtl=False)
        num_nodes = mesh.num_vertices
        edge_index = self.face_to_edge_index(mesh.indices)
        self.get_k_hop_neighbors(edge_index, num_nodes, K=10)

# 可视化
# mesh_o3d = o3d.geometry.TriangleMesh()
# mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
# mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)

# lines = []
# for i, neighbors in enumerate(knn_indices[:10]):  # 展示前 10 个顶点的 kNN
#     for neighbor in neighbors:
#         lines.append([i, neighbor])

# line_set = o3d.geometry.LineSet()
# line_set.points = o3d.utility.Vector3dVector(vertices)
# line_set.lines = o3d.utility.Vector2iVector(lines)
# o3d.visualization.draw_geometries([mesh_o3d, line_set])        


if __name__ == '__main__':
    Test(cuda=0).run()
