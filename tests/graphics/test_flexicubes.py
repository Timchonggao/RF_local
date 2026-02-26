# from __future__ import annotations

# from dataclasses import dataclass
# import torch

# from rfstudio.engine.task import Task
# from rfstudio.graphics import FlexiCubes
# from rfstudio.visualization import Visualizer


# @dataclass
# class TestFlexiCubes(Task):

#     viser: Visualizer = Visualizer(port=6789)

#     def run(self) -> None:
#         # # ==========================================
#         # # 1. 原有测试逻辑 (32分辨率基础测试)
#         # # ==========================================
#         # print("[Test 1] Running basic FlexiCubes generation...")
#         # flexicubes = FlexiCubes.from_resolution(32)
        
#         # # 创建一个半径 0.8 的球体
#         # sphere_sdfs = flexicubes.vertices.norm(dim=-1, keepdim=True) - 0.8 # [..., 1]
        
#         # # 创建一个边长范围 0.9 的立方体
#         # cube_sdfs = (flexicubes.vertices.abs() - 0.9).max(-1, keepdim=True).values

#         # ==========================================
#         # 2. 新增测试逻辑 (Coarse-to-Fine 上采样测试)
#         # ==========================================
#         print("[Test 2] Running Upsample (Coarse-to-Fine) test...")
        
#         # A. 初始化一个非常低分辨率的网格 (Res=16)
#         low_res = 16
#         fc_low = FlexiCubes.from_resolution(low_res)
        
#         # B. 在低分辨率上定义一个较小的球体 (半径 0.5)
#         # 注意：这里我们在低分辨率下计算 SDF
#         low_sdf_values = fc_low.vertices.norm(dim=-1, keepdim=True) - 0.5
#         fc_low = fc_low.replace(sdf_values=low_sdf_values)
        
#         # C. 调用 upsample 函数 (假设已添加到类中) 提升到高分辨率 (Res=64)
#         target_res = 64
#         fc_high = fc_low.upsample(target_res)
        
#         # D. 验证数据形状
#         print(f"    Low Res Grid:  {fc_low.resolution.tolist()}")
#         print(f"    High Res Grid: {fc_high.resolution.tolist()}")
#         print(f"    High Res SDF shape: {fc_high.sdf_values.shape}")
        
#         # 断言检查 (确保分辨率正确)
#         assert torch.all(fc_high.resolution == torch.tensor([target_res, target_res, target_res], device=fc_high.resolution.device))
#         print("    >> Upsample shape check passed.")

#         # ==========================================
#         # 3. 可视化所有结果
#         # ==========================================
#         with self.viser.customize() as handle:
#             # # 可视化 1: 原始球体
#             # mesh_sphere = flexicubes.replace(sdf_values=sphere_sdfs).dual_marching_cubes()[0]
#             # handle['Basic/sphere_res32'].show(mesh_sphere).configurate(normal_size=0.05)
            
#             # # 可视化 2: 原始立方体
#             # mesh_cube = flexicubes.replace(sdf_values=cube_sdfs).dual_marching_cubes()[0]
#             # handle['Basic/cube_res32'].show(mesh_cube).configurate(normal_size=0.05)

#             # 可视化 3: 低分辨率球体
#             mesh = fc_low.dual_marching_cubes()[0]
#             handle['Upsample_Test/sphere_16'].show(mesh).configurate(normal_size=0.03)
            
#             # 可视化 4: 上采样后的球体
#             # 它是通过 16 分辨率的粗糙 SDF 插值得到的，然后再插值到 64 分辨率
#             mesh_upsampled = fc_high.dual_marching_cubes()[0]
#             handle['Upsample_Test/sphere_16_to_64'].show(mesh_upsampled).configurate(normal_size=0.03)
            
#             print(f"Visualization ready at port {self.viser.port}")


# if __name__ == '__main__':
#     # 确保你有 GPU，否则去掉 cuda=0
#     TestFlexiCubes(cuda=0).run()


from __future__ import annotations

from dataclasses import dataclass
import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import FlexiCubes, Points  # 假设 Points 在此处，如不在请修改
from rfstudio.visualization import Visualizer


@dataclass
class TestFlexiCubes(Task):

    viser: Visualizer = Visualizer(port=6789)

    def run(self) -> None:
        # ==========================================
        # 1. Coarse-to-Fine 上采样流程
        # ==========================================
        print("[Step 1] Running Upsample (Coarse-to-Fine)...")
        
        # A. 初始化低分辨率 (Res=16) 并定义球体
        low_res = 16
        fc_low = FlexiCubes.from_resolution(low_res)
        # low_sdf_values = fc_low.vertices.norm(dim=-1, keepdim=True) - 0.5
        center = torch.tensor([0.3, -0.2, 0.5], device=fc_low.vertices.device, dtype=fc_low.vertices.dtype)
        radius = 0.4
        dist = (fc_low.vertices - center).norm(dim=-1, keepdim=True)
        # SDF = 距离 - 半径
        low_sdf_values = dist - radius
        fc_low = fc_low.replace(sdf_values=low_sdf_values)
        
        # B. 提取低分辨率 Mesh (作为后续的查询源/GT)
        # 这里的顶点 mesh_low.vertices 将作为我们感兴趣的 "查询点"
        mesh_low = fc_low.dual_marching_cubes()[0]
        print(f"    Low Res Mesh Vertices: {mesh_low.vertices.shape[0]}")

        # C. 上采样到高分辨率 (Res=64)
        target_res = 64
        fc_high = fc_low.upsample(target_res)
        
        # 简单验证
        assert torch.all(fc_high.resolution == torch.tensor([target_res]*3, device=fc_high.resolution.device))
        print("    >> Upsample successful.")

        # ==========================================
        # 2. 体素查询与外扩分析 (Query & Dilation)
        # ==========================================
        print("[Step 2] Querying High-Res Cubes with Dilation...")
        
        # A. 准备查询点：使用低分辨率 Mesh 的所有顶点
        query_positions = mesh_low.vertices # [N, 3]
        
        # B. 在高分辨率 Grid 中查询
        # dilation=1 表示 Moore Neighborhood (包含中心及周围一圈，共 3x3x3 = 27 个子体素)
        # 函数内部会自动处理去重
        unique_indices = fc_high.query_cubes(
            query_positions, 
            dilation=1
        )
        print(f"    Query Points: {query_positions.shape[0]}")
        print(f"    Activated Cubes (Unique): {unique_indices.shape[0]}")
        
        # C. 提取体素几何信息用于可视化
        # unique_indices 是 Cube 的线性索引 [M]
        # fc_high.indices 存储了每个 Cube 对应的 8 个顶点的索引 [Total_Cubes, 8]
        # fc_high.vertices 存储了网格顶点的实际坐标 [Total_Verts, 3]
        
        # 1. 找到被激活 Cube 的 8 个角点的索引
        activated_cube_vertex_indices = fc_high.indices[unique_indices] # [M, 8]
        
        # 2. 获取角点的实际坐标
        activated_cube_corners = fc_high.vertices[activated_cube_vertex_indices] # [M, 8, 3]
        
        # 3. 展平以便 Point Visualizer 使用
        cube_vis_positions = activated_cube_corners.reshape(-1, 3)
        
        flat_indices = activated_cube_vertex_indices.view(-1)
        unique_vertex_indices = torch.unique(flat_indices, sorted=True) # [N_unique]
        unique_vertex_positions = fc_high.vertices[unique_vertex_indices] # [N_unique, 3]
        
        flexicube_corners = fc_high.get_grid_corners() # [Total_Cubes, 8, 3]
        breakpoint()

        # ==========================================
        # 可视化 (Visualization)
        # ==========================================
        print(f"[Step 3] Visualizing results at port {self.viser.port}...")
        
        with self.viser.customize() as handle:
            # 1. 基础 Mesh 对比
            handle['Comparison/Low_Res_Input'].show(mesh_low)
            
            mesh_high = fc_high.dual_marching_cubes()[0]
            handle['Comparison/High_Res_Upsampled'].show(mesh_high)

            # 2. Query 分析可视化
            handle['Analysis/Query_original_points'].show(
                Points(positions=query_positions),
            )
            handle['Analysis/Query_Dilation_points'].show(
                Points(positions=cube_vis_positions)
            )
            handle['Analysis/Query_Dilation_unique_vertex_points'].show(
                Points(positions=unique_vertex_positions)
            )
            handle['Analysis/Flexicubes_corners'].show(
                Points(positions=flexicube_corners)
            ).configurate(
                point_size=(0.2),
            )
            


if __name__ == '__main__':
    TestFlexiCubes(cuda=0).run()
