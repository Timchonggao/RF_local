from __future__ import annotations

# import modules
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Literal, Optional, Tuple, Type, TypeVar
from jaxtyping import Float32, Int64

import numpy as np
from pytorch3d.ops import knn_points
from torch_geometric.utils import geodesic_distance
import open3d as o3d
import torch
from torch import Tensor
import torch.nn.functional as F

# import rfstudio modules
from rfstudio.graphics._mesh._texture import Texture2D
from rfstudio.utils.lazy_module import dr, o3d
from rfstudio.utils.tensor_dataclass import Float, Long, Size, TensorDataclass

from rfstudio.graphics._mesh._triangle_mesh import (
    BaseImages, DepthImages,
    Points,
    get_uniform_normal_from_hemisphere, safe_normalize,
    diff_poisson_surface_recon,
    _merge_materials,
    Texture2D,
    BaseShader, ShadingContext
)
IMG = TypeVar('IMG', bound=BaseImages)
T = TypeVar('T', bound='BaseShader')

# import rfstudio_ds modules
from ._obj import _load_obj
from .._cameras import DS_Cameras


@dataclass
class DS_TriangleMesh(TensorDataclass):
    '''
    Inherit from DS_TriangleMesh
        1. rewrite “from file” function to support read_mtl option
        2. support mesh simplify function to simplify mesh
        3. add vertex sdf flow attribute
        4. add vertex sdf flow to sceneflow calculation function
    '''

    num_vertices: int = Size.Dynamic
    num_faces: int = Size.Dynamic

    vertices: Tensor = Float[num_vertices, 3]
    indices: Tensor = Long[num_faces, 3]

    vertices_sdf_flow: Optional[Tensor] = Float[num_vertices, 1]
    vertices_scene_flow: Optional[Tensor] = Float[num_vertices, 3]
    face_scene_flow: Optional[Tensor] = Float[num_faces, 3, 3]

    normals: Optional[Tensor] = Float[num_vertices, 3]
    face_normals: Optional[Tensor] = Float[num_faces, 3, 3]
    uvs: Optional[Tensor] = Float[num_faces, 3, 2]
    kd: Optional[Texture2D] = Texture2D[...]
    ks: Optional[Texture2D] = Texture2D[...]

    @staticmethod
    def from_poisson_reconstruction(oriented_points: Points, *, depth: int = 8) -> DS_TriangleMesh:
        assert oriented_points.normals is not None
        positions = oriented_points.positions.detach().cpu().numpy()
        normals = oriented_points.normals.detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(positions)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        triangle_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)[0]
        vertices = torch.from_numpy(np.asarray(triangle_mesh.vertices)).float().to(oriented_points.device)
        indices = torch.from_numpy(np.asarray(triangle_mesh.triangles)).long().to(vertices.device)
        return DS_TriangleMesh(vertices=vertices, indices=indices)

    @staticmethod
    def from_diff_poisson_reconstruction(
        oriented_points: Points,
        *,
        resolution: int = 512,
        smoothness: float = 0.02,
    ) -> DS_TriangleMesh:
        assert oriented_points.normals is not None
        V, F, N = diff_poisson_surface_recon(
            oriented_points.positions,
            oriented_points.normals,
            resolution=resolution,
            smoothness=smoothness,
        )
        return DS_TriangleMesh(
            vertices=V,
            indices=F,
            normals=N,
        )

    @staticmethod
    def from_depth_fusion(
        depths: DepthImages,
        *,
        cameras: DS_Cameras,
        voxel_size: float = 0.01,
        sdf_trunc: float = 0.05,
        depth_scale: float = 1.0,
        depth_trunc: float = 4.0,
        alpha_trunc: float = 0.5,
        progress_handle: Optional[Callable] = None,
    ) -> DS_TriangleMesh:
        """
        Generate a DS_TriangleMesh using TSDF fusion from a set of cameras and corresponding depth images.

        Args:
            cameras (DS_Cameras): A DS_Cameras object containing camera parameters (intrinsics and extrinsics).
            depths (DepthImages): A DepthImages object containing depth maps captured by the cameras.

        Returns:
            DS_TriangleMesh: A triangle mesh generated from TSDF fusion.
        """
        assert len(cameras) == len(depths), "Number of cameras must match the number of depth images."

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        )

        cameras = cameras.detach().cpu()
        intrinsics = cameras.intrinsic_matrix.numpy()
        extrinsics = cameras.view_matrix.numpy()
        Ws = cameras.width.numpy()
        Hs = cameras.height.numpy()
        if progress_handle is not None:
            Ws = progress_handle(Ws)
        for W, H, intrin, extrin, depth_img in zip(Ws, Hs, intrinsics, extrinsics, depths.detach().cpu(), strict=True):
            depth_image = o3d.geometry.Image((depth_img[..., :1] * (depth_img[..., 1:] > alpha_trunc).float()).numpy())

            o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic()
            o3d_intrinsics.set_intrinsics(
                width=W,
                height=H,
                fx=intrin[0, 0],
                fy=intrin[1, 1],
                cx=intrin[0, 2],
                cy=intrin[1, 2],
            )

            volume.integrate(
                o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.zeros_like(depth_image)),  # No color, only depth
                    depth_image,
                    depth_scale=depth_scale,
                    depth_trunc=depth_trunc,
                    convert_rgb_to_intensity=False,
                ),
                o3d_intrinsics,
                extrin,
            )

        mesh = volume.extract_triangle_mesh()

        vertices = torch.from_numpy(np.asarray(mesh.vertices)).float().to(depths.device)
        indices = torch.from_numpy(np.asarray(mesh.triangles)).long().to(depths.device)

        return DS_TriangleMesh(vertices=vertices, indices=indices)

    @staticmethod
    def merge(*meshes: DS_TriangleMesh, only_geometry: bool = False) -> DS_TriangleMesh:
        vertices = []
        indices = []
        tfaces = []
        mfaces = []
        vcnt = 0
        for idx, mesh in enumerate(meshes):
            vertices.append(mesh.vertices)
            indices.append(mesh.indices + vcnt)
            if mesh.uvs is not None:
                tfaces.append(mesh.uvs)
            assert len(tfaces) in [idx + 1, 0]
            if only_geometry:
                continue
            if mesh.kd is not None:
                mfaces.append(mesh.indices.new_empty(mesh.num_faces).fill_(idx))
            assert len(mfaces) in [idx + 1, 0]
            vcnt += mesh.vertices.shape[0]
        vertices = torch.cat(vertices, dim=0)
        indices = torch.cat(indices, dim=0)
        tfaces = torch.cat(tfaces, dim=0) if tfaces else None
        mfaces = torch.cat(mfaces, dim=0) if mfaces else None
        assert vcnt == vertices.shape[0] and indices.max().item() + 1 == vcnt
        if only_geometry or mfaces is None:
            return DS_TriangleMesh(vertices=vertices, indices=indices, uvs=tfaces)
        kd, ks, tfaces = _merge_materials([(m.kd, m.ks) for m in meshes], tfaces=tfaces, mfaces=mfaces)
        return DS_TriangleMesh(vertices=vertices, indices=indices, uvs=tfaces, kd=kd, ks=ks)

    @staticmethod
    def create_sphere(*, radius: float = 1.0, resolution: int = 64) -> DS_TriangleMesh:
        triangle_mesh = o3d.geometry.TriangleMesh.create_sphere(
            radius=radius,
            resolution=resolution,
        )
        vertices = torch.from_numpy(np.asarray(triangle_mesh.vertices)).float()
        indices = torch.from_numpy(np.asarray(triangle_mesh.triangles)).long()
        return DS_TriangleMesh(
            vertices=vertices,
            indices=indices,
        )

    @staticmethod
    def create_cube(*, size: float = 1.0) -> DS_TriangleMesh:
        triangle_mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
        vertices = torch.from_numpy(np.asarray(triangle_mesh.vertices)).float() - size / 2
        indices = torch.from_numpy(np.asarray(triangle_mesh.triangles)).long()
        return DS_TriangleMesh(
            vertices=vertices,
            indices=indices,
        )

    @staticmethod
    def from_file(
        file: Path,
        *,
        kd_texture: Optional[Texture2D] = None,
        ks_texture: Optional[Texture2D] = None,
        read_mtl: Optional[bool] = None, # some obj's mtl lost, so we ban mtl read. default is None.
    ) -> DS_TriangleMesh:
        assert file.exists() and file.is_file()
        kd, ks = None, None
        mtl_override = (kd_texture is not None or ks_texture is not None)
        if file.suffix == '.ply':
            triangle_mesh = o3d.io.read_triangle_mesh(str(file))
            vertices = torch.from_numpy(np.asarray(triangle_mesh.vertices)).float()
            indices = torch.from_numpy(np.asarray(triangle_mesh.triangles)).long()
            assert len(indices) > 0
            if triangle_mesh.has_vertex_normals and len(triangle_mesh.vertex_normals) > 0:
                normals = torch.from_numpy(np.asarray(triangle_mesh.vertex_normals)).float()
            else:
                normals = None
            if triangle_mesh.has_triangle_normals and len(triangle_mesh.triangle_normals) > 0:
                face_normals = torch.from_numpy(np.asarray(triangle_mesh.triangle_normals)).float()
            else:
                face_normals = None
            uvs = None
        elif file.suffix == '.obj':
            normals = None
            vertices, indices, face_normals, uvs, kd, ks = _load_obj(file, mtl_override=mtl_override, read_mtl=read_mtl)
        else:
            raise ValueError(file.suffix)
        if mtl_override:
            kd, ks = kd_texture, ks_texture
        return DS_TriangleMesh(
            vertices=vertices[..., :3],
            indices=indices,
            normals=normals,
            face_normals=face_normals,
            uvs=uvs,
            kd=kd,
            ks=ks,
        )

    @torch.no_grad()
    def export(self, path: Path, *, only_geometry: bool = False) -> None:
        assert path.suffix in ['.obj', '.ply']
        if not only_geometry:
            raise NotImplementedError
        path.parent.mkdir(exist_ok=True, parents=True)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices.detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(self.indices.detach().cpu().numpy())
        if self.normals is not None:
            mesh.triangle_normals = o3d.utility.Vector3dVector(self.normals.detach().cpu().numpy())
        o3d.io.write_triangle_mesh(str(path), mesh)

    def build_texture_from_tensors_(
        self,
        tensors: Tensor,
        *,
        attrs: Literal['face', 'vertex', 'flat'],
        target: Literal['kd', 'ks'] = 'kd',
    ) -> DS_TriangleMesh:
        if attrs == 'flat':
            assert tensors.shape == (self.num_faces, 3)
            size = 1 << (torch.tensor(self.num_faces).log() / torch.tensor(4).log()).ceil().int().item()
            texture = tensors.new_zeros(size * size, 3)
            texture[:self.num_faces, :] = tensors
            texture = texture.view(size, size, 3)
            rng = torch.arange(self.num_faces, device=self.device)
            us = (rng % size) / (size - 1)
            vs = (rng // size) / (size - 1)
            uvs = torch.stack((us, vs), dim=-1).view(-1, 1, 2).repeat(1, 3, 1)
        elif attrs == 'vertex':
            assert tensors.shape == (self.num_vertices, 3)
            raise NotImplementedError
        elif attrs == 'face':
            assert tensors.shape == (self.num_faces, 3, 3)
            raise NotImplementedError
        else:
            raise ValueError(attrs)
        replaced = { 'uvs': uvs, target: Texture2D(data=texture) }
        return self.replace_(**replaced)

    def visible_faces(self, camera: DS_Cameras) -> Int64[Tensor, "N"]:
        assert camera.is_cuda and camera.shape == ()

        V = self.num_vertices
        ctx = dr.RasterizeCudaContext(camera.device)
        vertices = torch.cat((
            self.vertices,
            torch.ones_like(self.vertices[..., :1]),
        ), dim=-1).view(V, 4, 1)                          # [V, 4, 1]

        resolution = [camera.height.item(), camera.width.item()]
        mvp = camera.projection_matrix @ camera.view_matrix # [4, 4]
        projected = (mvp @ vertices).view(1, V, 4) # [1, V, 4]
        with dr.DepthPeeler(ctx, projected, self.indices.int(), resolution=resolution) as peeler:
            rast, _ = peeler.rasterize_next_layer() # [1, H, W, 4]
        visible: Tensor = (rast[..., -1:][rast[..., -1:] > 0].long() - 1)
        return visible.unique()

    def render(self, cameras: DS_Cameras, *, shader: BaseShader[IMG]) -> IMG:
        assert cameras.is_cuda

        V = self.num_vertices
        ctx = dr.RasterizeCudaContext(cameras.device)
        vertices = torch.cat((
            self.vertices,
            torch.ones_like(self.vertices[..., :1]),
        ), dim=-1).view(V, 4, 1)                          # [V, 4, 1]

        context = ShadingContext(mesh=self)

        if shader.culling:
            culling_normals, face_centers = context._culling_info

        images = []
        for camera in cameras.view(-1): # todo 思考如何 mini batch render: https://chatgpt.com/c/6856a491-e61c-8009-a2a8-b61ab2da612b
            camera_pos = camera.c2w[:, 3]
            resolution = [camera.height.item(), camera.width.item()]
            mvp = camera.projection_matrix @ camera.view_matrix # [4, 4]
            projected = (mvp @ vertices).view(1, V, 4) # [1, V, 4]

            context._clear()
            context._camera = camera
            if shader.culling:
                context._visibilities = ((camera_pos - face_centers) * culling_normals).sum(-1) > 0 # [F]
                indices = self.indices[context._visibilities, :].int()
            else:
                context._visibilities = None
                indices = self.indices.int()

            with dr.DepthPeeler(ctx, projected, indices, resolution=resolution) as peeler:
                rast, _ = peeler.rasterize_next_layer() # [1, H, W, 4]
                context._rast = rast

            colors = shader.shade(context)
            alphas = (rast[..., -1:] > 0).float() # [1, H, W, 1]
            rgba = torch.cat((colors, alphas), dim=-1)
            if shader.antialias:
                rgba = dr.antialias(rgba, rast, projected, indices)
            images.append(rgba.squeeze(0))
        return shader.get_image_class()(images)

    def compute_face_normals(self, *, fix: bool = False) -> DS_TriangleMesh:
        assert self.face_normals is None
        return self.replace().compute_face_normals_(fix=fix)

    def compute_vertex_normals(self, *, fix: bool = False) -> DS_TriangleMesh:
        assert self.normals is None
        return self.replace().compute_vertex_normals_(fix=fix)

    def compute_face_normals_(self, *, fix: bool = False) -> DS_TriangleMesh:
        assert self.face_normals is None
        F = self.num_faces
        expanded_inds = self.indices.view(F, 3, 1).expand(F, 3, 3) # [F, 3, 3]
        vertices = self.vertices.gather(
            dim=-2,
            index=expanded_inds.view(-1, 3)                        # [3F, 3]
        ).view(F, 3, 3)                                            # [F, 3, 3]

        normals = torch.cross(
            vertices[:, 1, :] - vertices[:, 0, :],
            vertices[:, 2, :] - vertices[:, 0, :],
            dim=-1,
        ) # [F, 3]
        normal_lengths = normals.norm(dim=-1, keepdim=True)        # [V, 1]
        if not fix:
            assert (normal_lengths > 1e-10).all()
            normals = normals / normal_lengths
        else:
            fixing = torch.tensor([0, 0, 1]).to(normals)
            normals = torch.where(normal_lengths > 1e-10, normals / normal_lengths.clamp_min(1e-10), fixing)
        return self.annotate_(face_normals=normals.view(F, 1, 3).expand(F, 3, 3))

    def compute_vertex_normals_(self, *, fix: bool = False) -> DS_TriangleMesh:
        assert self.normals is None
        F = self.num_faces
        expanded_inds = self.indices.view(F, 3, 1).expand(F, 3, 3) # [F, 3, 3]
        vertices = self.vertices.gather(
            dim=-2,
            index=expanded_inds.view(-1, 3)                        # [3F, 3]
        ).view(F, 3, 3)                                            # [F, 3, 3]
        weighted_face_normals = torch.cross(
            vertices[:, 1, :] - vertices[:, 0, :],
            vertices[:, 2, :] - vertices[:, 0, :],
            dim=-1,
        ).unsqueeze(-2).expand(expanded_inds.shape)                # [F, 3, 3]
        normals = torch.zeros_like(self.vertices)                  # [V, 3]
        normals.scatter_add_(
            dim=-2,
            index=expanded_inds.flatten(-3, -2),                   # [3F, 3]
            src=weighted_face_normals.flatten(-3, -2),             # [3F, 3]
        )
        normal_lengths = normals.norm(dim=-1, keepdim=True)        # [V, 1]
        if not fix:
            assert (normal_lengths > 1e-10).all()
            normals = normals / normal_lengths
        else:
            fixing = torch.tensor([0, 0, 1]).to(normals)
            normals = torch.where(normal_lengths > 1e-10, normals / normal_lengths.clamp_min(1e-10), fixing)
        return self.annotate_(normals=normals)

    @torch.no_grad()
    def compute_ambient_occlusion(self, cameras: Optional[DS_Cameras] = None) -> Float32[Tensor, "F"]:
        assert self.is_cuda

        if cameras is None:
            cameras = DS_Cameras.from_sphere(
                center=(0, 0, 0),
                up=(0, 0, 1),
                radius=3.,
                num_samples=128,
                resolution=(1024, 1024),
                device=self.device
            )

        V = self.num_vertices
        F = self.num_faces
        ctx = dr.RasterizeCudaContext(cameras.device)
        vertices = torch.cat((
            self.vertices,
            torch.ones_like(self.vertices[..., :1]),
        ), dim=-1).view(V, 4, 1)                          # [V, 4, 1]

        context = ShadingContext(mesh=self)

        culling_normals, face_centers = context._culling_info

        face_visibilities = torch.zeros(F, device=cameras.device) # [F]
        face_counts = torch.zeros(F, device=cameras.device) # [F]
        for camera in cameras.view(-1):
            camera_pos = camera.c2w[:, 3]
            resolution = [camera.height.item(), camera.width.item()]
            mvp = camera.projection_matrix @ camera.view_matrix # [4, 4]
            projected = (mvp @ vertices).view(1, V, 4) # [1, V, 4]
            view_directions = safe_normalize(camera_pos - face_centers) # [F, 3]

            context._clear()
            context._camera = camera
            context._visibilities = (view_directions * culling_normals).sum(-1) > 0 # [F]
            indices = self.indices[context._visibilities, :].int()

            with dr.DepthPeeler(ctx, projected, indices, resolution=resolution) as peeler:
                rast, _ = peeler.rasterize_next_layer() # [1, H, W, 4]
                context._rast = rast

            depths = torch.mul(
                context.camera.c2w[:3, 3] - context.global_positions,
                context.camera.c2w[:3, 2]
            ).sum(-1).flatten() # [H * W]

            sample_image_space = camera.project(face_centers) # [F, 3]
            sample_depths = sample_image_space[..., 2] # [F]
            H, W = rast.shape[1:3]
            xs = (sample_image_space[..., 0] + 0.5).long() # [F]
            ys = (sample_image_space[..., 1] + 0.5).long() # [F]
            depths = depths[xs.clamp(min=0, max=W - 1) + ys.clamp(min=0, max=H - 1) * W] # [F]
            valid = (context.alphas > 0).flatten()[xs.clamp(min=0, max=W - 1) + ys.clamp(min=0, max=H - 1) * W] # [F]
            valid = valid & (0 <= xs) & (xs < W) & (0 <= ys) & (ys < H) # [F]
            non_occluded = ((sample_depths < depths + 0.02) & valid).float() # [F]

            weights = (culling_normals * view_directions).sum(-1).clamp_min(1e-4) # [F]
            face_counts += valid.float() * weights
            face_visibilities += non_occluded * weights
        return face_visibilities / face_counts.clamp_min(1e-6)

    def subdivide(self) -> DS_TriangleMesh:
        F = self.num_faces
        V = self.num_vertices

        # with self.create_geometric_processing_context() as GP:
        #     sum_neighbors = GP.Vertex.scatter(GP.HalfEdge.VertexFrom.value(self.vertices)) # [V, 3]
        #     cnt_neighbors = GP.Vertex.scatter(GP.HalfEdge.VertexFrom.constant(1).long()) # [V, 1]
        #     inserted_vertices = GP.Vertex.gather(
        #         GP.UniqueEdge.VertexWinged.value(self.vertices) * 0.125,
        #         GP.UniqueEdge.VertexEnded.value(self.vertices) * 0.375,
        #     ) # [E, 3]
        #     new_indices = GP.Face.collect(GP.Vertex.index(), GP.UniqueEdge.index()) # [F, 6]

        with torch.no_grad():
            edges = self.indices[:, [1, 2, 2, 0, 0, 1]].view(F * 3, 2) # [F * 3, 2]
            sum_neighbors = self.vertices.scatter_reduce(
                dim=0,
                index=edges[:, 0:1].expand(F * 3, 3),                  # [F * 3, 3]
                src=self.vertices[edges[:, 1], :],                     # [F * 3, 3]
                reduce='sum',
                include_self=False,
            )                                                          # [V, 3]
            cnt_neighbors = edges.new_empty(V).scatter_reduce(
                dim=0,
                index=self.indices.flatten(),                          # [F * 3]
                src=torch.ones_like(edges[:, 0]),                      # [F * 3]
                reduce='sum',
                include_self=False,
            ).view(V, 1)                                               # [V, 1]
            assert cnt_neighbors.min().item() >= 3
            weights = torch.where(cnt_neighbors == 3, 7 / 16, 5 / 8)   # [V, 1]
            updated_vertices = torch.add(
                weights * self.vertices,
                (1 - weights) * (sum_neighbors / cnt_neighbors),
            )                                                          # [V, 3]

            edge_code = torch.stack((
                edges.min(1).values,
                edges.max(1).values,
            ), dim=-1)                                                 # [F * 3]
            unique_edges, inverse_indices = edge_code.unique(
                dim=0,
                return_inverse=True,
            )                                                          # [E, 3], [F * 3]\in[0, E)

            assert unique_edges.shape[0] * 2 == F * 3, "Not a manifold."

            inserted_vertices = self.vertices.new_zeros(unique_edges.shape[0], 3) # [E, 3]
            vertices = self.vertices[self.indices.view(F * 3), :].view(F, 3, 3)   # [F, 3, 3]
            wing_sum = 3 * vertices.sum(-2, keepdim=True) - vertices              # [F, 3, 3]
            inserted_vertices.scatter_add_(
                dim=0,
                index=inverse_indices.view(-1, 1).expand(F * 3, 3),               # [F * 3, 3]
                src=wing_sum.view(F * 3, 3) / 16,                                 # [F * 3, 3]
            )
            expanded_indices = torch.cat((
                self.indices,
                inverse_indices.view(F, 3) + V,
            ), dim=-1)                                                            # [F, 6]

        return DS_TriangleMesh(
            vertices=torch.cat((updated_vertices, inserted_vertices), dim=0),
            indices=expanded_indices[:, [0, 5, 4, 4, 3, 2, 3, 4, 5, 5, 1, 3]].view(F * 4, 3)
        )

    @torch.no_grad()
    def simplify(self, target_num_faces: int) -> DS_TriangleMesh:
        """
        使用 Open3D 的二次误差度量 (Quadric Error Metrics) 算法简化网格。

        Args:
            target_num_faces (int): 简化后网格的目标面数。该值必须小于当前网格的面数。

        Returns:
            DS_TriangleMesh: 一个包含简化后几何体的新 DS_TriangleMesh 对象。
            注意：在此实现中，原始网格的法线、UV坐标和纹理会被丢弃。
        """
        if not isinstance(target_num_faces, int) or target_num_faces <= 0:
            raise ValueError("target_num_faces 必须是一个正整数。")

        if target_num_faces >= self.num_faces:
            print(
                f"警告: 目标面数 ({target_num_faces}) 不小于当前面数 ({self.num_faces})。"
                f"将返回原始网格的副本。"
            )
            # 使用 .replace() 创建一个具有相同张量的副本
            return self.replace()

        print(f"开始简化网格，从 {self.num_faces} 个面减少到约 {target_num_faces} 个面...")

        o3d_mesh = o3d.geometry.TriangleMesh()
        # 将数据传递给 Open3D，需要在 CPU 上并使用 NumPy 数组
        o3d_mesh.vertices = o3d.utility.Vector3dVector(self.vertices.detach().cpu().numpy())
        o3d_mesh.triangles = o3d.utility.Vector3iVector(self.indices.detach().cpu().numpy())

        # 可选：在简化前清理网格可以提高稳定性
        o3d_mesh.remove_degenerate_triangles() # 移除退化三角形
        o3d_mesh.remove_duplicated_vertices() # 移除重复顶点
        o3d_mesh.remove_duplicated_triangles() # 移除重复三角形
        o3d_mesh.remove_non_manifold_edges() # 移除非流形边 (QEM算法通常需要流形网格)

        # simplify_quadric_decimation 函数返回简化后的网格
        simplified_o3d_mesh = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_num_faces
        )

        actual_num_faces = len(simplified_o3d_mesh.triangles)
        print(f"简化完成，实际面数为 {actual_num_faces}。")

        # 将结果转换回 NumPy 数组
        new_vertices_np = np.asarray(simplified_o3d_mesh.vertices)
        new_indices_np = np.asarray(simplified_o3d_mesh.triangles)

        # 创建 PyTorch 张量，保持原始的数据类型(dtype)和设备(device)
        new_vertices = torch.from_numpy(new_vertices_np).to(dtype=self.vertices.dtype, device=self.device)
        new_indices = torch.from_numpy(new_indices_np).to(dtype=self.indices.dtype, device=self.device)

        # 创建只包含基本几何信息的新对象
        return DS_TriangleMesh(vertices=new_vertices, indices=new_indices)

    def normalize(self, *, scale: float = 1.0) -> DS_TriangleMesh:
        max_bound = self.vertices.view(-1, 3).max(0).values
        min_bound = self.vertices.view(-1, 3).min(0).values
        assert (max_bound > min_bound).all()
        center = (max_bound + min_bound) / 2
        scale = 2 * scale / (max_bound - min_bound).max()
        return self.replace(vertices=(self.vertices - center) * scale)

    def translate(self, x: float, y: float, z: float, /) -> DS_TriangleMesh:
        return self.replace(vertices=self.vertices + torch.tensor([x, y, z]).to(self.vertices))

    def poisson_disk_sample(self, num_samples: int) -> Points:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices.detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(self.indices.detach().cpu().int().numpy())
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_poisson_disk(num_samples)
        return Points(
            positions=torch.from_numpy(np.asarray(pcd.points)).to(self.vertices),
            colors=None,
            normals=torch.from_numpy(np.asarray(pcd.normals)).to(self.vertices)
        )

    def uniformly_sample(
        self,
        num_samples: int,
        *,
        samples_per_face: Literal['random', 'uniform', 'uniform+nonzero'] = 'random',
        samples_in_face: Literal['random', 'r2'] = 'random',
        probs_per_face: Literal['uniform', 'area'] = 'area',
    ) -> Points:
        F = self.num_faces
        S = num_samples
        vertices = self.vertices.gather(
            dim=-2,
            index=self.indices.view(*self.shape, F * 3, 1).expand(*self.shape, F * 3, 3) # [..., 3F, 3]
        ).view(*self.shape, F, 3, 3)                                                     # [..., F, 3, 3]
        cross = torch.cross(
            vertices.detach()[..., 1, :] - vertices.detach()[..., 0, :],
            vertices.detach()[..., 2, :] - vertices.detach()[..., 0, :],
            dim=-1
        )                                                                                # [..., F, 3]
        areas_2x: Tensor = cross.norm(dim=-1)                                            # [..., F]
        if probs_per_face == 'area':
            probabilities = areas_2x / areas_2x.sum(-1, keepdim=True)                        # [..., F]
        else:
            assert probs_per_face == 'uniform' and samples_per_face in ['uniform', 'uniform+nonzero']
            probabilities = torch.empty_like(areas_2x).fill_(1.0 / F + 1e-12)
        if self.face_normals is None:
            face_normals = (cross / areas_2x[..., None]).unsqueeze(-2).repeat_interleave(3, dim=-2).contiguous()
        else:
            face_normals = self.face_normals

        if samples_per_face == 'random':
            dist = torch.distributions.Categorical(probabilities)
            sampled_indices = dist.sample([S]).view(S, -1).t().reshape(*self.shape, S)                # [..., S]
        elif samples_per_face in ['uniform', 'uniform+nonzero']:
            ideal_counts = probabilities * num_samples
            counts = ideal_counts.floor().long()                                                      # [..., F]
            if samples_per_face == 'uniform+nonzero' and (counts == 0).any():
                at_least = (1 / probabilities.min()).ceil().long().item()
                raise RuntimeError(
                    'To make sure each face has at least one sample, '
                    f'you should set `num_samples` to at least {at_least} (currently {num_samples}).'
                )
            fracs = ideal_counts - counts
            R = S - counts.sum().item()
            rng = torch.arange(F, device=self.device)
            if R > 0:
                dist = torch.distributions.Categorical(fracs)
                random_parts = dist.sample([R]).view(R, -1).t().reshape(*self.shape, R)               # [..., R]
                uniform_parts = rng.repeat_interleave(counts.view(-1)).view(*self.shape, S - R)       # [..., S-R]
                sampled_indices = torch.cat((uniform_parts, random_parts), dim=-1)                    # [..., S]
            else:
                sampled_indices = rng.repeat_interleave(counts.view(-1)).view(*self.shape, S)         # [..., S]
        else:
            raise ValueError('Argument `samples_per_face` must be any of random, uniform and uniform+nonzero')

        if samples_in_face == 'random':
            u = torch.rand(sampled_indices.shape, device=sampled_indices.device)
            v = torch.rand(sampled_indices.shape, device=sampled_indices.device).sqrt()
            # Barycentric coordinates for sampling within the triangles
            weights = torch.stack((v * (1 - u), u * v, 1 - v), dim=-1) # [..., S, 3]

        elif samples_in_face == 'r2':
            # See https://extremelearning.com.au/evenly-distributing-points-in-a-triangle/
            G = 1.3247179572447460259609088
            rng = torch.arange(num_samples, device=sampled_indices.device)
            u = (rng / G).frac()
            v = (rng / (G * G)).frac()
            flip = u + v > 1
            # Barycentric coordinates for sampling within the triangles
            u = torch.where(flip, 1 - u, u)
            v = torch.where(flip, 1 - v, v)
            weights = torch.stack((u, v, 1 - u - v), dim=-1)           # [..., S, 3]
        else:
            raise ValueError('Argument `samples_in_face` must be any of random and r2')

        sampled_vertices = vertices.gather(
            dim=-3,
            index=sampled_indices[..., None, None].expand(*self.shape, S, 3, 3),  # [..., S, 1, 1]
        )                                                                         # [..., S, 3, 3]
        positions = (sampled_vertices * weights[..., None]).sum(-2)               # [..., S, 3]
        colors = None
        sampled_normals = face_normals.gather(
            dim=-3,
            index=sampled_indices[..., None, None].expand(*self.shape, S, 3, 3),
        )                                                                         # [..., S, 3]
        normals = (sampled_normals * weights[..., None]).sum(-2)

        return Points(
            positions=positions,
            colors=colors,
            normals=normals / normals.norm(dim=-1, keepdim=True),
        )

    def spatial_aggregation(
        self,
        *,
        cameras: DS_Cameras,
        images: Float32[Tensor, "B H W C"],
        culling: bool = True,
    ) -> Tuple[Float32[Tensor, "N 3"], Float32[Tensor, "N C"]]:
        assert cameras.is_cuda

        V = self.num_vertices
        ctx = dr.RasterizeCudaContext(cameras.device)
        vertices = torch.cat((
            self.vertices,
            torch.ones_like(self.vertices[..., :1]),
        ), dim=-1).view(V, 4, 1)                          # [V, 4, 1]

        context = ShadingContext(mesh=self)

        if culling:
            culling_normals, face_centers = context._culling_info

        positions = []
        values = []
        C = images.shape[-1]
        for camera, value_map in zip(cameras.view(-1), images, strict=True):
            camera_pos = camera.c2w[:, 3]
            resolution = [camera.height.item(), camera.width.item()]
            mvp = camera.projection_matrix @ camera.view_matrix # [4, 4]
            projected = (mvp @ vertices).view(1, V, 4) # [1, V, 4] 所有顶点在当前摄像机视角下的投影坐标。

            context._clear()
            context._camera = camera # 清除之前帧的信息，重新绑定当前摄像机
            if culling:
                context._visibilities = ((camera_pos - face_centers) * culling_normals).sum(-1) > 0 # [F] 判断点乘结果为正，即法线朝向摄像机
                indices = self.indices[context._visibilities, :].int() # 只保留可见面片的
            else:
                context._visibilities = None
                indices = self.indices.int()

            if indices.shape[0] == 0:
                continue

            with dr.DepthPeeler(ctx, projected, indices, resolution=resolution) as peeler:
                # projected 和 indices 决定了要渲染的几何体。dr.DepthPeeler是深度优先的光栅化器，用于找出“最近”的面片
                rast, _ = peeler.rasterize_next_layer() # [1, H, W, 4] 光栅化结果，最后一个维度通常为 [u, v, z, id]，其中 id > 0 表示有效像素
                context._rast = rast # 保存当前光栅结果到上下文（供之后用）

            pos_map = context.global_positions.view(-1, 3) # 每个像素在世界坐标中的位置
            value_map = value_map.view(-1, C) # error map 的值
            mask = (rast[..., -1:] > 0).flatten() # rast[..., -1:] 是 id，值大于 0 表示该像素覆盖在某个面片上
            positions.append(pos_map[mask])
            values.append(value_map[mask]) # 用 mask 筛选有效的3D坐标和图像特征

        return torch.cat(positions), torch.cat(values)

    @torch.no_grad()
    def compute_euclidean_knn_pytorch3d(self, k: int) -> Tuple[Tensor, Tensor]:
        # 确保 vertices 形状为 [1, V, 3]（batch 维度）
        vertices = self.vertices.unsqueeze(0)  # [1, V, 3]
        
        # 使用 knn_points 计算 k 个最近邻（包括自身）
        knn_result = knn_points(
            vertices, vertices, K=k, return_sorted=True, norm=2
        )
        indices = knn_result.idx[0, :, :].long()  # [V, k]
        distances = knn_result.dists[0, :, :].sqrt()  # [V, k]; knn_points 返回平方距离, 所以这里需要开方
        
        return distances, indices  # 返回距离（非平方）和索引

    @torch.no_grad()
    def compute_geodesic_knn_pytorch3d(self, k: int) -> Tuple[Tensor, Tensor]:
        pass

    @torch.no_grad()
    def clean_knn_results(self, knn_distances, knn_indices, max_dist, min_valid_k_neighbors):
        """
        清洗 KNN 结果：替换非法邻居，并标记有效邻居不足的点。

        参数:
            knn_distances: [V, k] 距离张量
            knn_indices: [V, k] 索引张量
            max_dist: float，最大允许距离
            min_valid_k_neighbors: int，最小有效邻居数量

        返回:
            cleaned_distances: [V, k]
            cleaned_indices: [V, k]
            valid_mask_per_point: [V]，bool，True 表示该点有效
        """
        V, k = knn_distances.shape
        valid_mask = knn_distances <= max_dist                     # [V, k]
        valid_counts = valid_mask.sum(dim=1)                       # [V]

        # 标记哪些点是有效的（有效邻居数 >= 阈值）
        valid_point_mask = valid_counts >= min_valid_k_neighbors   # [V]

        # 初始化输出
        cleaned_distances = knn_distances.clone()
        cleaned_indices = knn_indices.clone()

        # 替换非法邻居：用当前点最大有效邻居代替
        temp_distances = knn_distances.clone()
        temp_distances[~valid_mask] = -1e6                         # 标记非法

        max_valid_distances, max_valid_indices = temp_distances.max(dim=1)
        max_valid_indices_exp = max_valid_indices.unsqueeze(1).expand_as(knn_indices)
        max_index_values = torch.gather(knn_indices, 1, max_valid_indices_exp)

        max_dist_expand = max_valid_distances.unsqueeze(1).expand_as(knn_distances)
        cleaned_distances[~valid_mask] = max_dist_expand[~valid_mask]
        cleaned_indices[~valid_mask] = max_index_values[~valid_mask]

        # 对无效点全部设为无效标记
        cleaned_distances[~valid_point_mask] = 0.0
        cleaned_indices[~valid_point_mask] = -1

        return cleaned_distances, cleaned_indices, valid_point_mask

    def solve_least_squares_velocity(
        self,
        A: torch.Tensor,            # [V, k, 6]
        b: torch.Tensor,            # [V, k, 1]
        method: Literal["solve", "lstsq", "pinv"] = "pinv",
        regularization: float = 1e-6
    ) -> torch.Tensor:
        """
        求解最小二乘问题 Ax = b，支持不同解法，返回 [V, 6, 1] 的解向量。

        Args:
            A: 设计矩阵 [V, k, 6]，每个点的局部约束。
            b: 目标值 [V, k, 1]，对应的 SDF flow。
            method: 解法类型：'solve'（默认）、'lstsq'、'pinv'。
            regularization: 正则化强度，仅对 'solve' 有效。

        Returns:
            velocities: 解向量 [V, 6, 1]
        """
        if method == "solve":
            AtA = torch.matmul(A.transpose(1, 2), A)  # [V, 6, 6]
            Atb = torch.matmul(A.transpose(1, 2), b)  # [V, 6, 1]
            lambda_eye = regularization * torch.eye(6, device=A.device).unsqueeze(0)  # [1, 6, 6]
            AtA_reg = AtA + lambda_eye
            velocities = -torch.linalg.solve(AtA_reg, Atb)  # [V, 6, 1]

        elif method == "lstsq":
            velocities = -torch.linalg.lstsq(A, b).solution  # [V, 6, 1]

        elif method == "pinv":
            AtA = torch.matmul(A.transpose(1, 2), A)  # [V, 6, 6]
            Atb = torch.matmul(A.transpose(1, 2), b)  # [V, 6, 1]
            velocities = -torch.matmul(torch.pinverse(AtA), Atb)  # [V, 6, 1]

        else:
            raise ValueError(f"Unknown method '{method}'. Use 'solve', 'lstsq', or 'pinv'.")

        return velocities  # [V, 6, 1]

    def compute_scene_flow_directional_magnitude_smoothness(
        self,
        scene_flow: torch.Tensor, # [N, 3]
        knn_indices: torch.Tensor, # [N, k]
        distances: torch.Tensor, # [N, k]
        valid_point_mask: Optional[torch.Tensor] = None,  # [N]
        sigma: float = 0.1, # Scale for distance-based weights in smoothing.
        alpha: float = 1.0, # Scale for magnitude-based weights in smoothing.
        beta: float = 1.0, # Scale for magnitude-based weights in smoothing.
        eps: float = 1e-6 # Epsilon for normalization.
    ) -> torch.Tensor:
        if valid_point_mask is not None:
            # 只保留有效点
            scene_flow = scene_flow[valid_point_mask]        # [V_valid, 3]
            knn_indices = knn_indices[valid_point_mask]      # [V_valid, k]
            distances = distances[valid_point_mask]          # [V_valid, k]

        # Gather邻居的 scene flow
        scene_flow_knn = scene_flow[knn_indices]             # [V_valid, k, 3]
        scene_flow_center = scene_flow.unsqueeze(1)          # [V_valid, 1, 3]

        # 方向一致性（cosine similarity）
        sf1 = F.normalize(scene_flow_center, dim=-1, eps=eps)   # [V_valid, 1, 3]
        sf2 = F.normalize(scene_flow_knn, dim=-1, eps=eps)       # [V_valid, k, 3]
        cos_sim = (sf1 * sf2).sum(dim=-1)                        # [V_valid, k]
        dir_loss = (1 - cos_sim).pow(2)                          # [V_valid, k]

        # 模长一致性
        mag1 = scene_flow_center.norm(dim=-1)                    # [V_valid, 1]
        mag2 = scene_flow_knn.norm(dim=-1)                       # [V_valid, k]
        mag_loss = (mag1 - mag2).pow(2)                          # [V_valid, k]

        # 加权求和
        total_loss = alpha * dir_loss + beta * mag_loss

        weights = torch.exp(-distances ** 2 / (2 * sigma ** 2))  # [V_valid, k]
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)  # 归一化
        weighted_loss = total_loss * weights

        return weighted_loss.sum() / scene_flow.shape[0]         # scalar

    def compute_scene_flow_test(
        self,
        k: int = 100,
        max_dist: float = 0.1,
        min_valid_k_neighbors: int = 12,
        method: Literal["solve", "lstsq", "pinv"] = "pinv",
        compute_smooth_loss: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        通俗地说：观察某点及其周围邻居的法线与流动情况，推断出这个点如何在三维空间中运动
        Compute scene flow for surface vertices using SDF flow and normals, with post-processing smoothing.
        Also compute smoothness loss for training.

        Args:
            k: Number of nearest neighbors for scene flow computation and smoothing.
            max_dist: Maximum distance for KNN search.
            min_valid_k_neighbors: Minimum required neighbors to consider a point non-isolated.
            method: Method for solving least squares problem.
            compute_smooth_loss: Whether to compute smoothness loss for training.

        Returns:
            Tuple containing:
            - scene_flow: Smoothed scene flow vectors, shape [V, 3].
            - smooth_loss: Smoothness loss, shape [1], for training.
        """
        assert self.vertices_sdf_flow is not None, "SDF flow must be provided for scene flow computation."
        if self.normals is None:
            self.compute_vertex_normals_(fix=True)
        
        # Compute KNN with valid neighbors
        knn_distances_, knn_indices_ = self.compute_euclidean_knn_pytorch3d(k)  # [V, k], [V, k]
        knn_distances, knn_indices, valid_point_mask = self.clean_knn_results(
            knn_distances_, knn_indices_, max_dist, min_valid_k_neighbors
        )  # [V, k], [V, k], [V] 清洗KNN + 得到有效点掩码

        # Build design matrix A and target b 只保留有效点的数据用于后续计算
        valid_knn_indices = knn_indices[valid_point_mask]              # [V_valid, k]
        points_knn = self.vertices[valid_knn_indices]                  # [V_valid, k, 3]
        normals_knn = self.normals[valid_knn_indices]                  # [V_valid, k, 3]
        sdf_flow_knn = self.vertices_sdf_flow[valid_knn_indices]       # [V_valid, k, 1]

        A = torch.cat([
            torch.cross(points_knn, normals_knn, dim=-1), 
            normals_knn
        ], dim=-1)  # [V_valid, k, 6] 构建设计矩阵 A 和目标向量 b； k（邻居数量）通常远大于 6，说明这是个 A 所描述的是一个过定（overdetermined）系统：有更多方程（k个）去拟合更少的未知数（6个

        velocities_valid = self.solve_least_squares_velocity(
            A, sdf_flow_knn, method=method
        )  # [V_valid, 6, 1]  最小二乘解 Ax = b，仅处理有效点

        # Compute scene flow: omega x p + v.   omega = velocities[..., :3, 0]  v = velocities[..., 3:, 0]
        vertices_valid = self.vertices[valid_point_mask]               # [V_valid, 3]
        scene_flow_valid = torch.cross(
            velocities_valid[..., :3, 0], vertices_valid, dim=1
        ) + velocities_valid[..., 3:, 0]  # [V_valid, 3]

        # Update vertices by scene flow
        scene_flow = torch.zeros_like(self.vertices)                   # [V, 3] 初始化全体 scene flow 为0，只填入有效点
        scene_flow[valid_point_mask] = scene_flow_valid # 更新有效点的 scene flow
        self.annotate_(vertices_scene_flow=scene_flow)

        if compute_smooth_loss:
            smooth_loss = self.compute_scene_flow_directional_magnitude_smoothness(
                scene_flow=scene_flow,
                knn_indices=knn_indices,
                distances=knn_distances,
                valid_point_mask=valid_point_mask,
                sigma=0.1,  # 可调：距离权重
                alpha=1.0,  # 可调：方向权重
                beta=1.0    # 可调：模长权重
            )
            return smooth_loss

    def compute_scene_flow(
        self,
        k: int = 100,
        max_dist: float = 0.4,
        min_valid_k_neighbors: int = 12,
        method: Literal["solve", "lstsq", "pinv"] = "pinv",
        compute_smooth_loss: bool = False,
    ) -> Optional[torch.Tensor]:
        """
        通俗地说：观察某点及其周围邻居的法线与流动情况，推断出这个点如何在三维空间中运动
        Compute scene flow for surface vertices using SDF flow and normals, with post-processing smoothing.
        Also compute smoothness loss for training.

        Args:
            k: Number of nearest neighbors for scene flow computation and smoothing.
            max_dist: Maximum distance for KNN search.
            min_valid_k_neighbors: Minimum required neighbors to consider a point non-isolated.
            method: Method for solving least squares problem.
            compute_smooth_loss: Whether to compute smoothness loss for training.

        Returns:
            Tuple containing:
            - scene_flow: Smoothed scene flow vectors, shape [V, 3].
            - smooth_loss: Smoothness loss, shape [1], for training.
        """
        assert self.vertices_sdf_flow is not None, "SDF flow must be provided for scene flow computation."
        if self.normals is None:
            self.compute_vertex_normals_(fix=True)

        mins, _ = torch.min(self.vertices, dim=0)
        maxs, _ = torch.max(self.vertices, dim=0)
        center = (mins + maxs) / 2
        size = maxs - mins   # [dx, dy, dz]
        mean_side = torch.mean(size)
        max_dist = mean_side * 0.3
        
        # Compute KNN with valid neighbors
        knn_distances_, knn_indices_ = self.compute_euclidean_knn_pytorch3d(k)  # [V, k], [V, k]
        knn_distances, knn_indices, valid_point_mask = self.clean_knn_results(
            knn_distances_, knn_indices_, max_dist, 1
        )  # [V, k], [V, k], [V] 清洗KNN + 得到有效点掩码

        # Build design matrix A and target b
        points_knn = self.vertices[knn_indices]  # [V, k, 3]
        normals_knn = self.normals[knn_indices]  # [V, k, 3]
        sdf_flow_knn = self.vertices_sdf_flow[knn_indices]  # [V, k, 1]

        A = torch.cat(
            [torch.cross(points_knn, normals_knn, dim=-1), normals_knn], 
            dim=-1
        )  # [V, k, 6] , k（邻居数量）通常远大于 6，说明这是个 A 所描述的是一个过定（overdetermined）系统：有更多方程（k个）去拟合更少的未知数（6个

        # use solve func to solve Ax = b
        velocities = self.solve_least_squares_velocity(
            A,
            sdf_flow_knn,
            method=method
        )  # [V, 6, 1]

        # Compute scene flow: omega x p + v.   omega = velocities[..., :3, 0]  v = velocities[..., 3:, 0]
        scene_flow = torch.cross(velocities[..., :3, 0], self.vertices, dim=1) + velocities[..., 3:, 0]  # [V, 3]
        if self.vertices_scene_flow is None:
            self.annotate_(vertices_scene_flow=scene_flow)
        else:
            self.replace_(vertices_scene_flow=scene_flow)

        if compute_smooth_loss:
            smooth_loss = self.compute_scene_flow_directional_magnitude_smoothness(
                scene_flow=scene_flow,
                knn_indices=knn_indices,
                distances=knn_distances,
                sigma=0.1,  # 可调：距离权重
                alpha=1.0,  # 可调：方向权重
                beta=1.0    # 可调：模长权重
            )
            return smooth_loss

    def get_next_frame_mesh(self, dt: float, compute_scene_flow: bool = False) -> DS_TriangleMesh:
        assert not (self.vertices_scene_flow is None and not compute_scene_flow), "Scene flow must be computed before next frame mesh can be generated."
        if self.vertices_scene_flow is None:
            self.compute_scene_flow()
        if compute_scene_flow:
            self.compute_scene_flow()
        vertices = self.vertices + self.vertices_scene_flow * dt  # Update vertices by scene flow
        
        return self.replace(vertices=vertices)
