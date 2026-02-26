from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch

from rfstudio.data import MeshViewSynthesisDataset, RelightDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.graphics import Cameras, Points, RGBImages, TriangleMesh
from rfstudio.graphics.shaders import PrettyShader
from rfstudio.io import dump_float32_image, open_video_renderer
from rfstudio.loss import LPIPSLoss, PSNRLoss, SSIMLoss
from rfstudio.model.density_primitives.geosplat_s4 import GeoSplatterS4
from rfstudio.trainer.geosplat_s4_trainer import GeoSplatS4Trainer
from rfstudio.ui import console
from rfstudio.utils.context import create_profiler
from rfstudio.utils.pretty import P

spot_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'spot',
    ),
    model=GeoSplatterS4(
        load=Path('exports') / 'spot.s3.pkl',
        background_color='white',
    ),
    experiment=Experiment(name='geosplat_s4', timestamp='spot'),
    trainer=GeoSplatS4Trainer(
        num_steps=100,
        batch_size=8,
        num_steps_per_val=10,
        mixed_precision=False,
        full_test_after_train=False,
    ),
    cuda=0,
    seed=1
)

garden_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'mip360' / 'masked_garden',
    ),
    model=GeoSplatterS4(
        load=Path('exports') / 'garden.s3.pkl',
        background_color='white',
    ),
    experiment=Experiment(name='geosplat_s4', timestamp='garden'),
    trainer=GeoSplatS4Trainer(
        num_steps=200,
        batch_size=8,
        num_steps_per_val=10,
        mixed_precision=False,
        full_test_after_train=False,
    ),
    cuda=0,
    seed=1
)

dami_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'damicornis',
    ),
    model=GeoSplatterS4(
        load=Path('exports') / 'dami.s3.pkl',
        background_color='white',
    ),
    experiment=Experiment(name='geosplat_s4', timestamp='dami'),
    trainer=GeoSplatS4Trainer(
        num_steps=100,
        batch_size=8,
        num_steps_per_val=10,
        mixed_precision=False,
        full_test_after_train=False,
    ),
    cuda=0,
    seed=1
)

blender_task = {}

for scene in ['chair', 'lego', 'ficus', 'mic', 'ship', 'materials', 'drums', 'hotdog']:
    blender_task[scene] = TrainTask(
        dataset=MeshViewSynthesisDataset(
            path=Path('data') / 'blender' / scene,
        ),
        model=GeoSplatterS4(
            load=Path('exports') / f'{scene}.s3.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s4', timestamp=scene),
        trainer=GeoSplatS4Trainer(
            num_steps=100,
            batch_size=8,
            num_steps_per_val=10,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1
    )

blender_task['lego_highres'] = TrainTask(
    dataset=RelightDataset(
        path=Path('data') / 'tensoir' / 'lego',
    ),
    model=GeoSplatterS4(
        load=Path('exports') / 'lego_highres.s3.pkl',
        background_color='white',
    ),
    experiment=Experiment(name='geosplat_s4', timestamp='lego_highres'),
    trainer=GeoSplatS4Trainer(
        num_steps=100,
        batch_size=8,
        num_steps_per_val=10,
        mixed_precision=False,
        full_test_after_train=False,
    ),
    cuda=0,
    seed=1
)

shiny_blender_task = {}

for scene in ['car', 'coffee', 'ball', 'helmet', 'teapot', 'toaster']:
    blender_task[scene] = TrainTask(
        dataset=MeshViewSynthesisDataset(
            path=Path('data') / 'refnerf' / scene,
        ),
        model=GeoSplatterS4(
            load=Path('exports') / f'{scene}.s3.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s4', timestamp=scene),
        trainer=GeoSplatS4Trainer(
            num_steps=100,
            batch_size=8,
            num_steps_per_val=10,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1
    )

s4r_task = {}

for scene in ['air_baloons', 'jugs', 'chair', 'hotdog']:
    if scene == 'air_baloons':
        desc = 's4r_air'
    else:
        desc = f's4r_{scene}'
    s4r_task[desc] = TrainTask(
        dataset=RelightDataset(
            path=Path('data') / 'Synthetic4Relight' / scene,
        ),
        model=GeoSplatterS4(
            load=Path('exports') / f'{desc}.s3.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s4', timestamp=desc),
        trainer=GeoSplatS4Trainer(
            num_steps=100,
            batch_size=8,
            num_steps_per_val=10,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1
    )

tsir_task = {}

for scene in ['lego', 'armadillo', 'ficus', 'hotdog']:
    if scene == 'armadillo':
        desc = 'tsir_arm'
    else:
        desc = f'tsir_{scene}'
    tsir_task[desc] = TrainTask(
        dataset=RelightDataset(
            path=Path('data') / 'tensoir' / scene,
        ),
        model=GeoSplatterS4(
            load=Path('exports') / f'{desc}.s3.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s4', timestamp=desc),
        trainer=GeoSplatS4Trainer(
            num_steps=100,
            batch_size=8,
            num_steps_per_val=10,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1
    )

sorb_task = {}

for scene in [
    'baking1',
    'ball2',
    'blocks2',
    'cactus1',
    'car4',
    'cup3',
    'curry1',
    'gnome7',
    'grogu1',
    'pepsi2',
    'pitcher1',
    'teapot1',
]:
    path = scene[:-1] + '_scene00' + scene[-1]
    scene = 'sorb_' + scene
    sorb_task[scene] = TrainTask(
        dataset=MeshViewSynthesisDataset(
            path=Path('data') / 'stanford_orb' / 'blender_LDR' / path,
        ),
        model=GeoSplatterS4(
            load=Path('exports') / f'{scene}.s3.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s4', timestamp=scene),
        trainer=GeoSplatS4Trainer(
            num_steps=100,
            batch_size=8,
            num_steps_per_val=10,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1
    )

@dataclass
class Evaler(Task):

    load: Path = ...

    step: Optional[int] = None

    only_psnr: bool = False

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            assert isinstance(model, GeoSplatterS4)
        dataset = train_task.dataset
        with console.progress(desc='Evaluating Test View', transient=True) as ptrack:
            psnrs = []
            ssims = []
            lpipss = []

            test_iter = dataset.get_test_iter(1)
            bg_color = model.get_background_color()
            for inputs, gt_outputs, indices in ptrack(test_iter, total=dataset.get_size(split='test')):
                pbra = model.render_report(inputs, gt_images=None, indices=None)[0]
                rgb = pbra.rgb2srgb().blend(bg_color).clamp(0, 1)
                gt_rgb = gt_outputs.blend(bg_color).clamp(0, 1)
                psnrs.append(PSNRLoss()(rgb, gt_rgb))
                if not self.only_psnr:
                    ssims.append(1 - SSIMLoss()(rgb, gt_rgb))
                    lpipss.append(LPIPSLoss()(rgb, gt_rgb))
            psnr = torch.stack(psnrs).mean()
            console.print(P@'PSNR: {psnr:.3f}')
            if not self.only_psnr:
                ssim = torch.stack(ssims).mean()
                lpips = torch.stack(lpipss).mean()
                console.print(P@'SSIM: {ssim:.4f}')
                console.print(P@'LPIPS: {lpips:.4f}')

@dataclass
class PBRRenderer(Task):

    load: Path = ...

    step: Optional[int] = None

    view: Literal['test', 'train'] = 'test'

    pretty: Literal['z', 'y', 'none'] = 'none'

    scale_factor: float = 1.0

    save_rgba: bool = False

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            dataset = train_task.dataset
            assert isinstance(model, GeoSplatterS4)
            assert isinstance(dataset, (MeshViewSynthesisDataset, RelightDataset))
        with console.progress(desc=f'Rendering {self.view.capitalize()} View', transient=True) as ptrack:
            bg_color = model.get_background_color()
            idx = 0
            loader = dataset.get_iter(split=self.view, batch_size=1, shuffle=False, infinite=False)
            mesh = TriangleMesh(vertices=model.mesh_v, indices=model.mesh_i.long())
            for inputs, gt_outputs, _ in ptrack(loader, total=dataset.get_size(split='test')):
                inputs = inputs.resize(self.scale_factor)
                pbra, vis, normal, _, _ = model.render_report(inputs, gt_images=None, indices=None)
                set_fmt = lambda x: x  # noqa: E731
                if self.save_rgba:
                    set_fmt = lambda x : torch.cat((x, gt_outputs.item()[..., 3:]), dim=-1)  # noqa: E731
                rgb = pbra.rgb2srgb().blend(bg_color).item().clamp(0, 1)
                albedo = vis[0].item().clamp(0, 1)
                roughness = vis[1].item()[..., 1:2].expand_as(rgb).clamp(0, 1)
                metallic = vis[1].item()[..., 2:3].expand_as(rgb).clamp(0, 1)
                if self.pretty != 'none':
                    pretty = mesh.render(
                        inputs.resize(2),
                        shader=PrettyShader(z_up=self.pretty=='z', wireframe=True),
                    ).resize_to(rgb.shape[1], rgb.shape[0]).rgb2srgb().blend(bg_color)
                    train_task.experiment.dump_image('pretty', index=idx, image=set_fmt(pretty.item().clamp(0, 1)))
                train_task.experiment.dump_image('pbr', index=idx, image=set_fmt(rgb))
                train_task.experiment.dump_image('normal', index=idx, image=set_fmt(normal.item().clamp(0, 1)))
                train_task.experiment.dump_image('raw_albedo', index=idx, image=set_fmt(albedo))
                train_task.experiment.dump_image('roughness', index=idx, image=set_fmt(roughness))
                train_task.experiment.dump_image('metallic', index=idx, image=set_fmt(metallic))
                train_task.experiment.dump_image('reference', index=idx, image=gt_outputs.item().clamp(0, 1))
                idx += 1
            envmap = model.get_envmap().as_cubemap(resolution=512).visualize(
                width=rgb.shape[1] * 2, height=rgb.shape[0]
            ).item().clamp(0, 1)
            train_task.experiment.dump_image('light', index=0, image=envmap)

@dataclass
class NVS(Task):

    load: Path = ...

    step: Optional[int] = None

    view: int = ...

    output: Path = ...

    with_gt: bool = False

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            assert isinstance(model, GeoSplatterS4)
        dataset = train_task.dataset
        test_view = dataset.get_inputs(split='test')[self.view]
        pbr = model.render_report(test_view.view(-1), indices=None, gt_images=None)[0]
        image = pbr.rgb2srgb().blend((1, 1, 1)).item()
        if self.with_gt:
            gt = dataset.get_gt_outputs(split='test')[self.view].blend((1, 1, 1)).item()
            image = torch.cat((image, gt), dim=1)
        dump_float32_image(self.output, image.clamp(0, 1))

@dataclass
class GeoNVS(Task):

    load: Path = ...

    step: Optional[int] = None

    view: int = ...

    output: Path = ...

    with_gt: bool = False

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            assert isinstance(model, GeoSplatterS4)
        dataset = train_task.dataset
        camera = dataset.get_inputs(split='test')[self.view]
        camera_pos = camera.c2w[:3, 3] # [3]
        gsplat = model.get_gsplat()
        visibilities = ((camera_pos - gsplat.gaussians.means) * gsplat.gaussians.colors).sum(-1) > 0
        gsplat.gaussians = gsplat.gaussians[visibilities]
        normals = gsplat.gaussians.colors
        depth_map = gsplat.render_depth(camera.view(-1)).visualize().item()
        gsplat.gaussians.replace_(colors=normals * 0.5 + 0.5)
        normal_map = gsplat.render_rgb(camera.view(-1)).item()
        image = torch.cat((depth_map, normal_map), dim=1)
        dump_float32_image(self.output, image.clamp(0, 1))

@dataclass
class GeoExport(Task):

    load: Path = ...

    output: Optional[Path] = None

    denoising_alpha_threshold: float = 0.01
    denoising_scale_threshold: float = 0.001
    resolution: int = 512
    fusing_alpha_threshold: float = 0.9

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load)
            model = train_task.model
            assert isinstance(model, GeoSplatterS4)
            assert isinstance(train_task.dataset, MeshViewSynthesisDataset)
            gsplat = model.get_gsplat()
            mask = gsplat.gaussians.scales.exp().max(-1).values > self.denoising_scale_threshold
            mask = mask & (gsplat.gaussians.opacities.sigmoid().squeeze(-1) > self.denoising_alpha_threshold)
            pcd = Points(positions=gsplat.gaussians.means[mask, :])
            gt_mesh = train_task.dataset.get_meta(split='train')
        with console.progress(desc='TSDF Fusion') as handle:
            cameras = Cameras.from_sphere(
                center=(0, 0, 0),
                up=(0, 0, 1),
                radius=3,
                num_samples=100,
                resolution=(800, 800),
                device=self.device
            )
            fused_mesh = TriangleMesh.from_depth_fusion(
                model.render_depth(cameras),
                cameras=cameras,
                progress_handle=handle,
                sdf_trunc=10. / self.resolution,
                voxel_size=2. / self.resolution,
                alpha_trunc=self.fusing_alpha_threshold,
            )
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
            gt_mesh.export(self.output / 'gt.ply', only_geometry=True)
            pcd.export(self.output / 'pcd.ply')
            fused_mesh.export(self.output / 'fused.ply', only_geometry=True)
        else:
            gt_mesh.export(train_task.experiment.dump_path / 'gt.ply', only_geometry=True)
            pcd.export(train_task.experiment.dump_path / 'pcd.ply')
            fused_mesh.export(train_task.experiment.dump_path / 'fused.ply', only_geometry=True)

@dataclass
class Relighter(Task):

    load: Path = ...

    envmap: Path = ...

    num_renders: int = 100

    pitch: float = 30

    radius: float = 3.

    yaw: float = 45

    resolution: Tuple[int, int] = (800, 800)

    hfov_degree: float = 40.

    use_test_view: bool = False

    rotate: Literal['scene', 'light'] = 'scene'

    envmap_concated: bool = False

    scaling: Literal['least-square', 'median'] = 'least-square'

    auto_video: bool = True

    z_up: bool = False

    envmap_as_bg: bool = False

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load)
            model = train_task.model
            assert isinstance(model, GeoSplatterS4)
            dataset = train_task.dataset
            assert isinstance(dataset, (RelightDataset, MeshViewSynthesisDataset))
        ref_camera = dataset.get_inputs(split='test')[0]
        cameras = Cameras.from_orbit(
            center=(0, 0, 0),
            up=(0, 0, 1) if self.z_up else (0, 1, 0),
            radius=self.radius,
            pitch_degree=self.pitch,
            num_samples=self.num_renders,
            resolution=self.resolution,
            hfov_degree=self.hfov_degree,
            device=ref_camera.device,
        )
        indices = torch.arange(self.num_renders, device=cameras.device)
        cameras = cameras[indices.roll(int(self.yaw / 360 * self.num_renders), dims=0)].contiguous()

        if self.use_test_view:
            cameras = dataset.get_inputs(split='test')

        bg_color = model.get_background_color()
        albedo_scaling = ref_camera.c2w.new_ones(3)

        if isinstance(dataset, RelightDataset):
            raw_bg = model.background_color
            test_inputs = dataset.get_inputs(split='test')[...]
            gt_albedos = dataset.get_meta(split='test')[0]
            model.background_color = 'black'
            with console.progress(desc='Estimating Albedo Scaling') as ptrack:
                albedo_scalings = []
                for inputs, gt_albedo in zip(ptrack(test_inputs), gt_albedos, strict=True):
                    vis = model.render_report(inputs, indices=None, gt_images=None)[1]
                    albedo = vis[0].srgb2rgb().item()
                    if self.scaling == 'least-square':
                        gt_albedo = gt_albedo.srgb2rgb().blend((0, 0, 0)).item()
                        albedo_scalings.append((albedo * gt_albedo).view(-1, 3).sum(0) / albedo.view(-1, 3).square().sum(0))
                    elif self.scaling == 'median':
                        gt_albedo = gt_albedo.srgb2rgb().item()
                        mask = (gt_albedo[..., 3:] > 0).expand_as(albedo)
                        albedo_scalings.append((gt_albedo[..., :3] / albedo.clamp_min(1e-3))[mask].view(-1, 3))
                    else:
                        raise ValueError(self.scaling)
                if self.scaling == 'least-square':
                    albedo_scaling = torch.stack(albedo_scalings).mean(0)
                elif self.scaling == 'median':
                    albedo_scaling = torch.cat(albedo_scalings).median(0).values
                else:
                    raise ValueError(self.scaling)
            model.background_color = raw_bg

        model.set_relight_envmap(self.envmap, albedo_scaling=albedo_scaling)
        base_envmap = model.envmap.as_cubemap(resolution=512)
        H, W = model.envmap.data.shape[:2]
        if self.z_up:
            base_envmap.z_up_to_y_up_()
            model.envmap = base_envmap.clone().as_latlng(width=W, height=H, apply_transform=True).compute_pdf_()

        name = self.envmap.stem if self.rotate == 'scene' else self.envmap.stem + '_rr'
        with console.progress(desc='Rendering') as ptrack:
            images = []
            for i, camera in enumerate(ptrack(cameras)):
                if self.rotate != 'scene':
                    camera = cameras[0]
                pbra = model.render_report(camera, gt_images=None, indices=None)[0]
                if not self.envmap_as_bg:
                    rgb = pbra.rgb2srgb().blend(bg_color).item().clamp(0, 1)
                else:
                    bg = model.envmap.as_cubemap(resolution=512).render(camera)
                    rgb = pbra.rgb2srgb().clamp(0, 1).blend_background(bg).item().clamp(0, 1)
                if self.envmap_concated:
                    if self.z_up:
                        envmap_temp = model.envmap.clone()
                        envmap_temp.y_up_to_z_up_()
                        envmap_vis = envmap_temp.visualize().resize_to(rgb.shape[0] * 2, rgb.shape[0])
                    else:
                        envmap_vis = model.envmap.visualize().resize_to(rgb.shape[0] * 2, rgb.shape[0])
                    rgb = torch.cat((rgb, envmap_vis.item().clamp(0, 1)), dim=1)
                train_task.experiment.dump_image(name, index=i, image=rgb)
                images.append(rgb)
                if self.rotate == 'light':
                    base_envmap.rotateY_(2 * torch.pi / self.num_renders)
                    model.envmap = base_envmap.as_latlng(width=W, height=H, apply_transform=True).compute_pdf_()

        if self.auto_video:
            with open_video_renderer(
                train_task.experiment.dump_path / (name + '.mp4'),
                fps=20,
            ) as renderer:
                for image in ptrack(images):
                    renderer.write(image)


@dataclass
class RelightEvaler(Task):

    load: Path = ...

    step: Optional[int] = None

    skip_nvs: bool = False

    skip_rlit: bool = False

    skip_mat: bool = False

    render_rlit: bool = False

    render_albedo: bool = False

    scaling: Literal['least-square', 'median'] = 'least-square'

    bg: Literal['black', 'white', 'default'] = 'white'

    fast: bool = False

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            assert isinstance(model, GeoSplatterS4)
            dataset = train_task.dataset
            assert isinstance(dataset, RelightDataset)
        raw_bg = model.background_color
        test_inputs = dataset.get_inputs(split='test')
        (
            gt_albedos,
            gt_roughnesses,
            gt_relights,
            gt_relight_envmaps,
        ) = dataset.get_meta(split='test')
        model.background_color = 'black'
        with console.progress(desc='Estimating Albedo Scaling') as ptrack:
            albedo_scalings = []
            for inputs, gt_albedo in zip(ptrack(test_inputs), gt_albedos, strict=True):
                vis = model.render_report(inputs, indices=None, gt_images=None)[1]
                albedo = vis[0].srgb2rgb().item()
                if self.scaling == 'least-square':
                    gt_albedo = gt_albedo.srgb2rgb().blend((0, 0, 0)).item()
                    albedo_scalings.append((albedo * gt_albedo).view(-1, 3).sum(0) / albedo.view(-1, 3).square().sum(0))
                elif self.scaling == 'median':
                    gt_albedo = gt_albedo.srgb2rgb().item()
                    mask = (gt_albedo[..., 3:] > 0).expand_as(albedo)
                    albedo_scalings.append((gt_albedo[..., :3] / albedo.clamp_min(1e-3))[mask].view(-1, 3))
                else:
                    raise ValueError(self.scaling)
            if self.scaling == 'least-square':
                albedo_scaling = torch.stack(albedo_scalings).mean(0)
            elif self.scaling == 'median':
                albedo_scaling = torch.cat(albedo_scalings).median(0).values
            else:
                raise ValueError(self.scaling)
        model.background_color = raw_bg if self.bg == 'default' else self.bg
        bg_color = model.get_background_color()
        if not self.skip_nvs:
            psnrs = []
            ssims = []
            lpipss = []
            with console.progress(desc='Evaluating NVS') as ptrack:
                test_iter = dataset.get_test_iter(1)
                for inputs, gt_outputs, indices in ptrack(test_iter, total=dataset.get_size(split='test')):
                    pbra = model.render_report(inputs, indices=None, gt_images=None)[0]
                    rgb = pbra.rgb2srgb().blend(bg_color).clamp(0, 1)
                    gt_rgb = gt_outputs.blend(bg_color).clamp(0, 1)
                    psnrs.append(PSNRLoss()(rgb, gt_rgb))
                    if not self.fast:
                        ssims.append(1 - SSIMLoss()(rgb, gt_rgb))
                        lpipss.append(LPIPSLoss()(rgb, gt_rgb))
            psnr = torch.stack(psnrs).mean()
            console.print(P@'NVS @ PSNR: {psnr:.3f}')
            if not self.fast:
                ssim = torch.stack(ssims).mean()
                lpips = torch.stack(lpipss).mean()
                console.print(P@'NVS @ SSIM: {ssim:.4f}')
                console.print(P@'NVS @ LPIPS: {lpips:.4f}')
        if not self.skip_rlit:
            for relight_idx, (relights, envmap) in enumerate(zip(gt_relights, gt_relight_envmaps, strict=True)):
                relight_idx += 1
                psnrs = []
                ssims = []
                lpipss = []
                with console.progress(desc=f'Evaluating Relighting #{relight_idx}') as ptrack:
                    model.set_relight_envmap(envmap, albedo_scaling=albedo_scaling)
                    for idx, (inputs, gt_outputs) in enumerate(zip(ptrack(test_inputs), relights, strict=True)):
                        pbra = model.render_report(inputs, indices=None, gt_images=None)[0]
                        rgb = pbra.rgb2srgb().blend(bg_color).clamp(0, 1)
                        gt_rgb = gt_outputs.blend(bg_color).clamp(0, 1)
                        psnrs.append(PSNRLoss()(rgb, gt_rgb))
                        if not self.fast:
                            ssims.append(1 - SSIMLoss()(rgb, gt_rgb))
                            lpipss.append(LPIPSLoss()(rgb, gt_rgb))
                        if self.render_rlit:
                            image = torch.cat((rgb.item(), gt_rgb.item()), dim=1).clamp(0, 1)
                            train_task.experiment.dump_image(f'relight{relight_idx}', index=idx, image=image)
                psnr = torch.stack(psnrs).mean()
                console.print(P@'RLIT[{relight_idx}] @ PSNR: {psnr:.3f}')
                if not self.fast:
                    ssim = torch.stack(ssims).mean()
                    lpips = torch.stack(lpipss).mean()
                    console.print(P@'RLIT[{relight_idx}] @ SSIM: {ssim:.4f}')
                    console.print(P@'RLIT[{relight_idx}] @ LPIPS: {lpips:.4f}')
        if not self.skip_mat:
            model.background_color = 'black'
            bg_color = model.get_background_color()
            roughness_mses = []
            psnrs = []
            ssims = []
            lpipss = []
            with console.progress(desc='Evaluating Albedo & Roughness') as ptrack:
                model.set_relight_envmap(gt_relight_envmaps[0], albedo_scaling=albedo_scaling)
                for idx, (inputs, gt_albedo) in enumerate(zip(ptrack(test_inputs), gt_albedos, strict=True)):
                    vis = model.render_report(inputs, indices=None, gt_images=None)[1]
                    if gt_roughnesses is not None:
                        roughness = vis[1].item()[..., 1:2] # [H, W, 1]
                        roughness_mses.append(
                            torch.nn.functional.mse_loss(
                                roughness,
                                gt_roughnesses[idx].blend(bg_color).item()[..., 0:1],
                            )
                        )
                    albedo = RGBImages([vis[0].item()]).clamp(0, 1)
                    gt_albedo = gt_albedo.blend(bg_color)
                    psnrs.append(PSNRLoss()(albedo, gt_albedo))
                    if not self.fast:
                        ssims.append(1 - SSIMLoss()(albedo, gt_albedo))
                        lpipss.append(LPIPSLoss()(albedo, gt_albedo))
                    if self.render_albedo:
                        image = torch.cat((albedo.item(), gt_albedo.item()), dim=1)
                        train_task.experiment.dump_image('albedo', index=idx, image=image.clamp(0, 1))
            psnr = torch.stack(psnrs).mean()
            console.print(P@'Albedo @ PSNR: {psnr:.3f}')
            if not self.fast:
                ssim = torch.stack(ssims).mean()
                lpips = torch.stack(lpipss).mean()
                console.print(P@'Albedo @ SSIM: {ssim:.4f}')
                console.print(P@'Albedo @ LPIPS: {lpips:.4f}')
            if gt_roughnesses is not None:
                roughness_mse = torch.stack(roughness_mses).mean()
                console.print(P@'Roughness @ MSE: {roughness_mse:.3f}')
            else:
                console.print(P@'Roughness @ MSE: N/A')

@dataclass
class ReportRefinement(Task):

    load: Path = ...

    step: Optional[int] = None

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            assert isinstance(model, GeoSplatterS4)
        origin_attrs = torch.load(model.load, map_location='cpu')
        splats = model.get_gsplat().gaussians.cpu()
        diff = (origin_attrs['means'] - getattr(splats, 'means')).abs().mean()
        console.print(P@"means: {diff:.4f}")
        diff = (origin_attrs['scales'].exp() - getattr(splats, 'scales').exp()).abs().mean()
        console.print(P@"scales: {diff:.4f}")
        diff = (
            origin_attrs['quats'] / origin_attrs['quats'].norm(dim=-1, keepdim=True) -
            getattr(splats, 'quats') / getattr(splats, 'quats').norm(dim=-1, keepdim=True)
        ).abs().mean()
        console.print(P@"quats: {diff:.4f}")
        diff = (origin_attrs['opacities'].sigmoid() - getattr(splats, 'opacities').sigmoid()).abs().mean()
        console.print(P@"opacities: {diff:.4f}")
        diff = (
            origin_attrs['normals'] / origin_attrs['normals'].norm(dim=-1, keepdim=True) -
            getattr(splats, 'colors') / getattr(splats, 'colors').norm(dim=-1, keepdim=True)
        ).abs().mean()
        console.print(P@"normals: {diff:.4f}")

@dataclass
class FPS(Task):

    load: Path = ...

    step: Optional[int] = None

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            assert isinstance(model, GeoSplatterS4)
            dataset = train_task.dataset
            assert isinstance(dataset, RelightDataset)
        test_inputs = dataset.get_inputs(split='test')[...].view(-1, 1)
        test_inputs = Cameras.cat([test_inputs] * 5, dim=0)
        with create_profiler() as profiler:
            model.profiling(test_inputs)
        fps = len(test_inputs) / profiler.duration
        console.print(P@'FPS: {fps:.2f}')

if __name__ == '__main__':
    TaskGroup(
        **s4r_task,
        **tsir_task,
        **blender_task,
        **shiny_blender_task,
        **sorb_task,
        spot=spot_task,
        dami=dami_task,
        garden=garden_task,
        fps=FPS(cuda=0),
        nvs=NVS(cuda=0),
        geoexport=GeoExport(cuda=0),
        pbr=PBRRenderer(cuda=0),
        geonvs=GeoNVS(cuda=0),
        rliteval=RelightEvaler(cuda=0),
        relight=Relighter(cuda=0),
        eval=Evaler(cuda=0),
        report=ReportRefinement(cuda=0),
    ).run()
