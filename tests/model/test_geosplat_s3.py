from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch

from rfstudio.data import MeshViewSynthesisDataset, RelightDataset
from rfstudio.engine.experiment import Experiment
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.engine.train import TrainTask
from rfstudio.graphics import Cameras, PBRAImages, Points, RGBImages, TextureCubeMap, TriangleMesh
from rfstudio.io import dump_float32_image, open_video_renderer
from rfstudio.loss import LPIPSLoss, PSNRLoss, SSIMLoss
from rfstudio.model.density_primitives.geosplat_s3 import GeoSplatterS3
from rfstudio.trainer.geosplat_s3_trainer import GeoSplatS3Trainer
from rfstudio.ui import console
from rfstudio.utils.pretty import P

spot_task = TrainTask(
    dataset=MeshViewSynthesisDataset(
        path=Path('data') / 'spot',
    ),
    model=GeoSplatterS3(
        load=Path('exports') / 'spot.s2.pkl',
        background_color='white',
    ),
    experiment=Experiment(name='geosplat_s3', timestamp='spot'),
    trainer=GeoSplatS3Trainer(
        num_steps=500,
        batch_size=8,
        num_steps_per_val=25,
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
    model=GeoSplatterS3(
        load=Path('exports') / 'dami.s2.pkl',
        background_color='white',
    ),
    experiment=Experiment(name='geosplat_s3', timestamp='dami'),
    trainer=GeoSplatS3Trainer(
        num_steps=500,
        batch_size=8,
        num_steps_per_val=25,
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
        model=GeoSplatterS3(
            load=Path('exports') / f'{scene}.s2.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s3', timestamp=scene),
        trainer=GeoSplatS3Trainer(
            num_steps=500,
            batch_size=8,
            num_steps_per_val=25,
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
        model=GeoSplatterS3(
            load=Path('exports') / f'{scene}.s2.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s3', timestamp=scene),
        trainer=GeoSplatS3Trainer(
            num_steps=500,
            batch_size=8,
            num_steps_per_val=25,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1
    )

dtu_task = {}

for scene in ['dtu24', 'dtu110', 'dtu65', 'dtu114', 'dtu118']:
    dtu_task[scene] = TrainTask(
        dataset=MeshViewSynthesisDataset(
            path=Path('data') / 'dtu' / scene.replace('dtu', 'dtu_scan'),
        ),
        model=GeoSplatterS3(
            load=Path('exports') / f'{scene}s2.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s3', timestamp=scene),
        trainer=GeoSplatS3Trainer(
            num_steps=1000,
            batch_size=8,
            num_steps_per_val=50,
            num_steps_per_save=250,
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
        model=GeoSplatterS3(
            load=Path('exports') / f'{desc}.s2.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s3', timestamp=desc),
        trainer=GeoSplatS3Trainer(
            num_steps=500,
            batch_size=8,
            num_steps_per_val=25,
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
        model=GeoSplatterS3(
            load=Path('exports') / f'{desc}.s2.pkl',
            background_color='white',
        ),
        experiment=Experiment(name='geosplat_s3', timestamp=desc),
        trainer=GeoSplatS3Trainer(
            num_steps=500,
            batch_size=8,
            num_steps_per_val=25,
            mixed_precision=False,
            full_test_after_train=False,
        ),
        cuda=0,
        seed=1
    )

@dataclass
class Export(Task):

    load: Path = ...
    output: Path = ...

    @torch.no_grad()
    def run(self) -> None:
        train_task = TrainTask.load_from_script(self.load)
        model = train_task.model
        assert isinstance(model, GeoSplatterS3)
        model.export(self.output)

@dataclass
class Evaler(Task):

    load: Path = ...

    step: Optional[int] = None

    no_shs: bool = False

    only_psnr: bool = False

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            assert isinstance(model, GeoSplatterS3)
        if self.no_shs:
            model.shs = torch.nn.Parameter(torch.empty_like(model.shs[:, 1:1, :]), requires_grad=False)
        dataset = train_task.dataset
        with console.progress(desc='Evaluating Test View', transient=True) as ptrack:
            psnrs = []
            ssims = []
            lpipss = []

            test_iter = dataset.get_test_iter(1)
            bg_color = model.get_background_color()
            for inputs, gt_outputs, indices in ptrack(test_iter, total=dataset.get_size(split='test')):
                pbra = model.render_report(inputs, indices=None)[0]
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

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            dataset = train_task.dataset
            assert isinstance(model, GeoSplatterS3)
            assert isinstance(dataset, (MeshViewSynthesisDataset, RelightDataset))
        with console.progress(desc='Rendering Test View', transient=True) as ptrack:
            bg_color = model.get_background_color()
            idx = 0
            for inputs, gt_outputs, _ in ptrack(dataset.get_test_iter(1), total=dataset.get_size(split='test')):
                pbra, vis, normal, _, _ = model.render_report(inputs, indices=None)
                rgb = pbra.rgb2srgb().blend(bg_color).item().clamp(0, 1)
                albedo = vis[0].item().clamp(0, 1)
                visibility = (1 - vis[1].item()[..., 0:1]).expand_as(rgb).clamp(0, 1)
                roughness = vis[1].item()[..., 1:2].expand_as(rgb).clamp(0, 1)
                metallic = vis[1].item()[..., 2:3].expand_as(rgb).clamp(0, 1)
                diffuse = model.render_diffuse(inputs).blend(bg_color)
                specular = model.render_specular(inputs).blend(bg_color)
                residual = model.render_residual(inputs, base=0.5, scale=2.5)
                train_task.experiment.dump_image('pbr', index=idx, image=rgb)
                train_task.experiment.dump_image('visibility', index=idx, image=visibility)
                train_task.experiment.dump_image('normal', index=idx, image=normal.item().clamp(0, 1))
                train_task.experiment.dump_image('albedo', index=idx, image=albedo)
                train_task.experiment.dump_image('roughness', index=idx, image=roughness)
                train_task.experiment.dump_image('metallic', index=idx, image=metallic)
                train_task.experiment.dump_image('residual', index=idx, image=residual.item().clamp(0, 1))
                train_task.experiment.dump_image('diffuse', index=idx, image=diffuse.item().clamp(0, 1))
                train_task.experiment.dump_image('specular', index=idx, image=specular.item().clamp(0, 1))
                train_task.experiment.dump_image('reference', index=idx, image=gt_outputs.item().clamp(0, 1))
                idx += 1
            envmap = TextureCubeMap(
                data=model.cubemap,
                transform=None,
            ).visualize(width=rgb.shape[1] * 2, height=rgb.shape[0]).item().clamp(0, 1)
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
            assert isinstance(model, GeoSplatterS3)
        dataset = train_task.dataset
        test_view = dataset.get_inputs(split='test')[self.view]
        pbr = model.render_report(test_view.view(-1), indices=None)[0]
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
            assert isinstance(model, GeoSplatterS3)
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
            assert isinstance(model, GeoSplatterS3)
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

    use_test_view: bool = False

    rotate: Literal['scene', 'light'] = 'scene'

    auto_video: bool = True

    z_up: bool = False

    envmap_as_bg: bool = False

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load)
            model = train_task.model
            assert isinstance(model, GeoSplatterS3)
            dataset = train_task.dataset
            assert isinstance(dataset, (RelightDataset, MeshViewSynthesisDataset))
        ref_camera = dataset.get_inputs(split='test')[0]
        cameras = Cameras.from_orbit(
            center=(0, 0, 0),
            up=(0, 0, 1) if self.z_up else (0, 1, 0),
            radius=self.radius,
            pitch_degree=self.pitch,
            num_samples=self.num_renders,
            device=ref_camera.device,
        )
        cameras.replace_(
            cx=(ref_camera.width * 0.5).expand_as(cameras.cx),
            cy=(ref_camera.height * 0.5).expand_as(cameras.cy),
            fx=ref_camera.fx.expand_as(cameras.fx),
            fy=ref_camera.fy.expand_as(cameras.fy),
            near=ref_camera.near.expand_as(cameras.near),
            far=ref_camera.far.expand_as(cameras.far),
            width=ref_camera.width.expand_as(cameras.width),
            height=ref_camera.height.expand_as(cameras.height),
        )
        indices = torch.arange(self.num_renders, device=cameras.device)
        cameras = cameras[indices.roll(int(self.yaw / 360 * self.num_renders), dims=0)].contiguous()

        if self.use_test_view:
            cameras = dataset.get_inputs(split='test')

        bg_color = model.get_background_color()
        model.set_relight_envmap(self.envmap, albedo_scaling=ref_camera.c2w.new_ones(3))
        if self.z_up:
            model.envmap.z_up_to_y_up_()

        if self.envmap_as_bg:
            cubemap = TextureCubeMap.from_image_file(self.envmap, resolution=1024, device=model.device)

        with console.progress(desc='Rendering') as ptrack:
            images = []
            for i, camera in enumerate(ptrack(cameras)):
                if self.rotate != 'scene':
                    camera = cameras[0]
                pbra = model.render_report(camera, indices=None)[0]
                if not self.envmap_as_bg:
                    rgb = pbra.rgb2srgb().blend(bg_color).item().clamp(0, 1)
                else:
                    bg = cubemap.replace(transform=model.envmap.transform).render(camera)
                    rgb = pbra.rgb2srgb().clamp(0, 1).blend_background(bg).item().clamp(0, 1)
                train_task.experiment.dump_image(self.envmap.stem, index=i, image=rgb)
                images.append(rgb)
                if self.rotate == 'light':
                    model.envmap.rotateY_(2 * torch.pi / self.num_renders)

        if self.auto_video:
            with open_video_renderer(
                train_task.experiment.dump_path / (self.envmap.stem + '.mp4'),
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

    bg: Literal['black', 'white', 'default'] = 'white'

    @torch.no_grad()
    def run(self) -> None:
        with console.status(desc='Loading Model'):
            train_task = TrainTask.load_from_script(self.load, step=self.step)
            model = train_task.model
            assert isinstance(model, GeoSplatterS3)
            dataset = train_task.dataset
            assert isinstance(dataset, RelightDataset)
        raw_bg = model.background_color
        test_inputs = dataset.get_inputs(split='test')
        (
            gt_albedos,
            gt_roughnesses,
            gt_relight1,
            gt_relight2,
            _,
            gt_relight1_envmap,
            gt_relight2_envmap,
        ) = dataset.get_meta(split='test')
        model.background_color = 'black'
        with console.progress(desc='Estimating Albedo Scaling') as ptrack:
            albedo_scalings = []
            for inputs, gt_albedo in ptrack(
                zip(test_inputs, gt_albedos, strict=True),
                total=test_inputs.shape[0],
            ):
                vis = model.render_report(inputs, indices=None, gt_images=None)[1]
                albedo = vis[0].srgb2rgb().item()
                if gt_roughnesses is None:
                    gt_albedo = gt_albedo.blend((0, 0, 0)).item()
                else:
                    gt_albedo = gt_albedo.srgb2rgb().blend((0, 0, 0)).item()
                albedo_scalings.append((albedo * gt_albedo).view(-1, 3).sum(0) / albedo.view(-1, 3).square().sum(0))
            albedo_scaling = torch.stack(albedo_scalings).mean(0)
        model.background_color = raw_bg if self.bg == 'default' else self.bg
        bg_color = model.get_background_color()
        if not self.skip_nvs:
            with console.progress(desc='Evaluating NVS') as ptrack:
                psnrs = []
                ssims = []
                lpipss = []
                test_iter = dataset.get_test_iter(1)
                for inputs, gt_outputs, indices in ptrack(test_iter, total=dataset.get_size(split='test')):
                    pbra = model.render_report(inputs, indices=None, gt_images=None)[0]
                    rgb = pbra.rgb2srgb().blend(bg_color).clamp(0, 1)
                    gt_rgb = gt_outputs.blend(bg_color).clamp(0, 1)
                    psnrs.append(PSNRLoss()(rgb, gt_rgb))
                    ssims.append(1 - SSIMLoss()(rgb, gt_rgb))
                    lpipss.append(LPIPSLoss()(rgb, gt_rgb))
                psnr = torch.stack(psnrs).mean()
                ssim = torch.stack(ssims).mean()
                lpips = torch.stack(lpipss).mean()
                console.print(P@'NVS @ PSNR: {psnr:.3f}')
                console.print(P@'NVS @ SSIM: {ssim:.4f}')
                console.print(P@'NVS @ LPIPS: {lpips:.4f}')
        if not self.skip_rlit:
            with console.progress(desc='Evaluating Relighting #1') as ptrack:
                psnrs = []
                ssims = []
                lpipss = []
                model.set_relight_envmap(gt_relight1_envmap, albedo_scaling=albedo_scaling)
                for idx, inputs, gt_outputs in ptrack(zip(
                    range(test_inputs.shape[0]),
                    test_inputs,
                    gt_relight1,
                ), total=test_inputs.shape[0]):
                    pbra = model.render_report(inputs, indices=None, gt_images=None)[0]
                    rgb = pbra.rgb2srgb().blend(bg_color).clamp(0, 1)
                    gt_rgb = gt_outputs.blend(bg_color).clamp(0, 1)
                    psnrs.append(PSNRLoss()(rgb, gt_rgb))
                    ssims.append(1 - SSIMLoss()(rgb, gt_rgb))
                    lpipss.append(LPIPSLoss()(rgb, gt_rgb))
                    if self.render_rlit:
                        image = torch.cat((rgb.item(), gt_rgb.item()), dim=1)
                        train_task.experiment.dump_image('relight1', index=idx, image=image.clamp(0, 1))
                psnr = torch.stack(psnrs).mean()
                ssim = torch.stack(ssims).mean()
                lpips = torch.stack(lpipss).mean()
                console.print(P@'RLIT[1] @ PSNR: {psnr:.3f}')
                console.print(P@'RLIT[1] @ SSIM: {ssim:.4f}')
                console.print(P@'RLIT[1] @ LPIPS: {lpips:.4f}')
            with console.progress(desc='Evaluating Relighting #2') as ptrack:
                psnrs = []
                ssims = []
                lpipss = []
                model.set_relight_envmap(gt_relight2_envmap, albedo_scaling=albedo_scaling)
                for idx, inputs, gt_outputs in ptrack(zip(
                    range(test_inputs.shape[0]),
                    test_inputs,
                    gt_relight2,
                ), total=test_inputs.shape[0]):
                    pbra = model.render_report(inputs, indices=None, gt_images=None)[0]
                    rgb = pbra.rgb2srgb().blend(bg_color).clamp(0, 1)
                    gt_rgb = gt_outputs.blend(bg_color).clamp(0, 1)
                    psnrs.append(PSNRLoss()(rgb, gt_rgb))
                    ssims.append(1 - SSIMLoss()(rgb, gt_rgb))
                    lpipss.append(LPIPSLoss()(rgb, gt_rgb))
                    if self.render_rlit:
                        image = torch.cat((rgb.item(), gt_rgb.item()), dim=1)
                        train_task.experiment.dump_image('relight2', index=idx, image=image.clamp(0, 1))
                psnr = torch.stack(psnrs).mean()
                ssim = torch.stack(ssims).mean()
                lpips = torch.stack(lpipss).mean()
                console.print(P@'RLIT[2] @ PSNR: {psnr:.3f}')
                console.print(P@'RLIT[2] @ SSIM: {ssim:.4f}')
                console.print(P@'RLIT[2] @ LPIPS: {lpips:.4f}')
        if not self.skip_mat:
            with console.progress(desc='Evaluating Albedo & Roughness') as ptrack:
                roughness_mses = []
                psnrs = []
                ssims = []
                lpipss = []
                is_tensoir = False
                if gt_roughnesses is None:
                    gt_roughnesses = gt_albedos
                    is_tensoir = True
                model.set_relight_envmap(gt_relight1_envmap, albedo_scaling=albedo_scaling)
                for idx, inputs, gt_roughness, gt_albedo in ptrack(
                    zip(range(test_inputs.shape[0]), test_inputs, gt_roughnesses, gt_albedos, strict=True),
                    total=test_inputs.shape[0],
                ):
                    vis = model.render_report(inputs, indices=None, gt_images=None)[1]
                    roughness = vis[1].item()[..., 1:2] # [H, W, 1]
                    roughness_mses.append(
                        torch.nn.functional.mse_loss(roughness, gt_roughness.blend(bg_color).item()[..., 0:1])
                    )
                    albedo = RGBImages([vis[0].item()]).clamp(0, 1)
                    if is_tensoir:
                        gt_albedo = PBRAImages([gt_albedo.item()]).rgb2srgb().blend(bg_color)
                    else:
                        gt_albedo = gt_albedo.blend(bg_color)
                    psnrs.append(PSNRLoss()(albedo, gt_albedo))
                    ssims.append(1 - SSIMLoss()(albedo, gt_albedo))
                    lpipss.append(LPIPSLoss()(albedo, gt_albedo))
                    if self.render_albedo:
                        image = torch.cat((albedo.item(), gt_albedo.item()), dim=1)
                        train_task.experiment.dump_image('albedo', index=idx, image=image.clamp(0, 1))
                psnr = torch.stack(psnrs).mean()
                ssim = torch.stack(ssims).mean()
                lpips = torch.stack(lpipss).mean()
                roughness_mse = torch.stack(roughness_mses).mean()
                console.print(P@'Albedo @ PSNR: {psnr:.3f}')
                console.print(P@'Albedo @ SSIM: {ssim:.4f}')
                console.print(P@'Albedo @ LPIPS: {lpips:.4f}')
                console.print(P@'Roughness @ MSE: {roughness_mse:.3f}')

if __name__ == '__main__':
    TaskGroup(
        **s4r_task,
        **tsir_task,
        **dtu_task,
        **blender_task,
        **shiny_blender_task,
        spot=spot_task,
        dami=dami_task,
        nvs=NVS(cuda=0),
        export=Export(cuda=0),
        geoexport=GeoExport(cuda=0),
        pbr=PBRRenderer(cuda=0),
        geonvs=GeoNVS(cuda=0),
        rliteval=RelightEvaler(cuda=0),
        relight=Relighter(cuda=0),
        eval=Evaler(cuda=0),
    ).run()
