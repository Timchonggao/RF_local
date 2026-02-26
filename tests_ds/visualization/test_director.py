from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.engine.task import Task, TaskGroup
from rfstudio.io import load_float32_image
from rfstudio.visualization import MovieAnimation, MovieDirector


@dataclass
class Main(Task):

    director: MovieDirector = MovieDirector(workspace=Path('temp') / 'demo', resolution='1080p')

    input: Path = Path('temp') / 'rust_teaser'

    export: bool = False

    target_mb: float = 64.0

    homepage: bool = True

    def run(self) -> None:
        self.director.to(self.device)

        if self.homepage:
            with self.director.stage('title_homepage', duration=40) as sd:
                sd.fade_out(20)
                sd[0.20:0.29, 0.04:] = MovieAnimation.StaticText(
                    'The Less You Depend, The More You Learn:',
                    bold=True,
                    align='left',
                )
                sd[0.31:0.4, 0.04:] = MovieAnimation.StaticText(
                    'Synthesizing Novel Views from Sparse, Unposed',
                    bold=True,
                    align='left',
                )
                sd[0.41:0.5, 0.04:] = MovieAnimation.StaticText(
                    'Images without Any 3D Knowledge',
                    bold=True,
                    align='left',
                )

                sd[0.64:0.7, 0.04:] = MovieAnimation.StaticText(
                    'Haoru Wang, Kai Ye, Yangyan Li, Wenzheng Chen, Baoquan Chen',
                    bold=False,
                    align='left',
                )
        else:
            with self.director.stage('title', duration=40) as sd:
                sd.fade_out(20)
                sd[0.20:0.27, 0.1:] = MovieAnimation.StaticText(
                    'The Less You Depend, The More You Learn:',
                    bold=True,
                    align='left',
                )
                sd[0.29:0.36, 0.1:] = MovieAnimation.StaticText(
                    'Synthesizing Novel Views from Sparse, Unposed',
                    bold=True,
                    align='left',
                )
                sd[0.38:0.45, 0.1:] = MovieAnimation.StaticText(
                    'Images without Any 3D Knowledge',
                    bold=True,
                    align='left',
                )

                sd[0.6:0.66, 0.1:] = MovieAnimation.StaticText(
                    'Anonymous NeurIPS submission',
                    bold=False,
                    align='left',
                )
                sd[0.68:0.74, 0.1:] = MovieAnimation.StaticText(
                    'Paper ID 6684',
                    bold=False,
                    align='left',
                )

        with self.director.stage('introduction', duration=200) as sd:
            sd.fade_in(20).fade_out(20)
            sd[0.19:0.75, 0.05:0.95] = MovieAnimation.ImageFrames(
                self.input,
                pattern="intro.png",
            )

        with self.director.stage('title_comp', duration=60) as sd:
            sd.fade_in(20).fade_out(20)
            sd[0.4:0.6, 0.1:0.9] = MovieAnimation.StaticText(
                'Qualitative Comparisons',
                bold=True,
            )

        for scene_idx, anim_desc in {
            1204: [slice(0.47, 0.72), slice(0.58, 0.83), slice(0.5, 1.0), slice(0.0, 0.5), 1],
            2382: [slice(0.3, 0.6), slice(0.27, 0.57), slice(0.5, 1.0), slice(0.5, 1.0), 1],
            2490: [slice(0.52, 0.92), slice(0.6, 1.0), slice(0.0, 0.65), slice(0.0, 0.65), 1],
            3788: [slice(0.38, 0.68), slice(0.1, 0.4), slice(0.0, 0.5), slice(0.5, 1.0), 1],
        }.items():
            with self.director.stage(f'comp_{scene_idx}', duration=290) as sd:
                sd.fade_in(20).fade_out(20)
                grid = MovieAnimation.GridContainer(
                    padding=30,
                    text_line_height=60,
                    rows={ ' ': 1. },
                    cols={ 'Inputs': 0.45, 'NoPoSplat': 1., 'Our UP-LVSM': 1., 'Ground Truth': 1. },
                    align='left',
                )
                with (self.input / f'{scene_idx:06d}' / 'metrics.json').open() as f:
                    item = json.load(f)
                    scene_name = item['summary']['scene_name']
                    frame_start = item['per_view'][0]['view']
                    filename = f'{scene_name}_frame_{frame_start}.mp4'
                context = load_float32_image(self.input / f'{scene_idx:06d}' / 'input.png')
                context = torch.cat((
                    context[:, :518],
                    torch.ones(30, 518, 3),
                    context[:, 518:],
                ), dim=0)
                grid[' ']['Inputs'] = MovieAnimation.StaticImage(context)
                grid[' ']['Our UP-LVSM'] = MovieAnimation.Highlight(
                    MovieAnimation.VideoFrames(
                        self.input / f'{scene_idx:06d}' / 'rendered_video.mp4',
                        padding='reflective',
                        duration=45,
                    ),
                    source=(anim_desc[0], anim_desc[1]),
                    target=(anim_desc[2], anim_desc[3]),
                    thickness=10,
                    start=75,
                    stop=185,
                    enable=anim_desc[-1],
                )
                grid[' ']['NoPoSplat'] = MovieAnimation.Highlight(
                    MovieAnimation.VideoFrames(
                        self.input / 'nopo' / filename,
                        padding='reflective',
                        duration=45,
                    ),
                    source=(anim_desc[0], anim_desc[1]),
                    target=(anim_desc[2], anim_desc[3]),
                    thickness=10,
                    start=75,
                    stop=185,
                    enable=anim_desc[-1],
                )
                grid[' ']['Ground Truth'] = MovieAnimation.Highlight(
                    MovieAnimation.VideoFrames(
                        self.input / 'gt' / filename,
                        padding='reflective',
                        duration=45,
                    ),
                    source=(anim_desc[0], anim_desc[1]),
                    target=(anim_desc[2], anim_desc[3]),
                    thickness=10,
                    start=75,
                    stop=185,
                    enable=anim_desc[-1],
                )
                sd[0.09:0.17, 0.055:] = MovieAnimation.StaticText(
                    'Qualitative Comparisons',
                    bold=True,
                    align='left',
                )
                sd[0.15:0.95, 0.005:0.98] = grid

        with self.director.stage('title_more', duration=60) as sd:
            sd.fade_in(20).fade_out(20)
            sd[0.4:0.6, 0.1:0.9] = MovieAnimation.StaticText(
                'More Results',
                bold=True,
            )

        with self.director.stage('more', duration=200) as sd:
            sd.fade_in(20).fade_out(20)

            grid_indices = [
                [7273, 2075, 2437, 2866],
                [3608, 6469, 6615, 5307],
                [7275, 5524, 6849, 6074],
            ]
            grid = MovieAnimation.GridContainer(
                padding=20,
                text_line_height=10,
                rows={ 'r0': 1.0, 'r1': 1.0, 'r2': 1.0 },
                cols={
                    'c00': 0.525, 'c01': 1.0,
                    'c10': 0.525, 'c11': 1.0,
                    'c20': 0.525, 'c21': 1.0,
                    'c30': 0.525, 'c31': 1.0,
                },
                align='left',
                hide_text=True,
            )
            for r_ind in range(3):
                for c_ind in range(4):
                    scene_idx = grid_indices[r_ind][c_ind]
                    with (self.input / f'{scene_idx:06d}' / 'metrics.json').open() as f:
                        item = json.load(f)
                        scene_name = item['summary']['scene_name']
                        frame_start = item['per_view'][0]['view']
                        filename = f'{scene_name}_frame_{frame_start}.mp4'
                    context = load_float32_image(self.input / f'{scene_idx:06d}' / 'input.png')
                    context = torch.cat((
                        context[:, :518],
                        torch.ones(45, 518, 3),
                        context[:, 518:],
                    ), dim=0)
                    context = torch.cat((torch.ones(context.shape[0], 80, 3), context), dim=1)
                    grid[f'r{r_ind}'][f'c{c_ind}0'] = MovieAnimation.StaticImage(context)
                    grid[f'r{r_ind}'][f'c{c_ind}1'] = MovieAnimation.VideoFrames(
                        self.input / f'{scene_idx:06d}' / 'rendered_video.mp4',
                        padding='reflective',
                        duration=45,
                    )
            sd[0.05:0.13, 0.03:] = MovieAnimation.StaticText(
                'More Results of Our UP-LVSM',
                bold=True,
                align='left',
            )
            sd[0.11:0.98, 0.005:0.98] = grid


        with self.director.stage('title_thanks', duration=20) as sd:
            sd.fade_in(18)
            sd[0.42:0.58, 0.1:0.9] = MovieAnimation.StaticText(
                'Thanks',
                bold=True,
            )

        if self.export:
            self.director.export(target_mb=self.target_mb)

@dataclass
class Homepage(Task):

    director: MovieDirector = MovieDirector(workspace=Path('temp') / 'homepage_demo', resolution='1080p')

    input: Path = Path('temp') / 'homepage'

    export: bool = False

    target_mb: float = 48.0

    def run(self) -> None:
        self.director.to(self.device)

        for name, desc in zip(
            ['indoor', 'outdoor', 'large_baseline', 'strong_reflection'],
            [
                'Indoor Results',
                'Outdoor Results',
                'Little Overlap',
                'Strong Reflection',
            ],
        ):
            with self.director.stage(name, duration=200) as sd:
                sd.fade_in(20).fade_out(20)

                grid = MovieAnimation.GridContainer(
                    padding=20,
                    text_line_height=10,
                    rows={ 'r0': 1.0, 'r1': 1.0 },
                    cols={
                        'c00': 0.525, 'c01': 1.0,
                        'c10': 0.525, 'c11': 1.0,
                        'c20': 0.525, 'c21': 1.0,
                    },
                    align='left',
                    hide_text=True,
                )
                ctx_list = list(sorted((self.input / name).glob("*.png")))
                video_list = list(sorted((self.input / name).glob("*.mp4")))
                for r_ind in range(2):
                    for c_ind in range(3):
                        idx = r_ind * 3 + c_ind
                        context = load_float32_image(ctx_list[idx])
                        context = torch.cat((
                            context[:, :518],
                            torch.ones(45, 518, 3),
                            context[:, 518:],
                        ), dim=0)
                        context = torch.cat((torch.ones(context.shape[0], 80, 3), context), dim=1)
                        grid[f'r{r_ind}'][f'c{c_ind}0'] = MovieAnimation.StaticImage(context)
                        grid[f'r{r_ind}'][f'c{c_ind}1'] = MovieAnimation.VideoFrames(
                            video_list[idx],
                            padding='reflective',
                            duration=45,
                        )
                sd[0.05:0.13, 0.03:] = MovieAnimation.StaticText(
                    desc,
                    bold=True,
                    align='left',
                )
                sd[0.09:0.92, 0.005:0.98] = grid
                sd[0.88:0.94, 0.03:0.97] = MovieAnimation.StaticText(
                    '       '.join(['Inputs        Synthesized'] * 3),
                    bold=True,
                    align='center',
                )

        if self.export:
            self.director.export(target_mb=self.target_mb)


if __name__ == '__main__':
    TaskGroup(
        main=Main(cuda=0),
        homepage=Homepage(cuda=0),
    ).run()
