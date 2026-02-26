from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, List

from rfstudio.engine.task import Task
from rfstudio.io import load_float32_image, open_video_renderer
from rfstudio.ui import console

from natsort import natsorted

@dataclass
class Image2Video(Task):

    input: Path = Path('/data3/gaochong/project/DiVa360/dataset/wolf/select_frames')

    output: Path = Path('/data3/gaochong/project/DiVa360/dataset/wolf/select_frames.mp4')

    fps: float = 48

    duration = 5

    downsample: Optional[float] = None

    target_mb: Optional[float] = None

    extension: Literal['png', 'jpg', 'jpeg'] = 'png'

    rjust_for_sort: str = '0'

    def run(self) -> None:

        image_list = list(self.input.glob(f"*.{self.extension}"))
        image_list = natsorted(image_list)

        if self.duration is not None:
            self.fps = max(1, int(len(image_list) / self.duration))
        
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with open_video_renderer(
            self.output,
            fps=self.fps,
            downsample=self.downsample,
            target_mb=self.target_mb,
        ) as renderer:
            with console.progress('Exporting...') as ptrack:
                for image_path in ptrack(image_list):
                    # renderer.write(load_float32_image(image_path))
                    renderer.write(load_float32_image(image_path, alpha_color=(1,1,1)))


@dataclass
class ExtractImages2Video(Task):

    input: Path = Path('/data3/gaochong/project/RadianceFieldStudio/data/diva360')

    fps: int = 48

    duration = None

    downsample: Optional[float] = None

    target_mb: Optional[float] = None

    extension: Literal['png', 'jpg', 'jpeg'] = 'png'

    def run(self) -> None:

        # 得到input下所有子文件夹的名字
        folder_list = [x for x in self.input.iterdir() if x.is_dir()]
        for folder in folder_list:
            image_path = Path(folder / 'extract_frames_1' / 'cam00')
            if not image_path.exists():
                continue
            output_path = Path(folder / 'cam00.mp4')
            if image_path.exists():
                print('Processing: ', image_path, output_path)
                
                image_list = list(image_path.glob(f"*.{self.extension}"))
                image_list = natsorted(image_list)

                if self.duration is not None:
                    self.fps = max(1, int(len(image_list) / self.duration))

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open_video_renderer(
                    output_path,
                    fps=self.fps,
                    downsample=self.downsample,
                    target_mb=self.target_mb,
                ) as renderer:
                    with console.progress('Exporting...') as ptrack:
                        for image_path in ptrack(image_list):
                            renderer.write(load_float32_image(image_path, alpha_color=(1,1,1)))


@dataclass
class ExtractMultiviewImages2Video(Task):

    cases: List[Path] = None

    fps: float = 24

    duration = 5

    downsample: Optional[float] = None

    target_mb: Optional[float] = None

    extension: Literal['png', 'jpg', 'jpeg'] = 'png'

    multiview_num: int = 6

    def run(self) -> None:
        for input in self.cases:
            print(f"Processing case: {input}")
            self.input = input

            # 得到input下所有子文件夹的名字
            image_list = list(self.input.glob(f"*.{self.extension}"))
            image_list = natsorted(image_list)

            # 按 multiview_num 拆分成multiview_num个列表
            total_images = len(image_list)
            split_size = total_images // self.multiview_num
            splits = [
                image_list[i * split_size : (i + 1) * split_size]
                for i in range(self.multiview_num)
            ]
            
            if self.duration is not None:
                self.fps = max(1, int(split_size / self.duration))
            
            for view_idx, split in enumerate(splits):
                output_path = self.input.parent / 'test_vis' / f"test_view{view_idx}.mp4"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open_video_renderer(
                    output_path,
                    fps=self.fps,
                    downsample=self.downsample,
                    target_mb=self.target_mb,
                ) as renderer:
                    with console.progress('Exporting...') as ptrack:
                        for image_path in ptrack(split):
                            renderer.write(load_float32_image(image_path))


if __name__ == '__main__':
    Image2Video().run()
    # ExtractImages2Video().run()
    
    
    # 定义多个 case
    # cases = [
    #     # Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/toy/test3/dump/test'),
    #     # Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/spidermanfight/test1/dump/test'),
    #     Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/deer/test1/dump/test')
    # ]
    # ExtractMultiviewImages2Video(cases=cases).run()



# from dataclasses import dataclass
# from pathlib import Path
# from typing import Literal, Optional

# from rfstudio.engine.task import Task
# from rfstudio.io import load_float32_image, open_video_renderer
# from rfstudio.ui import console

# import os
# from collections import defaultdict

# @dataclass
# class Image2Video(Task):

#     input: Path = Path("/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/cat/test/dump/eval/extract/orbit/gt/gt_image")

#     output: Path = Path("/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/cat/test/dump/eval/extract/orbit/gt/gt_image.mp4")

#     fps: float = 48

#     max_duration: Optional[float] = None

#     downsample: Optional[float] = None

#     target_mb: Optional[float] = None

#     extension: Literal['png', 'jpg', 'jpeg'] = 'png'

#     rjust_for_sort: str = '0'

#     def run(self) -> None:

#         image_list = list(self.input.glob(f"*.{self.extension}"))
#         maxlen = max(len(p.stem) for p in image_list)
#         image_list.sort(key=lambda p: p.stem.rjust(maxlen, self.rjust_for_sort))

#         if self.max_duration is not None:
#             image_list = image_list[:int(self.max_duration * self.fps)]

#         self.output.parent.mkdir(parents=True, exist_ok=True)
#         with open_video_renderer(
#             self.output,
#             fps=self.fps,
#             downsample=self.downsample,
#             target_mb=self.target_mb,
#         ) as renderer:
#             with console.progress('Exporting...') as ptrack:
#                 for image_path in ptrack(image_list):
#                     renderer.write(load_float32_image(image_path))

# def get_all_frame_file_paths(base_dir):
#     frames = [f for f in os.listdir(base_dir) if f.startswith("frame_") and os.path.isdir(os.path.join(base_dir, f))]
#     frames.sort(key=lambda x: int(x.split("_")[1]))

#     if not frames:
#         print("未找到 frame 文件夹")
#         return {}

#     first_frame_dir = os.path.join(base_dir, frames[0])
#     file_types = [f for f in os.listdir(first_frame_dir) if os.path.isfile(os.path.join(first_frame_dir, f))]

#     file_paths_dict = defaultdict(list)
#     for frame in frames:
#         frame_dir = os.path.join(base_dir, frame)
#         for ftype in file_types:
#             file_path = os.path.join(frame_dir, ftype)
#             if os.path.exists(file_path):
#                 file_paths_dict[ftype].append(file_path)
#             else:
#                 print(f"警告: {file_path} 不存在!")

#     return dict(file_paths_dict)

# def render_all_videos(base_dir, video_output_dir):
#     file_paths_dict = get_all_frame_file_paths(base_dir)

#     for file_type, paths in file_paths_dict.items():
#         num_frames = len(paths)
#         fps = int(num_frames / 3)
#         # 为每种类型创建一个临时目录，存放按帧排序的软链接（或拷贝）
#         temp_dir = Path(video_output_dir) / file_type.replace(".png", "")
#         temp_dir.mkdir(parents=True, exist_ok=True)

#         # 将图片路径按帧编号排序
#         sorted_paths = sorted(paths, key=lambda p: int(Path(p).parent.name.split("_")[1]))

#         # 创建软链接（避免拷贝大文件）
#         for i, src_path in enumerate(sorted_paths):
#             dst_path = temp_dir / f"{i:04d}.png"
#             if not dst_path.exists():
#                 os.symlink(src_path, dst_path)

#         # 输出视频文件路径
#         video_path = Path(video_output_dir) / f"{file_type.replace('.png', '')}.mp4"
#         video_path.parent.mkdir(parents=True, exist_ok=True)
        
#         # 生成视频
#         with open_video_renderer(
#                 video_path,
#                 fps=fps,
#         ) as renderer:
#             with console.progress('Exporting...') as ptrack:
#                 for image_path in ptrack(sorted_paths):
#                     renderer.write(load_float32_image(Path(image_path)))

# if __name__ == '__main__':
#     Image2Video().run()
    
#     # base_dir = Path("/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/toy/gather_image/test_view0")
#     # video_output_dir = base_dir / "videos"
#     # render_all_videos(base_dir, video_output_dir)
    
#     # base_dir_list = [
#     #     "/data3/gaochong/project/RadianceFieldStudio/outputs/diva_mv_d_joint/wolf/gather_image/test_view0",
#     #     "/data3/gaochong/project/RadianceFieldStudio/outputs/diva_mv_d_joint/dog/gather_image/test_view0",
#     #     "/data3/gaochong/project/RadianceFieldStudio/outputs/diva_mv_d_joint/k1_double_punch/gather_image/test_view0",
#     #     "/data3/gaochong/project/RadianceFieldStudio/outputs/diva_mv_d_joint/penguin/gather_image/test_view0",
                
#     #     # "/data3/gaochong/project/RadianceFieldStudio/outputs/cmupanonic_mv_d_joint/pizza1/gather_image/test_view5",
#     #     # "/data3/gaochong/project/RadianceFieldStudio/outputs/cmupanonic_mv_d_joint/ian3/gather_image/test_view4",
#     #     # "/data3/gaochong/project/RadianceFieldStudio/outputs/cmupanonic_mv_d_joint/hanggling_b2/gather_image/test_view5",
#     #     # "/data3/gaochong/project/RadianceFieldStudio/outputs/cmupanonic_mv_d_joint/cello1/gather_image/test_view1",
#     #     # "/data3/gaochong/project/RadianceFieldStudio/outputs/cmupanonic_mv_d_joint/band1/gather_image/test_view5",
        
#     #     # "/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/toy/gather_image/test_view5",
#     #     # "/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/spidermanfight/gather_image/test_view5",
#     #     # "/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/rabbit/gather_image/test_view1",
#     #     # "/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/lego/gather_image/test_view1",
#     #     # "/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/footballplayer/gather_image/test_view2",
#     #     # "/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/deer/gather_image/test_view0",
#     #     # "/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/cat/gather_image/test_view0",
#     # ]
#     # for base_dir in base_dir_list:
#     #     base_dir = Path(base_dir)
#     #     video_output_dir = base_dir / "videos"
#     #     render_all_videos(base_dir, video_output_dir)
