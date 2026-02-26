from __future__ import annotations

# import modulues
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt

import torch
from torch import Tensor
import numpy as np
import os
import pandas as pd
import re

# import rfstudio modules
from rfstudio.engine.task import Task, TaskGroup
from rfstudio.ui import console
from rfstudio.utils.pretty import P
from rfstudio.loss import ChamferDistanceMetric,PSNRLoss,LPIPSLoss,SSIMLoss
from rfstudio.graphics import Points, TriangleMesh
from rfstudio.graphics.shaders import PrettyShader, DepthShader, NormalShader
from rfstudio.graphics import DepthImages, VectorImages, PBRAImages

# import rfstudio_ds modules
from rfstudio_ds.engine.experiment import DS_Experiment
from rfstudio_ds.engine.train import DS_TrainTask
from rfstudio_ds.data import SyntheticDynamicMultiViewBlenderRGBADataset
from rfstudio_ds.model import D_Joint # rgb image loss optimization model
from rfstudio_ds.trainer import D_JointTrainer, JointRegularizationConfig # rgb image loss optimization trainer, dynamic nvdiffrec
from rfstudio_ds.graphics.spatial_sampler import NearestGridSampler
from rfstudio_ds.visualization import Visualizer
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh
from rfstudio_ds.model.density_field.components.encoding_4d import Grid4d_HashEncoding
from rfstudio_ds.nn import MLPDecoderNetwork, Grid4DDecoderNetwork
from rfstudio.io import load_float32_image, open_video_renderer, dump_float32_image
import natsort


@dataclass
class Concat_image_results(Task):

    def process(self, case_inputs, output, extract_type: str):

        frames_per_view = []
        for view_id in range(6):
            view_name = f"test_view{view_id}"
            print(f'processing view {view_name}, extract type {extract_type}')
            frames_per_method = []
            get_gt = False
            for method_path in case_inputs:
                if not get_gt:
                    assert 'dump' in str(method_path)
                    gt_frames = []
                    gt_dir = method_path / view_name / "gt" / f"gt_{extract_type}"
                    gt_files = natsort.natsorted(gt_dir.glob("*.png"))
                    print(f'{method_path} has {len(gt_files)} gt frames')
                    for gt_file in gt_files:
                        gt_frames.append(load_float32_image(gt_file))
                    frames_per_method.append(gt_frames)
                    get_gt = True
                frames = []
                pred_dir = method_path / view_name / "pred" / f"pred_{extract_type}"
                frame_files = natsort.natsorted(pred_dir.glob("*.png"))
                print(f'method {method_path} has {len(frame_files)} frames')
                for frame_file in frame_files:
                    frames.append(load_float32_image(frame_file))
                frames_per_method.append(frames)

            concat_frames = []
            num_frames = len(frames_per_method[0])
            for i in range(num_frames):
                gt_image = frames_per_method[0][i][:, 120:-120, :]
                ours = frames_per_method[1][i][:, 120:-120, :]

                dgmesh = frames_per_method[2][i][:, 120:-120, :]
                atgs = frames_per_method[3][i][:, 120:-120, :]
                d2g = frames_per_method[4][i][:, 120:-120, :]

                if extract_type == 'image':
                    d3g = frames_per_method[5][i][:, 120:-120, :]
                    grid4d = frames_per_method[6][i][:, 120:-120, :]
                    scgs = frames_per_method[7][i][:, 120:-120, :]

                    row1 = torch.cat([gt_image, grid4d, scgs, d3g], dim=1)
                    row2 = torch.cat([ours, d2g, dgmesh, atgs], dim=1)
                    concat_image = torch.cat([row1, row2], dim=0)
                else:
                    concat_image = torch.cat([gt_image, ours, d2g, dgmesh, atgs], dim=1)
                concat_frames.append(concat_image)
                output_path = output / view_name/ f"concat_{extract_type}" / f'frame{i}.png'
                output_path.parent.mkdir(exist_ok=True, parents=True)
                dump_float32_image(output_path, concat_image.clamp(0, 1))
                frames_per_view.append(concat_image)

            output_path = output / f'concat_{extract_type}_{view_name}.mp4'
            output_path.parent.mkdir(exist_ok=True, parents=True)
            print(f'writing {view_name} {extract_type} video...')
            with open_video_renderer(
                output_path,
                fps=max(1, int(num_frames / 5)),
            ) as renderer:
                for i in range(len(concat_frames)):
                    image = concat_frames[i]
                    assert image.min().item() >= 0 and image.max().item() <= 1
                    renderer.write(image)
        output_path = output / f'concat_{extract_type}_total_view.mp4'
        output_path.parent.mkdir(exist_ok=True, parents=True)
        print('writing total view video...')
        with open_video_renderer(
            output_path,
            fps=max(1, int(num_frames / 5)),
        ) as renderer:
            for i in range(len(frames_per_view)):
                image = frames_per_view[i]
                assert image.min().item() >= 0 and image.max().item() <= 1
                renderer.write(image)

    @torch.no_grad()
    def run(self) -> None:
        # cases = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        cases = ['cat', 'rabbit', 'lego', 'deer', 'spidermanfight']
        methods = ['ours', 'dg-mesh', 'atgs', 'dynamic-2dgs', 'deformable-3dgs', 'grid4d','sc-gs']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint')

        for case in cases:
            case_inputs = []
            for method in methods:
                if method == 'ours':
                    case_path = root / case /  'test' / 'dump' / 'eval' / 'extract'
                else:
                    case_path = root / case / 'baselines_eval' / method / 'extract'
                case_inputs.append(case_path)
                output = root / case / 'concat_eval'
            self.process(case_inputs, output, 'image')
            self.process(case_inputs, output, 'normal')
            self.process(case_inputs, output, 'mesh')


@dataclass
class Concat_eval_results(Task):
    
    def parse_eval_txt(self, path: Path, method: str):
        # 多个模式
        pattern_metrics = re.compile(
            r"Test view (\d+), Frame (\d+): PSNR: ([0-9\.Ee+-]+), SSIM: ([0-9\.Ee+-]+), LPIPS: ([0-9\.Ee+-]+)(?:, MAE: ([0-9\.Ee+-]+))?"
        )
        pattern_mean = re.compile(
            r"Test view, Mean PSNR: ([0-9\.Ee+-]+), Mean SSIM: ([0-9\.Ee+-]+), Mean LPIPS: ([0-9\.Ee+-]+)(?:, Mean Normal MAE: ([0-9\.Ee+-]+))?"
        )
        pattern_chamfer = re.compile(
            r"Chamfer Distance: ([0-9\.Ee+-]+)"
        )

        records = []
        chamfers = []
        with open(path, "r") as f:
            for line in f:
                m = pattern_metrics.search(line)
                if m:
                    view, frame, psnr, ssim, lpips, mae = m.groups()
                    records.append({
                        "method": method,
                        "view": int(view),
                        "frame": int(frame),
                        "PSNR": float(psnr),
                        "SSIM": float(ssim),
                        "LPIPS": float(lpips),
                        "MAE": float(mae) if mae is not None else np.nan,
                    })
                # else:
                #     records.append({
                #         "method": method,
                #         "view": 0,
                #         "frame": 0,
                #         "PSNR": 0.1,
                #         "SSIM": 0.1,
                #         "LPIPS": 0.1,
                #         "MAE": 0.1,
                #     })
                m2 = pattern_mean.search(line)
                if m2:
                    psnr, ssim, lpips, mae = m2.groups()
                    records.append({
                        "method": method,
                        "view": -1,
                        "frame": -1,
                        "PSNR": float(psnr),
                        "SSIM": float(ssim),
                        "LPIPS": float(lpips),
                        "MAE": float(mae) if mae is not None else np.nan,   # 确保列存在
                    })
                # else:
                #     records.append({
                #         "method": method,
                #         "view": -1,
                #         "frame": -1,
                #         "PSNR": 0.1,
                #         "SSIM": 0.1,
                #         "LPIPS": 0.1,
                #         "MAE": 0.1,
                #     })
                m3 = pattern_chamfer.search(line)
                if m3:
                    chamfers.append(float(m3.group(1)))
        # 强制补全所有可能的列，缺失的设为 None
        all_columns = ["method", "view", "frame", "PSNR", "SSIM", "LPIPS", "MAE"]
        df = pd.DataFrame(records, columns=all_columns)  # 👈 指定列，缺失自动填 NaN

        df = pd.DataFrame(records)
        chamfer_df = None
        if chamfers:
            chamfer_df = pd.DataFrame([{
                "method": method,
                "ChamferMean": float(f"{np.mean(chamfers):.7g}"),
                "ChamferSum": float(f"{np.sum(chamfers):.7g}")
            }])
        else:
            chamfer_df = pd.DataFrame([{
                "method": method,
                "ChamferMean": np.nan,
                "ChamferSum": np.nan
            }])
        return df, chamfer_df

    def process(self, case_inputs, output: Path, scale: Optional[int] = None):
        dfs = []
        chamfer_dfs = []
        for method, path in case_inputs:
            df, chamfer_df = self.parse_eval_txt(path, method)
            dfs.append(df)
            if chamfer_df is not None:
                chamfer_dfs.append(chamfer_df)
        df_all = pd.concat(dfs, ignore_index=True)

        # Chamfer 汇总
        if chamfer_dfs:
            chamfer_all = pd.concat(chamfer_dfs, ignore_index=True)
            chamfer_all.to_csv(output / "chamfer_summary.csv", index=False)

        # ====== Per-view + Average 分开绘制 ======
        # 忽略 view = -1
        per_view = df_all[df_all["view"] != -1].groupby(["method", "view"])["PSNR"].agg(["mean", "var"]).reset_index()

        # 计算 overall 平均
        overall = df_all[df_all["view"] != -1].groupby("method")["PSNR"].agg(["mean", "var"]).reset_index()

        # 方法排序（按 overall mean 排序）
        methods = overall.sort_values("mean", ascending=False)["method"].tolist()

        # 定义颜色映射（保持方法间一致）
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
        color_map = {m: colors[i] for i, m in enumerate(methods)}

        # ==== 绘制 ====
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 1]})

        # ----- 左边：逐 view -----
        views = sorted(per_view["view"].unique())
        width = 0.12
        x = range(len(views))
        for i, method in enumerate(methods):
            sub = per_view[per_view["method"] == method]
            ax1.bar([xx + i*width for xx in x], sub["mean"], yerr=sub["var"]**0.5,
                    width=width, label=method, alpha=0.7, capsize=3,
                    color=color_map[method])

        ax1.set_xticks([xx + width*(len(methods)/2) for xx in x])
        ax1.set_xticklabels([f"view {v}" for v in views])
        ax1.set_ylabel("PSNR (dB)")
        ax1.set_title("Per-view Mean ± Std")

        # y 轴范围缩放（保持两图一致）
        y_min = min(per_view["mean"].min(), overall["mean"].min()) - 5.5
        y_max = max(per_view["mean"].max(), overall["mean"].max()) + 5.5
        ax1.set_ylim(y_min, y_max)

        # ----- 右边：Average -----
        x2 = range(len(methods))
        for i, method in enumerate(methods):
            row = overall[overall["method"] == method].iloc[0]
            ax2.bar(i, row["mean"], yerr=row["var"]**0.5,
                    width=0.6, alpha=0.7, capsize=3,
                    color=color_map[method])

        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha="right")
        ax2.set_title("Average across views")
        ax2.set_ylim(y_min, y_max)

        # 公共图例
        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor=color_map[m], label=m) for m in methods]
        fig.legend(handles=legend_handles, title="Methods", loc="upper center", ncol=len(methods), frameon=True)

        fig.tight_layout(rect=[0, 0, 1, 0.92])  # 留空间给图例
        fig.savefig(output / "per_view_and_average_psnr.png", dpi=200)
        plt.close(fig)

        # 保存 csv
        per_view.to_csv(output / "per_view_psnr.csv", index=False)
        overall.to_csv(output / "overall_psnr.csv", index=False)

        # ===== 保存总体均值到 txt =====
        metrics = ["PSNR", "SSIM", "LPIPS", "MAE"]
        summary = df_all.groupby("method")[metrics].mean(numeric_only=True).reset_index()

        txt_path = output / "metrics_summary.txt"
        with open(txt_path, "w") as f:
            for _, row in summary.iterrows():
                f.write(f"Method: {row['method']}\n")
                 # 👇 对每个指标安全输出
                if not pd.isna(row.get("PSNR", np.nan)):
                    f.write(f"  Mean PSNR: {row['PSNR']:.6f}\n")
                if not pd.isna(row.get("SSIM", np.nan)):
                    f.write(f"  Mean SSIM: {row['SSIM']:.6f}\n")
                if not pd.isna(row.get("LPIPS", np.nan)):
                    f.write(f"  Mean LPIPS: {row['LPIPS']:.6f}\n")
                if not pd.isna(row.get("MAE", np.nan)):
                    f.write(f"  Mean Normal MAE: {row['MAE']:.6f}\n")
                if chamfer_dfs:
                    cham = chamfer_all[chamfer_all["method"] == row["method"]]
                    if not cham.empty:
                        f.write(f"  Mean Chamfer: {cham['ChamferMean'].values[0]:.7g}\n")
                        f.write(f"  Sum Chamfer: {cham['ChamferSum'].values[0]:.7g}\n")
                f.write("\n")
        
        # 新增：返回 summary，供 run 使用
        if chamfer_dfs:
            # merge chamfer summary
            summary = summary.merge(chamfer_all, on="method", how="left")
        return summary

    @torch.no_grad()
    def run(self) -> None:
        methods = ['ours', 'dynamic-2dgs', 'dg-mesh', 'deformable-3dgs', 'grid4d','sc-gs']
        # cases = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        # root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint')
        cases = ["dog", "k1_double_punch", "penguin", "wolf"]
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/realobject_mv_d_joint')
        # cases = ["band1", "cello1", "hanggling_b2", "ian3", "pizza1"]
        # root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/cmupanonic_mv_d_joint')

        all_summaries = []  # 新增：用于存储所有 case 的 summary

        for case in cases:
            case_inputs = []
            for method in methods:
                if method == 'ours':
                    case_path = root / case /  'test' / 'eval.txt'
                    # case_path = root / case /  'test1' / 'eval.txt' # for cmu eval, change the defualt path, need change default psnr metric
                # elif method in ['deformable-3dgs', 'grid4d','sc-gs']:
                #     case_path = root / case / 'baselines_eval_full' / method / 'eval.txt'
                # else:
                #     case_path = root / case / 'baselines_eval' / method / 'eval.txt'
                else:
                    case_path = root / case / 'baselines_eval_full' / method / 'eval.txt'
                case_inputs.append((method, case_path))
                output = root / case / 'concat_eval' / 'metrics_analysis'
                output.mkdir(exist_ok=True, parents=True)

            summary = self.process(case_inputs, output, None)  # 接收返回的 summary
            summary["case"] = case  # 保留 case 信息，方便 debug
            all_summaries.append(summary)

        # ===== 汇总所有 case =====
        all_df = pd.concat(all_summaries, ignore_index=True)
        # breakpoint()

        # 数值列分开处理
        metrics = ["PSNR", "SSIM", "LPIPS", "MAE", "ChamferMean"]
        mean_summary = all_df.groupby("method")[metrics].mean(numeric_only=True)

        # ChamferSum 单独做 sum
        if "ChamferSum" in all_df.columns:
            sum_chamfer = all_df.groupby("method")["ChamferSum"].sum()
            overall_summary = mean_summary.join(sum_chamfer, how="left").reset_index()
        else:
            overall_summary = mean_summary.reset_index()

        # 保存 CSV
        overall_csv = root / "metrics_summary.csv"
        overall_summary.to_csv(overall_csv, index=False)

        # 保存 TXT
        overall_txt = root / "metrics_summary.txt"
        with open(overall_txt, "w") as f:
            for _, row in overall_summary.iterrows():
                f.write(f"Method: {row['method']}\n")
                f.write(f"  Overall Mean PSNR: {row['PSNR']:.6f}\n")
                f.write(f"  Overall Mean SSIM: {row['SSIM']:.6f}\n")
                f.write(f"  Overall Mean LPIPS: {row['LPIPS']:.6f}\n")
                if not pd.isna(row.get("MAE", np.nan)):
                    f.write(f"  Overall Mean Normal MAE: {row['MAE']:.6f}\n")
                if "ChamferMean" in row:
                    f.write(f"  Overall Mean Chamfer: {row['ChamferMean']:.7g}\n")
                if "ChamferSum" in row:
                    f.write(f"  Overall Sum Chamfer: {row['ChamferSum']:.7g}\n")
                f.write("\n")


@dataclass
class Concat_pbr_results(Task):

    def process(self, case_input, output):

        frames_per_view = []
        for view_id in range(6):
            view_name = f"test_view{view_id}"
            print(f'processing view {view_name}')

            concat_frames = []
            pred_kd_dir = case_input / view_name / "pred" / "pbr_kd"
            pred_metallic_dir = case_input / view_name / "pred" / "pbr_metallic"
            pred_roughness_dir = case_input / view_name / "pred" / "pbr_roughness"
            pred_occ_dir = case_input / view_name / "pred" / "pbr_occ"
            num_frames = len(list(pred_kd_dir.glob("*.png")))
            for i in range(num_frames):
                kd = load_float32_image(pred_kd_dir / f"{i}.png")[:, 120:-120, :]
                metallic = load_float32_image(pred_metallic_dir / f"{i}.png")[:, 120:-120, :]
                roughness = load_float32_image(pred_roughness_dir / f"{i}.png")[:, 120:-120, :]
                occ = load_float32_image(pred_occ_dir / f"{i}.png")[:, 120:-120, :]
                concat_image = torch.cat([kd, metallic, roughness, occ], dim=1)
            
                concat_frames.append(concat_image)
                output_path = output / view_name/ f"concat_pred_pbr" / f'frame{i}.png'
                output_path.parent.mkdir(exist_ok=True, parents=True)
                dump_float32_image(output_path, concat_image.clamp(0, 1))
                frames_per_view.append(concat_image)

            output_path = output / f'concat_pred_pbr_{view_name}.mp4'
            output_path.parent.mkdir(exist_ok=True, parents=True)
            print(f'writing {view_name} concat_pred_pbr video...')
            with open_video_renderer(
                output_path,
                fps=max(1, int(num_frames / 5)),
            ) as renderer:
                for i in range(len(concat_frames)):
                    image = concat_frames[i]
                    assert image.min().item() >= 0 and image.max().item() <= 1
                    renderer.write(image)
        output_path = output / f'concat_pred_pbr_total_view.mp4'
        output_path.parent.mkdir(exist_ok=True, parents=True)
        print('writing total view video...')
        with open_video_renderer(
            output_path,
            fps=max(1, int(num_frames / 5)),
        ) as renderer:
            for i in range(len(frames_per_view)):
                image = frames_per_view[i]
                assert image.min().item() >= 0 and image.max().item() <= 1
                renderer.write(image)

    
    @torch.no_grad()
    def run(self) -> None:
        # cases = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        cases = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint')

        for case in cases:
            case_path = root / case /  'test' / 'dump' / 'eval' / 'extract'
            output = root / case / 'test' / 'dump' /'concat_eval'
            self.process(case_path, output)


@dataclass
class Concat_mesh_normal_sequence(Task):

    def process(self, case_input, output):

        frames_per_view = []
        for view_id in range(6):
            view_name = f"test_view{view_id}"
            print(f'processing view {view_name}')

            concat_frames = []
            concat_mesh_dir = case_input / view_name / "concat_mesh"
            concat_normal_dir = case_input / view_name / "concat_normal"

            num_frames = len(list(concat_normal_dir.glob("*.png")))
            for i in range(num_frames):
                concat_mesh = load_float32_image(concat_mesh_dir / f"frame{i}.png")
                concat_normal = load_float32_image(concat_normal_dir / f"frame{i}.png")
                concat_image = torch.cat([concat_normal, concat_mesh], dim=0)
            
                concat_frames.append(concat_image)
                output_path = output / view_name/ f"concat_mesh_normal" / f'frame{i}.png'
                output_path.parent.mkdir(exist_ok=True, parents=True)
                dump_float32_image(output_path, concat_image.clamp(0, 1))
                frames_per_view.append(concat_image)

            output_path = output / f'concat_mesh_normal_{view_name}.mp4'
            output_path.parent.mkdir(exist_ok=True, parents=True)
            print(f'writing {view_name} concat_mesh_normal video...')
            with open_video_renderer(
                output_path,
                fps=max(1, int(num_frames / 5)),
            ) as renderer:
                for i in range(len(concat_frames)):
                    image = concat_frames[i]
                    assert image.min().item() >= 0 and image.max().item() <= 1
                    renderer.write(image)
        output_path = output / f'concat_mesh_normal_total_view.mp4'
        output_path.parent.mkdir(exist_ok=True, parents=True)
        print('writing total view video...')
        with open_video_renderer(
            output_path,
            fps=max(1, int(num_frames / 5)),
        ) as renderer:
            for i in range(len(frames_per_view)):
                image = frames_per_view[i]
                assert image.min().item() >= 0 and image.max().item() <= 1
                renderer.write(image)

    @torch.no_grad()
    def run(self) -> None:
        # cases = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        cases = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint')

        for case in cases:
            case_path = root / case /  'test' / 'dump' / 'concat_eval'
            output = root / case / 'test' / 'dump' /'concat_eval'
            self.process(case_path, output)


if __name__ == '__main__':
    TaskGroup(
        concatimage = Concat_image_results(cuda=0), 
        concateval = Concat_eval_results(cuda=0), 
        concatpbr = Concat_pbr_results(cuda=0),
        concatmeshnormal = Concat_mesh_normal_sequence(cuda=0)
    ).run()
