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
from collections import defaultdict

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
class Eval_Synthetic_results(Task):

    # -------------------------------------------------------------
    # 解析 mesh_detail.txt
    # -------------------------------------------------------------
    def parse_mesh_detail(self, mesh_detail_path: Path):
        if not mesh_detail_path.exists():
            return {}

        text = mesh_detail_path.read_text()

        # 示例：
        # Frame 0 psdf Metrics: #V 10876 #F 21752 IN>5° 0.547490 CD 0.004652 F1 0.966982 ...
        re_mesh = re.compile(
            r"Frame\s+(\d+)\s+(\S+)\s+Metrics:\s+#V\s+([\d\.eE+-]+)\s+#F\s+([\d\.eE+-]+)"
            r"\s+IN>5°\s+([\d\.eE+-]+)\s+CD\s+([\d\.eE+-]+)\s+F1\s+([\d\.eE+-]+)"
            r"\s+ECD\s+([\d\.eE+-]+)\s+EF1\s+([\d\.eE+-]+)"
        )

        mesh_data = {}
        for m in re_mesh.finditer(text):
            frame = int(m.group(1))
            mesh_method = m.group(2)

            # ★ map mesh name → actual method name
            method = self.mesh_name_map.get(mesh_method, mesh_method)
            entry = {
                "frame": frame,
                "#V": float(m.group(3)),
                "#F": float(m.group(4)),
                "IN5": float(m.group(5)),
                "CD": float(m.group(6)),
                "F1": float(m.group(7)),
                "ECD": float(m.group(8)),
                "EF1": float(m.group(9)),
            }
            mesh_data.setdefault(method, []).append(entry)

        return mesh_data

    # -------------------------------------------------------------
    # 解析 mesh_summary.txt
    # -------------------------------------------------------------
    def parse_mesh_summary(self, mesh_summary_path: Path):
        if not mesh_summary_path.exists():
            return {}

        text = mesh_summary_path.read_text()

        re_sum = re.compile(
            r"(\S+)\s+Average Metrics:\s+#V\s+([\d\.eE+-]+)\s+#F\s+([\d\.eE+-]+)"
            r"\s+IN>5°\s+([\d\.eE+-]+)\s+CD\s+([\d\.eE+-]+)\s+F1\s+([\d\.eE+-]+)"
            r"\s+ECD\s+([\d\.eE+-]+)\s+EF1\s+([\d\.eE+-]+)"
        )

        summary = {}
        for m in re_sum.finditer(text):
            mesh_method = m.group(1)

            # ★ map mesh name → actual method name
            method = self.mesh_name_map.get(mesh_method, mesh_method)
            summary[method] = {
                "#V": float(m.group(2)),
                "#F": float(m.group(3)),
                "IN5": float(m.group(4)),
                "CD": float(m.group(5)),
                "F1": float(m.group(6)),
                "ECD": float(m.group(7)),
                "EF1": float(m.group(8)),
            }
        return summary

    # -------------------------------------------------------------
    # 主处理函数 process()
    # -------------------------------------------------------------
    def process(self, case_inputs, output: Path):
        self.mesh_name_map = {
            "d2dgs": "dynamic-2dgs",
            "dgmesh": "dg-mesh",
            "psdf": "psdf",
            "grid4d": "grid4d",
            "sc-gs": "sc-gs",
            "deformable-3dgs": "deformable-3dgs",
        }

        # -------------------------------
        # 正则
        # -------------------------------
        re_view_frame = re.compile(
            r"Test view (\d+), Frame (\d+): PSNR: ([\d\.eE+-]+), SSIM: ([\d\.eE+-]+), "
            r"LPIPS: ([\d\.eE+-]+), MAE: ([\d\.eE+-]+)"
        )
        re_chamfer = re.compile(r"Frame (\d+): Chamfer Distance: ([\d\.eE+-]+)")
        re_mean_chamfer = re.compile(r"Mean Chamfer Distance: ([\d\.eE+-]+)")
        re_mean_quality = re.compile(
            r"Mean PSNR: ([\d\.eE+-]+), Mean SSIM: ([\d\.eE+-]+), "
            r"Mean LPIPS: ([\d\.eE+-]+), Mean Normal MAE: ([\d\.eE+-]+)"
        )

        # ---------------------------------------------------------
        # 解析所有方法 eval.txt
        # ---------------------------------------------------------
        all_methods_frame_view = {}
        all_methods_chamfer = {}
        all_methods_summary = {}

        for method, eval_path in case_inputs:
            if not eval_path.exists():
                print(f"[WARN] eval.txt not found for method {method}: {eval_path}")
                continue

            text = eval_path.read_text()

            # ----------- A: Test view/frame ----------
            frame_view_list = []
            for m in re_view_frame.finditer(text):
                frame_view_list.append({
                    "view": int(m.group(1)),
                    "frame": int(m.group(2)),
                    "PSNR": float(m.group(3)),
                    "SSIM": float(m.group(4)),
                    "LPIPS": float(m.group(5)),
                    "MAE": float(m.group(6)),
                })
            all_methods_frame_view[method] = frame_view_list

            # ----------- B: Chamfer Distance ----------
            chamfers = []
            for m in re_chamfer.finditer(text):
                chamfers.append({
                    "frame": int(m.group(1)),
                    "chamfer": float(m.group(2)),
                })
            all_methods_chamfer[method] = chamfers

            # ----------- C: Mean Chamfer ----------
            m = re_mean_chamfer.search(text)
            mean_chamfer = float(m.group(1)) if m else None

            # ----------- D: Mean Image Quality ----------
            m = re_mean_quality.search(text)
            if m:
                mean_psnr = float(m.group(1))
                mean_ssim = float(m.group(2))
                mean_lpips = float(m.group(3))
                mean_mae = float(m.group(4))
            else:
                mean_psnr = mean_ssim = mean_lpips = mean_mae = None

            # ----------- 自动计算 mean -----------
            if frame_view_list and mean_psnr is None:
                psnrs = [x["PSNR"] for x in frame_view_list]
                ssims = [x["SSIM"] for x in frame_view_list]
                lpipss = [x["LPIPS"] for x in frame_view_list]
                maes = [x["MAE"] for x in frame_view_list]
                mean_psnr = sum(psnrs) / len(psnrs)
                mean_ssim = sum(ssims) / len(ssims)
                mean_lpips = sum(lpipss) / len(lpipss)
                mean_mae = sum(maes) / len(maes)

            if chamfers and mean_chamfer is None:
                mean_chamfer = sum(x["chamfer"] for x in chamfers) / len(chamfers)

            # 保存 summary
            all_methods_summary[method] = {
                "mean_psnr": mean_psnr,
                "mean_ssim": mean_ssim,
                "mean_lpips": mean_lpips,
                "mean_mae": mean_mae,
                "mean_chamfer": mean_chamfer,
            }

        # ---------------------------------------------------------
        # 解析 mesh_detail / mesh_summary
        # ---------------------------------------------------------
        mesh_detail_path = output / "mesh_detail.txt"
        mesh_summary_path = output / "mesh_summary.txt"

        mesh_detail = self.parse_mesh_detail(mesh_detail_path)
        mesh_summary = self.parse_mesh_summary(mesh_summary_path)

        # ---------------------------------------------------------
        # detail.txt 输出（标准化表格格式）
        # ---------------------------------------------------------
        # detail.txt 美观对齐
        detail_txt_path = output / "detail.txt"
        # 如果存在，则先删除
        if detail_txt_path.exists():
            os.remove(detail_txt_path)

        detail_columns = [
            "Frame", "View", "Method", "PSNR", "SSIM", "LPIPS", "MAE",
            "Chamfer", "#V", "#F", "IN5", "CD", "F1", "ECD", "EF1"
        ]

        col_widths = {
            "Frame": 6, "View": 6, "Method": 18, "PSNR": 20, "SSIM": 20, "LPIPS": 20, "MAE": 20,
            "Chamfer": 20, "#V": 20, "#F": 20, "IN5": 20, "CD": 20, "F1": 20, "ECD": 20, "EF1": 20
        }

        with detail_txt_path.open("w") as f:
            header = "".join(f"{col:<{col_widths[col]}}" for col in detail_columns)
            f.write(header + "\n")

            all_pairs = set()
            for method, lst in all_methods_frame_view.items():
                for x in lst:
                    all_pairs.add((x["frame"], x["view"]))
            all_pairs = sorted(all_pairs)

            for frame, view in all_pairs:
                for method in all_methods_frame_view.keys():
                    vf_list = all_methods_frame_view[method]
                    vf = next((x for x in vf_list if x["frame"] == frame and x["view"] == view), None)
                    if vf is None:
                        continue
                    chamfer_list = all_methods_chamfer[method]
                    ch = next((x for x in chamfer_list if x["frame"] == frame), None)
                    chamfer_value = ch["chamfer"] if ch else None
                    mesh_entries = mesh_detail.get(method, [])
                    mesh_frame = next((x for x in mesh_entries if x["frame"] == frame), {})

                    row_values = [
                        frame, view, method,
                        vf["PSNR"], vf["SSIM"], vf["LPIPS"], vf["MAE"],
                        chamfer_value,
                        mesh_frame.get("#V", ""), mesh_frame.get("#F", ""),
                        mesh_frame.get("IN5", ""), mesh_frame.get("CD", ""),
                        mesh_frame.get("F1", ""), mesh_frame.get("ECD", ""), mesh_frame.get("EF1", "")
                    ]

                    row = ""
                    for col, val in zip(detail_columns, row_values):
                        if isinstance(val, float):
                            row += f"{val:<{col_widths[col]}.8f}"
                        else:
                            row += f"{str(val):<{col_widths[col]}}"
                    f.write(row + "\n")

        # ---------------------------------------------------------
        # summary.txt 输出（标准化表格格式）
        # ---------------------------------------------------------
        # summary.txt 美观对齐输出
        summary_txt_path = output / "summary.txt"
        # 如果存在，则先删除
        if summary_txt_path.exists():
            os.remove(summary_txt_path)

        summary_columns = [
            "Method", "PSNR", "SSIM", "LPIPS", "MAE", "Chamfer",
            "#V", "#F", "IN5", "CD", "F1", "ECD", "EF1"
        ]

        # 为每列指定宽度
        col_widths = {
            "Method": 18,
            "PSNR": 20, "SSIM": 20, "LPIPS": 20, "MAE": 20, "Chamfer": 20,
            "#V": 20, "#F": 20, "IN5": 20, "CD": 20, "F1": 20, "ECD": 20, "EF1": 20
        }


        # 写表头
        with summary_txt_path.open("w") as f:
            header = "".join(f"{col:<{col_widths[col]}}" for col in summary_columns)
            f.write(header + "\n")

            for method, summary in all_methods_summary.items():
                mesh_summary_data = mesh_summary.get(method, {})

                row_values = [
                    method,
                    summary.get("mean_psnr", ""),
                    summary.get("mean_ssim", ""),
                    summary.get("mean_lpips", ""),
                    summary.get("mean_mae", ""),
                    summary.get("mean_chamfer", ""),
                    mesh_summary_data.get("#V", ""),
                    mesh_summary_data.get("#F", ""),
                    mesh_summary_data.get("IN5", ""),
                    mesh_summary_data.get("CD", ""),
                    mesh_summary_data.get("F1", ""),
                    mesh_summary_data.get("ECD", ""),
                    mesh_summary_data.get("EF1", ""),
                ]

                # 格式化每一列
                row = ""
                for col, val in zip(summary_columns, row_values):
                    if isinstance(val, float):
                        row += f"{val:<{col_widths[col]}.8f}"  # 小数点6位
                    else:
                        row += f"{str(val):<{col_widths[col]}}"
                f.write(row + "\n")
            
        # 保存 summary，并合并 mesh_summary
        for method, summary in all_methods_summary.items():
            mesh_summary_data = mesh_summary.get(method, {})
            for k, v in mesh_summary_data.items():
                summary[k] = v  # 添加 mesh_summary 指标

        return all_methods_summary

    
    @torch.no_grad()
    def run(self) -> None:
        print(f'Processing: eval synthetic results...')
        methods = ['psdf', 'dynamic-2dgs', 'dg-mesh', 'deformable-3dgs', 'grid4d','sc-gs']
        cases = ['toy', 'cat', 'rabbit', 'lego', 'footballplayer', 'deer', 'spidermanfight']
        # cases = ['toy']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint')
        
        case_summaries = []  # 用于存储所有 case 的 summary
        
        for case in cases:
            methods_eval_txt = []
            for method in methods:
                if method == 'psdf':
                    eval_txt = root / case /  'test' / 'eval.txt'
                elif method in ['deformable-3dgs', 'grid4d','sc-gs']:
                    eval_txt = root / case /  'baselines_eval_full' / method / 'eval.txt'
                else:
                    eval_txt = root / case /  'baselines_eval' / method / 'eval.txt'
                methods_eval_txt.append((method, eval_txt))
            output_root = root / case / 'gather_metrics'
            output_root.mkdir(parents=True, exist_ok=True)
            
            
            case_summary = self.process(methods_eval_txt, output_root)
            case_summary["case"] = case  # 保留 case 信息，方便 debug
            case_summaries.append(case_summary)
        
        # ---------------------------------------------------------
        # 计算跨案例平均
        # ---------------------------------------------------------
        from collections import defaultdict
        sum_metrics = defaultdict(lambda: defaultdict(float))
        count_metrics = defaultdict(int)

        for case_summary in case_summaries:
            for method, metrics in case_summary.items():
                if method == "case":
                    continue
                count_metrics[method] += 1
                for key, value in metrics.items():
                    if value is not None:
                        sum_metrics[method][key] += value

        avg_case_summary = {}
        for method, metrics in sum_metrics.items():
            avg_case_summary[method] = {}
            for key, total in metrics.items():
                avg_case_summary[method][key] = total / count_metrics[method]

        # ---------------------------------------------------------
        # 写 overall_summary.txt
        # ---------------------------------------------------------
        overall_summary_path = root / 'overall_summary.txt'
        summary_columns = [
            "Method", "PSNR", "SSIM", "LPIPS", "MAE", "Chamfer",
            "#V", "#F", "IN5", "CD", "F1", "ECD", "EF1"
        ]

        col_widths = {
            "Method": 18,
            "PSNR": 20, "SSIM": 20, "LPIPS": 20, "MAE": 20, "Chamfer": 20,
            "#V": 20, "#F": 20, "IN5": 20, "CD": 20, "F1": 20, "ECD": 20, "EF1": 20
        }

        # 删除已存在文件
        if overall_summary_path.exists():
            os.remove(overall_summary_path)

        with overall_summary_path.open("w") as f:
            header = "".join(f"{col:<{col_widths[col]}}" for col in summary_columns)
            f.write(header + "\n")

            for method, summary in avg_case_summary.items():
                # 因为平均计算时可能没有 mesh_summary 数据，这里补空字符串
                row_values = [
                    method,
                    summary.get("mean_psnr", ""),
                    summary.get("mean_ssim", ""),
                    summary.get("mean_lpips", ""),
                    summary.get("mean_mae", ""),
                    summary.get("mean_chamfer", ""),
                    summary.get("#V", ""),
                    summary.get("#F", ""),
                    summary.get("IN5", ""),
                    summary.get("CD", ""),
                    summary.get("F1", ""),
                    summary.get("ECD", ""),
                    summary.get("EF1", ""),
                ]

                row = ""
                for col, val in zip(summary_columns, row_values):
                    if isinstance(val, float):
                        row += f"{val:<{col_widths[col]}.8f}"
                    else:
                        row += f"{str(val):<{col_widths[col]}}"
                f.write(row + "\n")

        print(f"Overall summary saved to {overall_summary_path}")

@dataclass
class Eval_Synthetic_Ablation_results(Task):

    # -------------------------------------------------------------
    # 主处理函数 process()
    # -------------------------------------------------------------
    def process(self, case_inputs, output: Path):

        # -------------------------------
        # 正则
        # -------------------------------
        re_view_frame = re.compile(
            r"Test view (\d+), Frame (\d+): PSNR: ([\d\.eE+-]+), SSIM: ([\d\.eE+-]+), "
            r"LPIPS: ([\d\.eE+-]+), MAE: ([\d\.eE+-]+)"
        )
        re_chamfer = re.compile(r"Frame (\d+): Chamfer Distance: ([\d\.eE+-]+)")
        re_mean_chamfer = re.compile(r"Mean Chamfer Distance: ([\d\.eE+-]+)")
        re_mean_quality = re.compile(
            r"Mean PSNR: ([\d\.eE+-]+), Mean SSIM: ([\d\.eE+-]+), "
            r"Mean LPIPS: ([\d\.eE+-]+), Mean Normal MAE: ([\d\.eE+-]+)"
        )

        # ---------------------------------------------------------
        # 解析所有方法 eval.txt
        # ---------------------------------------------------------
        all_methods_frame_view = {}
        all_methods_chamfer = {}
        all_methods_summary = {}

        for method, eval_path in case_inputs:
            if not eval_path.exists():
                print(f"[WARN] eval.txt not found for method {method}: {eval_path}")
                continue

            text = eval_path.read_text()

            # ----------- A: Test view/frame ----------
            frame_view_list = []
            for m in re_view_frame.finditer(text):
                frame_view_list.append({
                    "view": int(m.group(1)),
                    "frame": int(m.group(2)),
                    "PSNR": float(m.group(3)),
                    "SSIM": float(m.group(4)),
                    "LPIPS": float(m.group(5)),
                    "MAE": float(m.group(6)),
                })
            all_methods_frame_view[method] = frame_view_list

            # ----------- B: Chamfer Distance ----------
            chamfers = []
            for m in re_chamfer.finditer(text):
                chamfers.append({
                    "frame": int(m.group(1)),
                    "chamfer": float(m.group(2)),
                })
            all_methods_chamfer[method] = chamfers

            # ----------- C: Mean Chamfer ----------
            m = re_mean_chamfer.search(text)
            mean_chamfer = float(m.group(1)) if m else None

            # ----------- D: Mean Image Quality ----------
            m = re_mean_quality.search(text)
            if m:
                mean_psnr = float(m.group(1))
                mean_ssim = float(m.group(2))
                mean_lpips = float(m.group(3))
                mean_mae = float(m.group(4))
            else:
                mean_psnr = mean_ssim = mean_lpips = mean_mae = None

            # ----------- 自动计算 mean -----------
            if frame_view_list and mean_psnr is None:
                psnrs = [x["PSNR"] for x in frame_view_list]
                ssims = [x["SSIM"] for x in frame_view_list]
                lpipss = [x["LPIPS"] for x in frame_view_list]
                maes = [x["MAE"] for x in frame_view_list]
                mean_psnr = sum(psnrs) / len(psnrs)
                mean_ssim = sum(ssims) / len(ssims)
                mean_lpips = sum(lpipss) / len(lpipss)
                mean_mae = sum(maes) / len(maes)

            if chamfers and mean_chamfer is None:
                mean_chamfer = sum(x["chamfer"] for x in chamfers) / len(chamfers)

            # 保存 summary
            all_methods_summary[method] = {
                "mean_psnr": mean_psnr,
                "mean_ssim": mean_ssim,
                "mean_lpips": mean_lpips,
                "mean_mae": mean_mae,
                "mean_chamfer": mean_chamfer,
            }


        # ---------------------------------------------------------
        # detail.txt 输出（标准化表格格式）
        # ---------------------------------------------------------
        # detail.txt 美观对齐
        detail_txt_path = output / "ablation_detail.txt"
        # 如果存在，则先删除
        if detail_txt_path.exists():
            os.remove(detail_txt_path)

        detail_columns = [
            "Frame", "View", "Method", "PSNR", "SSIM", "LPIPS", "MAE",
            "Chamfer",
        ]

        col_widths = {
            "Frame": 6, "View": 6, "Method": 30, "PSNR": 20, "SSIM": 20, "LPIPS": 20, "MAE": 20,
            "Chamfer": 20,
        }

        with detail_txt_path.open("w") as f:
            header = "".join(f"{col:<{col_widths[col]}}" for col in detail_columns)
            f.write(header + "\n")

            all_pairs = set()
            for method, lst in all_methods_frame_view.items():
                for x in lst:
                    all_pairs.add((x["frame"], x["view"]))
            all_pairs = sorted(all_pairs)

            for frame, view in all_pairs:
                for method in all_methods_frame_view.keys():
                    vf_list = all_methods_frame_view[method]
                    vf = next((x for x in vf_list if x["frame"] == frame and x["view"] == view), None)
                    if vf is None:
                        continue
                    chamfer_list = all_methods_chamfer[method]
                    ch = next((x for x in chamfer_list if x["frame"] == frame), None)
                    chamfer_value = ch["chamfer"] if ch else None

                    row_values = [
                        frame, view, method,
                        vf["PSNR"], vf["SSIM"], vf["LPIPS"], vf["MAE"],
                        chamfer_value,
                    ]

                    row = ""
                    for col, val in zip(detail_columns, row_values):
                        if isinstance(val, float):
                            row += f"{val:<{col_widths[col]}.8f}"
                        else:
                            row += f"{str(val):<{col_widths[col]}}"
                    f.write(row + "\n")

        # ---------------------------------------------------------
        # summary.txt 输出（标准化表格格式）
        # ---------------------------------------------------------
        # summary.txt 美观对齐输出
        summary_txt_path = output / "ablation_summary.txt"
        # 如果存在，则先删除
        if summary_txt_path.exists():
            os.remove(summary_txt_path)

        summary_columns = [
            "Method", "PSNR", "SSIM", "LPIPS", "MAE", "Chamfer",
        ]

        # 为每列指定宽度
        col_widths = {
            "Method": 30,
            "PSNR": 20, "SSIM": 20, "LPIPS": 20, "MAE": 20, "Chamfer": 20,
        }


        # 写表头
        with summary_txt_path.open("w") as f:
            header = "".join(f"{col:<{col_widths[col]}}" for col in summary_columns)
            f.write(header + "\n")

            for method, summary in all_methods_summary.items():

                row_values = [
                    method,
                    summary.get("mean_psnr", ""),
                    summary.get("mean_ssim", ""),
                    summary.get("mean_lpips", ""),
                    summary.get("mean_mae", ""),
                    summary.get("mean_chamfer", ""),
                ]

                # 格式化每一列
                row = ""
                for col, val in zip(summary_columns, row_values):
                    if isinstance(val, float):
                        row += f"{val:<{col_widths[col]}.8f}"  # 小数点6位
                    else:
                        row += f"{str(val):<{col_widths[col]}}"
                f.write(row + "\n")
            
        return all_methods_summary

    
    @torch.no_grad()
    def run(self) -> None:
        print(f'Processing: eval synthetic results...')
        methods = ['psdf', 'albation-geometry-10000', 'albation-geometry-10111', 'albation-geometry-11000', 'albation-geometry-11100','albation-geometry-11110', 'albation-shader-nopbr', 'albation-texture-kplane']
        cases = ['toy', 'rabbit']
        # cases = ['toy']
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint')
        
        case_summaries = []  # 用于存储所有 case 的 summary
        
        for case in cases:
            methods_eval_txt = []
            for method in methods:
                if method == 'psdf':
                    eval_txt = root / case /  'test' / 'eval.txt'
                else:
                    eval_txt = root / case / method / 'eval.txt'
                methods_eval_txt.append((method, eval_txt))
            output_root = root / case / 'gather_metrics'
            output_root.mkdir(parents=True, exist_ok=True)
            
            
            case_summary = self.process(methods_eval_txt, output_root)
            case_summary["case"] = case  # 保留 case 信息，方便 debug
            case_summaries.append(case_summary)
        
        # ---------------------------------------------------------
        # 计算跨案例平均
        # ---------------------------------------------------------
        from collections import defaultdict
        sum_metrics = defaultdict(lambda: defaultdict(float))
        count_metrics = defaultdict(int)

        for case_summary in case_summaries:
            for method, metrics in case_summary.items():
                if method == "case":
                    continue
                count_metrics[method] += 1
                for key, value in metrics.items():
                    if value is not None:
                        sum_metrics[method][key] += value

        avg_case_summary = {}
        for method, metrics in sum_metrics.items():
            avg_case_summary[method] = {}
            for key, total in metrics.items():
                avg_case_summary[method][key] = total / count_metrics[method]

        # ---------------------------------------------------------
        # 写 overall_summary.txt
        # ---------------------------------------------------------
        overall_summary_path = root / 'overall_ablation_summary.txt'
        summary_columns = [
            "Method", "PSNR", "SSIM", "LPIPS", "MAE", "Chamfer",
        ]

        col_widths = {
            "Method": 30,
            "PSNR": 20, "SSIM": 20, "LPIPS": 20, "MAE": 20, "Chamfer": 20,
        }

        # 删除已存在文件
        if overall_summary_path.exists():
            os.remove(overall_summary_path)

        with overall_summary_path.open("w") as f:
            header = "".join(f"{col:<{col_widths[col]}}" for col in summary_columns)
            f.write(header + "\n")

            for method, summary in avg_case_summary.items():
                # 因为平均计算时可能没有 mesh_summary 数据，这里补空字符串
                row_values = [
                    method,
                    summary.get("mean_psnr", ""),
                    summary.get("mean_ssim", ""),
                    summary.get("mean_lpips", ""),
                    summary.get("mean_mae", ""),
                    summary.get("mean_chamfer", ""),
                ]

                row = ""
                for col, val in zip(summary_columns, row_values):
                    if isinstance(val, float):
                        row += f"{val:<{col_widths[col]}.8f}"
                    else:
                        row += f"{str(val):<{col_widths[col]}}"
                f.write(row + "\n")

        print(f"Overall summary saved to {overall_summary_path}")
  
@dataclass
class Eval_Diva_results(Task):
    """新版：支持新格式 eval.txt / mesh_detail.txt / mesh_summary.txt"""

    # -------------------------------------------------------------
    # 解析 mesh_detail.txt
    # -------------------------------------------------------------
    def parse_mesh_detail(self, mesh_detail_path: Path):
        if not mesh_detail_path.exists():
            return {}

        text = mesh_detail_path.read_text()

        re_mesh = re.compile(
            r"Frame\s+(\d+)\s+(\S+)\s+Metrics:\s+#V\s+([\d\.eE+-]+)\s+#F\s+([\d\.eE+-]+)"
            r"\s+Aspect Ratio > 4 \(%\)\s+([\d\.eE+-]+)\s+Radius Ratio > 4 \(%\)\s+([\d\.eE+-]+)\s+Min Angle < 10 \(%\)\s+([\d\.eE+-]+)"
        )

        mesh_data = {}
        for m in re_mesh.finditer(text):
            frame = int(m.group(1))
            mesh_method = m.group(2)
            method = self.mesh_name_map.get(mesh_method, mesh_method)
            entry = {
                "frame": frame,
                "#V": float(m.group(3)),
                "#F": float(m.group(4)),
                "Aspect>4(%)": float(m.group(5)),
                "Radius>4(%)": float(m.group(6)),
                "MinAngle<10(%)": float(m.group(7)),
            }
            mesh_data.setdefault(method, []).append(entry)
        return mesh_data

    # -------------------------------------------------------------
    # 解析 mesh_summary.txt
    # -------------------------------------------------------------
    def parse_mesh_summary(self, mesh_summary_path: Path):
        if not mesh_summary_path.exists():
            return {}

        text = mesh_summary_path.read_text()

        re_sum = re.compile(
            r"(\S+)\s+Average Metrics:\s+#V\s+([\d\.eE+-]+)\s+#F\s+([\d\.eE+-]+)"
            r"\s+Aspect Ratio > 4 \(%\)\s+([\d\.eE+-]+)\s+Radius Ratio > 4 \(%\)\s+([\d\.eE+-]+)\s+Min Angle < 10 \(%\)\s+([\d\.eE+-]+)"
        )

        summary = {}
        for m in re_sum.finditer(text):
            mesh_method = m.group(1)
            method = self.mesh_name_map.get(mesh_method, mesh_method)
            summary[method] = {
                "#V": float(m.group(2)),
                "#F": float(m.group(3)),
                "Aspect>4(%)": float(m.group(4)),
                "Radius>4(%)": float(m.group(5)),
                "MinAngle<10(%)": float(m.group(6)),
            }
        return summary

    # -------------------------------------------------------------
    # 处理单个 case
    # -------------------------------------------------------------
    def process(self, case_inputs, output: Path):
        self.mesh_name_map = {
            "d2dgs": "dynamic-2dgs",
            "dgmesh": "dg-mesh",
            "psdf": "psdf",
            "grid4d": "grid4d",
            "sc-gs": "sc-gs",
            "deformable-3dgs": "deformable-3dgs",
        }

        # 匹配新格式 eval.txt
        re_view_frame = re.compile(
            r"Test view (\d+), Frame (\d+): PSNR: ([\d\.eE+-]+), SSIM: ([\d\.eE+-]+), LPIPS: ([\d\.eE+-]+)"
        )

        all_methods_frame_view = {}
        all_methods_summary = {}

        # ---------------------------------------------------------
        # 解析 eval.txt
        # ---------------------------------------------------------
        for method, eval_path in case_inputs:
            if not eval_path.exists():
                print(f"[WARN] eval.txt not found for {method}: {eval_path}")
                continue

            text = eval_path.read_text()

            frame_view_list = []
            for m in re_view_frame.finditer(text):
                frame_view_list.append({
                    "view": int(m.group(1)),
                    "frame": int(m.group(2)),
                    "PSNR": float(m.group(3)),
                    "SSIM": float(m.group(4)),
                    "LPIPS": float(m.group(5)),
                })

            all_methods_frame_view[method] = frame_view_list

            # 自动计算平均值
            if frame_view_list:
                psnrs = [x["PSNR"] for x in frame_view_list]
                ssims = [x["SSIM"] for x in frame_view_list]
                lpipss = [x["LPIPS"] for x in frame_view_list]

                mean_psnr = sum(psnrs) / len(psnrs)
                mean_ssim = sum(ssims) / len(ssims)
                mean_lpips = sum(lpipss) / len(lpipss)
            else:
                mean_psnr = mean_ssim = mean_lpips = None

            all_methods_summary[method] = {
                "mean_psnr": mean_psnr,
                "mean_ssim": mean_ssim,
                "mean_lpips": mean_lpips,
            }

        # ---------------------------------------------------------
        # 解析 mesh detail / summary
        # ---------------------------------------------------------
        mesh_detail = self.parse_mesh_detail(output / "mesh_detail.txt")
        mesh_summary = self.parse_mesh_summary(output / "mesh_summary.txt")

        # ---------------------------------------------------------
        # detail.txt 输出
        # ---------------------------------------------------------
        detail_txt_path = output / "detail.txt"
        if detail_txt_path.exists():
            os.remove(detail_txt_path)

        detail_columns = [
            "Frame", "View", "Method", "PSNR", "SSIM", "LPIPS",
            "#V", "#F", "Aspect>4(%)", "Radius>4(%)", "MinAngle<10(%)"
        ]
        col_widths = {c: 20 for c in detail_columns}

        with detail_txt_path.open("w") as f:
            f.write("".join(f"{c:<{col_widths[c]}}" for c in detail_columns) + "\n")

            all_pairs = sorted({(x["frame"], x["view"]) for lst in all_methods_frame_view.values() for x in lst})
            for frame, view in all_pairs:
                for method, lst in all_methods_frame_view.items():
                    vf = next((x for x in lst if x["frame"] == frame and x["view"] == view), None)
                    if vf is None:
                        continue
                    mesh_frame = next((x for x in mesh_detail.get(method, []) if x["frame"] == frame), {})
                    row_values = [
                        frame, view, method,
                        vf["PSNR"], vf["SSIM"], vf["LPIPS"],
                        mesh_frame.get("#V", ""), mesh_frame.get("#F", ""),
                        mesh_frame.get("Aspect>4(%)", ""), mesh_frame.get("Radius>4(%)", ""), mesh_frame.get("MinAngle<10(%)", "")
                    ]
                    row = "".join(
                        f"{val:<{col_widths[col]}.8f}" if isinstance(val, float) else f"{str(val):<{col_widths[col]}}"
                        for col, val in zip(detail_columns, row_values)
                    )
                    f.write(row + "\n")

        # ---------------------------------------------------------
        # summary.txt 输出
        # ---------------------------------------------------------
        summary_txt_path = output / "summary.txt"
        if summary_txt_path.exists():
            os.remove(summary_txt_path)

        summary_columns = [
            "Method", "PSNR", "SSIM", "LPIPS",
            "#V", "#F", "Aspect>4(%)", "Radius>4(%)", "MinAngle<10(%)"
        ]
        col_widths = {c: 20 for c in summary_columns}

        with summary_txt_path.open("w") as f:
            f.write("".join(f"{c:<{col_widths[c]}}" for c in summary_columns) + "\n")
            for method, summary in all_methods_summary.items():
                mesh_data = mesh_summary.get(method, {})
                row_values = [
                    method,
                    summary.get("mean_psnr", ""), summary.get("mean_ssim", ""), summary.get("mean_lpips", ""),
                    mesh_data.get("#V", ""), mesh_data.get("#F", ""),
                    mesh_data.get("Aspect>4(%)", ""), mesh_data.get("Radius>4(%)", ""), mesh_data.get("MinAngle<10(%)", "")
                ]
                row = "".join(
                    f"{val:<{col_widths[col]}.8f}" if isinstance(val, float) else f"{str(val):<{col_widths[col]}}"
                    for col, val in zip(summary_columns, row_values)
                )
                f.write(row + "\n")

        # 合并 mesh_summary
        for method, summary in all_methods_summary.items():
            summary.update(mesh_summary.get(method, {}))

        return all_methods_summary

    # -------------------------------------------------------------
    # 主运行函数
    # -------------------------------------------------------------
    @torch.no_grad()
    def run(self) -> None:
        print("Processing: eval results (new format)...")
        methods = ['psdf', 'dynamic-2dgs', 'dg-mesh', 'deformable-3dgs', 'grid4d', 'sc-gs']
        cases = ["dog", "k1_double_punch", "penguin", "wolf"]
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/diva_mv_d_joint')

        case_summaries = []
        for case in cases:
            methods_eval_txt = []
            for method in methods:
                if method == 'psdf':
                    eval_txt = root / case / 'test' / 'eval.txt'
                else:
                    eval_txt = root / case / 'baselines_eval_full' / method / 'eval.txt'
                methods_eval_txt.append((method, eval_txt))

            output_root = root / case / 'gather_metrics'
            output_root.mkdir(parents=True, exist_ok=True)
            case_summary = self.process(methods_eval_txt, output_root)
            case_summary["case"] = case
            case_summaries.append(case_summary)

        # ---------------------------------------------------------
        # 计算跨 case 平均
        # ---------------------------------------------------------
        sum_metrics = defaultdict(lambda: defaultdict(float))
        count_metrics = defaultdict(int)

        for case_summary in case_summaries:
            for method, metrics in case_summary.items():
                if method == "case":
                    continue
                count_metrics[method] += 1
                for k, v in metrics.items():
                    if isinstance(v, (float, int)):
                        sum_metrics[method][k] += v

        avg_case_summary = {
            method: {k: v / count_metrics[method] for k, v in metrics.items()}
            for method, metrics in sum_metrics.items()
        }

        # ---------------------------------------------------------
        # overall_summary.txt 输出
        # ---------------------------------------------------------
        overall_summary_path = root / 'overall_summary.txt'
        if overall_summary_path.exists():
            os.remove(overall_summary_path)

        summary_columns = [
            "Method", "PSNR", "SSIM", "LPIPS",
            "#V", "#F", "Aspect>4(%)", "Radius>4(%)", "MinAngle<10(%)"
        ]
        col_widths = {c: 20 for c in summary_columns}

        with overall_summary_path.open("w") as f:
            f.write("".join(f"{c:<{col_widths[c]}}" for c in summary_columns) + "\n")
            for method, summary in avg_case_summary.items():
                row_values = [
                    method,
                    summary.get("mean_psnr", ""), summary.get("mean_ssim", ""), summary.get("mean_lpips", ""),
                    summary.get("#V", ""), summary.get("#F", ""),
                    summary.get("Aspect>4(%)", ""), summary.get("Radius>4(%)", ""), summary.get("MinAngle<10(%)", "")
                ]
                row = "".join(
                    f"{val:<{col_widths[col]}.8f}" if isinstance(val, float) else f"{str(val):<{col_widths[col]}}"
                    for col, val in zip(summary_columns, row_values)
                )
                f.write(row + "\n")

        print(f"Overall summary saved to {overall_summary_path}")

@dataclass
class Eval_CMU_results(Task):
    # -------------------------------------------------------------
    # 解析 mesh_detail.txt
    # -------------------------------------------------------------
    def parse_mesh_detail(self, mesh_detail_path: Path):
        if not mesh_detail_path.exists():
            return {}

        text = mesh_detail_path.read_text()

        # 新格式示例：
        # [2025-11-10 10:18:24] Frame 0 psdf Metrics: #V 17902 #F 35792 Aspect Ratio > 4 (%) 2.785539 Radius Ratio > 4 (%) 7.700045 Min Angle < 10 (%) 0.374385
        re_mesh = re.compile(
            r"Frame\s+(\d+)\s+(\S+)\s+Metrics:\s+#V\s+([\d\.eE+-]+)\s+#F\s+([\d\.eE+-]+)"
            r"\s+Aspect Ratio > 4\s+\(%\)\s+([\d\.eE+-]+)"
            r"\s+Radius Ratio > 4\s+\(%\)\s+([\d\.eE+-]+)"
            r"\s+Min Angle < 10\s+\(%\)\s+([\d\.eE+-]+)"
        )

        mesh_data = {}
        for m in re_mesh.finditer(text):
            frame = int(m.group(1))
            mesh_method = m.group(2)
            method = self.mesh_name_map.get(mesh_method, mesh_method)
            entry = {
                "frame": frame,
                "#V": float(m.group(3)),
                "#F": float(m.group(4)),
                "Aspect>4(%)": float(m.group(5)),
                "Radius>4(%)": float(m.group(6)),
                "MinAngle<10(%)": float(m.group(7)),
            }
            mesh_data.setdefault(method, []).append(entry)

        return mesh_data

    # -------------------------------------------------------------
    # 解析 mesh_summary.txt
    # -------------------------------------------------------------
    def parse_mesh_summary(self, mesh_summary_path: Path):
        if not mesh_summary_path.exists():
            return {}

        text = mesh_summary_path.read_text()

        # 新格式：
        # psdf Average Metrics: #V 17902 #F 35792 Aspect Ratio > 4 (%) 2.785539 Radius Ratio > 4 (%) 7.700045 Min Angle < 10 (%) 0.374385
        re_sum = re.compile(
            r"(\S+)\s+Average Metrics:\s+#V\s+([\d\.eE+-]+)\s+#F\s+([\d\.eE+-]+)"
            r"\s+Aspect Ratio > 4\s+\(%\)\s+([\d\.eE+-]+)"
            r"\s+Radius Ratio > 4\s+\(%\)\s+([\d\.eE+-]+)"
            r"\s+Min Angle < 10\s+\(%\)\s+([\d\.eE+-]+)"
        )

        summary = {}
        for m in re_sum.finditer(text):
            mesh_method = m.group(1)
            method = self.mesh_name_map.get(mesh_method, mesh_method)
            summary[method] = {
                "#V": float(m.group(2)),
                "#F": float(m.group(3)),
                "Aspect>4(%)": float(m.group(4)),
                "Radius>4(%)": float(m.group(5)),
                "MinAngle<10(%)": float(m.group(6)),
            }
        return summary

    # -------------------------------------------------------------
    # 主处理函数 process()
    # -------------------------------------------------------------
    def process(self, case_inputs, output: Path):
        self.mesh_name_map = {
            "d2dgs": "dynamic-2dgs",
            "dgmesh": "dg-mesh",
            "psdf": "psdf",
            "grid4d": "grid4d",
            "sc-gs": "sc-gs",
            "deformable-3dgs": "deformable-3dgs",
        }

        # 新 eval.txt 格式：
        # [2025-09-22 07:08:18] Frame 0: Chamfer Distance: 0.0009810025803744793
        re_chamfer = re.compile(r"Frame\s+(\d+):\s+Chamfer Distance:\s+([\d\.eE+-]+)")

        all_methods_chamfer = {}
        all_methods_summary = {}

        for method, eval_path in case_inputs:
            if not eval_path.exists():
                print(f"[WARN] eval.txt not found for method {method}: {eval_path}")
                continue

            text = eval_path.read_text()
            chamfers = []
            for m in re_chamfer.finditer(text):
                chamfers.append({"frame": int(m.group(1)), "chamfer": float(m.group(2))})
            all_methods_chamfer[method] = chamfers

            if chamfers:
                mean_chamfer = sum(x["chamfer"] for x in chamfers) / len(chamfers)
            else:
                mean_chamfer = None

            all_methods_summary[method] = {"mean_chamfer": mean_chamfer}

        # ---------------------------------------------------------
        # 解析 mesh_detail / mesh_summary
        # ---------------------------------------------------------
        mesh_detail_path = output / "mesh_detail.txt"
        mesh_summary_path = output / "mesh_summary.txt"

        mesh_detail = self.parse_mesh_detail(mesh_detail_path)
        mesh_summary = self.parse_mesh_summary(mesh_summary_path)

        # ---------------------------------------------------------
        # detail.txt 输出
        # ---------------------------------------------------------
        detail_txt_path = output / "detail.txt"
        if detail_txt_path.exists():
            os.remove(detail_txt_path)

        detail_columns = [
            "Frame", "Method", "Chamfer", "#V", "#F",
            "Aspect>4(%)", "Radius>4(%)", "MinAngle<10(%)"
        ]
        col_widths = {col: 20 for col in detail_columns}
        col_widths["Frame"] = 8
        col_widths["Method"] = 18

        with detail_txt_path.open("w") as f:
            header = "".join(f"{col:<{col_widths[col]}}" for col in detail_columns)
            f.write(header + "\n")

            all_frames = sorted({x["frame"] for lst in all_methods_chamfer.values() for x in lst})

            for frame in all_frames:
                for method in all_methods_chamfer.keys():
                    chamfer_list = all_methods_chamfer[method]
                    ch = next((x for x in chamfer_list if x["frame"] == frame), None)
                    chamfer_value = ch["chamfer"] if ch else None
                    mesh_entries = mesh_detail.get(method, [])
                    mesh_frame = next((x for x in mesh_entries if x["frame"] == frame), {})

                    row_values = [
                        frame,
                        method,
                        chamfer_value,
                        mesh_frame.get("#V", ""),
                        mesh_frame.get("#F", ""),
                        mesh_frame.get("Aspect>4(%)", ""),
                        mesh_frame.get("Radius>4(%)", ""),
                        mesh_frame.get("MinAngle<10(%)", ""),
                    ]

                    row = ""
                    for col, val in zip(detail_columns, row_values):
                        if isinstance(val, float):
                            row += f"{val:<{col_widths[col]}.8f}"
                        else:
                            row += f"{str(val):<{col_widths[col]}}"
                    f.write(row + "\n")

        # ---------------------------------------------------------
        # summary.txt 输出
        # ---------------------------------------------------------
        summary_txt_path = output / "summary.txt"
        if summary_txt_path.exists():
            os.remove(summary_txt_path)

        summary_columns = [
            "Method", "Chamfer", "#V", "#F",
            "Aspect>4(%)", "Radius>4(%)", "MinAngle<10(%)"
        ]
        col_widths = {col: 20 for col in summary_columns}
        col_widths["Method"] = 18

        with summary_txt_path.open("w") as f:
            header = "".join(f"{col:<{col_widths[col]}}" for col in summary_columns)
            f.write(header + "\n")

            for method, summary in all_methods_summary.items():
                mesh_summary_data = mesh_summary.get(method, {})
                row_values = [
                    method,
                    summary.get("mean_chamfer", ""),
                    mesh_summary_data.get("#V", ""),
                    mesh_summary_data.get("#F", ""),
                    mesh_summary_data.get("Aspect>4(%)", ""),
                    mesh_summary_data.get("Radius>4(%)", ""),
                    mesh_summary_data.get("MinAngle<10(%)", ""),
                ]
                row = ""
                for col, val in zip(summary_columns, row_values):
                    if isinstance(val, float):
                        row += f"{val:<{col_widths[col]}.8f}"
                    else:
                        row += f"{str(val):<{col_widths[col]}}"
                f.write(row + "\n")

        # ---------------------------------------------------------
        # 合并 mesh_summary
        # ---------------------------------------------------------
        for method, summary in all_methods_summary.items():
            mesh_summary_data = mesh_summary.get(method, {})
            for k, v in mesh_summary_data.items():
                summary[k] = v

        return all_methods_summary

    # -------------------------------------------------------------
    # 运行入口 run()
    # -------------------------------------------------------------
    @torch.no_grad()
    def run(self) -> None:
        print(f'Processing: eval results (Chamfer + Mesh)...')
        methods = ['psdf', 'dynamic-2dgs', 'dg-mesh', 'deformable-3dgs', 'grid4d','sc-gs']
        cases = ["band1", "cello1", "hanggling_b2", "ian3", "pizza1"]
        root = Path('/data3/gaochong/project/RadianceFieldStudio/outputs/cmupanonic_mv_d_joint')

        case_summaries = []

        for case in cases:
            methods_eval_txt = []
            for method in methods:
                if method == 'psdf':
                    eval_txt = root / case / 'test' / 'eval.txt'
                else:
                    eval_txt = root / case / 'baselines_eval_full' / method / 'eval.txt'
                methods_eval_txt.append((method, eval_txt))

            output_root = root / case / 'gather_metrics'
            output_root.mkdir(parents=True, exist_ok=True)

            case_summary = self.process(methods_eval_txt, output_root)
            case_summary["case"] = case
            case_summaries.append(case_summary)

        # ---------------------------------------------------------
        # 计算跨案例平均
        # ---------------------------------------------------------
        sum_metrics = defaultdict(lambda: defaultdict(float))
        count_metrics = defaultdict(int)

        for case_summary in case_summaries:
            for method, metrics in case_summary.items():
                if method == "case":
                    continue
                count_metrics[method] += 1
                for key, value in metrics.items():
                    if value is not None:
                        sum_metrics[method][key] += value

        avg_case_summary = {}
        for method, metrics in sum_metrics.items():
            avg_case_summary[method] = {
                key: total / count_metrics[method] for key, total in metrics.items()
            }

        # ---------------------------------------------------------
        # 写 overall_summary.txt
        # ---------------------------------------------------------
        overall_summary_path = root / 'overall_summary.txt'
        if overall_summary_path.exists():
            os.remove(overall_summary_path)

        summary_columns = [
            "Method", "Chamfer", "#V", "#F",
            "Aspect>4(%)", "Radius>4(%)", "MinAngle<10(%)"
        ]
        col_widths = {col: 20 for col in summary_columns}
        col_widths["Method"] = 18

        with overall_summary_path.open("w") as f:
            header = "".join(f"{col:<{col_widths[col]}}" for col in summary_columns)
            f.write(header + "\n")

            for method, summary in avg_case_summary.items():
                row_values = [
                    method,
                    summary.get("mean_chamfer", ""),
                    summary.get("#V", ""),
                    summary.get("#F", ""),
                    summary.get("Aspect>4(%)", ""),
                    summary.get("Radius>4(%)", ""),
                    summary.get("MinAngle<10(%)", ""),
                ]
                row = ""
                for col, val in zip(summary_columns, row_values):
                    if isinstance(val, float):
                        row += f"{val:<{col_widths[col]}.8f}"
                    else:
                        row += f"{str(val):<{col_widths[col]}}"
                f.write(row + "\n")

        print(f"Overall summary saved to {overall_summary_path}")


if __name__ == '__main__':
    TaskGroup(
        evalsyn = Eval_Synthetic_results(cuda=0),
        evalsynablation = Eval_Synthetic_Ablation_results(cuda=0),
        evaldiva = Eval_Diva_results(cuda=0),
        evalcmu = Eval_CMU_results(cuda=0),
    ).run()

