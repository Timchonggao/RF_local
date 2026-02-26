from __future__ import annotations

# import modules
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Literal, List, Union
import matplotlib.pyplot as plt

import torch
from torch import Tensor



def plot_curve(
    times, 
    gt_data: Optional[Tensor] = None, gt_flow: Optional[Tensor] = None, 
    pred_data: Optional[Tensor] = None, pred_flow: Optional[Tensor] = None, 
    pred_data_static_component: Optional[Tensor] = None,
    pred_data_poly_component: Optional[Tensor] = None, pred_flow_poly_component: Optional[Tensor] = None,
    pred_data_low_freq_component: Optional[Tensor] = None, pred_flow_low_freq_component: Optional[Tensor] = None,
    pred_data_mid_freq_component: Optional[Tensor] = None, pred_flow_mid_freq_component: Optional[Tensor] = None,
    pred_data_high_freq_component: Optional[Tensor] = None, pred_flow_high_freq_component: Optional[Tensor] = None,
    pred_data_wavelet_component: Optional[Tensor] = None, pred_flow_wavelet_component: Optional[Tensor] = None,
    # pred_poly: Optional[Tensor] = None, pred_poly_flow: Optional[Tensor] = None,
    info_dict: Dict = {}, save_path: Optional[str] = None, 
    figsize: tuple = (10, 8), title: Optional[str] = "Points PredictedSDF vs Ground Truth SDF"
):
    """
    绘制预测和真实SDF曲线，并显示info_dict中的多项式和傅里叶系数信息。
    
    Args:
        times: 时间序列，tensor
        gt_data: 真实SDF，tensor
        gt_flow: 真实SDF流场，tensor
        pred_data: 预测SDF，tensor
        pred_flow: 预测SDF流场，tensor
        pred_data_static_component: 预测SDF静态分量，tensor
        pred_data_poly_component: 预测SDF多项式分量，tensor
        pred_flow_poly_component: 预测SDF多项式流场，tensor
        pred_data_low_freq_component: 预测SDF低频分量，tensor
        pred_flow_low_freq_component: 预测SDF低频流场，tensor
        pred_data_mid_freq_component: 预测SDF中频分量，tensor
        pred_flow_mid_freq_component: 预测SDF中频流场，tensor
        pred_data_high_freq_component: 预测SDF高频分量，tensor
        pred_flow_high_freq_component: 预测SDF高频流场，tensor
        # pred_poly: 预测SDF多项式，tensor
        # pred_poly_flow: 预测SDF多项式流场，tensor
        info_dict: 包含点信息、网格索引和系数的字典
        save_path: 保存图表的路径（可选）
        figsize: 图表尺寸，默认为(10, 8)
    """
    # 创建带两行的子图布局：上部画曲线，下部放文本
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.35, 0.8, 0.6])  # 上部占60%高度
    text_ax = fig.add_axes([0.1, 0.05, 0.8, 0.25])  # 下部占25%高度

    # 绘制曲线
    assert pred_data is not None or gt_data is not None or gt_flow is not None or pred_flow is not None, "No data to plot"
    if gt_data is not None:
        ax.plot(times, gt_data, label='GT SDF', linestyle='-', color='red')
    if gt_flow is not None:
        ax.plot(times, gt_flow, label='GT SDF Flow', linestyle='--', color='red')
    if pred_data is not None:
        ax.plot(times, pred_data, label='Pred SDF', linestyle='-', color='blue')
    if pred_flow is not None:
        ax.plot(times, pred_flow, label='Pred SDF Flow', linestyle='--', color='blue')

    component_style = {
        "linestyle": ":",
        "linewidth": 1.5,
    }
    flow_component_style = {
        "linestyle": "-",
        "linewidth": 1.5,
        "alpha": 0.6,
    }
    if pred_data_static_component is not None:
        ax.plot(times, pred_data_static_component, label='Static Component', color='darkcyan', **component_style)

    if pred_data_poly_component is not None:
        ax.plot(times, pred_data_poly_component, label='Poly Component', color='orange', **component_style)

    if pred_flow_poly_component is not None:
        ax.plot(times, pred_flow_poly_component, label='Poly Flow', color='orange', **flow_component_style)

    if pred_data_low_freq_component is not None:
        ax.plot(times, pred_data_low_freq_component, label='Low-Freq Component', color='olive', **component_style)

    if pred_flow_low_freq_component is not None:
        ax.plot(times, pred_flow_low_freq_component, label='Low-Freq Flow', color='olive', **flow_component_style)

    if pred_data_mid_freq_component is not None:
        ax.plot(times, pred_data_mid_freq_component, label='Mid-Freq Component', color='brown', **component_style)

    if pred_flow_mid_freq_component is not None:
        ax.plot(times, pred_flow_mid_freq_component, label='Mid-Freq Flow', color='brown', **flow_component_style)

    if pred_data_high_freq_component is not None:
        ax.plot(times, pred_data_high_freq_component, label='High-Freq Component', color='teal', **component_style)

    if pred_flow_high_freq_component is not None:
        ax.plot(times, pred_flow_high_freq_component, label='High-Freq Flow', color='teal', **flow_component_style)

    if pred_data_wavelet_component is not None:
        ax.plot(times, pred_data_wavelet_component, label='Wavelet Component', color='purple', **component_style)

    if pred_flow_wavelet_component is not None:
        ax.plot(times, pred_flow_wavelet_component, label='Wavelet Flow', color='purple', **flow_component_style)

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('SDF', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=8)

    # 构建系数信息字符串
    def build_coefficient_string(info_dict):
        parts = []
        # 点信息
        vertices = info_dict.get('vertices')
        grid_indices = info_dict.get('grid_indices')
        static_sdf = info_dict.get('static_sdf_values')
        if all(x is not None for x in [vertices, grid_indices, static_sdf]):
            point_info = (
                f"Vertex: ({vertices[0]:.2f}, {vertices[1]:.2f}, {vertices[2]:.2f}) | "
                f"Grid Index: {grid_indices} | Static SDF: {static_sdf.item():.4f}"
            )
            parts.append(point_info)

        # 多项式系数
        poly_coeff = info_dict.get('sdf_curve_poly_coefficient')
        if poly_coeff is not None and len(poly_coeff) > 0:
            poly_terms = [
                f"{coef:.4f}*x^{i+1}" for i, coef in enumerate(poly_coeff)
            ]
            parts.append("Poly: " + (" + ".join(poly_terms) if poly_terms else "0"))

        # 傅里叶系数
        freq_keys = [
            ('Low', 'sdf_curve_low_freq_fourier_coefficient'),
            ('Mid', 'sdf_curve_mid_freq_fourier_coefficient'),
            ('High', 'sdf_curve_high_freq_fourier_coefficient')
        ]
        for freq, key in freq_keys:
            coeff = info_dict.get(key)
            if coeff is not None and len(coeff) > 0:
                terms = []
                for i in range(0, len(coeff), 2):
                    a, b = coeff[i], coeff[i + 1]
                    terms.append(f"{a:.4f}*Cos()")
                    terms.append(f"{b:.4f}*sin()")
                parts.append(f"{freq} Freq: " + (" + ".join(terms) if terms else "0"))

        return "\n".join(parts)

    # 在下部区域添加系数和点信息
    coefficient_text = build_coefficient_string(info_dict)
    y_pos = 0.9
    for line in coefficient_text.split("\n"):
        color = "black"
        if "Low Freq" in line:
            color = "blue"
        elif "Mid Freq" in line:
            color = "green"
        elif "High Freq" in line:
            color = "red"
        text_ax.text(0.5, y_pos, line, fontsize=10, ha='center', va='top', color=color, wrap=True)
        y_pos -= 0.3
    text_ax.axis('off')

    # 保存或显示
    try:
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close(fig)
    
def plot_cube_curve(
    times, 
    gt_data: Optional[Tensor] = None, gt_flow: Optional[Tensor] = None, 
    pred_data: Optional[Tensor] = None, pred_flow: Optional[Tensor] = None, 
    info_dict: Dict = {}, save_path: Optional[str] = None, 
    figsize: tuple = (10, 8), title: Optional[str] = "Points PredictedSDF vs Ground Truth SDF",
    highlight_indices: Optional[List[int]] = None
):
    """
    绘制预测和真实SDF曲线，并显示info_dict中的多项式和傅里叶系数信息。不一样的是，每个cube中有8个点，所以绘制8条曲线。
    在指定时间点（通过索引）使用五角星标记值。

    Args:
        times: 时间序列，tensor：(N)
        gt_data: 真实SDF，tensor：(N, 8)
        gt_flow: 真实SDF流场，tensor
        pred_data: 预测SDF，tensor
        pred_flow: 预测SDF流场，tensor
        info_dict: 包含点信息、网格索引和系数的字典
        save_path: 保存图表的路径（可选）
        figsize: 图表尺寸，默认为(10, 8)
        title: 图表标题
        highlight_indices: 需要高亮显示的时间点索引列表（可选）
    """
    # 创建带两行的子图布局：上部画曲线，下部放文本
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0.1, 0.35, 0.8, 0.6])  # 上部占60%高度
    text_ax = fig.add_axes([0.1, 0.05, 0.8, 0.25])  # 下部占25%高度

    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    assert gt_data is not None or pred_data is not None, "No data to plot"

    # 绘制每个点（8个）在时间序列上的曲线并标记指定点
    if gt_data is not None:
        for i in range(gt_data.shape[1]):
            ax.plot(times, gt_data[:, i], label=f'GT SDF P{i}', linestyle='-', color=colors[i % len(colors)], alpha=0.5)
            if highlight_indices is not None:
                for idx in highlight_indices:
                    if idx < len(times):
                        ax.scatter(times[idx], gt_data[idx, i], marker='*', s=100, color=colors[i % len(colors)], edgecolors='black')
    
    if gt_flow is not None:
        for i in range(gt_flow.shape[1]):
            ax.plot(times, gt_flow[:, i], label=f'GT Flow P{i}', linestyle='--', color=colors[i % len(colors)], alpha=0.5)
            if highlight_indices is not None:
                for idx in highlight_indices:
                    if idx < len(times):
                        ax.scatter(times[idx], gt_flow[idx, i], marker='*', s=100, color=colors[i % len(colors)], edgecolors='black')
    
    if pred_data is not None:
        for i in range(pred_data.shape[1]):
            ax.plot(times, pred_data[:, i], label=f'Pred SDF P{i}', linestyle='-', color=colors[i % len(colors)])
            if highlight_indices is not None:
                for idx in highlight_indices:
                    if idx < len(times):
                        ax.scatter(times[idx], pred_data[idx, i], marker='*', s=100, color=colors[i % len(colors)], edgecolors='black')
    
    if pred_flow is not None:
        for i in range(pred_flow.shape[1]):
            ax.plot(times, pred_flow[:, i], label=f'Pred Flow P{i}', linestyle='--', color=colors[i % len(colors)])
            if highlight_indices is not None:
                for idx in highlight_indices:
                    if idx < len(times):
                        ax.scatter(times[idx], pred_flow[idx, i], marker='*', s=100, color=colors[i % len(colors)], edgecolors='black')

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('SDF', fontsize=10)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=8)

    # 构建系数信息字符串
    def build_coefficient_string(info_dict):
        parts = []
        # 点信息
        vertices = info_dict.get('mean_cube_positions')
        grid_indices = info_dict.get('cube_indices')
        if all(x is not None for x in [vertices, grid_indices]):
            point_info = (
                f"mean_cube_positions: ({vertices[0]:.2f}, {vertices[1]:.2f}, {vertices[2]:.2f}) | "
                f"cube_indices: {grid_indices.tolist()} "
            )
            parts.append(point_info)

        return "\n".join(parts)

    # 在下部区域添加系数和点信息
    coefficient_text = build_coefficient_string(info_dict)
    y_pos = 0.9
    for line in coefficient_text.split("\n"):
        color = "black"
        text_ax.text(0.5, y_pos, line, fontsize=10, ha='center', va='top', color=color, wrap=True)
        y_pos -= 0.3
    text_ax.axis('off')

    # 保存或显示
    try:
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
    except Exception as e:
        print(f"Error saving plot: {e}")
    finally:
        plt.close(fig)
