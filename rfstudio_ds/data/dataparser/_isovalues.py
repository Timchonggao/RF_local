'''
定义一个IsoValue类,专注于分析和可视化这些体视值数据
'''

from __future__ import annotations
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import TypeAlias, Literal, Dict, List, Tuple, Union, Optional
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torchvision.utils import make_grid

import os

from concurrent.futures import ThreadPoolExecutor

class DataProcessor:
    """负责数据预处理和时间序列提取"""
    def __init__(self, data: np.ndarray, times: np.ndarray, pred_data: Optional[np.ndarray] = None):
        # TODO: 支持GPU版本
        self.data = self._convert_to_numpy(data)
        self.times = self._convert_to_numpy(times)
        self.pred_data = self._convert_to_numpy(pred_data) if pred_data is not None else None
        self.is_3d = len(data.shape) == 4
    
    @property
    def data_shape(self) -> tuple:
        return self.data.shape
    
    def _convert_to_numpy(self, data) -> np.ndarray:
        if isinstance(data, torch.Tensor):
            return data.detach().numpy()
        return data

    def extract_series(self, index: Optional[list[int]] = None, is_pred: bool = False, 
                      downsample_type: str = 'none', ratio: int = 1, stride: int = 1) -> np.ndarray:
        """
        提取时间序列数据
            :param index: 需要提取时间序列的坐标索引，格式为[y, x, z] 或者 [y, x];如果为None，则提取所有数据;
            :param is_pred: 是否提取预测数据
            :param downsample_type: 降采样类型，'none'表示不降采样，'uniform'表示均匀降采样
            :param ratio: 降采样空间步长
            :param stride: 降采样时间步长
            :return: 时间序列数据
        """
        data = self.pred_data if is_pred else self.data
        if downsample_type == 'uniform':
            data = (data[::stride, ::ratio, ::ratio, ::ratio] if self.is_3d 
                    else data[::stride, ::ratio, ::ratio])
            if index:
                index = [i // ratio for i in index]
        # TODO: 实现 'lerp' 降采样
        return (data[:, index[0], index[1], index[2]] if self.is_3d else data[:, index[0], index[1]]) if index else data

    def extract_spatial_features(
        self,
        is_pred: bool = False,
        downsample_type: str = 'uniform',
        spatial_downsample_ratio: int = 2,
        time_stride: int = 1
    ) -> Tuple[np.ndarray, List[str]]:
        """
        执行降采样并提取所有点的特征
            :param is_pred: 是否提取预测数据的特征
            :param downsample_type: 降采样类型 ('none', 'uniform')
            :param spatial_downsample_ratio: 空间降采样比率（默认 8，例如 128→16）
            :param time_stride: 时间维度步长（默认 1，不降采样）
            :return: (特征矩阵 [N, F], 特征名称列表)
        """
        # 提取降采样后的数据
        sampled_data = self.extract_series(
            index=None,  # 提取所有点
            is_pred=is_pred,
            downsample_type=downsample_type,
            ratio=spatial_downsample_ratio,
            stride=time_stride
        )

        # 重塑为 [N, T] 矩阵
        T = sampled_data.shape[0]  # 时间步长
        N = np.prod(sampled_data.shape[1:])  # 空间点数
        data_matrix = sampled_data.reshape(T, N).T  # [N, T]

        # 计算时间步长
        dt = (self.times[1] - self.times[0]) * time_stride

        # 调用 StatisticalAnalyzer 的批量特征提取方法
        return StatisticalAnalyzer.batch_extract_features(data_matrix, dt)


class StatisticalAnalyzer:
    @staticmethod
    def basic_stats(series: np.ndarray) -> dict:
        """
        计算时间序列的基本统计量
            :param series: 输入时间序列
            :return: 包含基本统计量的字典
        """
        return {
            'mean': np.mean(series),
            'std': np.std(series),
            'max': np.max(series),
            'min': np.min(series),
            'median': np.median(series),
            'skewness': skew(series),
            'kurtosis': kurtosis(series)
        }

    @staticmethod
    def derivative_analysis(series: np.ndarray, dt: float) -> dict:
        """
        计算时间序列的一阶导数及其统计特性
            :param series: 输入时间序列
            :param dt: 时间间隔（用于数值微分）
            :return: 包含导数及其统计量的字典
        """
        # 计算一阶导数
        derivative = np.diff(series) / dt
        
        # 计算统计特性
        stats = {
            'derivative': derivative,  # 导数序列
            'deriv_abs_mean': np.mean(np.abs(derivative)),  # 绝对值均值
            'deriv_abs_std': np.std(np.abs(derivative)),  # 绝对值标准差
            'deriv_abs_max': np.max(np.abs(derivative)),  # 绝对值最大值
            'deriv_abs_min': np.min(np.abs(derivative)),  # 绝对值最小值
            'zero_crossings': int(np.sum(np.diff(np.sign(derivative)) != 0))  # 过零点次数
        }
        return stats
    
    @staticmethod
    def fft_analysis(series: np.ndarray, dt: float, n_samples: int) -> dict:
        """
        对时间序列进行傅里叶变换分析
            :param series: 输入时间序列，形状为 [T,]
            :param dt: 时间间隔（用于计算频率）
            :param n_samples: 采样点数（时间序列长度）
            :return: 包含频谱特性的字典
        """
        # 执行 FFT
        fft_result = fft(series)
        frequencies = fftfreq(n_samples, dt)
        amplitudes = np.abs(fft_result) / n_samples  # 归一化幅度

        # 只保留正频率部分
        positive_mask = frequencies >= 0
        frequencies = frequencies[positive_mask]
        amplitudes = amplitudes[positive_mask]

        # 找到主频率（不排除零频率）
        dominant_idx = np.argmax(amplitudes)
        dominant_freq = float(frequencies[dominant_idx])
        dominant_amplitude = float(amplitudes[dominant_idx])

        return {
            'dominant_frequency': dominant_freq,
            'dominant_amplitude': dominant_amplitude,
            'frequencies': frequencies,
            'amplitudes': amplitudes
        }
    
    @staticmethod
    def batch_extract_features(data: np.ndarray, dt: float = 1.0) -> Tuple[np.ndarray, List[str]]:
        """
        批量提取时间序列特征
            :param data: 形状为 [N, T] 的时间序列矩阵，N 为点数，T 为时间步长
            :param dt: 时间步长（用于计算导数和频率）
            :return: (特征矩阵 [N, F], 特征名称列表)
        """
        N, T = data.shape
        features = []
        feature_names = []

        # 预计算导数
        derivatives = np.diff(data, axis=1) / dt  # [N, T-1]

        # 统计特征
        stats_features = np.stack([
            np.mean(data, axis=1),          # 均值
            np.std(data, axis=1),           # 标准差
            np.max(data, axis=1),           # 最大值
            np.min(data, axis=1),           # 最小值
            skew(data, axis=1),             # 偏度
            kurtosis(data, axis=1),         # 峰度
        ], axis=1)
        features.append(stats_features)
        feature_names += ['mean', 'std', 'max', 'min', 'skewness', 'kurtosis']

        # 导数特征
        deriv_stats = np.stack([
            np.mean(np.abs(derivatives), axis=1),    # 导数绝对值均值
            np.std(np.abs(derivatives), axis=1),     # 导数绝对值标准差
            np.max(np.abs(derivatives), axis=1),     # 导数绝对值最大值
            np.min(np.abs(derivatives), axis=1),     # 导数绝对值最小值
            np.count_nonzero(np.diff(np.sign(derivatives), axis=1) != 0, axis=1)  # 过零点次数
        ], axis=1)
        features.append(deriv_stats)
        feature_names += ['deriv_mean', 'deriv_std', 'zero_crossings']
        
        # 频域特征
        fft_result = fft(data, axis=1)[:, :T//2]  # 取正频率部分
        freqs = fftfreq(T, dt)[:T//2]
        amplitudes = np.abs(fft_result) / T       # 归一化幅度

        # 主频率和对应幅度
        dominant_freq_idx = np.argmax(amplitudes, axis=1)
        dominant_freqs = freqs[dominant_freq_idx]
        dominant_amps = amplitudes[np.arange(N), dominant_freq_idx]
        
        freq_features = np.stack([
            dominant_freqs,
            dominant_amps,
        ], axis=1)
        features.append(freq_features)
        feature_names += ['dominant_freq', 'dominant_amp']
        
        # 合并特征
        return np.concatenate(features, axis=1), feature_names


class Visualizer:
    """负责可视化时间序列及其分析结果"""
    def __init__(self, figsize: Tuple[float, float] = (12, 6)):
        self.figsize = figsize

    def _render_to_array(self, fig: plt.Figure) -> np.ndarray:
        """将 matplotlib 图表渲染为 numpy 数组"""
        canvas = FigureCanvas(fig)
        canvas.draw()
        image_argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        image_argb = image_argb.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        image_rgb = image_argb[..., 1:]  # 去掉 alpha 通道
        return image_rgb.copy()
    
    def _beautify_plot(self, ax: plt.Axes, title: str, xlabel: str, ylabel: str, 
                       legend_loc: str = 'best') -> None:
        """美化图表"""
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if ax.get_legend_handles_labels()[0]:  # 仅在有图例时添加
            ax.legend(loc=legend_loc)
        ax.grid(True, linestyle='--', alpha=0.7)
        # plt.tight_layout()

    def _add_annotations(self, ax: plt.Axes, texts: List[str]) -> None:
        """在图表内部自动添加标注"""
        for i, text in enumerate(texts):
            # 使用数据坐标系，定位到右上角附近，稍微偏移避免重叠
            x_max, y_max = ax.get_xlim()[1], ax.get_ylim()[1]
            x_pos = x_max * 0.95  # 靠近右侧 95% 位置
            y_pos = y_max * (0.95 - i * 0.15)  # 从顶部 95% 开始向下偏移
            ax.annotate(text, xy=(x_pos, y_pos), xycoords='data',
                        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8),
                        ha='right', va='top', fontsize=9)


    def _build_function_string(self, pred_coefficient):
        """
        根据系数构建函数字符串，分行显示多项式和傅里叶项，并用不同颜色区分频率。
        
        Args:
            pred_coefficient: 包含多项式和傅里叶系数的字典
        
        Returns:
            字符串，表示完整的函数 F(x)，包含换行符
        """
        poly_coeffient = pred_coefficient['poly']  # (degree + 1,)
        fourier_low_freq_coeffient = pred_coefficient['fourier']['low']  # (2 * degree,)
        omega_low = pred_coefficient['fourier']['omega_low']  # (degree,)
        fourier_mid_freq_coeffient = pred_coefficient['fourier']['mid']  # (2 * degree,)
        omega_mid = pred_coefficient['fourier']['omega_mid']  # (degree,)
        fourier_high_freq_coeffient = pred_coefficient['fourier']['high']  # (2 * degree,)
        omega_high = pred_coefficient['fourier']['omega_high']  # (degree,)

        # 初始化函数字符串部分
        coefficient_text_parts = []

        # 1. 多项式部分
        poly_text = "Poly: "
        if poly_coeffient is not None and len(poly_coeffient) > 0:
            poly_terms = []
            for i, coef in enumerate(poly_coeffient):
                if abs(coef) < 1e-6:  # 忽略接近0的系数
                    continue
                if i == 0:
                    term = f"{coef:.4f}"
                elif i == 1:
                    term = f"{coef:.4f}*x"
                else:
                    term = f"{coef:.4f}*x^{i}"
                poly_terms.append(term)
            if poly_terms:
                poly_text += " + ".join(poly_terms)
            else:
                poly_text += "0"
        coefficient_text_parts.append(poly_text)

        # 2. 傅里叶部分（分频率存储，便于后续颜色区分）
        fourier_low_text = "Low Freq: "
        fourier_mid_text = "Mid Freq: "
        fourier_high_text = "High Freq: "
        low_terms, mid_terms, high_terms = [], [], []

        # 低阶傅里叶
        if fourier_low_freq_coeffient is not None and len(fourier_low_freq_coeffient) > 0:
            for i in range(0, len(fourier_low_freq_coeffient), 2):
                a = fourier_low_freq_coeffient[i]  # cos项系数
                b = fourier_low_freq_coeffient[i + 1]  # sin项系数
                omega = omega_low[i // 2]  # 对应的频率
                if abs(a) > 1e-6:
                    low_terms.append(f"{a:.4f}*C({omega:.2f}*x)")
                if abs(b) > 1e-6:
                    low_terms.append(f"{b:.4f}*S({omega:.2f}*x)")
            if low_terms:
                fourier_low_text += " + ".join(low_terms)
            else:
                fourier_low_text += "0"
            coefficient_text_parts.append(fourier_low_text)

        # 中阶傅里叶
        if fourier_mid_freq_coeffient is not None and len(fourier_mid_freq_coeffient) > 0:
            for i in range(0, len(fourier_mid_freq_coeffient), 2):
                a = fourier_mid_freq_coeffient[i]
                b = fourier_mid_freq_coeffient[i + 1]
                omega = omega_mid[i // 2]
                if abs(a) > 1e-6:
                    mid_terms.append(f"{a:.4f}*C({omega:.2f}*x)")
                if abs(b) > 1e-6:
                    mid_terms.append(f"{b:.4f}*S({omega:.2f}*x)")
            if mid_terms:
                fourier_mid_text += " + ".join(mid_terms)
            else:
                fourier_mid_text += "0"
            coefficient_text_parts.append(fourier_mid_text)

        # 高阶傅里叶
        if fourier_high_freq_coeffient is not None and len(fourier_high_freq_coeffient) > 0:
            for i in range(0, len(fourier_high_freq_coeffient), 2):
                a = fourier_high_freq_coeffient[i]
                b = fourier_high_freq_coeffient[i + 1]
                omega = omega_high[i // 2]
                if abs(a) > 1e-6:
                    high_terms.append(f"{a:.4f}*C({omega:.2f}*x)")
                if abs(b) > 1e-6:
                    high_terms.append(f"{b:.4f}*S({omega:.2f}*x)")
            if high_terms:
                fourier_high_text += " + ".join(high_terms)
            else:
                fourier_high_text += "0"
            coefficient_text_parts.append(fourier_high_text)

        # 用换行符拼接所有部分
        return "\n".join(coefficient_text_parts)

    def render_curve(self, times: np.ndarray, series_dict: Dict[str, np.ndarray], 
                    title: str, stats_dict: Optional[Dict[str, Dict[str, float]]] = None,
                    return_type: Literal['numpy', 'tensor'] = 'numpy',
                    pred_coefficient: Optional[Dict] = None) -> np.ndarray:
        # 创建带两行的子图布局：上部画曲线，下部放文本
        fig = plt.figure(figsize=(self.figsize[0], self.figsize[1] + 2))  # 增加高度
        # ax = fig.add_axes([0.1, 0.25, 0.8, 0.7])  # 上部占70%高度
        # text_ax = fig.add_axes([0.1, 0.05, 0.8, 0.15])  # 下部占15%高度，用于文本
        ax = fig.add_axes([0.1, 0.35, 0.8, 0.6])  # 上部占60%高度，从0.35到0.95
        text_ax = fig.add_axes([0.1, 0.05, 0.8, 0.25])  # 下部占25%高度，从0.05到0.30
        
        # 绘制曲线
        for label, series in series_dict.items():
            linestyle = '--' if 'Pred' in label else '-'
            ax.plot(times, series, label=label, color='blue' if 'GT' in label else 'red', linestyle=linestyle)

        if stats_dict:
            texts = [f"{label}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, Max={stats['max']:.2f}, Min={stats['min']:.2f}, Skew={stats['skewness']:.2f}"
                    for label, stats in stats_dict.items()]
            self._add_annotations(ax, texts)

        # 美化主图表
        self._beautify_plot(ax, title, "Time", "Value")

        # 在下部区域添加 pred_coefficient 并设置颜色
        if pred_coefficient:
            coefficient_text = self._build_function_string(pred_coefficient)
            lines = coefficient_text.split("\n")
            y_pos = 0.9  # 从顶部开始
            for line in lines:
                if "Poly:" in line:
                    color = "black"  # 多项式用黑色
                elif "Low Freq:" in line:
                    color = "blue"  # 低频用蓝色
                elif "Mid Freq:" in line:
                    color = "green"  # 中频用绿色
                elif "High Freq:" in line:
                    color = "red"  # 高频用红色
                else:
                    color = "black"
                text_ax.text(0.5, y_pos, line, fontsize=12, ha='center', va='top', color=color, wrap=True)
                y_pos -= 0.3  # 每行下降0.3个单位
            text_ax.axis('off')  # 隐藏下部轴

        image = self._render_to_array(fig)
        plt.close(fig)
        return image if return_type == 'numpy' else torch.from_numpy(image)

    def render_derivative(self, times: np.ndarray, deriv_dict: Dict[str, Dict[str, Union[np.ndarray, float]]], 
                          title: str, return_type: Literal['numpy', 'tensor'] = 'numpy') -> np.ndarray:
        fig, ax = plt.subplots(figsize=self.figsize)
        deriv_times = times[:-1]
        colors = {'GT': 'blue', 'Pred': 'red'}

        for label, stats in deriv_dict.items():
            derivative = stats['derivative']
            linestyle = '--' if 'Pred' in label else '-'
            ax.plot(deriv_times, derivative, label=f"{label} Derivative", 
                    color=colors.get(label, 'blue'), linestyle=linestyle)

        # 添加统计信息到图表内部
        texts = []
        for label, stats in deriv_dict.items():
            stats_text = (
                f"{label}: Mean Abs={stats['deriv_abs_mean']:.2f}, "
                f"Std Abs={stats['deriv_abs_std']:.2f}, Zero Crossings={stats['zero_crossings']}"
            )
            texts.append(stats_text)
        self._add_annotations(ax, texts)

        self._beautify_plot(ax, title, "Time", "Derivative Value")
        image = self._render_to_array(fig)
        plt.close(fig)
        return image if return_type == 'numpy' else torch.from_numpy(image)

    def render_fft_spectrum(self, fft_dict: Dict[str, Dict[str, Union[np.ndarray, float]]], 
                           title: str, return_type: Literal['numpy', 'tensor'] = 'numpy') -> np.ndarray:
        fig, ax = plt.subplots(figsize=self.figsize)
        colors = {'GT': 'blue', 'Pred': 'red'}
        linestyles = {'GT': '-', 'Pred': '--'}

        for label, stats in fft_dict.items():
            frequencies = stats['frequencies']
            amplitudes = stats['amplitudes']
            dominant_freq = stats['dominant_frequency']
            ax.plot(frequencies, amplitudes, label=f"{label} Spectrum", 
                    color=colors.get(label, 'blue'), linestyle=linestyles.get(label, '-'))
            ax.axvline(x=dominant_freq, color=colors.get(label, 'blue'), linestyle=':', 
                       label=f"{label} Dominant Freq: {dominant_freq:.2f} Hz")

        # 添加统计信息到图表内部
        texts = []
        for label, stats in fft_dict.items():
            stats_text = (
                f"{label}: Dominant Freq={stats['dominant_frequency']:.2f} Hz, "
                f"Dominant Amp={stats['dominant_amplitude']:.2f}"
            )
            texts.append(stats_text)
        self._add_annotations(ax, texts)

        self._beautify_plot(ax, title, "Frequency (Hz)", "Amplitude")
        image = self._render_to_array(fig)
        plt.close(fig)
        return image if return_type == 'numpy' else torch.from_numpy(image)

    def render_3d_volume(self, data: np.ndarray, title: str, 
                         return_type: Literal['numpy', 'tensor'] = 'numpy') -> np.ndarray:
        """
        绘制3D体视图（散点颜色与数值相关，无值限制）
            :param data: 体数据，格式为 (H, W, D)
            :param title: 图表标题
            :param return_type: 返回类型 ('numpy' 或 'tensor')
            :return: 渲染后的图像
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 生成所有点的坐标
        h, w, d = data.shape
        x, y, z = np.indices((h, w, d))
        x = x.flatten()
        y = y.flatten()
        z = z.flatten()
        values = data.flatten()

        # 绘制散点图，颜色与数值相关
        scatter = ax.scatter(x, y, z, c=values, cmap='viridis', s=30)
        
        # 调整颜色条位置（底部）并缩小
        cbar = fig.colorbar(scatter, ax=ax, label='Value', orientation='horizontal', 
                           fraction=0.03, pad=0.1)  # fraction 控制大小，pad 控制间距

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

        # 设置视角以优化显示
        # ax.view_init(elev=20, azim=45)  # 调整仰角和方位角，使体数据更立体

        # 优化布局
        plt.tight_layout()

        image = self._render_to_array(fig)
        plt.close(fig)
        return image if return_type == 'numpy' else torch.from_numpy(image)



class IsoValue:
    """整合类，协调各模块"""
    def __init__(self, data: np.ndarray, times: np.ndarray, pred_data: Optional[np.ndarray] = None):
        self.dataprocessor = DataProcessor(data, times, pred_data)
        self.visualizer = Visualizer()

    def points_statistics_analysis(self, points: Dict[Tuple[float], List[int]] = None) -> dict:
        """
        对指定点的统计分析，包括基本统计量、导数分析和FFT分析
        :param points: 点坐标到索引的映射，例如 {(x, y, z): [y_idx, x_idx, z_idx]}
        :return: 统计分析结果字典
        """
        if not points:
            return {}
        
        points_results = {}
        dt = self.dataprocessor.times[1] - self.dataprocessor.times[0]
        n_samples = len(self.dataprocessor.times)
        for coord, idx in points.items():
            points_results[coord] = {}
            # 提取数据序列并计算基本统计量
            series = self.dataprocessor.extract_series(index=idx)
            points_results[coord]['basic_stats'] = StatisticalAnalyzer.basic_stats(series)
            points_results[coord]['derivative_analysis'] = StatisticalAnalyzer.derivative_analysis(series, dt)
            points_results[coord]['fft_analysis'] = StatisticalAnalyzer.fft_analysis(series, dt, n_samples)

            # 如果存在预测数据，提取预测序列并更新统计结果
            if self.dataprocessor.pred_data is not None:
                pred_series = self.dataprocessor.extract_series(index=idx, is_pred=True)
                points_results[coord]['basic_stats_pred'] = StatisticalAnalyzer.basic_stats(pred_series)
                points_results[coord]['derivative_analysis_pred'] = StatisticalAnalyzer.derivative_analysis(pred_series, dt)
                points_results[coord]['fft_analysis_pred'] = StatisticalAnalyzer.fft_analysis(pred_series, dt, n_samples)
        return points_results
    
    def points_statistics_visualization(self, points: Dict[Tuple[float, ...], List[int]], 
                                        plot_types: List[Literal['curve', 'derivative', 'fft']] = ['curve'],
                                        return_type: Literal['numpy', 'tensor'] = 'numpy',
                                        pred_coefficients: Optional[Dict] = None,) -> Dict[Tuple[float, ...], Dict[str, Union[np.ndarray, torch.Tensor]]]:
        """
        可视化指定点的统计分析结果
        :param points: 点坐标到索引的映射
        :param plot_types: 要绘制的图表类型列表
        :param save_dir: 可选的保存目录
        :return: 包含图表类型的图像字典
        """
        if not points:
            return {}
        results = {}
        times = self.dataprocessor.times
        points_results = self.points_statistics_analysis(points)
        
        for coord, idx in points.items():
            results[coord] = {}
            series = self.dataprocessor.extract_series(index=idx)
            deriv_dict = {'GT': points_results[coord]['derivative_analysis']}
            fft_dict = {'GT': points_results[coord]['fft_analysis']}
            stats_dict = {'GT': points_results[coord]['basic_stats']}
            if pred_coefficients is not None:
                pred_coefficient = pred_coefficients[coord]

            if self.dataprocessor.pred_data is not None:
                pred_series = self.dataprocessor.extract_series(index=idx, is_pred=True)
                deriv_dict['Pred'] = points_results[coord]['derivative_analysis_pred']
                fft_dict['Pred'] = points_results[coord]['fft_analysis_pred']
                stats_dict['Pred'] = points_results[coord]['basic_stats_pred']

            for plot_type in plot_types:
                if plot_type == 'curve':
                    title = f"Time Series at Point {coord}"
                    series_dict = {'GT': series}
                    if self.dataprocessor.pred_data is not None:
                        series_dict['Pred'] = pred_series
                    results[coord][f"{coord}_time_series_curve"] = self.visualizer.render_curve(
                        times, series_dict, title, stats_dict=stats_dict, return_type=return_type,pred_coefficient=pred_coefficient)
                elif plot_type == 'derivative':
                    title = f"Derivative Analysis at Point {coord}"
                    results[coord][f"{coord}_derivative"] = self.visualizer.render_derivative(
                        times, deriv_dict, title, return_type=return_type)
                elif plot_type == 'fft':
                    title = f"FFT Spectrum at Point {coord}"
                    results[coord][f"{coord}_fft_spectrum"] = self.visualizer.render_fft_spectrum(
                        fft_dict, title, return_type=return_type)
        return results

    def visualize_3d_volume(
        self,
        time_steps: Union[int, List[int], None] = 0,
        is_pred: bool = False,
        downsample_ratio: int = 2,
        return_type: Literal['numpy', 'tensor'] = 'numpy'
    ) -> Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]]:
        """
        绘制 3D 体视图，支持单一或多个时间步。
            :param time_steps: 要可视化的时间步（单一 int、多个 List[int] 或 None 表示所有）
            :param is_pred: 是否使用预测数据
            :param downsample_ratio: 空间降采样比率
            :param return_type: 返回类型 ('numpy' 或 'tensor')
            :return: 渲染后的图像或图像列表
        """
        if not self.dataprocessor.is_3d:
            raise ValueError("3D volume visualization is only supported for 3D data")

        # 提取降采样数据
        data = self.dataprocessor.extract_series(
            is_pred=is_pred,
            downsample_type='uniform',
            ratio=downsample_ratio
        )
        if data is None:
            raise ValueError("Data not available (prediction data may be missing)")

        max_time = data.shape[0]
        # 处理 time_steps 参数
        if time_steps is None:
            time_steps = list(range(max_time))
        elif isinstance(time_steps, int):
            time_steps = [time_steps]
        elif not isinstance(time_steps, list):
            raise ValueError("time_steps must be int, List[int], or None")

        # 验证时间步范围
        for t in time_steps:
            if t < 0 or t >= max_time:
                raise ValueError(f"Time step {t} is out of range [0, {max_time-1}]")

        # 生成图像
        images = []
        for time_step in time_steps:
            data_3d = data[time_step, :, :, :]
            title = f"3D Volume at Time Step {time_step} ({'Pred' if is_pred else 'GT'})"
            image = self.visualizer.render_3d_volume(data_3d, title, return_type=return_type)
            images.append(image)

        # 根据 time_steps 类型返回结果
        return images[0] if len(images) == 1 else images


    def spatial_statistics_analysis_visualization(
        self,
        is_pred: bool = False,
        downsample_type: str = 'uniform',
        spatial_downsample_ratio: int = 2,
        time_stride: int = 1,
        use_pca: bool = False,
        n_components: int = 3,
        analysis_types: List[Literal['direct', 'clustering', 'anomaly']] = ['clustering'],
        n_clusters: int = 3,  # 聚类分析参数
        contamination: float = 0.1,  # 异常检测参数
        return_type: Literal['numpy', 'tensor'] = 'numpy'
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        对提取的空间特征进行统计分析和3D体视图可视化。
        支持直接对原始特征分析或先进行 PCA 降维后再分析。
            :param is_pred: 是否使用预测数据
            :param downsample_type: 降采样类型 ('none', 'uniform')
            :param spatial_downsample_ratio: 空间降采样比率（默认 2，例如 128→64）
            :param time_stride: 时间维度步长（默认 1，不降采样）
            :param use_pca: 是否对特征进行 PCA 降维
            :param n_components: PCA 降维后的维度数（仅当 use_pca=True 时有效）
            :param analysis_types: 分析类型列表，支持 'direct'（直接可视化）、'clustering'（聚类）、'anomaly'（异常检测）
            :param n_clusters: 聚类分析的簇数（仅对 clustering 有效）
            :param contamination: 异常检测的预期异常比例（仅对 anomaly 有效）
            :param return_type: 返回类型 ('numpy' 或 'tensor')
            :return: 字典，键为特征名称/主成分编号+分析类型，值为对应的3D体视图图像
        """
        if not self.dataprocessor.is_3d:
            raise ValueError("Spatial statistics visualization is only supported for 3D data")

        # 提取空间特征
        features, feature_names = self.dataprocessor.extract_spatial_features(
            is_pred=is_pred,
            downsample_type=downsample_type,
            spatial_downsample_ratio=spatial_downsample_ratio,
            time_stride=time_stride
        )

        # 获取降采样后的空间维度
        original_shape = self.dataprocessor.data_shape[1:]  # 原始形状 [H, W, D]
        downsampled_shape = tuple(s // spatial_downsample_ratio for s in original_shape)  # 降采样后的形状 [H', W', D']
        N_expected = np.prod(downsampled_shape)  # 预期点数

        if features.shape[0] != N_expected:
            raise ValueError(f"Feature matrix size {features.shape[0]} does not match expected number of points {N_expected}")

        # 初始化结果字典
        visualization_results = {}

        if use_pca:
            # PCA 降维路径
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)  # [N, F]
            pca = PCA(n_components=n_components)
            processed_features = pca.fit_transform(features_scaled)  # [N, n_components]
            feature_labels = [f"pca_component_{i+1}" for i in range(n_components)]
            # 打印解释方差比
            for i, ratio in enumerate(pca.explained_variance_ratio_):
                print(f"PCA Component {i+1} explained variance ratio: {ratio:.4f}")
        else:
            # 原始特征路径
            processed_features = features  # [N, F]
            feature_labels = feature_names

        # 对每个特征或主成分进行分析
        for i, label in enumerate(feature_labels):
            # 提取当前特征向量 [N,]
            feature_vector = processed_features[:, i]

            # 计算统计分析
            stats = StatisticalAnalyzer.basic_stats(feature_vector)
            stats_text = (
                f"Mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                f"Min={stats['min']:.2f}, Max={stats['max']:.2f}, "
            )
            print(f"Feature {label} statistics: {stats_text}")

            # 重塑为降采样后的3D形状 [H', W', D']
            feature_3d = feature_vector.reshape(downsampled_shape)

            # 标准化当前特征（仅针对当前特征）
            scaler = StandardScaler()
            feature_scaled = scaler.fit_transform(feature_vector.reshape(-1, 1))  # [N, 1]

            # 根据分析类型处理
            for analysis_type in analysis_types:
                if analysis_type == 'direct' or (not use_pca and analysis_type in ['clustering', 'anomaly']):
                    # 直接可视化（PCA 时始终支持，原始特征时仅作为默认）
                    title = f"3D {'PCA Component' if use_pca else 'Feature'} Visualization: {label} ({'Pred' if is_pred else 'GT'})"
                    image = self.visualizer.render_3d_volume(
                        data=feature_3d,
                        title=title,
                        return_type=return_type
                    )
                    visualization_results[label if analysis_type == 'direct' else f"{label}_{analysis_type}"] = image

                if analysis_type == 'clustering':
                    # 聚类分析
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(feature_scaled)  # [N,]
                    cluster_3d = cluster_labels.reshape(downsampled_shape)

                    cluster_title = f"3D Clustering on {label} (n_clusters={n_clusters}, {'Pred' if is_pred else 'GT'})"
                    cluster_image = self.visualizer.render_3d_volume(
                        data=cluster_3d,
                        title=cluster_title,
                        return_type=return_type
                    )
                    visualization_results[f"{label}_clustering"] = cluster_image

                elif analysis_type == 'anomaly':
                    # 异常检测
                    iso_forest = IsolationForest(contamination=contamination, random_state=42)
                    anomaly_scores = iso_forest.fit_predict(feature_scaled)  # [N,]，-1 表示异常，1 表示正常
                    anomaly_3d = anomaly_scores.reshape(downsampled_shape)

                    anomaly_title = f"3D Anomaly Detection on {label} (cont={contamination}, {'Pred' if is_pred else 'GT'})"
                    anomaly_image = self.visualizer.render_3d_volume(
                        data=anomaly_3d,
                        title=anomaly_title,
                        return_type=return_type
                    )
                    visualization_results[f"{label}_anomaly"] = anomaly_image

        return visualization_results


# Type aliases for better readability
Point: TypeAlias = Tuple[float, float, float]
Index: TypeAlias = List[int]
PointIndexMap: TypeAlias = Dict[Point, Index]

def index_to_coord(x_idx: int, y_idx: int, z_idx: int, x_min: float, y_min: float, z_min: float, cell_size: float) -> Tuple[float, float, float]:
    """Convert grid indices to physical coordinates."""
    x = float(x_min + x_idx * cell_size)
    y = float(y_min + y_idx * cell_size)
    z = float(z_min + z_idx * cell_size)
    return (x, y, z)

def coord_to_index(point: Tuple[float, float, float], x_min: float, y_min: float, z_min: float, cell_size: float) -> List[int]:
    """Convert physical coordinates to grid indices."""
    x_idx = int((point[0] - x_min) / cell_size)
    y_idx = int((point[1] - y_min) / cell_size)
    z_idx = int((point[2] - z_min) / cell_size)
    return [y_idx, x_idx, z_idx]  # Order matches IsoValue [y, x, z]

# Base class for shared visualization logic
class DynamicBase:
    """Base class for dynamic models providing common visualization methods."""
    
    def __init__(self, iso_values: np.ndarray, times: np.ndarray):
        self.iso_values = iso_values
        self.times = times
        self.iso_value_analyzer = IsoValue(data=iso_values, times=times)

    def points_statistics_visualization(
        self,
        point_coordinates: List[List[int]],
        times: Optional[np.ndarray] = None,
        iso_values: Optional[np.ndarray] = None,
        plot_types: List[Literal['curve', 'derivative', 'fft']] = ['curve'],
        return_type: Literal['numpy', 'tensor'] = 'numpy'
    ) -> Dict[Point, Dict[str, Union[np.ndarray, torch.Tensor]]]:
        """Render statistical plots for specified points."""
        if iso_values is not None and times is not None:
            analyzer = IsoValue(data=iso_values, times=times)
        else:
            analyzer = self.iso_value_analyzer
        
        point_index_map = self.get_points_coordinate_index(point_coordinates)
        return analyzer.points_statistics_visualization(
            points=point_index_map,
            plot_types=plot_types,
            return_type=return_type
        )

    def visualize_3d_volume(
        self,
        time_steps: Union[int, List[int], None] = 0,
        downsample_ratio: int = 2,
        return_type: Literal['numpy', 'tensor'] = 'numpy'
    ) -> Union[np.ndarray, torch.Tensor]:
        """Render 3D volume at a specific time step."""
        return self.iso_value_analyzer.visualize_3d_volume(
            time_steps=time_steps,
            downsample_ratio=downsample_ratio,
            return_type=return_type
        )

    def spatial_statistics_analysis_visualization(
        self,
        downsample_type: str = 'uniform',
        spatial_downsample_ratio: int = 2,
        time_stride: int = 1,
        use_pca: bool = False,
        n_components: int = 3,
        analysis_types: List[Literal['direct', 'clustering', 'anomaly']] = ['clustering'],
        n_clusters: int = 3,  # 聚类分析参数
        contamination: float = 0.1,  # 异常检测参数
        return_type: Literal['numpy', 'tensor'] = 'numpy'
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        对提取的空间特征进行统计分析和3D体视图可视化。
        支持直接对原始特征分析或先进行 PCA 降维后再分析。
            :param downsample_type: 降采样类型 ('none', 'uniform')
            :param spatial_downsample_ratio: 空间降采样比率（默认 2，例如 128→64）
            :param time_stride: 时间维度步长（默认 1，不降采样）
            :param use_pca: 是否对特征进行 PCA 降维
            :param n_components: PCA 降维后的维度数（仅当 use_pca=True 时有效）
            :param analysis_types: 分析类型列表，支持 'direct'（直接可视化）、'clustering'（聚类）、'anomaly'（异常检测）
            :param n_clusters: 聚类分析的簇数（仅对 clustering 有效）
            :param contamination: 异常检测的预期异常比例（仅对 anomaly 有效）
            :param return_type: 返回类型 ('numpy' 或 'tensor')
            :return: 字典，键为特征名称/主成分编号+分析类型，值为对应的3D体视图图像
        """
        return self.iso_value_analyzer.spatial_statistics_analysis_visualization(
            downsample_type=downsample_type,
            spatial_downsample_ratio=spatial_downsample_ratio,
            time_stride=time_stride,
            use_pca=use_pca,
            n_components=n_components,
            analysis_types=analysis_types,
            n_clusters=n_clusters,
            contamination=contamination,
            return_type=return_type
        )

    def get_points_coordinate_index(self, point_coordinates: List[List[int]]) -> PointIndexMap:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_points_coordinate_index")


class DynamicMesh(DynamicBase):
    """A dynamic mesh class for loading and analyzing time-varying SDF data."""
    #TODO parse mesh file and extract mesh sdf data

    def __init__(
            self, 
            sdf_file_dir: str,
            padding_enabled: bool = False, 
            padding_type: Literal["reflect", "replicate", "circular"] = "reflect", 
            padding_size: int = 20
    ):
        self.sdf_file_dir = sdf_file_dir
        self.padding_enabled = padding_enabled
        self.padding_type = padding_type
        self.padding_size = padding_size
        """Initialize mesh parameters and load data."""
        self.__load_and_sort_sdf_files()
        self.__initialize_mesh_boundaries()
        self.__generate_time_series()
        if self.padding_enabled:
            self.__apply_padding()
        super().__init__(self.iso_values, self.times)

    def __convert_mesh_sequence_to_sdf_files(self,):
        pass

    def __load_and_sort_sdf_files(self):
        """Load and sort SDF files from the specified directory."""
        from natsort import natsorted
        self.sdf_files = natsorted([f for f in os.listdir(self.sdf_file_dir) if f.endswith('.npy')])
        if not self.sdf_files:
            raise ValueError(f"No SDF files found in directory: {self.sdf_file_dir}")

    def __initialize_mesh_boundaries(self):
        """Determine grid boundaries based on the first SDF file."""
        temp_sdf = np.load(os.path.join(self.sdf_file_dir, self.sdf_files[0]))
        self.x_min, self.x_max = 0, temp_sdf.shape[1] - 1
        self.y_min, self.y_max = 0, temp_sdf.shape[0] - 1
        self.z_min, self.z_max = 0, temp_sdf.shape[2] - 1
        self.cell_size = 1
        del temp_sdf

    def __generate_time_series(self):
        """Generate time series and load SDF data in parallel."""
        self.times = np.linspace(0, 1, len(self.sdf_files))
        # self.times = np.arange(0, len(self.sdf_files)) # no time normalization
        self.iso_values = self.__get_iso_values().astype(np.float32)

    def __load_sdf_file(self, sdf_file: str) -> np.ndarray:
        """Load a single SDF file."""
        return np.load(os.path.join(self.sdf_file_dir, sdf_file))

    def __get_iso_values(self) -> np.ndarray:
        """Load SDF data for all time steps in parallel."""
        shape = (
            len(self.times),
            self.y_max - self.y_min + 1,
            self.x_max - self.x_min + 1,
            self.z_max - self.z_min + 1
        )
        iso_value_array = np.zeros(shape)
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.__load_sdf_file, self.sdf_files))
        for idx, data in enumerate(results):
            iso_value_array[idx, :, :, :] = data
        return iso_value_array

    def __apply_padding(self):
        """Apply padding based on the specified type."""
        padding_strategies = {
            "reflect": self.__reflect_padding_sdf,
            "replicate": self.__replicate_padding_sdf,
            "circular": self.__circular_padding_sdf
        }
        padding_method = padding_strategies.get(self.padding_type)
        if padding_method:
            padding_method()
        else:
            raise ValueError(f"Invalid padding type: {self.padding_type}")

    def __reflect_padding_sdf(self):
        """Apply reflection padding."""
        start_padding = self.iso_values[1:self.padding_size + 1, :, :, :][::-1, :, :, :]
        end_padding = self.iso_values[-self.padding_size - 1:-1, :, :, :][::-1, :, :, :]
        self.iso_values = np.concatenate((start_padding, self.iso_values, end_padding), axis=0)
        self.__update_time_series()

    def __replicate_padding_sdf(self):
        """Apply replication padding."""
        first_frame = self.iso_values[0:1, :, :, :]
        last_frame = self.iso_values[-1:, :, :, :]
        start_padding = np.repeat(first_frame, self.padding_size, axis=0)
        end_padding = np.repeat(last_frame, self.padding_size, axis=0)
        self.iso_values = np.concatenate((start_padding, self.iso_values, end_padding), axis=0)
        self.__update_time_series()

    def __circular_padding_sdf(self):
        """Apply circular padding."""
        start_padding = self.iso_values[-self.padding_size:, :, :, :]
        end_padding = self.iso_values[:self.padding_size, :, :, :]
        self.iso_values = np.concatenate((start_padding, self.iso_values, end_padding), axis=0)
        self.__update_time_series()

    def __update_time_series(self):
        """Update time series after padding."""
        self.times = np.linspace(0, 1, len(self.iso_values))

    def __validate_index(self, x_idx: int, y_idx: int, z_idx: int):
        """Validate grid indices."""
        assert 0 <= x_idx <= (self.x_max - self.x_min), f"X index {x_idx} out of range"
        assert 0 <= y_idx <= (self.y_max - self.y_min), f"Y index {y_idx} out of range"
        assert 0 <= z_idx <= (self.z_max - self.z_min), f"Z index {z_idx} out of range"

    def get_points_coordinate_index(self, point_coordinates: List[List[int]]) -> PointIndexMap:
        """Map physical coordinates to grid indices."""
        point_index_map: PointIndexMap = {}
        for point in point_coordinates:
            if len(point) != 3:
                raise ValueError(f"Each point must have 3 coordinates, got {point}")
            indices = coord_to_index(tuple(point), self.x_min, self.y_min, self.z_min, self.cell_size)
            self.__validate_index(indices[1], indices[0], indices[2])  # Adjust order for [y, x, z]
            point_index_map[tuple(point)] = indices
        return point_index_map
