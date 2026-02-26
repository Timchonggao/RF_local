from __future__ import annotations

# import modules
from dataclasses import dataclass

import torch

# # import rfstudio modules
from rfstudio.nn import Module


@dataclass
class DDDM(Module):
    """Discrete Dual Domain Model"""
    poly_degree: int = 3
    fourier_low_degree: int = 3   # 低阶傅里叶项数
    fourier_mid_degree: int = 6   # 中阶傅里叶项数
    fourier_high_degree: int = 9  # 高阶傅里叶项数
    num_points: int = 128**3

    def __setup__(self) -> None:
        self.fourier_degree = self.fourier_low_degree + self.fourier_mid_degree + self.fourier_high_degree

        # 多项式系数
        if self.poly_degree > 0:
            self.poly_coeffs = torch.nn.Parameter(torch.zeros(self.num_points, self.poly_degree + 1, device=self.device, requires_grad=True))
            self.poly_coeffs.data[:, 0] = torch.rand(self.num_points) * 0.5 - 0.05 # for isocubes sdf initialization
        else:
            self.poly_coeffs = torch.nn.Parameter(torch.empty(0))
    
        # 傅里叶系数
        if self.fourier_degree > 0:
            # 傅里叶系数分组
            if self.fourier_low_degree > 0:
                self.fourier_low_coeffs = torch.nn.Parameter(torch.zeros(self.num_points, self.fourier_low_degree * 2, device=self.device, requires_grad=True))
            else:
                self.fourier_low_coeffs = torch.nn.Parameter(torch.empty(0))
            if self.fourier_mid_degree > 0:
                self.fourier_mid_coeffs = torch.nn.Parameter(torch.zeros(self.num_points, self.fourier_mid_degree * 2, device=self.device, requires_grad=True))
            else:
                self.fourier_mid_coeffs = torch.nn.Parameter(torch.empty(0))
            if self.fourier_high_degree > 0:
                self.fourier_high_coeffs = torch.nn.Parameter(torch.zeros(self.num_points, self.fourier_high_degree * 2, device=self.device, requires_grad=True))
            else:
                self.fourier_high_coeffs = torch.nn.Parameter(torch.empty(0))

            self.omega_base = 2 * torch.pi * torch.arange(1, self.fourier_degree + 1, device=self.device)
            self.omega_low = self.omega_base[:self.fourier_low_degree]
            self.omega_mid = self.omega_base[self.fourier_low_degree:self.fourier_low_degree + self.fourier_mid_degree]
            self.omega_high = self.omega_base[self.fourier_low_degree + self.fourier_mid_degree:]
    
    def __call__(self, input_t: torch.Tensor, model_stage: int = -1) -> torch.Tensor:
        assert self.device == input_t.device, "Model and input should be on the same device"
        if self.omega_base.device!= self.device:
            self.omega_base = self.omega_base.to(self.device)
            self.omega_low = self.omega_low.to(self.device)
            self.omega_mid = self.omega_mid.to(self.device)
            self.omega_high = self.omega_high.to(self.device)

        batch_size = input_t.shape[0]
        input_t = input_t.unsqueeze(1).expand(-1, self.num_points) # expand to (batch_size, num_points)

        # 定义阶段对应的最大阶数
        max_stages = [0, 1, 2, 3]  # 0: poly, 1: low, 2: mid, 3: high
        curr_stage = min(model_stage, max_stages[-1]) if model_stage >= 0 else max_stages[-1]

        # 多项式部分
        y_poly = torch.zeros((batch_size, self.num_points), device=self.device)
        if self.poly_degree > 0 and curr_stage >= 0: # step > 0
            for i in range(self.poly_degree + 1):
                y_poly += self.poly_coeffs[:, i].unsqueeze(0).expand(batch_size, -1) * (input_t ** i)

        # 傅里叶部分
        y_fourier = torch.zeros((batch_size, self.num_points), device=self.device)
        # 低阶傅里叶
        if self.fourier_low_degree > 0 and curr_stage >= 1:
            a_b_low = self.fourier_low_coeffs.reshape(-1, self.fourier_low_degree, 2)
            omega_low = self.omega_low.unsqueeze(0).expand(batch_size, -1)
            for i in range(self.fourier_low_degree):
                a_i = a_b_low[:, i, 0]
                b_i = a_b_low[:, i, 1]
                y_fourier += a_i.unsqueeze(0) * torch.cos(omega_low[:, i].unsqueeze(1) * input_t) + \
                            b_i.unsqueeze(0) * torch.sin(omega_low[:, i].unsqueeze(1) * input_t)

        # 中阶傅里叶
        if self.fourier_mid_degree > 0 and curr_stage >= 2:
            a_b_mid = self.fourier_mid_coeffs.reshape(-1, self.fourier_mid_degree, 2)
            omega_mid = self.omega_mid.unsqueeze(0).expand(batch_size, -1)
            for i in range(self.fourier_mid_degree):
                a_i = a_b_mid[:, i, 0]
                b_i = a_b_mid[:, i, 1]
                y_fourier += a_i.unsqueeze(0) * torch.cos(omega_mid[:, i].unsqueeze(1) * input_t) + \
                            b_i.unsqueeze(0) * torch.sin(omega_mid[:, i].unsqueeze(1) * input_t)
        
        # 高阶傅里叶
        if self.fourier_high_degree > 0 and curr_stage >= 3:
            a_b_high = self.fourier_high_coeffs.reshape(-1, self.fourier_high_degree, 2)
            omega_high = self.omega_high.unsqueeze(0).expand(batch_size, -1)
            for i in range(self.fourier_high_degree):
                a_i = a_b_high[:, i, 0]
                b_i = a_b_high[:, i, 1]
                y_fourier += a_i.unsqueeze(0) * torch.cos(omega_high[:, i].unsqueeze(1) * input_t) + \
                            b_i.unsqueeze(0) * torch.sin(omega_high[:, i].unsqueeze(1) * input_t)
        
        return y_poly, y_fourier

    def _get_coefficients_for_points(
            self,
            points_index_map: dict,
        ):
        """
        提取指定点的模型系数（多项式和傅里叶）。
        
        Args:
            self: FourierPolyFitter模型
            points_index_map: 点的索引字典，格式为 {(x,y,z): [y_index, x_index, z_index], ...}
        
        Returns:
            字典，键为点坐标，值为包含多项式和傅里叶系数的字典
        """
        # 将三维索引转换为一维索引
        def to_1d_index(y_idx: int, x_idx: int, z_idx: int, res: int) -> int:
            return y_idx * res ** 2 + x_idx * res + z_idx

        coefficients = {}
        
        # 遍历所有点
        for coord, idx in points_index_map.items():
            coefficients[coord] = {}
            # idx = to_1d_index(i, j, k, self.resolution)
            y_idx, x_idx, z_idx = idx
            coff_idx = to_1d_index(y_idx, x_idx, z_idx, self.resolution)
            
            # 提取多项式系数
            poly_coeffs = self.poly_coeffs[coff_idx].detach().cpu().numpy() if self.poly_degree > 0 else None
            coefficients[coord]['poly'] = poly_coeffs

            # 提取傅里叶系数
            fourier_coeffs = {}
            if self.fourier_low_degree > 0:
                fourier_low_coeffs = self.fourier_low_coeffs[coff_idx].detach().cpu().numpy()
                fourier_coeffs['low'] = fourier_low_coeffs
                fourier_coeffs['omega_low'] = self.omega_low.detach().cpu().numpy()
            if self.fourier_mid_degree > 0:
                fourier_mid_coeffs = self.fourier_mid_coeffs[coff_idx].detach().cpu().numpy()
                fourier_coeffs['mid'] = fourier_mid_coeffs
                fourier_coeffs['omega_mid'] = self.omega_mid.detach().cpu().numpy()
            if self.fourier_high_degree > 0:
                fourier_high_coeffs = self.fourier_high_coeffs[coff_idx].detach().cpu().numpy()
                fourier_coeffs['high'] = fourier_high_coeffs
                fourier_coeffs['omega_high'] = self.omega_high.detach().cpu().numpy()
            
            coefficients[coord]['fourier'] = fourier_coeffs
        return coefficients
