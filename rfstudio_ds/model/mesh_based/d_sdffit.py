from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Any, Tuple, List, Optional, List, Dict
from jaxtyping import Float32

import torch
from torch import Tensor

from rfstudio.nn import Module
from rfstudio.utils.decorator import chains
from rfstudio_ds.graphics import DS_Cameras, DS_TriangleMesh, DualDomain4DIsoCubes, DualDomain4DFlexiCubes


@dataclass
class D_SDFFit(Module):

    # geometry setting
    geometry: Literal['gt', 'DD_isocubes', 'DD_flexicubes'] = 'DD_isocubes' # method to generate geometry, gt: ground truth, DD_isocubes: Dual-Domain 4D isosurface cubes; 
    geometry_resolution: int = 128
    geometry_scale: float = 1.0
    # dynamic geometry setting
    time_resolution: int = 240
    poly_degree: int = 3
    low_freq_fourier_bands: List[int] = field(default_factory=lambda: [1, 3])
    mid_freq_fourier_bands: List[int] = field(default_factory=lambda: [4, 9])
    high_freq_fourier_bands: List[int] = field(default_factory=lambda: [10, 18])
    wavelet_name: Literal['haar', 'db2', 'db4',] = 'db2'
    wavelet_level: int = 3  
    # geometry regularization setting
    reg_time_tv_able: bool = False
    reg_coeff_tv_able: bool = False
    reg_wavelet_sparse_able: bool = False
    
    def __setup__(self) -> None:
        if self.geometry == 'DD_isocubes':
            self.geometric_repr = DualDomain4DIsoCubes.from_resolution(
                self.geometry_resolution,
                scale=self.geometry_scale,
                poly_degree=self.poly_degree,
                low_freq_fourier_bands=self.low_freq_fourier_bands,
                mid_freq_fourier_bands=self.mid_freq_fourier_bands,
                high_freq_fourier_bands=self.high_freq_fourier_bands,
                time_frames=torch.linspace(0, 1, self.time_resolution),
                wavelet_name=self.wavelet_name,
                wavelet_level=self.wavelet_level,
            )
            self.static_sdf_params = torch.nn.Parameter(self.geometric_repr.static_sdf_values.clone())
            self.sdf_curve_poly_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_poly_coefficient.clone())
            self.sdf_curve_low_freq_fourier_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_low_freq_fourier_coefficient.clone())
            self.sdf_curve_mid_freq_fourier_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_mid_freq_fourier_coefficient.clone())
            self.sdf_curve_high_freq_fourier_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_high_freq_fourier_coefficient.clone())
            self.sdf_curve_low_freq_fourier_omega = torch.nn.Parameter(self.geometric_repr.sdf_curve_low_freq_fourier_omega.clone())
            self.sdf_curve_mid_freq_fourier_omega = torch.nn.Parameter(self.geometric_repr.sdf_curve_mid_freq_fourier_omega.clone())
            self.sdf_curve_high_freq_fourier_omega = torch.nn.Parameter(self.geometric_repr.sdf_curve_high_freq_fourier_omega.clone())
            self.sdf_curve_wavelet_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_wavelet_coefficient.clone())
            self.deform_params = torch.nn.Parameter(torch.empty(0)) # disable in isocubes when 4d reconstruct
            self.weight_params = torch.nn.Parameter(torch.empty(0)) # disable in isocubes
        elif self.geometry == 'DD_flexicubes':
            self.geometric_repr = DualDomain4DFlexiCubes.from_resolution(
                self.geometry_resolution,
                scale=self.geometry_scale,
                poly_degree=self.poly_degree,
                low_freq_fourier_bands=self.low_freq_fourier_bands,
                mid_freq_fourier_bands=self.mid_freq_fourier_bands,
                high_freq_fourier_bands=self.high_freq_fourier_bands,
                time_frames=torch.linspace(0, 1, self.time_resolution),
                wavelet_name=self.wavelet_name,
                wavelet_level=self.wavelet_level,
            )
            self.static_sdf_params = torch.nn.Parameter(self.geometric_repr.static_sdf_values.clone())
            self.sdf_curve_poly_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_poly_coefficient.clone())
            self.sdf_curve_low_freq_fourier_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_low_freq_fourier_coefficient.clone())
            self.sdf_curve_mid_freq_fourier_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_mid_freq_fourier_coefficient.clone())
            self.sdf_curve_high_freq_fourier_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_high_freq_fourier_coefficient.clone())
            self.sdf_curve_low_freq_fourier_omega = torch.nn.Parameter(self.geometric_repr.sdf_curve_low_freq_fourier_omega.clone())
            self.sdf_curve_mid_freq_fourier_omega = torch.nn.Parameter(self.geometric_repr.sdf_curve_mid_freq_fourier_omega.clone())
            self.sdf_curve_high_freq_fourier_omega = torch.nn.Parameter(self.geometric_repr.sdf_curve_high_freq_fourier_omega.clone())
            self.sdf_curve_wavelet_coefficient = torch.nn.Parameter(self.geometric_repr.sdf_curve_wavelet_coefficient.clone())
            self.deform_params = torch.nn.Parameter(torch.empty(0)) # disable in isocubes when 4d reconstruct
            self.weight_params = torch.nn.Parameter(torch.empty(0)) # disable in isocubes
        else:
            raise ValueError(self.geometry)
        
        self.dynamic_model_stage: dict = field(default_factory=dict)
        
        self.reg_coeff_tv_weight = 0.0
        self.reg_time_tv_weight = 0.0
        self.reg_wavelet_sparse_weight = 0.0
        # self.reg_scene_flow_smoothness_weight = 0.0

    def get_sdfs(self, t: Tensor = None) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.no_grad():
            if self.geometric_repr.device != self.device:
                self.geometric_repr.swap_(self.geometric_repr.to(self.device))
        assert t is not None
        if self.geometry == 'DD_isocubes':
            geometric_repr = self.geometric_repr.replace(
                static_sdf_values = self.static_sdf_params,
                sdf_curve_poly_coefficient = self.sdf_curve_poly_coefficient,
                sdf_curve_low_freq_fourier_coefficient = self.sdf_curve_low_freq_fourier_coefficient,
                sdf_curve_mid_freq_fourier_coefficient = self.sdf_curve_mid_freq_fourier_coefficient,
                sdf_curve_high_freq_fourier_coefficient = self.sdf_curve_high_freq_fourier_coefficient,
                sdf_curve_low_freq_fourier_omega = self.sdf_curve_low_freq_fourier_omega,
                sdf_curve_mid_freq_fourier_omega = self.sdf_curve_mid_freq_fourier_omega,
                sdf_curve_high_freq_fourier_omega = self.sdf_curve_high_freq_fourier_omega,
                sdf_curve_wavelet_coefficient = self.sdf_curve_wavelet_coefficient,
            )
            sdfs, sdf_flows = geometric_repr.query_sdf_at_times(t=t,model_stage=self.dynamic_model_stage, compute_sdf_flow=True)
            time_tv_loss = sdf_flows.abs().mean() * self.reg_time_tv_weight if self.reg_time_tv_weight > 0.0 else 0.0
            coeff_tv_loss = geometric_repr.compute_coeff_tv_loss() * self.reg_coeff_tv_weight if self.reg_coeff_tv_able else 0.0
            wavelet_sparse_loss = geometric_repr.compute_wavelet_sparse_loss() * self.reg_wavelet_sparse_weight if self.reg_wavelet_sparse_able else 0.0
            return sdfs, sdf_flows, time_tv_loss, coeff_tv_loss, wavelet_sparse_loss
        elif self.geometry == 'DD_flexicubes':
            geometric_repr = self.geometric_repr.replace(
                static_sdf_values = self.static_sdf_params,
                sdf_curve_poly_coefficient = self.sdf_curve_poly_coefficient,
                sdf_curve_low_freq_fourier_coefficient = self.sdf_curve_low_freq_fourier_coefficient,
                sdf_curve_mid_freq_fourier_coefficient = self.sdf_curve_mid_freq_fourier_coefficient,
                sdf_curve_high_freq_fourier_coefficient = self.sdf_curve_high_freq_fourier_coefficient,
                sdf_curve_low_freq_fourier_omega = self.sdf_curve_low_freq_fourier_omega,
                sdf_curve_mid_freq_fourier_omega = self.sdf_curve_mid_freq_fourier_omega,
                sdf_curve_high_freq_fourier_omega = self.sdf_curve_high_freq_fourier_omega,
                sdf_curve_wavelet_coefficient = self.sdf_curve_wavelet_coefficient,
            )
            sdfs, sdf_flows = geometric_repr.query_sdf_at_times(t=t,model_stage=self.dynamic_model_stage, compute_sdf_flow=True)
            time_tv_loss = sdf_flows.abs().mean() * self.reg_time_tv_weight if self.reg_time_tv_able else 0.0
            coeff_tv_loss = geometric_repr.compute_coeff_tv_loss() * self.reg_coeff_tv_weight if self.reg_coeff_tv_able else 0.0
            wavelet_sparse_loss = geometric_repr.compute_wavelet_sparse_loss() * self.reg_wavelet_sparse_weight if self.reg_wavelet_sparse_able else 0.0
            return sdfs, sdf_flows, time_tv_loss, coeff_tv_loss, wavelet_sparse_loss
        else:
            raise ValueError(self.geometry)

    def render_report(
            self, 
            times: Tensor,
            trainer_mode: Literal['train', 'val', 'test', 'orbit_vis', 'fix_vis'] = 'train',
        ) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        if trainer_mode != 'train':
            with torch.no_grad():
                sdfs, sdf_flows, time_tv_loss, coeff_tv_loss, wavelet_sparse_loss = self.get_sdfs(t=times)
        else:
            sdfs, sdf_flows, time_tv_loss, coeff_tv_loss, wavelet_sparse_loss = self.get_sdfs(t=times)
        reg_loss_dict = {
            'reg_time_tv': time_tv_loss,
            'reg_coeff_tv': coeff_tv_loss,
            'reg_wavelet_sparse': wavelet_sparse_loss,
        }
        return sdfs, sdf_flows, reg_loss_dict

    def _get_info_for_points(
            self,
            indices: Tensor,
        ):
        with torch.no_grad():
            if self.geometric_repr.device != self.device:
                self.geometric_repr.swap_(self.geometric_repr.to(self.device))

        if self.geometry == 'DD_isocubes':
            geometric_repr = self.geometric_repr.replace(
                static_sdf_values = self.static_sdf_params,
                sdf_curve_poly_coefficient = self.sdf_curve_poly_coefficient,
                sdf_curve_low_freq_fourier_coefficient = self.sdf_curve_low_freq_fourier_coefficient,
                sdf_curve_mid_freq_fourier_coefficient = self.sdf_curve_mid_freq_fourier_coefficient,
                sdf_curve_high_freq_fourier_coefficient = self.sdf_curve_high_freq_fourier_coefficient,
                sdf_curve_low_freq_fourier_omega = self.sdf_curve_low_freq_fourier_omega,
                sdf_curve_mid_freq_fourier_omega = self.sdf_curve_mid_freq_fourier_omega,
                sdf_curve_high_freq_fourier_omega = self.sdf_curve_high_freq_fourier_omega,
                sdf_curve_wavelet_coefficient = self.sdf_curve_wavelet_coefficient,
            )
        elif self.geometry == 'DD_flexicubes':
            geometric_repr = self.geometric_repr.replace(
                static_sdf_values = self.static_sdf_params,
                sdf_curve_poly_coefficient = self.sdf_curve_poly_coefficient,
                sdf_curve_low_freq_fourier_coefficient = self.sdf_curve_low_freq_fourier_coefficient,
                sdf_curve_mid_freq_fourier_coefficient = self.sdf_curve_mid_freq_fourier_coefficient,
                sdf_curve_high_freq_fourier_coefficient = self.sdf_curve_high_freq_fourier_coefficient,
                sdf_curve_low_freq_fourier_omega = self.sdf_curve_low_freq_fourier_omega,
                sdf_curve_mid_freq_fourier_omega = self.sdf_curve_mid_freq_fourier_omega,
                sdf_curve_high_freq_fourier_omega = self.sdf_curve_high_freq_fourier_omega,
                sdf_curve_wavelet_coefficient = self.sdf_curve_wavelet_coefficient,
            )
        points_infos = geometric_repr.get_curve_coefficients_and_coords(indices)
        return points_infos
        
    
    def get_background_color(self) -> Float32[Tensor, "3 "]:
        if self.background_color == 'black':
            return torch.zeros(3)
        if self.background_color == 'white':
            return torch.ones(3)
        if self.training:
            return torch.rand(3)
        return torch.tensor([0.1490, 0.1647, 0.2157])

    @chains
    def as_module(self, *, field_name: Literal['poly_coeffs', 'fourier_low_coeffs', 'fourier_mid_coeffs', 'fourier_high_coeffs']) -> Any:
        def parameters(self) -> Any:
            params = {
                'static_sdf_params': self.static_sdf_params,
                'dddm_poly_coeffs': self.sdf_curve_poly_coefficient,
                'dddm_fourier_low_coeffs': self.sdf_curve_low_freq_fourier_coefficient,
                'dddm_fourier_mid_coeffs': self.sdf_curve_mid_freq_fourier_coefficient,
                'dddm_fourier_high_coeffs': self.sdf_curve_high_freq_fourier_coefficient,
                'dddm_fourier_low_omega': self.sdf_curve_low_freq_fourier_omega,
                'dddm_fourier_mid_omega': self.sdf_curve_mid_freq_fourier_omega,
                'dddm_fourier_high_omega': self.sdf_curve_high_freq_fourier_omega,
                'dddm_wavelet_coeffs': self.sdf_curve_wavelet_coefficient,
            }[field_name]
            return [params]
        return parameters
