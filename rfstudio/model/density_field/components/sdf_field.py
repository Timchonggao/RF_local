from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from rfstudio.graphics import RaySamples
from rfstudio.model.density_field.components.encoding import PosEncoding
from rfstudio.nn import MLP, Module, ParameterModule
from rfstudio.utils.decorator import lazy


@dataclass
class SDFMLP(Module):

    '''SDF Network for NeuS'''

    layers: List[int] = ...
    skip_connections: List[int] = field(default_factory=list)
    init_bias: float = 0.8
    weight_norm: bool = True
    invert_init: bool = False
    position_encoding: PosEncoding = PosEncoding(num_frequencies=6, max_freq_exp=5.0)

    def __setup__(self) -> None:
        assert len(self.layers) > 1
        assert all([layer != -1 for layer in self.layers[1:]]), "Only the first layer can be set to auto size"
        assert 0 not in self.skip_connections, "Skip connection at layer 0 doesn't make sense."

        self.skip_connection_set = set(self.skip_connections)
        nn_layers = []
        for i, in_dim, out_dim in zip(range(len(self.layers) - 1), self.layers[:-1], self.layers[1:]):
            if i in self.skip_connection_set:
                in_dim = -1 if self.layers[0] == -1 else (in_dim + self.layers[0])
            nn_layers.append(
                nn.Linear(in_dim, out_dim)
                if in_dim != -1
                else nn.LazyLinear(out_dim)
            )
        self.nn_layers = nn.ModuleList(nn_layers)
        self.softplus = nn.Softplus(beta=100)
        self._loaded = False

    def _load_from_state_dict(
        self,
        state_dict: Dict,
        prefix: str,
        local_metadata: Dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        self._loaded = True
        if self.weight_norm:
            lst = []
            for i, layer in enumerate(self.nn_layers):
                if isinstance(layer, nn.LazyLinear):
                    in_dim = state_dict[prefix + f'nn_layers.{i}.weight_v'].shape[-1]
                    layer.in_features = in_dim
                    if layer.has_uninitialized_params():
                        layer.weight.materialize((layer.out_features, in_dim))
                        if layer.bias is not None:
                            layer.bias.materialize((layer.out_features, ))
                    lst.append(nn.utils.weight_norm(layer))
                elif isinstance(layer, nn.Linear):
                    lst.append(nn.utils.weight_norm(layer))
                else:
                    lst.append(layer)
            self.nn_layers = nn.ModuleList(lst)
        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @lazy
    def initialize_weights(self, init_dim: Optional[int] = None) -> None:
        if self._loaded:
            return
        for i, in_dim, out_dim, layer in zip(
            range(len(self.nn_layers)),
            self.layers[:-1],
            self.layers[1:],
            self.nn_layers,
            strict=True
        ):
            if isinstance(layer, nn.LazyLinear):
                assert init_dim is not None
                in_dim = init_dim if in_dim == -1 else (in_dim + init_dim)
                layer.in_features = in_dim
                if layer.has_uninitialized_params():
                    layer.weight.materialize((layer.out_features, in_dim))
                    layer.bias.materialize((layer.out_features, ))
            if i == len(self.nn_layers) - 1:
                if not self.invert_init:
                    torch.nn.init.normal_(layer.weight, mean=(torch.pi / in_dim) ** 0.5, std=1e-4)
                    torch.nn.init.constant_(layer.bias, -self.init_bias)
                else:
                    torch.nn.init.normal_(layer.weight, mean=-(torch.pi / in_dim) ** 0.5, std=1e-4)
                    torch.nn.init.constant_(layer.bias, self.init_bias)
            elif i == 0:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(layer.weight[:, :3], 0.0, (2 / out_dim) ** 0.5)
            elif i in self.skip_connection_set:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, (2 / out_dim) ** 0.5)
                torch.nn.init.constant_(layer.weight[:, -(init_dim - 3):], 0.0)
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, (2 / out_dim) ** 0.5)
        if self.weight_norm:
            self.nn_layers = nn.ModuleList([
                torch.nn.utils.weight_norm(layer) if isinstance(layer, nn.Linear) else layer
                for layer in self.nn_layers
            ])

    def __call__(self, inputs: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs out_dim"]:
        inputs = self.position_encoding(inputs)
        x = inputs
        self.initialize_weights(x.shape[-1])
        for i, layer in enumerate(self.nn_layers):
            if i in self.skip_connection_set:
                x = torch.cat((x, inputs), dim=-1) / (2 ** 0.5)
            x = layer(x)
            if i < len(self.nn_layers) - 1:
                x = self.softplus(x)
        return x

    def get_sdf_gradient(
        self,
        inputs: Float[Tensor, "*bs 3"],
    ) -> Tuple[Float[Tensor, "*bs 1"], Float[Tensor, "*bs out_dim"], Float[Tensor, "*bs 3"]]:
        inputs = inputs.clone().requires_grad_(True)
        with torch.enable_grad():
            features = self.__call__(inputs)
            sdf = features[..., :1]
            assert sdf.isfinite().all()
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=inputs,
            grad_outputs=torch.ones_like(sdf, device=sdf.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        assert gradients.isfinite().all()
        return sdf, features[..., 1:], gradients

    def get_sdf(self, inputs: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs 1"]:
        features = self.__call__(inputs)
        sdf = features[..., :1]
        assert sdf.isfinite().all()
        return sdf


@dataclass
class SDFField(Module):

    """NeuS Field"""

    direction_encoding: PosEncoding = PosEncoding(num_frequencies=4, max_freq_exp=3.0)
    sdf_mlp: SDFMLP = SDFMLP(
        layers=[-1, 256, 256, 256, 256, 256, 256, 256, 257],
        skip_connections=[4],
    )
    color_mlp: MLP = MLP(
        layers=[-1, 256, 256, 256, 256, 3],
        activation='sigmoid',
        weight_norm=True,
    )
    deviation: ParameterModule = ParameterModule(shape=[1], init_value=0.1)

    def get_sdf(self, positions: Float[Tensor, "... 3"]) -> Float[Tensor, "... 1"]:
        return self.sdf_mlp.get_sdf(positions)

    def get_variance(self) -> Float[Tensor, "1"]:
        return (self.deviation.params * 10.0).exp().clamp(1e-6, 1e6)

    def get_alphas(
        self,
        samples: RaySamples,
        *,
        cos_anneal_ratio: float,
        sdf: Optional[Float[Tensor, "... 1"]] = None,
        gradients: Optional[Float[Tensor, "... 1"]] = None,
    ) -> Float[Tensor, "... 1"]:
        if sdf is None or gradients is None:
            sdf, _, gradients = self.sdf_mlp.get_sdf_gradient(samples.positions)
        inv_s = self.get_variance() # [1]
        true_cos = (samples.directions[..., None, :] * gradients).sum(-1, keepdim=True) # [..., S, 1]

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive, [..., S, 1]

        # Estimate signed distances at section points
        distances = samples.distances
        estimated_next_sdf = sdf + iter_cos * distances * 0.5
        estimated_prev_sdf = sdf - iter_cos * distances * 0.5

        prev_cdf = (estimated_prev_sdf * inv_s).sigmoid()
        next_cdf = (estimated_next_sdf * inv_s).sigmoid()
        p = prev_cdf - next_cdf
        c = prev_cdf
        alphas = ((p + 1e-5) / (c + 1e-5)).clamp(0.0, 1.0)
        return alphas

    def get_outputs(self, ray_samples: RaySamples, *, cos_anneal_ratio: float) -> Tuple[RaySamples, Tensor]:
        positions = ray_samples.positions
        encoded_dir = self.direction_encoding(
            ray_samples.directions /
            ray_samples.directions.norm(dim=-1, keepdim=True)
        ) # [..., E]
        sdf, geom_feats, gradients = self.sdf_mlp.get_sdf_gradient(positions)
        alpha = self.get_alphas(ray_samples, cos_anneal_ratio=cos_anneal_ratio, sdf=sdf, gradients=gradients)
        encoded_dir = encoded_dir[..., None, :].expand(
            *encoded_dir.shape[:-1],
            alpha.shape[-2],
            encoded_dir.shape[-1],
        )                                                              # [..., S, E]
        color = self.color_mlp(torch.cat((
            positions,
            gradients,
            encoded_dir,
            geom_feats,
        ), dim=-1))                                                    # [..., S, E]
        eikonal_loss = ((gradients.norm(2, dim=-1) - 1) ** 2).mean(-1)
        return ray_samples.annotate(colors=color, alphas=alpha).get_weighted(), eikonal_loss
