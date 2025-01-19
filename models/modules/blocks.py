from abc import abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch.nn.functional
from e3nn import nn, o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from typing import Dict, Optional, Union
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, to_dense_batch
from math import pi as PI
from .zero_message import *


class HIL(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cutoff: float, num_layers: int, edge_channels: int):
        super(HIL, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.edge_channels = edge_channels

        # self.mlp_radial = torch.nn.Sequential(torch.nn.Linear(8, self.in_channels), torch.nn.SiLU())

        self.aggregate = torch.nn.ModuleList([node_update(self.in_channels, self.out_channels) for _ in range(self.num_layers)])
        self.message = torch.nn.ModuleList([edge_update(self.in_channels, self.out_channels, self.edge_channels, self.cutoff) for _ in range(self.num_layers)])

    def forward(self, node_feats, edge_feats, edge_index, dist):
        for mess, agg in zip(self.message, self.aggregate):
            edge_attr = mess(node_feats, dist, edge_feats, edge_index)
            node_feats = agg(node_feats, edge_attr, edge_index)

        return node_feats

def mask_head(x: torch.Tensor, head: torch.Tensor, num_heads: int) -> torch.Tensor:
    mask = torch.zeros(x.shape[0], x.shape[1] // num_heads, num_heads, device=x.device)
    idx = torch.arange(mask.shape[0], device=x.device)
    mask[idx, :, head] = 1
    mask = mask.permute(0, 2, 1).reshape(x.shape)
    return x * mask


def _broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(
        src: torch.Tensor,
        index: torch.Tensor,
        dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None,
        reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(
            self,
            node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.linear(node_attrs)

class NodeChargeEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.node_linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)
        self.charge_linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)
        self.out_linear = o3.Linear(irreps_in=o3.Irreps('256x0e'), irreps_out=irreps_out)

    def forward(
            self,
            node_attrs: torch.Tensor,
            formal_charge: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]

        node_feats = self.node_linear(node_attrs)
        charge_feats = self.charge_linear(formal_charge.unsqueeze(-1))

        node_feats = self.out_linear(torch.cat([node_feats, charge_feats], dim=-1))

        return node_feats

class HiddenReadoutBlock(torch.nn.Module):
    def __init__(
            self,
            irreps_in: o3.Irreps,
            MLP_irreps: o3.Irreps,
            gate: Optional[Callable],
            irrep_out: o3.Irreps = o3.Irreps("0e"),
            num_heads: int = 1,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])

    def forward(
            self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return x  # [n_nodes, len(heads)]


class energy_layer(torch.nn.Module):
    def __init__(
            self,
            irreps_in: o3.Irreps,
            MLP_irreps: o3.Irreps,
            gate: Optional[Callable],
            irrep_out: o3.Irreps = o3.Irreps("0e"),
            num_heads: int = 1,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.num_heads = num_heads
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        # self.linear_2 = o3.Linear(irreps_in=self.hidden_irreps, irreps_out=irrep_out)

    def forward(
            self, x: torch.Tensor, heads: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return x

class InteractionBlock(torch.nn.Module):
    def __init__(
            self,
            node_attrs_irreps: o3.Irreps,
            node_feats_irreps: o3.Irreps,
            edge_attrs_irreps: o3.Irreps,
            edge_feats_irreps: o3.Irreps,
            target_irreps: o3.Irreps,
            hidden_irreps: o3.Irreps,
            avg_num_neighbors: float,
            radial_MLP: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
            self,
            idx: int,
            node_attrs: torch.Tensor,
            node_feats: torch.Tensor,
            edge_attrs: torch.Tensor,
            edge_feats: torch.Tensor,
            edge_index: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


nonlinearities = {1: torch.nn.functional.silu, -1: torch.tanh}


class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(
            self,
            r_max: float,
            num_bessel: int,
            num_polynomial_cutoff: int,
    ):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
            self,
            edge_lengths: torch.Tensor,  # [n_edges, 1]
    ):
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]

        radial = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        return radial * cutoff  # [n_edges, n_basis]


class Spherical_block(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, gate=torch.nn.functional.silu):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.pre_activation = nn.Activation(irreps_in=self.irreps_in, acts=[gate])
        self.pre_linear = o3.Linear(
            self.irreps_in, self.irreps_out, internal_weights=True, shared_weights=True
        )
        self.post_activation = nn.Activation(irreps_in=self.irreps_in, acts=[gate])
        self.post_linear = o3.Linear(
            self.irreps_in, self.irreps_out, internal_weights=True, shared_weights=True
        )

    def forward(self, xs):
        ys = xs
        xs = self.pre_activation(xs)
        xs = self.pre_linear(xs)
        xs = self.post_activation(xs)
        xs = self.post_linear(xs)
        xs = ys + xs

        return xs


class SphericalConv(InteractionBlock):
    def _setup(self) -> None:
        self.value = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        # input_dim = 128
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.irreps_out, self.node_attrs_irreps, self.irreps_out
        )

    def forward(self,
                idx: int,
                node_attrs: torch.Tensor,
                node_feats: torch.Tensor,
                edge_attrs: torch.Tensor,
                edge_feats: torch.Tensor,
                edge_index: torch.Tensor,
                ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]

        node_feats = self.value(node_feats)

        num_nodes = node_feats.shape[0]
        tp_weights = self.conv_tp_weights(edge_feats)

        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        # if idx > 0:
        #     message = message + node_feats
        return message  # [n_nodes, irreps]


class BesselBasis(torch.nn.Module):
    """
    Equation (7)
    """

    def __init__(self, r_max: float, num_basis=8, trainable=False):
        super().__init__()

        bessel_weights = (
            np.pi
            / r_max
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_weights)
        else:
            self.register_buffer("bessel_weights", bessel_weights)

        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(np.sqrt(2.0 / r_max), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )

class PolynomialCutoff(torch.nn.Module):
    """
    Equation (8)
    """

    p: torch.Tensor
    r_max: torch.Tensor

    def __init__(self, r_max: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "r_max", torch.tensor(r_max, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # yapf: disable
        envelope = (
                1.0
                - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.r_max, self.p)
                + self.p * (self.p + 2.0) * torch.pow(x / self.r_max, self.p + 1)
                - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.r_max, self.p + 2)
        )
        # yapf: enable

        # noinspection PyUnresolvedReferences
        return envelope * (x < self.r_max)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_max={self.r_max})"


# Based on mir-group/nequip
def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
) -> Tuple[o3.Irreps, List]:
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: List[Tuple[int, o3.Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions

def linear_out_irreps(irreps: o3.Irreps, target_irreps: o3.Irreps) -> o3.Irreps:
    # Assuming simplified irreps
    irreps_mid = []
    for _, ir_in in irreps:
        found = False

        for mul, ir_out in target_irreps:
            if ir_in == ir_out:
                irreps_mid.append((mul, ir_out))
                found = True
                break

        if not found:
            raise RuntimeError(f"{ir_in} not in {target_irreps}")

    return o3.Irreps(irreps_mid)
