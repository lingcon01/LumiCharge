# coding: utf-8

import math
import os
from argparse import Namespace
from collections import defaultdict

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter
from e3nn import o3
import torch.nn.functional as F
import logging
from torch_geometric.nn import radius_graph
from .modules.zero_message import ShiftedSoftplus

from .modules.blocks import (
    LinearNodeEmbeddingBlock,
    energy_layer,
    SphericalConv,
    RadialEmbeddingBlock,
    HIL,
)


class DecNet(nn.Module):
    """Neural network for computing Hamiltonian/Overlap matrices in a rotationally equivariant way"""

    def __init__(
            self,
            order=None,  # 1 maximum order of spherical harmonics features 定义球谐函数的最高阶次（控制旋转对称性）
            basis_functions=None,  # exp-bernstein 定义径向基函数类型，用于对分子间的距离进行参数化
            num_basis_functions=128,
            # type of radial basis functions (exp-gaussian/exp-bernstein/gaussian/bernstein)
            cutoff=None,  # 15.0 cutoff distance (default is 15 Bohr)
            num_elements=1,
            radial_MLP=None,
            avg_num_neighbors=3,
            correlation=3,
            num_interactions=3,
            heads=["dft"],
            # hidden_irreps=o3.Irreps("128x0e+128x1o+128x2e+128x3o+128x4e"),
            hidden_irreps=o3.Irreps("128x0e+128x1o+128x2e"),
            MLP_irreps=o3.Irreps("64x0e"),
            gate=ShiftedSoftplus(),
            r_max=5.0,
            num_bessel=8,
            num_polynomial_cutoff=6,
    ):
        super(DecNet, self).__init__()

        # variables to control the flow of the forward graph
        # (calculate full_hamiltonian/core_hamiltonian/overlap_matrix/energy/forces?)
        self.create_graph = True  # can be set to False if the NN is only used for inference

        self.order = order

        if isinstance(correlation, int):
            correlation = [correlation] * num_interactions

        # Embedding
        node_attr_irreps = o3.Irreps([(num_elements, (0, 1))])
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))])
        self.node_embedding = LinearNodeEmbeddingBlock(
            irreps_in=node_attr_irreps, irreps_out=node_feats_irreps
        )

        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )

        edge_feats_irreps = o3.Irreps(f"{self.radial_embedding.out_dim}x0e")

        sh_irreps = o3.Irreps.spherical_harmonics(order)
        num_features = hidden_irreps.count(o3.Irrep(0, 1))
        interaction_irreps = (sh_irreps * num_features).sort()[0].simplify()
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        if radial_MLP is None:
            radial_MLP = [64, 64, 64]

        inter = SphericalConv(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=avg_num_neighbors,
            radial_MLP=radial_MLP,
        )
        self.interactions = torch.nn.ModuleList([inter])

        for i in range(num_interactions - 1):
            hidden_irreps_out = hidden_irreps
            inter = SphericalConv(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=avg_num_neighbors,
                radial_MLP=radial_MLP,
            )
            self.interactions.append(inter)

        self.scalar_inter = HIL(128, 128, 5.0, 3, 8)

        self.edge_embedding = nn.Linear(8, 128)

        self.vec2sca = energy_layer(hidden_irreps, o3.Irreps("128x0e"), gate)

        self.sca_embedding = nn.Linear(128, 256)

        self.readout_layer = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1))

        # store hyperparameter values
        self.basis_functions = basis_functions
        self.cutoff = cutoff
        self.num_basis_functions = num_basis_functions

    def get_number_of_parameters(self):
        num = 0
        for param in self.parameters():
            if param.requires_grad:
                num += param.numel()
        return num

    @staticmethod
    def calculate_distances_and_directions(R, idx_i, idx_j):
        Ri = torch.gather(
            R,
            -2,
            idx_i.view(*(1,) * len(R.shape[:-2]), -1, 1).repeat(*R.shape[:-2], 1, R.size(-1)),
        )

        Rj = torch.gather(
            R,
            -2,
            idx_j.view(*(1,) * len(R.shape[:-2]), -1, 1).repeat(*R.shape[:-2], 1, R.size(-1)),
        )

        # Ri = R[idx_i]
        # Rj = R[idx_j]
        rij = Rj - Ri  # displacement vectors
        dij = torch.norm(rij, dim=-1, keepdim=True)  # distances
        uij = rij / dij  # unit displacement vectors
        return dij, uij

    """
    Computes the Hamiltonian/Overlap matrix

    inputs:
        R: Cartesian coordinates of shape [batch_size, num_atoms, 3]
    outputs:
        matrix: Hamiltonian/Overlap matrix of shape [batch_size, num_orbitals, num_orbitals]
    """  # coding: utf-8

    def forward(self, atoms_batch):
        R = atoms_batch.pos.to(float)
        Z = atoms_batch.x.to(float)
        batch = atoms_batch.batch.to(float)

        self.edge_index = radius_graph(R, r=5.0, batch=batch)
        self.idx_i, self.idx_j = self.edge_index[0], self.edge_index[1]

        node_feats = self.node_embedding(Z)

        # compute radial basis functions and spherical harmonics
        dij, uij = self.calculate_distances_and_directions(R, self.idx_i, self.idx_j)
        rbf = self.radial_embedding(dij)

        sph = self.spherical_harmonics(uij)

        # Interactions
        for idx, interaction in enumerate(self.interactions):
            node_feats = interaction(
                idx=idx,
                node_attrs=Z,
                node_feats=node_feats,
                edge_attrs=sph,
                edge_feats=rbf,
                edge_index=self.edge_index,
            )

        node_sca = self.vec2sca(node_feats)

        # node_sca = self.sca_embedding(node_sca)

        node_scalar = self.scalar_inter(node_feats=node_sca, edge_feats=rbf, edge_index=self.edge_index, dist=dij)

        # node_scalar, node_vector = self.scalar_inter(node_sca=node_sca, node_vec=node_vec, edge_feats=rbf, edge_index=self.edge_index, dist=dij)

        node_out = torch.cat([node_sca, node_scalar], dim=-1)

        node_charge = self.readout_layer(node_out)

        return node_charge.squeeze(-1)


