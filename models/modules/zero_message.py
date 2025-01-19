from math import pi as PI
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear
from torch_scatter import scatter
from torch_geometric.nn import radius_graph
from typing import List, Optional
import logging
from e3nn import o3


class edge_update(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super(edge_update, self).__init__()
        self.cutoff = cutoff
        self.linear = Linear(hidden_channels, num_filters, bias=False)
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)

    def forward(self, node_feats, dist, rbf, edge_index):
        j, _ = edge_index
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        W = self.mlp(rbf) * C.view(-1, 1)
        node_feats = self.linear(node_feats)
        edge_feats = node_feats[j] * W
        return edge_feats

class edge_update_mix(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff):
        super(edge_update_mix, self).__init__()
        self.cutoff = cutoff
        self.linear_sca = Linear(hidden_channels, num_filters, bias=False)
        self.linear_vec = o3.Linear(irreps_in=o3.Irreps('128x1o+128x2e'), irreps_out=o3.Irreps('128x1o+128x2e'))
        # self.linear_comb = Linear(num_filters * 2, num_filters, bias=False)
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear_sca.weight)
        # torch.nn.init.xavier_uniform_(self.linear_vec.weight)
        # torch.nn.init.xavier_uniform_(self.linear_comb.weight)
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[0].bias.data.fill_(0)

    def forward(self, node_sca, node_vec, dist, rbf, edge_index):
        j, _ = edge_index
        C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        W = self.mlp(rbf) * C.view(-1, 1)
        node_vec = self.linear_vec(node_vec)
        node_sca = self.linear(node_sca).unsqueeze(-1)
        node_feats = torch.cat([node_sca, node_vec], dim=-1)
        edge_feats = node_feats[j] * W.unsqueeze(1)
        return edge_feats

class node_update_mix(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters):
        super(node_update_mix, self).__init__()
        self.act = ShiftedSoftplus()
        self.linear_1 = Linear(num_filters, hidden_channels)
        self.linear_2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_2.bias.data.fill_(0)

    def forward(self, node_sca, node_vec, edge_feats, edge_index):
        _, i = edge_index
        agg = scatter(edge_feats, i, dim=0, dim_size=node_sca.size()[0])
        agg_vec = agg[:, 1:, :]
        agg = agg.sum(dim=1)
        agg = self.linear_1(agg)
        agg = self.act(agg)
        agg = self.linear_2(agg)

        node_sca = node_sca + agg
        node_vec = node_vec + agg_vec

        return node_sca, node_vec


class node_update(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters):
        super(node_update, self).__init__()
        self.act = ShiftedSoftplus()
        self.linear_1 = Linear(num_filters, hidden_channels)
        self.linear_2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear_1.weight)
        self.linear_1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.linear_2.weight)
        self.linear_2.bias.data.fill_(0)

    def forward(self, node_feats, edge_feats, edge_index):
        _, i = edge_index
        agg = scatter(edge_feats, i, dim=0, dim_size=node_feats.size()[0])
        agg = self.linear_1(agg)
        agg = self.act(agg)
        agg = self.linear_2(agg)

        return node_feats + agg


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift