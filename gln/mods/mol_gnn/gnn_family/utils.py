from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from functools import partial
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch_geometric.nn.conv import MessagePassing

from torch_scatter import scatter_add, scatter_mean
from torch_scatter import scatter_max as orig_smax
from torch_scatter import scatter_min as orig_smin

from gln.mods.mol_gnn.mg_clib.mg_lib import MGLIB
from gln.mods.mol_gnn.torch_util import MLP, NONLINEARITIES


class GNNEmbedding(nn.Module):
    def __init__(self, embed_dim, dropout=None):
        super(GNNEmbedding, self).__init__()
        self.embed_dim = embed_dim
        if dropout is not None and dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = lambda x: x

    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, graph_list):
        selected = []
        sublist = []
        for i, g in enumerate(graph_list):
            if g is not None:
                selected.append(i)
                sublist.append(g)
        if len(sublist):
            embed, nodes_info = self.get_feat(sublist)
            embed = self.dropout(embed)
            if nodes_info is not None:
                g_idx, node_embed = nodes_info
                node_embed = self.dropout(embed)
                nodes_info = (g_idx, node_embed)
        else:
            embed = None
            nodes_info = None
        if len(sublist) == len(graph_list):
            return embed, nodes_info

        full_embed = torch.zeros(len(graph_list), self.embed_dim, dtype=torch.float32)
        if self.is_cuda():
            full_embed = full_embed.cuda()
        if embed is not None:
            full_embed[selected] = embed
        return full_embed, None

    def get_feat(self, graph_list):
        raise NotImplementedError


class ReadoutNet(nn.Module):
    def __init__(self, node_state_dim, output_dim, max_lv, act_func, out_method, readout_agg, act_last, bn):
        super(ReadoutNet, self).__init__()

        self.out_method = out_method
        self.max_lv = max_lv
        self.readout_agg = get_agg(readout_agg)
        self.act_last = act_last
        self.act_func = NONLINEARITIES[act_func]
        self.readout_funcs = []
        self.bn = bn
        if output_dim is None:
            self.embed_dim = node_state_dim
            for i in range(self.max_lv + 1):
                self.readout_funcs.append(lambda x: x)
        else:
            self.embed_dim = output_dim
            for i in range(self.max_lv + 1):
                self.readout_funcs.append(nn.Linear(node_state_dim, output_dim))
                if self.out_method == 'last':
                    break
            self.readout_funcs = nn.ModuleList(self.readout_funcs)

        if self.out_method == 'gru':
            self.final_cell = nn.GRUCell(self.embed_dim, self.embed_dim)
        if self.bn:
            out_bn = [nn.BatchNorm1d(self.embed_dim) for _ in range(self.max_lv + 1)]
            self.out_bn = nn.ModuleList(out_bn)

    def forward(self, list_node_states, g_idx, num_graphs):
        assert len(list_node_states) == self.max_lv + 1
        if self.out_method == 'last':
            out_states = self.readout_funcs[0](list_node_states[-1])        
            if self.act_last:
                out_states = self.act_func(out_states)            
            graph_embed = self.readout_agg(out_states, g_idx, dim=0, dim_size=num_graphs)
            return graph_embed, (g_idx, out_states)

        list_node_embed = [self.readout_funcs[i](list_node_states[i]) for i in range(self.max_lv + 1)]
        if self.act_last:
            list_node_embed = [self.act_func(e) for e in list_node_embed]
        if self.bn:
            list_node_embed = [self.out_bn[i](e) for i, e in enumerate(list_node_embed)]
        list_graph_embed = [self.readout_agg(e, g_idx, dim=0, dim_size=num_graphs) for e in list_node_embed]
        out_embed = list_graph_embed[0]

        for i in range(1, self.max_lv + 1):
            if self.out_method == 'gru':
                out_embed = self.final_cell(list_graph_embed[i], out_embed)
            elif self.out_method == 'sum' or self.out_method == 'mean':
                out_embed += list_graph_embed[i]
            else:
                raise NotImplementedError
        
        if self.out_method == 'mean':
            out_embed /= self.max_lv + 1

        return out_embed, (None, None)

         
def scatter_max(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    return orig_smax(src, index, dim, out, dim_size, fill_value)[0]


def scatter_min(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    return orig_smin(src, index, dim, out, dim_size, fill_value)[0]


def get_agg(agg_type):
    if agg_type == 'sum':
        return scatter_add
    elif agg_type == 'mean':
        return scatter_mean
    elif agg_type == 'max':
        return scatter_max
    elif agg_type == 'min':
        return scatter_min
    else:
        raise NotImplementedError


def prepare_gnn(graph_list, is_cuda):
    node_feat, edge_feat = MGLIB.PrepareBatchFeature(graph_list)
    if is_cuda:
        node_feat = node_feat.cuda()
        edge_feat = edge_feat.cuda()
    edge_to_idx, edge_from_idx, g_idx = MGLIB.PrepareIndices(graph_list)
    if is_cuda:
        edge_to_idx = edge_to_idx.cuda()
        edge_from_idx = edge_from_idx.cuda()
        g_idx = g_idx.cuda()
    return node_feat, edge_feat, edge_from_idx, edge_to_idx, g_idx
