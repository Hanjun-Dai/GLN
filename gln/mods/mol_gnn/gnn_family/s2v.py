from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from gln.mods.mol_gnn.gnn_family.utils import GNNEmbedding, prepare_gnn, get_agg, ReadoutNet
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import NNConv, Set2Set
from gln.mods.mol_gnn.torch_util import MLP, NONLINEARITIES
from torch_scatter import scatter_add


class _MeanFieldLayer(MessagePassing):
    def __init__(self, latent_dim):
        super(_MeanFieldLayer, self).__init__()
        self.conv_params = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, edge_index):
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, x=x)

    def update(self, aggr_out):
        return self.conv_params(aggr_out)


class S2vMeanFieldV2(GNNEmbedding):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv=3, act_func='relu', readout_agg='sum', act_last=True, out_method='last', bn=True, dropout=None):
        if output_dim > 0:
            embed_dim = output_dim
        else:
            embed_dim = latent_dim
        super(S2vMeanFieldV2, self).__init__(embed_dim, dropout)
        self.latent_dim = latent_dim
        self.output_dim = output_dim        
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.bn = bn
        self.max_lv = max_lv
        self.act_func = NONLINEARITIES[act_func]
        self.w_n2l = nn.Linear(num_node_feats, latent_dim)
        if num_edge_feats > 0:
            self.w_e2l = [nn.Linear(num_edge_feats, latent_dim) for _ in range(self.max_lv + 1)]
            self.w_e2l = nn.ModuleList(self.w_e2l)
        lm_layer = lambda: _MeanFieldLayer(latent_dim)        
        conv_layers = [lm_layer() for _ in range(max_lv)]
        self.conv_layers = nn.ModuleList(conv_layers)        
        self.conv_l2 = [nn.Linear(latent_dim, latent_dim) for _ in range(self.max_lv)]
        self.conv_l2 = nn.ModuleList(self.conv_l2)

        self.readout_net = ReadoutNet(node_state_dim=latent_dim,
                                      output_dim=output_dim,
                                      max_lv=max_lv,
                                      act_func=act_func,
                                      out_method='last',
                                      readout_agg=readout_agg,
                                      act_last=act_last,
                                      bn=bn)
        if self.bn:
            msg_bn = [nn.BatchNorm1d(latent_dim) for _ in range(self.max_lv + 1)]
            hidden_bn = [nn.BatchNorm1d(latent_dim) for _ in range(self.max_lv)]
            self.msg_bn = nn.ModuleList(msg_bn)
            self.hidden_bn = nn.ModuleList(hidden_bn)
        else:
            self.msg_bn = [lambda x: x for _ in range(self.max_lv + 1)]
            self.hidden_bn = [lambda x: x for _ in range(self.max_lv)]

    def get_feat(self, graph_list):
        node_feat, edge_feat, edge_from_idx, edge_to_idx, g_idx = prepare_gnn(graph_list, self.is_cuda())
        input_node_linear = self.w_n2l(node_feat)
        input_message = input_node_linear
        if edge_feat is not None:
            input_edge_linear = self.w_e2l[0](edge_feat)
            e2npool_input = scatter_add(input_edge_linear, edge_to_idx, dim=0, dim_size=node_feat.shape[0])
            input_message += e2npool_input
        input_potential = self.act_func(input_message)
        input_potential = self.msg_bn[0](input_potential)

        cur_message_layer = input_potential
        all_embeds = [cur_message_layer]
        edge_index = [edge_from_idx, edge_to_idx]
        edge_index = torch.stack(edge_index)        
        for lv in range(self.max_lv):
            node_linear = self.conv_layers[lv](cur_message_layer, edge_index)
            edge_linear = self.w_e2l[lv + 1](edge_feat)
            e2npool_input = scatter_add(edge_linear, edge_to_idx, dim=0, dim_size=node_linear.shape[0])            
            merged_hidden = self.act_func(node_linear + e2npool_input)
            merged_hidden = self.hidden_bn[lv](merged_hidden)
            residual_out = self.conv_l2[lv](merged_hidden) + cur_message_layer
            cur_message_layer = self.act_func(residual_out)
            cur_message_layer = self.msg_bn[lv + 1](cur_message_layer)
            all_embeds.append(cur_message_layer)
        return self.readout_net(all_embeds, g_idx, len(graph_list))
