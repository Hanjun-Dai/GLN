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


class EmbedMeanField(GNNEmbedding):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv=3, act_func='tanh', readout_agg='sum', share_params=True, act_last=True, dropout=None):
        if output_dim > 0:
            embed_dim = output_dim
        else:
            embed_dim = latent_dim
        super(EmbedMeanField, self).__init__(embed_dim, dropout)
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats

        self.max_lv = max_lv
        self.act_func = NONLINEARITIES[act_func]
        self.w_n2l = nn.Linear(num_node_feats, latent_dim)
        if num_edge_feats > 0:
            self.w_e2l = nn.Linear(num_edge_feats, latent_dim)
        
        lm_layer = lambda: _MeanFieldLayer(latent_dim)
        if share_params:
            self.conv_layer = lm_layer()
            self.conv_layers = [lambda x, y: self.conv_layer(x, y) for _ in range(max_lv)]
        else:
            conv_layers = [lm_layer() for _ in range(max_lv)]
            self.conv_layers = nn.ModuleList(conv_layers)

        self.readout_net = ReadoutNet(node_state_dim=latent_dim,
                                      output_dim=output_dim,
                                      max_lv=max_lv,
                                      act_func=act_func,
                                      out_method='last',
                                      readout_agg=readout_agg,
                                      act_last=act_last,
                                      bn=False)

    def get_feat(self, graph_list):
        node_feat, edge_feat, edge_from_idx, edge_to_idx, g_idx = prepare_gnn(graph_list, self.is_cuda())
        input_node_linear = self.w_n2l(node_feat)
        input_message = input_node_linear
        if edge_feat is not None:
            input_edge_linear = self.w_e2l(edge_feat)
            e2npool_input = scatter_add(input_edge_linear, edge_to_idx, dim=0, dim_size=node_feat.shape[0])
            input_message += e2npool_input
        input_potential = self.act_func(input_message)

        cur_message_layer = input_potential
        all_embeds = [cur_message_layer]
        edge_index = [edge_from_idx, edge_to_idx]
        edge_index = torch.stack(edge_index)
        for lv in range(self.max_lv):
            node_linear = self.conv_layers[lv](cur_message_layer, edge_index)            
            merged_linear = node_linear + input_message
            cur_message_layer = self.act_func(merged_linear)
            all_embeds.append(cur_message_layer)

        return self.readout_net(all_embeds, g_idx, len(graph_list))
