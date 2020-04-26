from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from gln.mods.mol_gnn.gnn_family.utils import GNNEmbedding, prepare_gnn, get_agg
from torch_geometric.nn import NNConv, Set2Set
from gln.mods.mol_gnn.torch_util import MLP, NONLINEARITIES
from torch_geometric.nn.conv import MessagePassing


class GGNNConv(MessagePassing):
    def __init__(self, node_state_dim, num_edge_feats, edge_hidden_sizes, 
                 aggr='add', act_func='elu'):
        if aggr == 'sum':
            aggr = 'add'
        super(GGNNConv, self).__init__(aggr=aggr)

        self.node_state_dim = node_state_dim
        self.num_edge_feats = num_edge_feats
        self.edge_hidden_sizes = edge_hidden_sizes
        self.message_net = MLP(input_dim=self.node_state_dim * 2 + num_edge_feats, 
                               hidden_dims=self.edge_hidden_sizes,
                               nonlinearity=act_func,
                               act_last=act_func)
        self.cell = nn.GRUCell(self.node_state_dim, 
                                self.node_state_dim)

    def forward(self, x, edge_index, edge_features):
        prop_out = self.propagate(edge_index, x=x, edge_features=edge_features)
        new_states = self.cell(prop_out, x)
        return new_states

    def message(self, x_i, x_j, edge_features):
        if edge_features is None:
            edge_inputs = torch.cat((x_i, x_j), dim=-1)
        else:
            edge_inputs = torch.cat((x_i, x_j, edge_features), dim=-1)
        return self.message_net(edge_inputs)


class GGNN(GNNEmbedding):
    def __init__(self, node_state_dim, output_dims, edge_hidden_sizes, 
                 num_node_feats, num_edge_feats, max_lv=3, msg_aggregate_type='sum',
                 readout_agg='sum', share_params=False, act_func='elu', out_method='last', dropout=None):
        if output_dims is None:
            embed_dim = node_state_dim
        else:
            if isinstance(output_dims, str):
                embed_dim = int(output_dims.split('-')[-1])
            else:
                embed_dim = output_dims[-1]
        super(GGNN, self).__init__(embed_dim, dropout)
        self.out_method = out_method
        self.node_state_dim = node_state_dim
        if isinstance(edge_hidden_sizes, str):
            edge_hidden_sizes += '-%d' % node_state_dim
        else:
            edge_hidden_sizes += [node_state_dim]
        lm_layer = lambda: GGNNConv(node_state_dim, num_edge_feats, edge_hidden_sizes,
                                    aggr=msg_aggregate_type, act_func=act_func)
        if share_params:
            self.ggnn_layer = lm_layer()
            self.layers = [lambda x: self.ggnn_layer(x)] * max_lv
        else:
            self.layers = [lm_layer() for _ in range(max_lv)]
            self.layers = nn.ModuleList(self.layers)
        self.max_lv = max_lv
        self.node2hidden = nn.Linear(num_node_feats, node_state_dim)
        self.readout_agg = get_agg(readout_agg)

        self.readout_funcs = []
        if output_dims is None:
            for i in range(self.max_lv + 1):
                self.readout_funcs.append(lambda x: x)
        else:
            for i in range(self.max_lv + 1):
                mlp = MLP(input_dim=node_state_dim, 
                          hidden_dims=output_dims, 
                          nonlinearity=act_func,
                          act_last=act_func)
                self.readout_funcs.append(mlp)
                if self.out_method == 'last':
                    break
            self.readout_funcs = nn.ModuleList(self.readout_funcs)
        if self.out_method == 'gru':
            self.final_cell = nn.GRUCell(self.embed_dim, self.embed_dim)
 
    def get_feat(self, graph_list):
        node_feat, edge_feat, edge_from_idx, edge_to_idx, g_idx = prepare_gnn(graph_list, self.is_cuda())
        edge_index = [edge_from_idx, edge_to_idx]
        edge_index = torch.stack(edge_index)
        node_states = self.node2hidden(node_feat)
        init_embed = self.readout_funcs[-1](node_states)
        outs = self.readout_agg(init_embed, g_idx, dim=0, dim_size=len(graph_list))
        for i in range(self.max_lv):
            layer = self.layers[i]
            new_states = layer(node_states, edge_index, edge_feat)
            node_states = new_states

            if self.out_method == 'last':
                continue

            out_states = self.readout_funcs[i](node_states)

            graph_embed = self.readout_agg(out_states, g_idx,
                                           dim=0, dim_size=len(graph_list))
            if self.out_method == 'gru':
                outs = self.final_cell(graph_embed, outs)
            else:
                outs += graph_embed

        if self.out_method == 'last':
            out_states = self.readout_funcs[0](node_states)

            graph_embed = self.readout_agg(out_states, g_idx,
                                           dim=0, dim_size=len(graph_list))
            return graph_embed, (g_idx, out_states)
        else:
            if self.out_method == 'mean':
                outs /= self.max_lv + 1
            return outs, None

