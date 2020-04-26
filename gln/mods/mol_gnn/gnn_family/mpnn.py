from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from gln.mods.mol_gnn.gnn_family.utils import GNNEmbedding, prepare_gnn
from torch_geometric.nn import NNConv, Set2Set
from gln.mods.mol_gnn.torch_util import MLP, NONLINEARITIES


class MPNN(GNNEmbedding):
    def __init__(self, latent_dim, output_dim, num_node_feats, num_edge_feats, max_lv=3, 
                act_func='elu', msg_aggregate_type='mean', dropout=None):
        if output_dim > 0:
            embed_dim = output_dim
        else:
            embed_dim = latent_dim            
        super(MPNN, self).__init__(embed_dim, dropout)
        if msg_aggregate_type == 'sum':
            msg_aggregate_type = 'add'
        self.max_lv = max_lv
        self.readout = nn.Linear(2 * latent_dim, self.embed_dim)
        self.lin0 = torch.nn.Linear(num_node_feats, latent_dim)
        net = MLP(input_dim=num_edge_feats, 
                  hidden_dims=[128, latent_dim * latent_dim],
                  nonlinearity=act_func)
        self.conv = NNConv(latent_dim, latent_dim, net, aggr=msg_aggregate_type, root_weight=False)

        self.act_func = NONLINEARITIES[act_func]
        self.gru = nn.GRU(latent_dim, latent_dim)
        self.set2set = Set2Set(latent_dim, processing_steps=3)

    def get_feat(self, graph_list):
        node_feat, edge_feat, edge_from_idx, edge_to_idx, g_idx = prepare_gnn(graph_list, self.is_cuda())
        out = self.act_func(self.lin0(node_feat))
        h = out.unsqueeze(0)
        edge_index = [edge_from_idx, edge_to_idx]
        edge_index = torch.stack(edge_index)
        for lv in range(self.max_lv):
            m = self.act_func(self.conv(out, edge_index, edge_feat))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        out = self.set2set(out, g_idx)
        out = self.readout(out)

        return out, None
