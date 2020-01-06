from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from gln.mods.mol_gnn.gnn_family.utils import GNNEmbedding
from gln.mods.mol_gnn.torch_util import MLP, NONLINEARITIES
from torch_scatter import scatter_add
from torch_sparse import spmm


class MorganFp(GNNEmbedding):
    def __init__(self, feat_dim, hidden_size, num_hidden, feat_mode='dense', act_func='elu', dropout=0):
        super(MorganFp, self).__init__(hidden_size, dropout)
        self.feat_mode = feat_mode
        self.feat_dim = feat_dim
        if self.feat_mode == 'dense':
            self.mlp = MLP(input_dim=feat_dim,
                           hidden_dims=[hidden_size] * num_hidden,
                           nonlinearity=act_func,
                           dropout=dropout,
                           act_last=act_func)
        else:
            self.input_linear = nn.Linear(feat_dim, hidden_size)
            if num_hidden > 1:
                self.mlp = MLP(input_dim=hidden_size,
                               hidden_dims=[hidden_size] * (num_hidden - 1),
                               nonlinearity=act_func,
                               dropout=dropout,
                               act_last=act_func)
            else:
                self.mlp = lambda x: x

    def get_fp(self, graph_list):
        feat_indices = []
        row_indices = []
        for i, mol in enumerate(graph_list):
            feat = [t % self.feat_dim for t in mol.fingerprints]
            row_indices += [i] * len(feat)
            feat_indices += feat
        assert len(row_indices) == len(feat_indices)
        sp_indices = torch.LongTensor([row_indices, feat_indices])
        vals = torch.ones(len(row_indices), dtype=torch.float32)

        if self.is_cuda():
            sp_indices = sp_indices.cuda()
            vals = vals.cuda()
        
        if self.feat_mode == 'dense':
            sp_feat = torch.sparse.FloatTensor(sp_indices, vals, torch.Size([len(graph_list), self.feat_dim]))
            sp_feat = sp_feat.to_dense()            
            return sp_feat
        else:
            return sp_indices, vals

    def get_feat(self, graph_list):        
        if self.feat_mode == 'dense':
            dense_feat = self.get_fp(graph_list)
        else:
            sp_indices, vals = self.get_fp(graph_list)
            w = self.input_linear.weight
            b = self.input_linear.bias
            dense_feat = spmm(sp_indices, vals, len(graph_list), w.transpose(0, 1)) + b

        return self.mlp(dense_feat), None
