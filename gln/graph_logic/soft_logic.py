from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import os
import rdkit
from rdkit import Chem
import csv
from gln.mods.mol_gnn.mol_utils import SmartsMols
from gln.mods.mol_gnn.torch_util import MLP, glorot_uniform

from gln.common.consts import DEVICE
from gln.data_process.data_info import DataInfo
from torch_scatter import scatter_max, scatter_add, scatter_mean
from gln.mods.torchext import jagged_log_softmax
from gln.graph_logic import get_gnn
from gln.graph_logic.graph_feat import DeepsetTempFeaturizer, DeepsetReactFeaturizer

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def jagged_forward(list_graph, list_of_list_cand, graph_enc, cand_enc, att_func, list_target_pos=None, normalize=True):
    graph_embed = graph_enc(list_graph)

    flat_cands = []
    rep_indices = []
    prefix_sum = []
    offset = 0
    for i, l in enumerate(list_of_list_cand):
        for c in l:
            flat_cands.append(c)
        rep_indices += [i] * len(l)
        offset += len(l)
        prefix_sum.append(offset)
    
    cand_embed = cand_enc(flat_cands)
    rep_indices = torch.LongTensor(rep_indices).to(DEVICE)
    prefix_sum = torch.LongTensor(prefix_sum).to(DEVICE)

    graph_embed = torch.gather(graph_embed, 0, rep_indices.view(-1, 1).expand(-1, graph_embed.shape[1]))
    
    logits = att_func(graph_embed, cand_embed)
    if normalize:
        log_prob = jagged_log_softmax(logits, prefix_sum)
    else:
        log_prob = logits
    
    if list_target_pos is None:
        return log_prob

    offset = 0
    target_pos = []
    for i, l in enumerate(list_of_list_cand):
        idx = list_target_pos[i]
        target_pos.append(offset + idx)
        offset += len(l)
    target_pos = torch.LongTensor(target_pos).to(DEVICE)

    return log_prob[target_pos]


class ReactionProbCalc(nn.Module):
    def __init__(self, args):    
        super(ReactionProbCalc, self).__init__()

        self.prod_enc = get_gnn(args)
        self.react_enc = DeepsetReactFeaturizer(args)
        if args.att_type == 'inner_prod':
            self.att_func = lambda x, y: torch.sum(x * y, dim=1).view(-1)
        elif args.att_type == 'mlp':
            self.pred = MLP(2 * args.embed_dim, [args.mlp_hidden, 1], nonlinearity='relu')
            self.att_func = lambda x, y: self.pred(torch.cat((x, y), dim=1)).view(-1)
        elif args.att_type == 'bilinear':
            self.bilin = nn.Bilinear(args.embed_dim, args.embed_dim, 1)
            self.att_func = lambda x, y: self.bilin(x, y).view(-1)
        else:
            raise NotImplementedError

    def forward(self, list_mols, list_of_list_reactions, list_target_pos=None):
        if list_target_pos is None:
            list_target_pos = [0] * len(list_mols)

        log_prob = jagged_forward(list_mols, list_of_list_reactions, 
                                  graph_enc=lambda x: self.prod_enc(x)[0],
                                  cand_enc=lambda x: self.react_enc(x),
                                  att_func=self.att_func,
                                  list_target_pos=list_target_pos)
        return log_prob

    def inference(self, list_mols, list_of_list_reactions):
        logits = jagged_forward(list_mols, list_of_list_reactions, 
                                graph_enc=lambda x: self.prod_enc(x)[0],
                                cand_enc=lambda x: self.react_enc(x),
                                att_func=self.att_func,
                                list_target_pos=None,
                                normalize=False)
        return logits


class OnehotEmbedder(nn.Module):
    def __init__(self, list_keys, fn_getkey, embed_size):
        super(OnehotEmbedder, self).__init__()
        self.key_idx = {}
        for i, key in enumerate(list_keys):
            self.key_idx[key] = i
        self.embed_size = embed_size
        self.fn_getkey = fn_getkey
        self.embedding = nn.Embedding(len(list_keys) + 1, embed_size)
        glorot_uniform(self)

    def forward(self, list_objs):
        indices = []
        for obj in list_objs:
            key = self.fn_getkey(obj)
            if key is None:
                indices.append(len(self.key_idx))
            else:
                indices.append(self.key_idx[key])
        indices = torch.LongTensor(indices).to(DEVICE)
        return self.embedding(indices)


class ActiveProbCalc(nn.Module):
    def __init__(self, args):
        super(ActiveProbCalc, self).__init__()
        self.prod_enc = get_gnn(args)
        if args.tpl_enc == 'deepset':
            self.tpl_enc = DeepsetTempFeaturizer(args)
        elif args.tpl_enc == 'onehot':
            self.tpl_enc = OnehotEmbedder(list_keys=DataInfo.unique_templates,
                                          fn_getkey=lambda x: x,
                                          embed_size=args.embed_dim)        
        else:
            raise NotImplementedError
        if args.att_type == 'inner_prod':
            self.att_func = lambda x, y: torch.sum(x * y, dim=1).view(-1)
        elif args.att_type == 'mlp':
            self.pred = MLP(2 * args.embed_dim, [args.mlp_hidden, 1], nonlinearity='relu')
            self.att_func = lambda x, y: self.pred(torch.cat((x, y), dim=1)).view(-1)
        elif args.att_type == 'bilinear':
            self.bilin = nn.Bilinear(args.embed_dim, args.embed_dim, 1)
            self.att_func = lambda x, y: self.bilin(x, y).view(-1)
        else:
            raise NotImplementedError

    def forward(self, list_mols, list_of_list_templates, list_target_pos=None):
        if list_target_pos is None:
            list_target_pos = [0] * len(list_mols)
        log_prob = jagged_forward(list_mols, list_of_list_templates, 
                                  graph_enc=lambda x: self.prod_enc(x)[0],
                                  cand_enc=lambda x: self.tpl_enc(x),
                                  att_func=self.att_func, 
                                  list_target_pos=list_target_pos)

        return log_prob

    def inference(self, list_mols, list_of_list_templates):
        logits = jagged_forward(list_mols, list_of_list_templates, 
                                graph_enc=lambda x: self.prod_enc(x)[0],
                                cand_enc=lambda x: self.tpl_enc(x),
                                att_func=self.att_func,
                                list_target_pos=None,
                                normalize=False)
        return logits


class CenterProbCalc(nn.Module):
    def __init__(self, args):
        super(CenterProbCalc, self).__init__()
        self.prod_enc = get_gnn(args)
        if args.subg_enc == 'onehot':
            self.prod_center_enc = OnehotEmbedder(list_keys=DataInfo.prod_cano_smarts, 
                                                  fn_getkey=lambda m: m.name if m is not None else None,
                                                  embed_size=args.embed_dim)
            self.prod_embed_func = lambda x: self.prod_center_enc(x)
        else:
            self.prod_center_enc = get_gnn(args, gm=args.subg_enc)
            self.prod_embed_func = lambda x: self.prod_center_enc(x)[0]
        if args.att_type == 'inner_prod':
            self.att_func = lambda x, y: torch.sum(x * y, dim=1).view(-1)
        elif args.att_type == 'mlp':
            self.pred = MLP(2 * args.embed_dim, [args.mlp_hidden, 1], nonlinearity=args.act_func)
            self.att_func = lambda x, y: self.pred(torch.cat((x, y), dim=1)).view(-1)
        elif args.att_type == 'bilinear':
            self.bilin = nn.Bilinear(args.embed_dim, args.embed_dim, 1)
            self.att_func = lambda x, y: self.bilin(x, y).view(-1)
        else:
            raise NotImplementedError

    def forward(self, list_mols, list_of_list_centers, list_target_pos=None):
        if list_target_pos is None:
            list_target_pos = [0] * len(list_mols)        
        log_prob = jagged_forward(list_mols, list_of_list_centers, 
                                  graph_enc=lambda x: self.prod_enc(x)[0],
                                  cand_enc=lambda x: self.prod_embed_func(x),
                                  att_func=self.att_func,
                                  list_target_pos=list_target_pos)

        return log_prob

    def inference(self, list_mols, list_of_list_centers):
        logits = jagged_forward(list_mols, list_of_list_centers, 
                                graph_enc=lambda x: self.prod_enc(x)[0],
                                cand_enc=lambda x: self.prod_embed_func(x),
                                att_func=self.att_func,
                                list_target_pos=None,
                                normalize=False)
        return logits
