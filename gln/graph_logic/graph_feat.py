from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import os
import rdkit
from rdkit import Chem
import csv
from gln.mods.mol_gnn.torch_util import MLP
from gln.mods.mol_gnn.gnn_family.utils import get_agg
from gln.mods.mol_gnn.mol_utils import SmartsMols, SmilesMols
from gln.common.consts import DEVICE
from gln.data_process.data_info import DataInfo
from gln.mods.mol_gnn.mg_clib import NUM_NODE_FEATS, NUM_EDGE_FEATS

from gln.graph_logic import get_gnn

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TempFeaturizer(nn.Module):
    def __init__(self, args):
        super(TempFeaturizer, self).__init__()


class DeepsetTempFeaturizer(TempFeaturizer):
    def __init__(self, args):
        super(DeepsetTempFeaturizer, self).__init__(args)
        self.prod_gnn = get_gnn(args)
        self.react_gnn = get_gnn(args)
        self.reactants_agg = get_agg('sum')

        self.readout = MLP(args.embed_dim * 2, [args.mlp_hidden, args.embed_dim], nonlinearity='relu')

    def forward(self, template_list):
        list_prod = []
        list_rxtants = []
        rxtant_indices = []
        for i, temp in enumerate(template_list):
            prod, _, reacts = temp.split('>')
            prod = DataInfo.get_cano_smarts(prod)
            list_prod.append(SmartsMols.get_mol_graph(prod))
            reacts = reacts.split('.')
            for r in reacts:
                r = DataInfo.get_cano_smarts(r)
                list_rxtants.append(SmartsMols.get_mol_graph(r))

            rxtant_indices += [i] * len(reacts)

        prods, _ = self.prod_gnn(list_prod)
        rxtants, _ = self.react_gnn(list_rxtants)
        rxtants = F.relu(rxtants)
        
        rxtant_indices = torch.LongTensor(rxtant_indices).to(DEVICE)
        rxtants = self.reactants_agg(rxtants, rxtant_indices.view(-1, 1).expand(-1, rxtants.shape[1]),
                                    dim=0, dim_size=len(template_list))

        feats = torch.cat((prods, rxtants), dim=1)
        out = self.readout(feats)
        return out


class DeepsetReactFeaturizer(nn.Module):

    def __init__(self, args):
        super(DeepsetReactFeaturizer, self).__init__()

        self.react_gnn = get_gnn(args)
        self.reactants_agg = get_agg('sum')

    def forward(self, reaction_list):
        list_prod = []
        list_cata = []
        list_rxtants = []
        rxtant_indices = []
        for i, react in enumerate(reaction_list):
            reactants, cata, prod = react.split('>')
            list_prod.append(SmilesMols.get_mol_graph(prod))
            list_cata.append(SmilesMols.get_mol_graph(cata))
            reactants = reactants.split('.')
            for r in reactants:
                list_rxtants.append(SmilesMols.get_mol_graph(r))

            rxtant_indices += [i] * len(reactants)
    
        rxtants, _ = self.react_gnn(list_rxtants)

        rxtant_indices = torch.LongTensor(rxtant_indices).to(DEVICE)
        rxtants = self.reactants_agg(rxtants, rxtant_indices.view(-1, 1).expand(-1, rxtants.shape[1]),
                                    dim=0, dim_size=len(reaction_list))
        return rxtants
