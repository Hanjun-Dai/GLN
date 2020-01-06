from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import os
import rdkit
from rdkit import Chem
import csv
from gln.mods.mol_gnn.mol_utils import SmartsMols, SmilesMols
from gln.common.consts import DEVICE, t_float
from torch_scatter import scatter_max, scatter_add, scatter_mean
from gln.graph_logic.graph_feat import get_gnn
from gln.graph_logic.soft_logic import OnehotEmbedder, CenterProbCalc, ActiveProbCalc, ReactionProbCalc

import torch
from gln.mods.torchext import jagged_log_softmax

from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gln.data_process.data_info import DataInfo


class GraphPath(nn.Module):
    def __init__(self, args):
        super(GraphPath, self).__init__()

        # predict the template
        self.tpl_fwd_predicate = ActiveProbCalc(args)        
        # predict the center
        self.prod_center_predicate = CenterProbCalc(args)
        # predict the entire reaction
        self.reaction_predicate = ReactionProbCalc(args)
        self.retro_during_train = args.retro_during_train

    def forward(self, samples):
        prods = []
        list_of_list_centers = []
        list_of_list_tpls = []
        list_of_list_reacts = []

        for sample in samples:
            prods.append(SmilesMols.get_mol_graph(sample.prod))

            list_centers = [sample.center] + sample.neg_centers
            list_of_list_centers.append([SmartsMols.get_mol_graph(c) for c in list_centers])

            list_tpls = [sample.template] + sample.neg_tpls
            list_of_list_tpls.append(list_tpls)
            if self.retro_during_train:    
                list_reacts = [sample.reaction] + sample.neg_reactions
                list_of_list_reacts.append(list_reacts)

        center_log_prob = self.prod_center_predicate(prods, list_of_list_centers)
        tpl_log_prob = self.tpl_fwd_predicate(prods, list_of_list_tpls)

        loss = -torch.mean(center_log_prob) - torch.mean(tpl_log_prob)
        if self.retro_during_train:
            react_log_prob = self.reaction_predicate(prods, list_of_list_reacts)
            loss = loss - torch.mean(react_log_prob)

        return loss
