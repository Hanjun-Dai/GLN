from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import rdkit
from rdkit import Chem
import os
import numpy as np
import torch
import pickle as cp
import math
from scipy.special import softmax
from gln.data_process.data_info import DataInfo, load_bin_feats
from gln.mods.mol_gnn.mol_utils import SmartsMols, SmilesMols
from gln.common.reactor import Reactor
from gln.graph_logic.logic_net import GraphPath


class RetroGLN(object):
    def __init__(self, dropbox, model_dump):
        """
        Args:
            dropbox: the dropbox folder
            model_dump: the ckpt folder, which contains model dump and model args
        """
        assert os.path.isdir(model_dump)

        arg_file = os.path.join(model_dump, 'args.pkl')
        with open(arg_file, 'rb') as f:
            self.args = cp.load(f)
            self.args.dropbox = dropbox

        DataInfo.init(dropbox, self.args)
        load_bin_feats(dropbox, self.args)

        model_file = os.path.join(model_dump, 'model.dump')
        self.gln = GraphPath(self.args)
        self.gln.load_state_dict(torch.load(model_file))
        self.gln.cuda()
        self.gln.eval()

        self.prod_center_maps = {}
        self.cached_smarts = None

    def _ordered_tpls(self, cano_prod, beam_size, rxn_type):
        if (rxn_type, cano_prod) not in self.prod_center_maps:
            mol = Chem.MolFromSmiles(cano_prod)
            if mol is None:
                return None
            if self.cached_smarts is None:
                self.cached_smarts = []
                print('caching smarts centers')
                for sm in DataInfo.prod_cano_smarts:
                    self.cached_smarts.append(Chem.MolFromSmarts(sm))

            prod_center_cand_idx = []
            for i, sm in enumerate(self.cached_smarts):
                if sm is not None and mol.HasSubstructMatch(sm):
                    prod_center_cand_idx.append(i)
            self.prod_center_maps[(rxn_type, cano_prod)] = prod_center_cand_idx
        prod_center_cand_idx = self.prod_center_maps[(rxn_type, cano_prod)]

        # infer the reaction center
        if not len(prod_center_cand_idx):
            return None
        prod_center_mols = [SmartsMols.get_mol_graph(DataInfo.prod_cano_smarts[m]) for m in prod_center_cand_idx]
        prod_mol = SmilesMols.get_mol_graph(cano_prod)
        prod_center_scores = self.gln.prod_center_predicate.inference([prod_mol], [prod_center_mols])
        prod_center_scores = prod_center_scores.view(-1).data.cpu().numpy()
        top_centers = np.argsort(-1 * prod_center_scores)[:beam_size]
        top_center_scores = [prod_center_scores[i] for i in top_centers]
        top_center_mols = [prod_center_mols[i] for i in top_centers]
        top_center_smarts = [DataInfo.prod_cano_smarts[prod_center_cand_idx[i]] for i in top_centers]

        # infer the template
        list_of_list_tpls = []
        for i, c in enumerate(top_center_smarts):
            assert c in DataInfo.unique_tpl_of_prod_center
            if not rxn_type in DataInfo.unique_tpl_of_prod_center[c]:
                continue
            tpl_indices = DataInfo.unique_tpl_of_prod_center[c][rxn_type]
            tpls = [DataInfo.unique_templates[t][1] for t in tpl_indices]
            list_of_list_tpls.append(tpls)
        if not len(list_of_list_tpls):
            return None
        tpl_scores = self.gln.tpl_fwd_predicate.inference([prod_mol] * len(top_center_mols), list_of_list_tpls)
        tpl_scores = tpl_scores.view(-1).data.cpu().numpy()

        idx = 0
        tpl_with_scores = []
        for i, c in enumerate(top_center_scores):
            for tpl in list_of_list_tpls[i]:
                t_score = tpl_scores[idx]
                tot_score = c + t_score
                tpl_with_scores.append((tot_score, tpl))
                idx += 1
        tpl_with_scores = sorted(tpl_with_scores, key=lambda x: -1 * x[0])

        return tpl_with_scores

    def run(self, raw_prod, beam_size, topk, rxn_type='UNK'):
        """
        Args:
            raw_prod: the single product smiles
            beam_size: the size for beam search
            topk: top-k prediction of reactants
            rxn_type: (optional) reaction type
        Return:
            a dictionary with the following keys:
            {
                'reactants': the top-k prediction of reactants
                'template': the list of corresponding reaction templates used
                'scores': the scores for the corresponding predictions, in descending order
            }
            if no valid reactions are found, None will be returned
        """
        cano_prod = DataInfo.get_cano_smiles(raw_prod)
        prod_mol = SmilesMols.get_mol_graph(cano_prod)
        tpl_with_scores = self._ordered_tpls(cano_prod, beam_size, rxn_type)

        if tpl_with_scores is None:
            return None
        # filter out invalid tpls
        list_of_list_reacts = []
        list_reacts = []
        list_tpls = []
        num_tpls = 0
        num_reacts = 0
        for prod_tpl_score, tpl in tpl_with_scores:
            pred_mols = Reactor.run_reaction(raw_prod, tpl)
            if pred_mols is not None and len(pred_mols):
                num_tpls += 1
                list_of_list_reacts.append(pred_mols)
                num_reacts += len(pred_mols)
                list_tpls.append((prod_tpl_score, tpl))
                if num_tpls >= beam_size:
                    break

        list_rxns = []
        for i in range(len(list_of_list_reacts)):
            list_rxns.append([DataInfo.get_cano_smiles(r) + '>>' + cano_prod for r in list_of_list_reacts[i]])
        if len(list_rxns) and len(list_tpls):
            react_scores = self.gln.reaction_predicate.inference([prod_mol] * len(list_tpls), list_rxns)
            react_scores = react_scores.view(-1).data.cpu().numpy()

            idx = 0
            final_joint = []
            for i, (prod_tpl_score, tpl) in enumerate(list_tpls):
                for reacts in list_of_list_reacts[i]:
                    r_score = react_scores[idx]
                    tot_score = prod_tpl_score + r_score
                    final_joint.append((tot_score, tpl, reacts))
                    idx += 1
            final_joint = sorted(final_joint, key=lambda x: -1 * x[0])[:topk]
            scores = [t[0] for t in final_joint]
            scores = softmax([each_score for each_score in scores])
            list_reacts = [t[2] for t in final_joint]
            ret_tpls = [t[1] for t in final_joint]
            result = {'template': ret_tpls,
                    'reactants': list_reacts,
                    'scores': scores}
        else:
            result = {'template': [],
                        'reactants': [],
                        'scores': []}
        return result
