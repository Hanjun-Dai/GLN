from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import os
import rdkit
from rdkit import Chem
import random
import csv
import sys
from itertools import chain
from collections import defaultdict
from gln.common.cmd_args import cmd_args
from gln.data_process.data_info import DataInfo, load_train_reactions
from tqdm import tqdm
from gln.common.reactor import Reactor
from collections import Counter

import multiprocessing
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')


def find_tpls(cur_task):
    idx, (rxn_type, rxn) = cur_task
    reactants, _, raw_prod = rxn.split('>')

    prod = DataInfo.get_cano_smiles(raw_prod)

    if not (rxn_type, prod) in DataInfo.prod_center_maps:
        return None
    reactants = DataInfo.get_cano_smiles(reactants)
    prod_center_cand_idx = DataInfo.prod_center_maps[(rxn_type, prod)]
    
    neg_reactants = set()
    pos_tpl_idx = {}
    tot_tpls = 0
    for center_idx in prod_center_cand_idx:
        c = DataInfo.prod_cano_smarts[center_idx]
        assert c in DataInfo.unique_tpl_of_prod_center

        tpl_indices = DataInfo.unique_tpl_of_prod_center[c][rxn_type]
        tot_tpls += len(tpl_indices)
        for tpl_idx in tpl_indices:
            cur_t, tpl = DataInfo.unique_templates[tpl_idx]
            assert cur_t == rxn_type
            pred_mols = Reactor.run_reaction(prod, tpl)
            if pred_mols is None or len(pred_mols) == 0:
                continue            
            for pred in pred_mols:
                if pred != reactants:
                    neg_reactants.add(pred)
                else:
                    pos_tpl_idx[tpl_idx] = (len(tpl_indices), len(pred_mols))
    return (idx, pos_tpl_idx, neg_reactants)


def get_writer(fname, header):
    f = open(os.path.join(cmd_args.save_dir, 'np-%d' % cmd_args.num_parts, fname), 'w')
    writer = csv.writer(f)
    writer.writerow(header) 
    return f, writer


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    DataInfo.init(cmd_args.dropbox, cmd_args)
    
    fn_pos = lambda idx: get_writer('pos_tpls-part-%d.csv' % idx, ['tpl_idx', 'pos_tpl_idx', 'num_tpl_compete', 'num_react_compete'])
    fn_neg = lambda idx: get_writer('neg_reacts-part-%d.csv' % idx, ['sample_idx', 'neg_reactants'])

    if cmd_args.num_parts <= 0:
        num_parts = cmd_args.num_cores
        DataInfo.load_cooked_part('train', load_graphs=False)
    else:
        num_parts = cmd_args.num_parts

    train_reactions = load_train_reactions(cmd_args)
    n_train = len(train_reactions)
    part_size = n_train // num_parts + 1

    if cmd_args.part_num > 0:
        prange = range(cmd_args.part_id, cmd_args.part_id + cmd_args.part_num)
    else:
        prange = range(num_parts)
    for pid in prange:
        f_pos, writer_pos = fn_pos(pid)
        f_neg, writer_neg = fn_neg(pid)
        if cmd_args.num_parts > 0:
            DataInfo.load_cooked_part('train', part=pid, load_graphs=False)
        part_tasks = []
        idx_range = list(range(pid * part_size, min((pid + 1) * part_size, n_train)))
        for i in idx_range:
            part_tasks.append((i, train_reactions[i]))

        pool = multiprocessing.Pool(cmd_args.num_cores)
        for result in tqdm(pool.imap_unordered(find_tpls, part_tasks), total=len(idx_range)):
            if result is None:
                continue
            idx, pos_tpl_idx, neg_reactions = result
            idx = str(idx)
            neg_keys = neg_reactions
            
            if cmd_args.max_neg_reacts > 0:
                neg_keys = list(neg_keys)
                random.shuffle(neg_keys)
                neg_keys = neg_keys[:cmd_args.max_neg_reacts]
            for pred in neg_keys:
                writer_neg.writerow([idx, pred])
            for key in pos_tpl_idx:
                nt, np = pos_tpl_idx[key]
                writer_pos.writerow([idx, key, nt, np])
            f_pos.flush()
            f_neg.flush()
        f_pos.close()
        f_neg.close()
        pool.close()
        pool.join()
