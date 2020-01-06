from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import rdkit
from rdkit import Chem
import csv
import sys
import os
from tqdm import tqdm
import pickle as cp
from collections import defaultdict
from gln.common.cmd_args import cmd_args
from gln.mods.mol_gnn.mol_utils import SmartsMols, SmilesMols


if __name__ == '__main__':
    file_root = os.path.join(cmd_args.dropbox, 'cooked_' + cmd_args.data_name, 'tpl-%s' % cmd_args.tpl_name)
    if cmd_args.fp_degree > 0:
        SmilesMols.set_fp_degree(cmd_args.fp_degree)
        SmartsMols.set_fp_degree(cmd_args.fp_degree)

    if cmd_args.retro_during_train:
        part_folder = os.path.join(file_root, 'np-%d' % cmd_args.num_parts)
        if cmd_args.part_num > 0:
            prange = range(cmd_args.part_id, cmd_args.part_id + cmd_args.part_num)
        else:
            prange = range(cmd_args.num_parts)
        for pid in prange:
            with open(os.path.join(part_folder, 'neg_reacts-part-%d.csv' % pid), 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in tqdm(reader):
                    reacts = row[-1]
                    for t in reacts.split('.'):
                        SmilesMols.get_mol_graph(t)
                    SmilesMols.get_mol_graph(reacts)
            SmilesMols.save_dump(os.path.join(part_folder, 'neg_graphs-part-%d' % pid))
            SmilesMols.clear()
        sys.exit()

    with open(os.path.join(file_root, '../cano_smiles.pkl'), 'rb') as f:
        smiles_cano_map = cp.load(f)

    with open(os.path.join(file_root, 'prod_cano_smarts.txt'), 'r') as f:
        prod_cano_smarts = [row.strip() for row in f.readlines()]

    with open(os.path.join(file_root, 'react_cano_smarts.txt'), 'r') as f:
        react_cano_smarts = [row.strip() for row in f.readlines()]


    for mol in tqdm(smiles_cano_map):
        SmilesMols.get_mol_graph(smiles_cano_map[mol])
    SmilesMols.save_dump(os.path.join(cmd_args.save_dir, '../graph_smiles'))

    for smarts in tqdm(prod_cano_smarts + react_cano_smarts):
        SmartsMols.get_mol_graph(smarts)
    SmartsMols.save_dump(cmd_args.save_dir + '/graph_smarts')
