from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import rdkit
from rdkit import Chem
import csv
import os
from tqdm import tqdm
import pickle as cp
from collections import defaultdict
from gln.common.cmd_args import cmd_args
from gln.common.mol_utils import cano_smarts, cano_smiles, smarts_has_useless_parentheses


def process_centers():
    prod_cano_smarts = set()
    react_cano_smarts = set()

    smarts_cano_map = {}
    pbar = tqdm(retro_templates)
    for template in pbar:
        sm_prod, _, sm_react = template.split('>')
        if smarts_has_useless_parentheses(sm_prod):
            sm_prod = sm_prod[1:-1]
        
        smarts_cano_map[sm_prod] = cano_smarts(sm_prod)[1]
        prod_cano_smarts.add(smarts_cano_map[sm_prod])
        
        for r_smarts in sm_react.split('.'):            
            smarts_cano_map[r_smarts] = cano_smarts(r_smarts)[1]
            react_cano_smarts.add(smarts_cano_map[r_smarts])
        pbar.set_description('# prod centers: %d, # react centers: %d' % (len(prod_cano_smarts), len(react_cano_smarts)))
    print('# prod centers: %d, # react centers: %d' % (len(prod_cano_smarts), len(react_cano_smarts)))

    with open(os.path.join(cmd_args.save_dir, 'prod_cano_smarts.txt'), 'w') as f:
        for s in prod_cano_smarts:
            f.write('%s\n' % s)
    with open(os.path.join(cmd_args.save_dir, 'react_cano_smarts.txt'), 'w') as f:
        for s in react_cano_smarts:
            f.write('%s\n' % s)
    with open(os.path.join(cmd_args.save_dir, 'cano_smarts.pkl'), 'wb') as f:
        cp.dump(smarts_cano_map, f, cp.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    tpl_file = os.path.join(cmd_args.save_dir, 'templates.csv')

    retro_templates = []
    
    with open(tpl_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in tqdm(reader):
            retro_templates.append(row[header.index('retro_templates')])

    raw_data_root = os.path.join(cmd_args.dropbox, cmd_args.data_name)
    rxn_smiles = []
    for phase in ['train', 'val', 'test']:
        csv_file = os.path.join(raw_data_root, 'raw_%s.csv' % phase)
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rxn_idx = header.index('reactants>reagents>production')
            for row in tqdm(reader):
                rxn_smiles.append(row[rxn_idx])

    process_centers()
