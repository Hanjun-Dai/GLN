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


def process_smiles():
    all_symbols = set()

    smiles_cano_map = {}
    for rxn in tqdm(rxn_smiles):
        reactants, _, prod = rxn.split('>')
        mols = reactants.split('.') + [prod]
        for sm in mols:
            m, cano_sm = cano_smiles(sm)
            if m is not None:
                for a in m.GetAtoms():
                    all_symbols.add((a.GetAtomicNum(), a.GetSymbol()))
            if sm in smiles_cano_map:
                assert smiles_cano_map[sm] == cano_sm
            else:
                smiles_cano_map[sm] = cano_sm
    print('num of smiles', len(smiles_cano_map))
    set_mols = set()
    for s in smiles_cano_map:
        set_mols.add(smiles_cano_map[s])
    print('# unique smiles', len(set_mols))    
    with open(os.path.join(cmd_args.save_dir, 'cano_smiles.pkl'), 'wb') as f:
        cp.dump(smiles_cano_map, f, cp.HIGHEST_PROTOCOL)
    print('# unique atoms:', len(all_symbols))
    all_symbols = sorted(list(all_symbols))
    with open(os.path.join(cmd_args.save_dir, 'atom_list.txt'), 'w') as f:
        for a in all_symbols:
            f.write('%d\n' % a[0])


if __name__ == '__main__':

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

    process_smiles()
    
