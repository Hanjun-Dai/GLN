from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import random
import csv
import os
import sys
import re
from tqdm import tqdm
from rdkit import Chem
import pickle as cp


def get_rxn_smiles(prod, reactants):
    prod_smi = Chem.MolToSmiles(prod, True)

    # Get rid of reactants when they don't contribute to this prod
    prod_maps = set(re.findall('\:([[0-9]+)\]', prod_smi))
    reactants_smi_list = []
    for mol in reactants:
        if mol is None:
            continue
        used = False
        for a in mol.GetAtoms():
            if a.HasProp('molAtomMapNumber'):
                if a.GetProp('molAtomMapNumber') in prod_maps:
                    used = True 
                else:
                    a.ClearProp('molAtomMapNumber')
        if used:
            reactants_smi_list.append(Chem.MolToSmiles(mol, True))

    reactants_smi = '.'.join(reactants_smi_list)
    return '{}>>{}'.format(reactants_smi, prod_smi)


if __name__ == '__main__':
    seed = 19260817
    np.random.seed(seed)
    random.seed(seed)
    fname = sys.argv[1]
    split_mode = 'multi' # single or multi

    pt = re.compile(r':(\d+)]')
    cnt = 0
    clean_list = []
    set_rxn = set()
    num_single = 0
    num_multi = 0
    bad_mapping = 0
    bad_prod = 0
    missing_map = 0
    raw_num = 0
    with open(fname, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        print(header)
        pbar = tqdm(reader)
        bad_rxn = 0
        for row in pbar:
            rxn_smiles = row[header.index('ReactionSmiles')]
            all_reactants, reagents, prods = rxn_smiles.split('>')
            all_reactants = all_reactants.split()[0]  # remove ' |f:1...'
            prods = prods.split()[0] # remove ' |f:1...'
            if '.' in prods:
                num_multi += 1
            else:
                num_single += 1
            if split_mode == 'single' and '.' in prods:  # multiple prods
                continue
            rids = ','.join(sorted(re.findall(pt, all_reactants)))
            pids = ','.join(sorted(re.findall(pt, prods)))
            if rids != pids:  # mapping is not 1:1
                bad_mapping += 1
                continue
            reactants = [Chem.MolFromSmiles(smi) for smi in all_reactants.split('.')]
            
            for sub_prod in prods.split('.'):
                mol_prod = Chem.MolFromSmiles(sub_prod)
                if mol_prod is None:  # rdkit is not able to parse the product
                    bad_prod += 1
                    continue
                # Make sure all have atom mapping
                if not all([a.HasProp('molAtomMapNumber') for a in mol_prod.GetAtoms()]):
                    missing_map += 1
                    continue
                
                raw_num += 1
                rxn_smiles = get_rxn_smiles(mol_prod, reactants)
                if not rxn_smiles in set_rxn:
                    clean_list.append((row[header.index('PatentNumber')], rxn_smiles))
                    set_rxn.add(rxn_smiles)
            pbar.set_description('select: %d, dup: %d' % (len(clean_list), raw_num))
    print('# clean', len(clean_list))
    print('single', num_single, 'multi', num_multi)
    print('bad mapping', bad_mapping)
    print('bad prod', bad_prod)
    print('missing map', missing_map)
    print('raw extracted', raw_num)
    
    random.shuffle(clean_list)

    num_val = num_test = int(len(clean_list) * 0.1)

    out_folder = '.'
    for phase in ['val', 'test', 'train']:
        fout = os.path.join(out_folder, 'raw_%s.csv' % phase)
        with open(fout, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'reactants>reagents>production'])

            if phase == 'val':
                r = range(num_val)
            elif phase == 'test':
                r = range(num_val, num_val + num_test)
            else:
                r = range(num_val + num_test, len(clean_list))
            for i in r:
                rxn_smiles = clean_list[i][1].split('>')
                result = []
                for r in rxn_smiles:
                    if len(r.strip()):
                        r = r.split()[0]
                    result.append(r)
                rxn_smiles = '>'.join(result)
                writer.writerow([clean_list[i][0], rxn_smiles])
