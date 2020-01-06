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
import multiprocessing


def find_edges(task):
    idx, rxn_type, smiles = task
    smiles = smiles_cano_map[smiles]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return idx, rxn_type, smiles, None
    list_centers = []
    for i, (sm_center, center_mol) in enumerate(prod_center_mols):
        if center_mol is None:
            continue
        if not rxn_type in smarts_type_set[sm_center]:
            continue
        if mol.HasSubstructMatch(center_mol):
            list_centers.append(str(i))
    if len(list_centers) == 0:
        return idx, rxn_type, smiles, None
    centers = ' '.join(list_centers)
    return idx, rxn_type, smiles, centers


if __name__ == '__main__':
    with open(os.path.join(cmd_args.save_dir, '../cano_smiles.pkl'), 'rb') as f:
        smiles_cano_map = cp.load(f)

    with open(os.path.join(cmd_args.save_dir, 'cano_smarts.pkl'), 'rb') as f:
        smarts_cano_map = cp.load(f)

    with open(os.path.join(cmd_args.save_dir, 'prod_cano_smarts.txt'), 'r') as f:
        prod_cano_smarts = [row.strip() for row in f.readlines()]

    prod_center_mols = []
    for sm in tqdm(prod_cano_smarts):
        prod_center_mols.append((sm, Chem.MolFromSmarts(sm)))

    print('num of prod centers', len(prod_center_mols))
    print('num of smiles', len(smiles_cano_map))

    csv_file = os.path.join(cmd_args.save_dir, 'templates.csv')

    smarts_type_set = defaultdict(set)
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        tpl_idx = header.index('retro_templates')
        type_idx = header.index('class')
        for row in reader:
            rxn_type = row[type_idx]
            template = row[tpl_idx]
            sm_prod, _, _ = template.split('>')
            if smarts_has_useless_parentheses(sm_prod):
                sm_prod = sm_prod[1:-1]            
            sm_prod = smarts_cano_map[sm_prod]
            smarts_type_set[sm_prod].add(rxn_type)

    if cmd_args.num_parts <= 0:
        num_parts = cmd_args.num_cores
    else:
        num_parts = cmd_args.num_parts

    pool = multiprocessing.Pool(cmd_args.num_cores)

    raw_data_root = os.path.join(cmd_args.dropbox, cmd_args.data_name)
    for out_phase in ['train', 'val', 'test']:
        csv_file = os.path.join(raw_data_root, 'raw_%s.csv' % out_phase)        

        rxn_smiles = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            rxn_idx = header.index('reactants>reagents>production')
            type_idx = header.index('class')            
            for row in tqdm(reader):
                rxn_smiles.append((row[type_idx], row[rxn_idx]))

        part_size = min(len(rxn_smiles) // num_parts + 1, len(rxn_smiles))

        for pid in range(num_parts):        
            idx_range = range(pid * part_size, min((pid + 1) * part_size, len(rxn_smiles)))

            local_results = [None] * len(idx_range)

            tasks = []            
            for i, idx in enumerate(idx_range):
                rxn_type, rxn = rxn_smiles[idx]
                reactants, _, prod = rxn.split('>')
                tasks.append((i, rxn_type, prod))                
            for result in tqdm(pool.imap_unordered(find_edges, tasks), total=len(tasks)):
                i, rxn_type, smiles, centers = result
                local_results[i] = (rxn_type, smiles, centers)
            out_folder = os.path.join(cmd_args.save_dir, 'np-%d' % num_parts)
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)
            fout = open(os.path.join(out_folder, '%s-prod_center_maps-part-%d.csv' % (out_phase, pid)), 'w')
            writer = csv.writer(fout)
            writer.writerow(['smiles', 'class', 'centers'])

            for i in range(len(local_results)):
                rxn_type, smiles, centers = local_results[i]
                if centers is not None:
                    writer.writerow([smiles, rxn_type, centers])
            fout.close()
