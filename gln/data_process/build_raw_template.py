from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import csv
import os
from tqdm import tqdm
import pickle as cp
import multiprocessing
from gln.common.cmd_args import cmd_args
from gln.mods.rdchiral.template_extractor import extract_from_reaction

def get_writer(fname, header):
    output_name = os.path.join(cmd_args.save_dir, fname)
    fout = open(output_name, 'w')
    writer = csv.writer(fout)
    writer.writerow(header)
    return fout, writer

def get_tpl(task):
    idx, row_idx, rxn_smiles = task
    react, reagent, prod = rxn_smiles.split('>')
    reaction = {'_id': row_idx, 'reactants': react, 'products': prod}
    template = extract_from_reaction(reaction)
    return idx, template

if __name__ == '__main__':
    fname = os.path.join(cmd_args.dropbox, cmd_args.data_name, 'raw_train.csv')
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]

    pool = multiprocessing.Pool(cmd_args.num_cores)
    tasks = []
    for idx, row in tqdm(enumerate(rows)):
        row_idx, _, rxn_smiles = row
        tasks.append((idx, row_idx, rxn_smiles))

    fout, writer = get_writer('proc_train_singleprod.csv', ['id', 'class', 'rxn_smiles', 'retro_templates'])
    fout_failed, failed_writer = get_writer('failed_template.csv', ['id', 'class', 'rxn_smiles', 'err_msg'])

    for result in tqdm(pool.imap_unordered(get_tpl, tasks), total=len(tasks)):
        idx, template = result
        row_idx, rxn_type, rxn_smiles = rows[idx]

        if 'reaction_smarts' in template:
            writer.writerow([row_idx, rxn_type, rxn_smiles, template['reaction_smarts']])            
            fout.flush()
        else:
            failed_writer.writerow([row_idx, rxn_type, rxn_smiles, template['err_msg']])
            fout_failed.flush()

    fout.close()
    fout_failed.close()
