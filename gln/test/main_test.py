from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import os
import sys
import rdkit
from rdkit import Chem
import random
import csv
from gln.common.cmd_args import cmd_args
from gln.data_process.data_info import DataInfo, load_center_maps
from gln.test.model_inference import RetroGLN
from gln.common.evaluate import get_score, canonicalize

from tqdm import tqdm
import torch

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

import argparse
cmd_opt = argparse.ArgumentParser(description='Argparser for test only')
cmd_opt.add_argument('-model_for_test', default=None, help='model for test')
local_args, _ = cmd_opt.parse_known_args()


def load_raw_reacts(name):
    print('loading raw', name)
    args = cmd_args
    csv_file = os.path.join(args.dropbox, args.data_name, 'raw_%s.csv' % name) 
    reactions = []
    print('loading templates')
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)        
        for row in tqdm(reader):
            reactions.append((row[1], row[2]))
    print('num %s:' % name, len(reactions))
    return reactions


def rxn_data_gen(phase, model):
    list_reactions = load_raw_reacts(phase)

    eval_cnt = 0
    for pid in range(cmd_args.num_parts):        
        fname = os.path.join(cmd_args.dropbox, 'cooked_' + cmd_args.data_name, 'tpl-%s' % cmd_args.tpl_name, 'np-%d' % cmd_args.num_parts, '%s-prod_center_maps-part-%d.csv' % (phase, pid))
        model.prod_center_maps = load_center_maps(fname)

        tot_num = len(list_reactions)
        if cmd_args.num_parts > 1:
            part_size = tot_num // cmd_args.num_parts + 1
        else:
            part_size = tot_num
        indices = range(pid * part_size, min((pid + 1) * part_size, tot_num))

        for idx in indices:
            rxn_type, rxn = list_reactions[idx]
            _, _, raw_prod = rxn.split('>')
            eval_cnt += 1
            yield rxn_type, rxn, raw_prod
    assert eval_cnt == len(list_reactions)


def eval_model(phase, model, fname_pred):
    case_gen = rxn_data_gen(phase, model)

    cnt = 0
    topk_scores = [0.0] * cmd_args.topk
    
    pbar = tqdm(case_gen)

    fpred = open(fname_pred, 'w')
    for rxn_type, rxn, raw_prod in pbar:
        pred_struct = model.run(raw_prod, cmd_args.beam_size, cmd_args.topk, rxn_type=rxn_type)
        reactants, _, prod = rxn.split('>')
        if pred_struct is not None and len(pred_struct['reactants']):
            predictions = pred_struct['reactants']
        else:
            predictions = [prod]
        s = 0.0
        reactants = canonicalize(reactants)
        for i in range(cmd_args.topk):
            if i < len(predictions):
                pred = predictions[i]
                pred = canonicalize(pred)
                predictions[i] = pred
                cur_s = (pred == reactants)
            else:
                cur_s = s
            s = max(cur_s, s)
            topk_scores[i] += s
        cnt += 1
        if pred_struct is None or len(pred_struct['reactants']) == 0:
            predictions = []
        fpred.write('%s %s %d\n' % (rxn_type, rxn, len(predictions)))
        for i in range(len(predictions)):
            fpred.write('%s %s\n' % (pred_struct['template'][i], predictions[i]))
        msg = 'average score'
        for k in range(0, min(cmd_args.topk, 10), 3):
            msg += ', t%d: %.4f' % (k + 1, topk_scores[k] / cnt)
        pbar.set_description(msg)
    fpred.close()
    h = '========%s results========' % phase
    print(h)
    for k in range(cmd_args.topk):
        print('top %d: %.4f' % (k + 1, topk_scores[k] / cnt))
    print('=' * len(h))

    f_summary = '.'.join(fname_pred.split('.')[:-1]) + '.summary'
    with open(f_summary, 'w') as f:
        f.write('type overall\n')
        for k in range(cmd_args.topk):
            f.write('top %d: %.4f\n' % (k + 1, topk_scores[k] / cnt))


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    if local_args.model_for_test is None: # test all
        i = 0
        while True:
            model_dump = os.path.join(cmd_args.save_dir, 'model-%d.dump' % i)
            if not os.path.isdir(model_dump):
                break
            local_args.model_for_test = model_dump
            model = RetroGLN(cmd_args.dropbox, local_args.model_for_test)
            print('testing', local_args.model_for_test)
            for phase in ['val', 'test']:
                fname_pred = os.path.join(cmd_args.save_dir, '%s-%d.pred' % (phase, i))
                eval_model(phase, model, fname_pred)
            i += 1
    else:
        model = RetroGLN(cmd_args.dropbox, local_args.model_for_test)
        print('testing', local_args.model_for_test)
        fname_pred = os.path.join(cmd_args.save_dir, 'test.pred')
        eval_model('test', model, fname_pred)
