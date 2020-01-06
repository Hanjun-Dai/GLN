from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import random
import os
import rdkit
from rdkit import Chem
import csv
from gln.mods.mol_gnn.mol_utils import SmartsMols, SmilesMols
from multiprocessing import Process, Queue
import time

from gln.data_process.data_info import DataInfo, load_train_reactions
from gln.common.reactor import Reactor


class DataSample(object):
    def __init__(self, prod, center, template, label=None, neg_centers=None, neg_tpls=None,
                 reaction=None, neg_reactions=None):
        self.prod = prod
        self.center = center
        self.template = template
        self.label = label
        self.neg_centers = neg_centers
        self.neg_tpls = neg_tpls
        self.reaction = reaction
        self.neg_reactions = neg_reactions


def _rand_sample_except(candidates, exclude, k=None):
    assert len(candidates)
    if k is None:
        if len(candidates) == 1:
            assert exclude is None or candidates[0] == exclude
            return candidates[0]
        else:
            while True:
                c = np.random.choice(candidates)
                if exclude is None or c != exclude:
                    break
            return c
    else:        
        if k <= 0 or len(candidates) <= k:
            return [c for c in candidates if exclude is None or c != exclude]
        cand_indices = np.random.permutation(len(candidates))[:k]        
        selected = []
        for i in cand_indices:
            c = candidates[i]
            if exclude is None or c != exclude:
                selected.append(c)
            if k <= 0:
                continue
            if len(selected) >= k:
                break
        return selected


def worker_softmax(worker_id, seed, args):
    np.random.seed(seed)
    random.seed(seed)
    num_epochs = 0
    part_id = 0
    train_reactions = load_train_reactions(args)
    while True:
        if num_epochs % args.epochs_per_part == 0:
            DataInfo.load_cooked_part('train', part_id)
            tot_num = len(train_reactions)
            part_size = tot_num // args.num_parts + 1
            indices = range(part_id * part_size, min((part_id + 1) * part_size, tot_num))
            indices = list(indices)
            part_id = (part_id + 1) % args.num_parts
        random.shuffle(indices)
        for sample_idx in indices:
            rxn_type, rxn_smiles = train_reactions[sample_idx]

            if sample_idx in DataInfo.train_pos_maps:
                pos_tpls, weights = DataInfo.train_pos_maps[sample_idx]
                pos_tpl_idx = pos_tpls[np.argmax(np.random.multinomial(1, weights))]
                rxn_type, rxn_template = DataInfo.unique_templates[pos_tpl_idx]
            else:
                continue

            reactants, _, prod = rxn_smiles.split('>')
            cano_prod = DataInfo.smiles_cano_map[prod]
            sm_prod, _, _ = rxn_template.split('>')
            cano_sm_prod = DataInfo.smarts_cano_map[sm_prod]

            # negative samples of prod centers
            assert (rxn_type, cano_prod) in DataInfo.prod_center_maps
            prod_center_cand_idx = DataInfo.prod_center_maps[(rxn_type, cano_prod)]
            
            neg_center_idxes = _rand_sample_except(prod_center_cand_idx, DataInfo.prod_smarts_idx[cano_sm_prod], args.neg_num)
            neg_centers = [DataInfo.prod_cano_smarts[c] for c in neg_center_idxes]

            # negative samples of templates
            assert cano_sm_prod in DataInfo.unique_tpl_of_prod_center
            assert rxn_type in DataInfo.unique_tpl_of_prod_center[cano_sm_prod]
            neg_tpl_idxes = _rand_sample_except(DataInfo.unique_tpl_of_prod_center[cano_sm_prod][rxn_type], pos_tpl_idx, args.neg_num)
            tpl_cand_idx = []
            for c in neg_centers:
                tpl_cand_idx += DataInfo.unique_tpl_of_prod_center[c][rxn_type]
            if len(tpl_cand_idx):
                neg_tpl_idxes += _rand_sample_except(tpl_cand_idx, pos_tpl_idx, args.neg_num)
            neg_tpls = [DataInfo.unique_templates[i][1] for i in neg_tpl_idxes]

            sample = DataSample(prod=cano_prod, center=cano_sm_prod, template=rxn_template,
                                neg_centers=neg_centers, neg_tpls=neg_tpls)

            if args.retro_during_train:
                sample.reaction = DataInfo.get_cano_smiles(reactants) + '>>' + cano_prod
                sample.neg_reactions = []
                if len(DataInfo.neg_reactions_all[sample_idx]):
                    neg_reacts = DataInfo.neg_reactions_all[sample_idx]
                    if len(neg_reacts):
                        neg_reactants = _rand_sample_except(neg_reacts, None, args.neg_num)
                        sample.neg_reactions = [DataInfo.neg_reacts_list[r] + '>>' + cano_prod for r in neg_reactants]
            if len(sample.neg_tpls) or len(sample.neg_reactions):
                yield (worker_id, sample)
        num_epochs += 1


def worker_process(worker_func, worker_id, seed, data_q, *args):
    worker_gen = worker_func(worker_id, seed, *args)
    for t in worker_gen:
        data_q.put(t)


def data_gen(num_workers, worker_func, worker_args, max_qsize=16384, max_gen=-1, timeout=60):
    cnt = 0
    data_q = Queue(max_qsize)

    if num_workers == 0:  # single process generator
        worker_gen = worker_func(-1, np.random.randint(10000), *worker_args)
        while True:
            worker_id, data_sample = next(worker_gen)
            yield data_sample
            cnt += 1
            if max_gen > 0 and cnt >= max_gen:
                break
        return

    worker_procs = [Process(target=worker_process, args=[worker_func, i, np.random.randint(10000), data_q] + worker_args) for i in range(num_workers)]
    for p in worker_procs:
        p.start()
    last_update = [time.time()] * num_workers    
    while True:
        if data_q.empty():
            time.sleep(0.1)
        if not data_q.full():
            for i in range(num_workers):
                if time.time() - last_update[i] > timeout:
                    print('worker', i, 'is dead')
                    worker_procs[i].terminate()
                    while worker_procs[i].is_alive():  # busy waiting for the stop of the process
                        time.sleep(0.01)
                    worker_procs[i] = Process(target=worker_process, args=[worker_func, i, np.random.randint(10000), data_q] + worker_args)
                    print('worker', i, 'restarts')
                    worker_procs[i].start()
                    last_update[i] = time.time()            
        try:
            sample = data_q.get_nowait()
        except:
            continue
        cnt += 1
        worker_id, data_sample = sample
        last_update[worker_id] = time.time()
        yield data_sample
        if max_gen > 0 and cnt >= max_gen:
            break

    print('stopping')
    for p in worker_procs:
        p.terminate()
    for p in worker_procs:
        p.join()
