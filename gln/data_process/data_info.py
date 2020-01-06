from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from tqdm import tqdm
import csv
import os
import pickle as cp
from collections import defaultdict
import numpy as np

from gln.common.mol_utils import cano_smarts, cano_smiles
from gln.common.cmd_args import cmd_args
from gln.common.evaluate import canonicalize
from gln.common.mol_utils import smarts_has_useless_parentheses
from gln.mods.mol_gnn.mol_utils import SmilesMols, SmartsMols


def load_bin_feats(dropbox, args):
    print('loading smiles feature dump')
    file_root = os.path.join(dropbox, 'cooked_' + args.data_name, 'tpl-%s' % args.tpl_name)
    SmartsMols.set_fp_degree(args.fp_degree)
    load_feats = args.subg_enc != 'ecfp' or args.tpl_enc != 'onehot'
    load_fp = args.subg_enc == 'ecfp'
    SmartsMols.load_dump(os.path.join(file_root, 'graph_smarts'), load_feats=load_feats, load_fp=load_fp)
    SmilesMols.set_fp_degree(args.fp_degree)
    SmilesMols.load_dump(os.path.join(file_root, '../graph_smiles'), load_feats=args.gm != 'ecfp', load_fp=args.gm == 'ecfp')


def load_center_maps(fname):
    prod_center_maps = {}
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in tqdm(reader):
            smiles, rxn_type, indices = row
            indices = [int(t) for t in indices.split()]
            prod_center_maps[(rxn_type, smiles)] = indices
    avg_sizes = [len(prod_center_maps[key]) for key in prod_center_maps]
    print('average # centers per mol:', np.mean(avg_sizes))
    return prod_center_maps


def load_train_reactions(args):
    train_reactions = []
    raw_data_root = os.path.join(args.dropbox, args.data_name)
    with open(os.path.join(raw_data_root, 'raw_train.csv'), 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        pos = header.index('reactants>reagents>production') if 'reactants>reagents>production' in header else -1
        c_idx = header.index('class')
        for row in reader:
            train_reactions.append((row[c_idx], row[pos]))
    print('# raw train loaded', len(train_reactions))
    return train_reactions


class DataInfo(object):

    @classmethod
    def load_cooked_part(cls, phase, part, load_graphs=True):
        args = cls.args
        load_feats = args.gm != 'ecfp'
        load_fp = not load_feats
        if cls.cur_part is not None and cls.cur_part == part:
            return
        file_root = os.path.join(args.dropbox, 'cooked_' + args.data_name, 'tpl-%s' % args.tpl_name, 'np-%d' % args.num_parts)
        assert phase == 'train'
        # load neg reactant features
        if load_graphs and args.retro_during_train:
            if cls.cur_part is not None:
                SmilesMols.remove_dump(os.path.join(file_root, 'neg_graphs-part-%d' % cls.cur_part))
            SmilesMols.load_dump(os.path.join(file_root, 'neg_graphs-part-%d' % part), additive=True, load_feats=load_feats, load_fp=load_fp)

        if args.gen_method != 'none':  # load pos-tpl map
            print('loading positive tpls')
            cls.train_pos_maps = defaultdict(list)
            fname = 'pos_tpls-part-%d.csv' % part
            with open(os.path.join(file_root, fname), 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    tpl_idx = int(row[0])
                    cls.train_pos_maps[tpl_idx].append((int(row[1]), int(row[2])))
            print('# pos tpls', len(cls.train_pos_maps))
            for key in cls.train_pos_maps:
                pos = cls.train_pos_maps[key]
                weights = np.array([1.0 / float(x[1]) for x in pos])
                weights /= np.sum(weights)
                tpls = [x[0] for x in pos]
                cls.train_pos_maps[key] = (tpls, weights)
        else:
            cls.train_pos_maps = None

        if args.retro_during_train:  # load negative reactions
            print('loading negative reactions')
            cls.neg_reacts_ids = {}
            cls.neg_reacts_list = []
            cls.neg_reactions_all = defaultdict(set)
            fname = 'neg_reacts.csv' if part is None else 'neg_reacts-part-%d.csv' % part
            with open(os.path.join(file_root, fname), 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in tqdm(reader):
                    sample_idx, reacts = row
                    if not reacts in cls.neg_reacts_ids:
                        idx = len(cls.neg_reacts_ids)
                        cls.neg_reacts_ids[reacts] = idx
                        cls.neg_reacts_list.append(reacts)
                    idx = cls.neg_reacts_ids[reacts]                        
                    cls.neg_reactions_all[int(row[0])].add(idx)
            for key in cls.neg_reactions_all:
                cls.neg_reactions_all[key] = list(cls.neg_reactions_all[key])

        cls.prod_center_maps = {}
        print('loading training prod center maps')
        fname = 'train-prod_center_maps-part-%d.csv' % part
        fname = os.path.join(file_root, fname)
        cls.prod_center_maps = load_center_maps(fname)

        cls.cur_part = part

    @classmethod
    def init(cls, dropbox, args):
        cls.args = args
        cls.args.dropbox = dropbox
        file_root = os.path.join(dropbox, 'cooked_' + args.data_name, 'tpl-%s' % args.tpl_name)
        print('loading data info from', file_root)
        
        # load training
        tpl_file = os.path.join(file_root, 'templates.csv')

        cls.unique_templates = set()
        print('loading templates')
        with open(tpl_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            tpl_idx = header.index('retro_templates')
            rt_idx = header.index('class')
            for row in tqdm(reader):
                tpl = row[tpl_idx]
                center, r_a, r_c = tpl.split('>')
                if smarts_has_useless_parentheses(center):
                    center = center[1:-1]
                tpl = '>'.join([center, r_a, r_c])
                rxn_type = row[rt_idx]
                cls.unique_templates.add((rxn_type, tpl))
        cls.unique_templates = sorted(list(cls.unique_templates))
        cls.idx_of_template = {}
        for i, tpl in enumerate(cls.unique_templates):
            cls.idx_of_template[tpl] = i
        print('# unique templates', len(cls.unique_templates))

        with open(os.path.join(file_root, '../cano_smiles.pkl'), 'rb') as f:
            cls.smiles_cano_map = cp.load(f)

        with open(os.path.join(file_root, 'cano_smarts.pkl'), 'rb') as f:
            cls.smarts_cano_map = cp.load(f)

        with open(os.path.join(file_root, 'prod_cano_smarts.txt'), 'r') as f:
            cls.prod_cano_smarts = [row.strip() for row in f.readlines()]
        
        cls.prod_smarts_idx = {}
        for i in range(len(cls.prod_cano_smarts)):
            cls.prod_smarts_idx[cls.prod_cano_smarts[i]] = i

        cls.unique_tpl_of_prod_center = defaultdict(lambda: defaultdict(list))
        for i, row in enumerate(cls.unique_templates):
            rxn_type, tpl = row
            center = tpl.split('>')[0]
            cano_center = cls.smarts_cano_map[center]
            cls.unique_tpl_of_prod_center[cano_center][rxn_type].append(i)

        cls.cur_part = None

    @classmethod
    def get_cano_smiles(cls, smiles):
        if smiles in cls.smiles_cano_map:
            return cls.smiles_cano_map[smiles]
        ans = canonicalize(smiles)
        cls.smiles_cano_map[smiles] = ans
        return ans

    @classmethod
    def get_cano_smarts(cls, smarts):
        if smarts in cls.smarts_cano_map:
            return cls.smarts_cano_map[smarts]
        ans = cano_smarts(smarts)[1]
        cls.smarts_cano_map[smarts] = ans
        return ans
