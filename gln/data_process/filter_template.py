from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import csv
from tqdm import tqdm
from gln.common.cmd_args import cmd_args
from collections import Counter, defaultdict


if __name__ == '__main__':
    proc_file = os.path.join(cmd_args.save_dir, '../proc_train_singleprod.csv')

    unique_tpls = Counter()
    tpl_types = defaultdict(set)
    with open(proc_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(header)
        for row in tqdm(reader):
            tpl = row[header.index('retro_templates')]
            rxn_type = row[header.index('class')]
            tpl_types[tpl].add(rxn_type)
            unique_tpls[tpl] += 1

    print('total # templates', len(unique_tpls))

    used_tpls = []
    for x in unique_tpls:
        if unique_tpls[x] >= cmd_args.tpl_min_cnt:
            used_tpls.append(x)
    print('num templates after filtering', len(used_tpls))

    out_file = os.path.join(cmd_args.save_dir, 'templates.csv')
    with open(out_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'retro_templates'])
        for x in used_tpls:
            for t in tpl_types[x]:
                writer.writerow([t, x])
