from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import os
import sys
import rdkit
from rdkit import Chem
import random
import pickle as cp
import csv
from gln.common.cmd_args import cmd_args
from gln.common.consts import t_float, DEVICE
from gln.data_process.data_info import load_bin_feats, DataInfo
from gln.graph_logic.logic_net import GraphPath
from gln.training.data_gen import data_gen, worker_softmax
from tqdm import tqdm
import torch
import torch.optim as optim

from gln.common.reactor import Reactor
from gln.common.evaluate import get_score, canonicalize
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')


def main_train():
    data_root = os.path.join(cmd_args.dropbox, cmd_args.data_name)
    train_sample_gen = data_gen(cmd_args.num_data_proc, worker_softmax, [cmd_args], max_gen=-1)

    if cmd_args.init_model_dump is not None:
        graph_path.load_state_dict(torch.load(cmd_args.init_model_dump))

    optimizer = optim.Adam(graph_path.parameters(), lr=cmd_args.learning_rate)

    for epoch in range(cmd_args.num_epochs):

        pbar = tqdm(range(1, 1 + cmd_args.iters_per_val))
        
        for it in pbar:
            samples = [next(train_sample_gen) for _ in range(cmd_args.batch_size)]
            optimizer.zero_grad()
            loss = graph_path(samples)
            loss.backward()

            if cmd_args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(graph_path.parameters(), max_norm=cmd_args.grad_clip)

            optimizer.step()
            pbar.set_description('epoch %.2f, loss %.4f' % (epoch + it / cmd_args.iters_per_val, loss.item()))

        if epoch % cmd_args.epochs2save == 0:
            out_folder = os.path.join(cmd_args.save_dir, 'model-%d.dump' % epoch)
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)            
            torch.save(graph_path.state_dict(), os.path.join(out_folder, 'model.dump'))
            with open(os.path.join(out_folder, 'args.pkl'), 'wb') as f:
                cp.dump(cmd_args, f, cp.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    DataInfo.init(cmd_args.dropbox, cmd_args)
    load_bin_feats(cmd_args.dropbox, cmd_args)
    graph_path = GraphPath(cmd_args).to(DEVICE)
    main_train()
