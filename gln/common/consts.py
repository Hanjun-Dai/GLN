from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import argparse
import logging
import numpy as np
import torch
import torch.nn as nn

t_float = torch.float32
np_float = np.float32
str_float = "float32"

opts = argparse.ArgumentParser(description='gpu option')
opts.add_argument('-gpu', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')

args, _ = opts.parse_known_args()
if torch.cuda.is_available() and args.gpu >= 0:
    DEVICE = torch.device('cuda:' + str(args.gpu))
    print('use gpu indexed: %d' % args.gpu)
else:
    DEVICE = torch.device('cpu')
    print('use cpu')

