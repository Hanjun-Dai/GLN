from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import pickle as cp

cmd_opt = argparse.ArgumentParser(description='Argparser for retrosyn_graph')
cmd_opt.add_argument('-save_dir', default='.', help='result output root')
cmd_opt.add_argument('-dropbox', default=None, help='dropbox folder')
cmd_opt.add_argument('-cooked_root', default=None, help='cooked data root folder')
cmd_opt.add_argument('-init_model_dump', default=None, help='model dump')
cmd_opt.add_argument('-data_name', default=None, help='dataset name')
cmd_opt.add_argument('-tpl_name', default=None, help='template name')
cmd_opt.add_argument('-tpl_min_cnt', default=0, type=int, help='template min cnt (for filtering)')
cmd_opt.add_argument('-phase', default=None, help='phase')
cmd_opt.add_argument('-is_training', default=True, type=eval, help='is training')
cmd_opt.add_argument('-split_mode', default='single', help='single/multi/ignore')

cmd_opt.add_argument('-bn', default=True, type=eval, help='using bn?')
cmd_opt.add_argument('-file_for_eval', default=None, help='file for evaluation')
cmd_opt.add_argument('-model_for_eval', default=None, help='model for evaluation')
cmd_opt.add_argument('-num_cores', default=1, type=int, help='# cpu cores')
cmd_opt.add_argument('-num_parts', default=1, type=int, help='num of parts to split')

cmd_opt.add_argument('-part_id', default=0, type=int, help='part id')
cmd_opt.add_argument('-epochs2save', default=1, type=int, help='epochs to save')
cmd_opt.add_argument('-max_neg_reacts', default=0, type=int, help='max neg')
cmd_opt.add_argument('-part_num', default=0, type=int, help='part num')
cmd_opt.add_argument('-eval_func', default='acc', help='acc/mix_f1')

cmd_opt.add_argument('-neg_sample', default='local', help='local/all')
cmd_opt.add_argument('-num_data_proc', default=0, type=int, help='num of data process')
cmd_opt.add_argument('-topk', default=1, type=int, help='topk eval')
cmd_opt.add_argument('-neg_num', default=-1, type=int, help='num of negative samples')
cmd_opt.add_argument('-beam_size', default=1, type=int, help='beam search size')
cmd_opt.add_argument('-gm', default='mean_field', help='choose gnn module')
cmd_opt.add_argument('-fp_degree', default=0, type=int, help='fingerprint? [>0, 0]')

cmd_opt.add_argument('-latent_dim', default=64, type=int, help='latent dim of gnn')
cmd_opt.add_argument('-embed_dim', default=128, type=int, help='embedding dim of gnn')

cmd_opt.add_argument('-mlp_hidden', default=256, type=int, help='hidden dims in mlp')
cmd_opt.add_argument('-seed', default=19260817, type=int, help='seed')

cmd_opt.add_argument('-max_lv', default=3, type=int, help='# layers of gnn')
cmd_opt.add_argument('-eval_start_idx', default=0, type=int, help='model idx for eval')

cmd_opt.add_argument('-ggnn_update_type', default='gru', help='use gru or mlp for update state')
cmd_opt.add_argument('-msg_agg_type', default='sum', help='how to aggregate the message')
cmd_opt.add_argument('-att_type', default='inner_prod', help='mlp/inner_prod/bilinear')

cmd_opt.add_argument('-readout_agg_type', default='sum', help='how to aggregate all node embeddings')
cmd_opt.add_argument('-logic_net', default='gpath', help='gpath/mlp')

cmd_opt.add_argument('-node_dims', default='128', help='hidden dims for node uptate')
cmd_opt.add_argument('-edge_dims', default='128', help='hidden dims for edge update')
cmd_opt.add_argument('-act_func', default='tanh', help='default activation function')
cmd_opt.add_argument('-gnn_out', default='last', help='last/gru/sum/mean')
cmd_opt.add_argument('-act_last', default=True, type=eval, help='activation of last embedding layer')
cmd_opt.add_argument('-subg_enc', default='mean_field', help='subgraph embedding method')
cmd_opt.add_argument('-tpl_enc', default='deepset', help='template embedding method')

cmd_opt.add_argument('-neg_local', default=False, type=eval, help='local or global neg reaction?')

cmd_opt.add_argument('-gnn_share_param', default=False, type=eval, help='share params across layers')
cmd_opt.add_argument('-learning_rate', default=1e-3, type=float, help='learning rate')
cmd_opt.add_argument('-grad_clip', default=5, type=float, help='clip gradient')
cmd_opt.add_argument('-dropout', default=0, type=float, help='dropout')
cmd_opt.add_argument('-fp_dim', default=2048, type=int, help='dim of fp')
cmd_opt.add_argument('-gen_method', default='none', help='none/uniform/weighted')

cmd_opt.add_argument('-test_during_train', default=False, type=eval, help='do fast testing during training')
cmd_opt.add_argument('-test_mode', default='model', help='model/file')
cmd_opt.add_argument('-num_epochs', default=10000, type=int, help='number of training epochs')
cmd_opt.add_argument('-epochs_per_part', default=1, type=int, help='number of epochs per part')
cmd_opt.add_argument('-iters_per_val', default=1000, type=int, help='number of iterations per evaluation')
cmd_opt.add_argument('-batch_size', default=64, type=int, help='batch size for training')
cmd_opt.add_argument('-retro_during_train', type=eval, default=False, help='doing retrosynthesis during training?')


cmd_args, _ = cmd_opt.parse_known_args()

if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

from gln.mods.rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun

print(cmd_args)
