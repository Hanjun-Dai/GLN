from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from gln.mods.mol_gnn.gnn_family import EmbedMeanField, GGNN, MPNN, MorganFp, S2vMeanFieldV2
from gln.mods.mol_gnn.mg_clib import NUM_NODE_FEATS, NUM_EDGE_FEATS


def get_gnn(args, gm=None):
    if gm is None:
        gm = args.gm
    if gm == 'mean_field':
        gnn = EmbedMeanField(latent_dim=args.latent_dim,
                             output_dim=args.embed_dim,
                             num_node_feats=NUM_NODE_FEATS,
                             num_edge_feats=NUM_EDGE_FEATS,
                             max_lv=args.max_lv,
                             act_func=args.act_func,
                             readout_agg=args.readout_agg_type,
                             act_last=args.act_last,
                             dropout=args.dropout)
    elif gm == 's2v_v2':
        gnn = S2vMeanFieldV2(latent_dim=args.latent_dim,
                             output_dim=args.embed_dim,
                             num_node_feats=NUM_NODE_FEATS,
                             num_edge_feats=NUM_EDGE_FEATS,
                             max_lv=args.max_lv,
                             act_func=args.act_func,
                             readout_agg=args.readout_agg_type,
                             act_last=args.act_last,
                             out_method=args.gnn_out,
                             bn=args.bn,
                             dropout=args.dropout)
    elif gm == 'ggnn':
        gnn = GGNN(node_state_dim=args.latent_dim,
                   output_dims=[args.embed_dim], 
                   edge_hidden_sizes=[args.latent_dim],
                   num_node_feats=NUM_NODE_FEATS, 
                   num_edge_feats=NUM_EDGE_FEATS,
                   max_lv=args.max_lv,
                   msg_aggregate_type=args.msg_agg_type,
                   readout_agg=args.readout_agg_type,
                   share_params=args.gnn_share_param, 
                   act_func=args.act_func,
                   dropout=args.dropout)
    elif gm == 'mpnn':
        gnn = MPNN(latent_dim=args.latent_dim,
                   output_dim=args.embed_dim,
                   num_node_feats=NUM_NODE_FEATS,
                   num_edge_feats=NUM_EDGE_FEATS,
                   max_lv=args.max_lv,
                   msg_aggregate_type=args.msg_agg_type,
                   act_func=args.act_func,
                   dropout=args.dropout)
    elif gm == 'ecfp':
        gnn = MorganFp(feat_dim=args.fp_dim,
                       hidden_size=args.embed_dim,
                       num_hidden=1,
                       feat_mode='dense',
                       act_func=args.act_func,
                       dropout=args.dropout)
    else:
        raise NotImplementedError
    return gnn
