#!/bin/bash

dropbox=../../../dropbox
data_name=$1
tpl_name=default
gm=mean_field
act=relu
msg_dim=128
embed_dim=256
neg_size=64
lv=3
tpl_enc=deepset
subg_enc=mean_field
graph_agg=max
retro=True
bn=True
gen=weighted
gnn_out=last
neg_sample=all
att_type=bilinear

save_dir=$HOME/scratch/results/gln/$data_name/tpl-$tpl_name/${gm}-${act}-lv-${lv}-l-${msg_dim}-e-${embed_dim}-gagg-${graph_agg}-retro-${retro}-gen-${gen}-ng-${neg_size}-bn-${bn}-te-${tpl_enc}-se-${subg_enc}-go-${gnn_out}-ns-${neg_sample}-att-${att_type}

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2

python ../main.py \
    -gm $gm \
    -fp_degree 2 \
    -neg_sample $neg_sample \
    -att_type $att_type \
    -gnn_out $gnn_out \
    -tpl_enc $tpl_enc \
    -subg_enc $subg_enc \
    -latent_dim $msg_dim \
    -bn $bn \
    -gen_method $gen \
    -retro_during_train $retro \
    -neg_num $neg_size \
    -embed_dim $embed_dim \
    -readout_agg_type $graph_agg \
    -act_func $act \
    -act_last True \
    -max_lv $lv \
    -dropbox $dropbox \
    -data_name $data_name \
    -save_dir $save_dir \
    -tpl_name $tpl_name \
    -f_atoms $dropbox/cooked_$data_name/atom_list.txt \
    -iters_per_val 3000 \
    -gpu 0 \
    -topk 50 \
    -beam_size 50 \
    -num_parts 1 \

