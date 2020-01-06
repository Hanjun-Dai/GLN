#!/bin/bash

dropbox=../../dropbox
data_name=$1
tpl_name=default

save_dir=$data_name-results

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

export CUDA_VISIBLE_DEVICES=0

python main_test.py \
    -dropbox $dropbox \
    -data_name $data_name \
    -save_dir $save_dir \
    -model_for_test $dropbox/$data_name.ckpt \
    -tpl_name $tpl_name \
    -f_atoms $dropbox/cooked_$data_name/atom_list.txt \
    -topk 50 \
    -beam_size 50 \
    -gpu 0 \

