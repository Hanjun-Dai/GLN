#!/bin/bash

dropbox=../../dropbox
data_name=schneider50k
tpl_name=default
num_cores=4
num_parts=1

save_dir=$dropbox/cooked_$data_name/tpl-$tpl_name


python build_all_reactions.py \
    -dropbox $dropbox \
    -phase cooking \
    -data_name $data_name \
    -save_dir $save_dir \
    -tpl_name $tpl_name \
    -f_atoms $dropbox/cooked_$data_name/atom_list.txt \
    -num_cores $num_cores \
    -num_parts $num_parts \
    -gpu -1 \
    $@

