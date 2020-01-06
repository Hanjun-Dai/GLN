#!/bin/bash

dropbox=../../dropbox
data=schneider50k
tpl=default
fp_degree=2
num_parts=1

save_dir=$dropbox/cooked_$data/tpl-$tpl

for r in False True; do

python dump_graphs.py \
    -dropbox $dropbox \
    -data_name $data \
    -tpl_name $tpl \
    -save_dir $save_dir \
    -f_atoms $dropbox/cooked_$data/atom_list.txt \
    -num_parts $num_parts \
    -fp_degree $fp_degree \
    -retro_during_train $r \
    $@

done
