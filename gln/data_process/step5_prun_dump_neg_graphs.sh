#!/bin/bash

dropbox='../../dropbox'
data=uspto_full
tpl=new
tpl_min_cnt=4
fp_degree=2
num_parts=40
cooked_root=../../cooked_data

save_dir=$cooked_root/$data/tpl-$tpl-mincnt-$tpl_min_cnt

N=$num_parts
(
for ((s=0;s<40;s+=1))
do
    ((i=i%N)); ((i++==0)) && wait

python dump_graphs.py \
    -dropbox $dropbox \
    -data_name $data \
    -tpl_min_cnt $tpl_min_cnt \
    -cooked_root $cooked_root \
    -tpl_name $tpl \
    -save_dir $save_dir \
    -f_atoms $cooked_root/$data/atom_list.txt \
    -num_parts $num_parts \
    -part_id $s \
    -part_num 1 \
    -fp_degree $fp_degree \
    -retro_during_train True \
    &

done
)
