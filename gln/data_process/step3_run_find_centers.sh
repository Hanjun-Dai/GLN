#!/bin/bash

dropbox=../../dropbox
data=schneider50k
tpl=default
num_cores=8
num_parts=1

save_dir=$dropbox/cooked_$data/tpl-$tpl

python find_centers.py \
    -dropbox $dropbox \
    -data_name $data \
    -tpl_name $tpl \
    -save_dir $save_dir \
    -num_cores $num_cores \
    -num_parts $num_parts \
    $@
