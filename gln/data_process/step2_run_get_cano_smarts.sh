#!/bin/bash


dropbox=../../dropbox
data=schneider50k
tpl=default

save_dir=$dropbox/cooked_$data/tpl-$tpl


python get_canonical_smarts.py \
    -dropbox $dropbox \
    -data_name $data \
    -save_dir $save_dir \
    -tpl_name $tpl \
