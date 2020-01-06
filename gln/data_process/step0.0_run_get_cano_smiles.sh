#!/bin/bash


dropbox=../../dropbox
data=schneider50k

save_dir=$dropbox/cooked_$data


python get_canonical_smiles.py \
    -dropbox $dropbox \
    -data_name $data \
    -save_dir $save_dir
