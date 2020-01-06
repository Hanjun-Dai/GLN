#!/bin/bash

dropbox=../../dropbox
data_name=schneider50k
tpl_name=default_tpl

python graph_feat.py \
    -dropbox $dropbox \
    -data_name $data_name \
    -tpl_name $tpl_name \
    -f_atoms $dropbox/cooked_$data_name/atom_list.txt \
    $@

