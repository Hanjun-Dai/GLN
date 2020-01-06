#!/bin/bash


dropbox=../../dropbox
data=schneider50k
num_cores=4

save_dir=$dropbox/cooked_${data}

if [ ! -e $save_dir ]; 
then
    mkdir -p $save_dir
fi

python build_raw_template.py \
    -dropbox $dropbox \
    -data_name $data \
    -save_dir $save_dir \
    -num_cores $num_cores \
