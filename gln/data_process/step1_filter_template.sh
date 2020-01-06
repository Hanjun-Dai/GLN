#!/bin/bash


dropbox=../../dropbox
data=schneider50k
version=default

save_dir=$dropbox/cooked_${data}/tpl-$version

if [ ! -e $save_dir ]; 
then
    mkdir -p $save_dir
fi

python filter_template.py \
    -dropbox $dropbox \
    -data_name $data \
    -tpl_name $version \
    -save_dir $save_dir \
