#!/bin/bash

model="piunet"
this_dir=`pwd`

log_dir="$this_dir/log_dir/$model/"
mkdir -p $log_dir

save_dir="$this_dir/Results/$model/"
mkdir -p $save_dir


CUDA_VISIBLE_DEVICES=0 python Code/$model/main.py --log_dir $log_dir --save_dir $save_dir
