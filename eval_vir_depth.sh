#!/bin/bash
param1=$1
#echo "aaa"+$param1
source ~/anaconda3/etc/profile.d/conda.sh
conda  activate obinsert_backup
cd /home/niangao/disk1/_PycharmProjects/PycharmProjects/VirConv
python bash_generate_depth.py --n $param1
#python sys.py