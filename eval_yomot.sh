#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda  activate obtracking
cd /home/niangao/disk1/_PycharmProjects/PycharmProjects/YONTD-MOT
cp -r /home/niangao/disk1/_PycharmProjects/PycharmProjects/YONTD-MOT/oxts /home/niangao/disk1/_PycharmProjects/PycharmProjects/YONTD-MOT/data/KITTI/training
python kitti_main.py
#python sys.py