#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda  activate pcd
cd ./openpcdet
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
cd ./tools
python test.py --cfg_file ./cfgs/kitti_models/second.yaml --batch_size 2 --ckpt ../premodel/second_7862.pth --save_to_file
#python sys.py