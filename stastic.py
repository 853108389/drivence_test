import os

import numpy as np
import pandas as pd
from natsort import natsorted

import config
from config import common_config


def get_aug_data_seg_tabel():
    base_dir = common_config.aug_datasets_dirname
    column_names = list(range(0, 1))
    row_names = ['0000', '0002', '0005', '0007', '0010', '0011', '0014', '0018']
    vv = np.inf
    vv2 = 0
    df = pd.DataFrame(None, index=row_names, columns=column_names)
    data_num_pre = 0
    for p in natsorted(os.listdir(base_dir)):
        params = p.split("_")
        data_num = params[1]
        if data_num != data_num_pre:
            vv = np.inf
            vv2 = 0
            data_num_pre = data_num
        seed_num = int(params[3])
        ls = natsorted(os.listdir(os.path.join(base_dir, p, "KITTI", "training", "image_02", data_num)))
        v = int(ls[0].split(".")[0])
        v2 = int(ls[-1].split(".")[0])
        vv = min(v, vv)
        vv2 = max(v2, vv2)
        df.at[data_num, 0] = f"{vv}-{vv2}"
    print(df)


def get_aug_data_num_tabel():
    base_dir = common_config.aug_datasets_dirname
    column_names = list(range(0, 10))
    row_names = ['0000', '0002', '0005', '0007', '0010', '0011', '0014', '0018']
    df = pd.DataFrame(None, index=row_names, columns=column_names)
    for p in natsorted(os.listdir(base_dir)):
        params = p.split("_")
        data_num = params[1]
        seed_num = int(params[3])
        v = len(os.listdir(os.path.join(base_dir, p, "KITTI", "training", "image_02", data_num)))
        df.at[data_num, seed_num] = v
    df.to_csv(config.common_config.result_path + "/aug_data_num.csv")
    print(df)


def get_jmodt_faults_tabel():
    base_dir = common_config.result_path
    base_dir = os.path.join(base_dir, "jmodt")
    column_names = list(range(0, 10))
    row_names = ['0000', '0002', '0005', '0007', '0010', '0011', '0014', '0018']
    df = pd.DataFrame(None, index=row_names, columns=column_names)
    for p in natsorted(os.listdir(base_dir)):
        params = p.split("_")
        data_num = params[1]
        seed_num = int(params[3])
        with open(os.path.join(base_dir, p, "mot_data/val/eval/car/summary_car.txt")) as f:
            res = f.readlines()
            MOTA = res[1].split("(MOTA)")[1].strip()
        v = MOTA
        df.at[data_num, seed_num] = v
    print(df)
    df.to_csv(config.common_config.result_path + "/jmodt.csv")


def get_dfmot_faults_tabel():
    base_dir = common_config.result_path
    base_dir = os.path.join(base_dir, "dfmot")
    column_names = list(range(0, 10))
    row_names = ['0000', '0002', '0005', '0007', '0010', '0011', '0014', '0018']
    df = pd.DataFrame(None, index=row_names, columns=column_names)
    for p in natsorted(os.listdir(base_dir)):
        params = p.split("_")
        data_num = params[1]
        seed_num = int(params[3])
        with open(os.path.join(base_dir, p, "eval/car/summary_car.txt")) as f:
            res = f.readlines()
            MOTA = res[1].split("(MOTA)")[1].strip()
        v = MOTA
        df.at[data_num, seed_num] = v
    print(df)
    df.to_csv(config.common_config.result_path + "/dfmot.csv")


if __name__ == '__main__':
    get_aug_data_num_tabel()
