import os

import pandas as pd
from natsort import natsorted

import config
from config import common_config


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
    print(df)


def get_default_tabel():
    column_names = list(range(0, 30))
    row_names = ['0000', '0002', '0005', '0007', '0010', '0011', '0014', '0018']
    df = pd.DataFrame(None, index=row_names, columns=column_names)
    return df


def get_dfmot_faults_tabel():
    name = "dfmot"
    base_dir = common_config.result_path
    base_dir = os.path.join(base_dir, f"{name}_hota")
    df_HOTA = get_default_tabel()
    df_DetA = get_default_tabel()
    df_AssA = get_default_tabel()
    for p in natsorted(os.listdir(base_dir)):
        params = p.split("_")
        data_num = params[1]
        seed_num = int(params[3])
        with open(os.path.join(base_dir, p, "car_summary.txt")) as f:
            res = f.readlines()
            params = res[1].split(" ")
            HOTA = params[0]
            DetA = params[1]
            AssA = params[2]
            df_HOTA.at[data_num, seed_num] = HOTA
            df_DetA.at[data_num, seed_num] = DetA
            df_AssA.at[data_num, seed_num] = AssA

    out_path = os.path.join(config.common_config.stastic_path, name)
    os.makedirs(out_path, exist_ok=True)
    df_HOTA.to_csv(out_path + f"/{name}_hota.csv")
    df_DetA.to_csv(out_path + f"/{name}_detA.csv")
    df_AssA.to_csv(out_path + f"/{name}_assA.csv")


def get_jmodt_faults_tabel():
    base_dir = common_config.result_path
    base_dir = os.path.join(base_dir, "jmodt_hota")

    df_HOTA = get_default_tabel()
    df_DetA = get_default_tabel()
    df_AssA = get_default_tabel()
    for p in natsorted(os.listdir(base_dir)):
        params = p.split("_")
        data_num = params[1]
        seed_num = int(params[3])
        with open(os.path.join(base_dir, p, "mot_data/car_summary.txt")) as f:
            res = f.readlines()
            params = res[1].split(" ")
            HOTA = params[0]
            DetA = params[1]
            AssA = params[2]
            df_HOTA.at[data_num, seed_num] = HOTA
            df_DetA.at[data_num, seed_num] = DetA
            df_AssA.at[data_num, seed_num] = AssA

    out_path = os.path.join(config.common_config.stastic_path, "jmodt")
    print(out_path)
    os.makedirs(out_path, exist_ok=True)
    df_HOTA.to_csv(out_path + "/jmodt_hota.csv")
    df_DetA.to_csv(out_path + "/jmodt_detA.csv")
    df_AssA.to_csv(out_path + "/jmodt_assA.csv")


def get_virmot_faults_tabel():
    name = "virmot"
    base_dir = common_config.result_path
    base_dir = os.path.join(base_dir, name)
    df_HOTA = get_default_tabel()
    df_DetA = get_default_tabel()
    df_AssA = get_default_tabel()
    for p in natsorted(os.listdir(base_dir)):
        params = p.split("_")
        print(p)
        data_num = params[1]
        seed_num = int(params[3])
        print(p)
        with open(os.path.join(base_dir, p, "tracking/car_summary.txt")) as f:
            res = f.readlines()
            params = res[1].split(" ")
            HOTA = params[0]
            DetA = params[1]
            AssA = params[2]
            df_HOTA.at[data_num, seed_num] = HOTA
            df_DetA.at[data_num, seed_num] = DetA
            df_AssA.at[data_num, seed_num] = AssA

    out_path = os.path.join(config.common_config.stastic_path, name)
    df_HOTA["mean"] = df_HOTA.mean(axis=1)
    print(out_path)
    os.makedirs(out_path, exist_ok=True)
    df_HOTA.to_csv(out_path + f"/{name}_hota.csv")
    df_DetA.to_csv(out_path + f"/{name}_detA.csv")
    df_AssA.to_csv(out_path + f"/{name}_assA.csv")


def get_yomot_faults_tabel():
    name = "yomot"
    base_dir = common_config.result_path
    base_dir = os.path.join(base_dir, name)
    df_HOTA = get_default_tabel()
    df_DetA = get_default_tabel()
    df_AssA = get_default_tabel()
    for p in natsorted(os.listdir(base_dir)):
        params = p.split("_")
        data_num = params[1]
        seed_num = int(params[3])
        with open(os.path.join(base_dir, p, "tracking/car_summary.txt")) as f:
            res = f.readlines()
            params = res[1].split(" ")
            HOTA = params[0]
            DetA = params[1]
            AssA = params[2]
            df_HOTA.at[data_num, seed_num] = HOTA
            df_DetA.at[data_num, seed_num] = DetA
            df_AssA.at[data_num, seed_num] = AssA

    out_path = os.path.join(config.common_config.stastic_path, name)
    os.makedirs(out_path, exist_ok=True)
    df_HOTA.to_csv(out_path + f"/{name}_hota.csv")
    df_DetA.to_csv(out_path + f"/{name}_detA.csv")
    df_AssA.to_csv(out_path + f"/{name}_assA.csv")


if __name__ == '__main__':
    get_virmot_faults_tabel()
