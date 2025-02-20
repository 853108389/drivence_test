import os

import pandas as pd
from natsort import natsorted

import config


def systems():
    base_dir = os.path.join(config.common_config.project_dirname, "_aug_op_datasets")
    for system in ["jmodt_hota", "dfmot_hota", "virmot", "yomot"]:
        column_names = ["Lose Tracked", "Partly Tracked", "Fragmentation", "ID Switch"]
        row_names = ["ID", "LD", "FL", "SU", "SD", "CW"]
        df_system = pd.DataFrame(0, index=column_names, columns=row_names)
        for op in os.listdir(base_dir):
            op_path = os.path.join(base_dir, op)
            for task_name in natsorted(os.listdir(op_path)):
                base_path = os.path.join(config.common_config.result_path, system, task_name)
                if system == "dfmot_hota":
                    csv_path = os.path.join(base_path, "car_detailed.csv")
                elif system == "jmodt_hota":
                    csv_path = os.path.join(base_path, "mot_data", "car_detailed.csv")
                elif system == "virmot":
                    csv_path = os.path.join(base_path, "tracking", "car_detailed.csv")
                elif system == "yomot":
                    csv_path = os.path.join(base_path, "tracking", "car_detailed.csv")
                df = pd.read_csv(csv_path)
                df_system.at["Partly Tracked", op] += df.loc[0, "PT"]
                df_system.at["Lose Tracked", op] += df.loc[0, "ML"]
                df_system.at["Fragmentation", op] += df.loc[0, "Frag"]
                df_system.at["ID Switch", op] += df.loc[0, "IDSW"]
        base_path = f"{config.common_config.project_dirname}/result_stastic/op_faults"
        os.makedirs(base_path, exist_ok=True)
        print(f"{base_path}/{system}_faults.csv")
        df_system.to_csv(f"{base_path}/{system}_faults.csv")
        print(df_system)


def ops():
    base_dir = os.path.join(config.common_config.project_dirname, "_aug_op_datasets")
    for op in os.listdir(base_dir):
        column_names = ["Lose Tracked", "Partly Tracked", "Fragmentation", "ID Switch"]
        row_names = ["virmot", "jmodt_hota", "dfmot_hota", "yomot"]
        df_op = pd.DataFrame(0, index=row_names, columns=column_names)
        for system in ["virmot", "jmodt_hota", "dfmot_hota", "yomot"]:
            op_path = os.path.join(base_dir, op)
            for task_name in natsorted(os.listdir(op_path)):
                base_path = os.path.join(config.common_config.result_path, system, task_name)
                if system == "dfmot_hota":
                    csv_path = os.path.join(base_path, "car_detailed.csv")
                elif system == "jmodt_hota":
                    csv_path = os.path.join(base_path, "mot_data", "car_detailed.csv")
                elif system == "virmot":
                    csv_path = os.path.join(base_path, "tracking", "car_detailed.csv")
                elif system == "yomot":
                    csv_path = os.path.join(base_path, "tracking", "car_detailed.csv")
                df = pd.read_csv(csv_path)
                df_op.at[system, "Partly Tracked"] += df.loc[0, "PT"]
                df_op.at[system, "Lose Tracked"] += df.loc[0, "ML"]
                df_op.at[system, "Fragmentation"] += df.loc[0, "Frag"]
                df_op.at[system, "ID Switch"] += df.loc[0, "IDSW"]
        base_path = f"{config.common_config.project_dirname}/result_stastic/op_faults"
        os.makedirs(base_path, exist_ok=True)
        df_op.to_csv(f"{base_path}/{op}_faults.csv")


if __name__ == '__main__':
    ops()
