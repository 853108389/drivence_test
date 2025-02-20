import os
import shutil
from collections import defaultdict

from natsort import natsorted

import config

if __name__ == '__main__':
    base_path = f"{config.common_config.project_dirname}/TrackGen/queue"
    all_data_dict = defaultdict(int)
    for seq in natsorted(os.listdir(base_path)):
        data_path1 = os.path.join(base_path, seq)
        data_path = os.path.join(data_path1, "_seed_pool")
        data_dict = defaultdict(int)
        for seed in natsorted(os.listdir(data_path)):
            seed_path = os.path.join(data_path, seed, "npc_insert")
            npc_num = 0
            for npc in natsorted(os.listdir(seed_path)):
                trajectory_path = os.path.join(seed_path, npc)
                params = os.path.splitext(npc)[0].split("_")
                is_avalibel = int(params[6])
                if is_avalibel:
                    npc_num += 1
                    op = str(params[1])
            if npc_num >= 1 and op == "FL":
                data_dict[op] += 1
                all_data_dict[op] += 1
                file_name = f"task_{seq}_seed_{seed}_trajectories"
                src_path = os.path.join(config.common_config.project_dirname, "_aug_datasets", file_name, "KITTI")
                target_base_path = os.path.join(config.common_config.project_dirname, "_aug_op_datasets", op,
                                                file_name)
                target_path = os.path.join(target_base_path, "KITTI")
                os.makedirs(target_base_path, exist_ok=True)
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                shutil.copytree(src_path, target_path)

        print(f"seq: {seq}, data_dict: {sorted(data_dict.items())}")
    print(all_data_dict)
