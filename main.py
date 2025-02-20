import os
import os
import shutil
import traceback

import open3d as o3d
from natsort import natsorted

from TrackGen.main import run
from config import common_config
from data_gen.demo2 import perpare
from data_gen.demo2 import run as run2

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def check_avalible_trajectory(base_dir, seq):
    npc_dir = os.path.join(common_config.project_dirname, "TrackGen", "queue", f"{seq}", "npc")
    trajectory_dir = os.path.join(base_dir, "npc_insert")
    trajectory_paths = natsorted(os.listdir(trajectory_dir))
    count = 0
    if os.path.exists(common_config.trakcing_insert_dir):
        shutil.rmtree(common_config.trakcing_insert_dir)
    if os.path.exists(common_config.trakcing_background_dir):
        shutil.rmtree(common_config.trakcing_background_dir)
    os.makedirs(common_config.trakcing_insert_dir, exist_ok=True)
    shutil.copytree(npc_dir, common_config.trakcing_background_dir)
    for trajectory_path in trajectory_paths:
        trajectory_path2 = os.path.join(trajectory_dir, trajectory_path)
        params = os.path.splitext(trajectory_path)[0].split("_")
        is_avalibel = int(params[6])

        if is_avalibel:
            shutil.copy(trajectory_path2, common_config.trakcing_insert_dir)
            count += 1

    if len(trajectory_paths) > 0:
        print(f"Trajectory generated successfully, generating {count}/{len(trajectory_paths)} npcs")
        return True
    else:
        return False


def main(data_num, seed_num, ischeck=True, is_delete=False, object_name=None):
    seq = f"{data_num:04d}"
    base_dir = os.path.join(common_config.project_dirname, "TrackGen", "queue", f"{seq}", "_seed_pool",
                            f"{seed_num}")
    inrest_path = os.path.join(base_dir, "npc_insert")
    if ischeck and os.path.exists(inrest_path) and len(os.listdir(inrest_path)) > 0:
        ...
    else:
        if is_delete and os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        run(data_num, [seed_num])

    if check_avalible_trajectory(base_dir, seq):
        perpare(data_num, obj_name=object_name)
        run2(data_num, seed_num)
    else:
        print("Trajectory generation failed")


if __name__ == '__main__':
    data_num = 0
    seed_num = 0
    obj_name = "2e8c4fd40a1be2fa5f38ed4497f2c53c"
    while True:
        basepath = os.path.join(common_config.aug_datasets_dirname,
                                f"task_{data_num:04d}_seed_{seed_num}_trajectories", "KITTI", "training")
        image_path = os.path.join(basepath, "image_02")
        main(data_num, seed_num, ischeck=True, is_delete=False,
             object_name=obj_name)

