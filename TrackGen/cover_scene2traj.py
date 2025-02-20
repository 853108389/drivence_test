import os
import shutil
from collections import defaultdict

import numpy as np
from natsort import natsorted

import config
from core.scene_info import SceneInfo
from my_utils2 import Oxts
from tracking_uitls.tracking_utils import MyCalibration

queue_dir = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/_queue/KITTI"


def seed_5():
    tasks = natsorted(os.listdir(queue_dir))
    save_dir = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/TrackGen/queue/{}/_seed_pool/{}/npc_insert"
    for task in tasks:
        if "task_0011" not in task:
            continue
        sequence = task.split("_")[1]
        seed_num = task.split("_")[3]
        oxts_base_path = os.path.join(config.common_config.oxts_path, f"{sequence}.txt")
        calib_base_path = os.path.join(config.common_config.tracking_dataset_path, "calib_fill", f"{sequence}.txt")
        calib_info = MyCalibration(calib_base_path)
        pose = Oxts(oxts_base_path)
        task_dir = os.path.join(queue_dir, task)
        if "seed_5" in task:

            result_dict = defaultdict(list)
            frame_dict = defaultdict(list)
            data_nums = natsorted(os.listdir(task_dir))
            for data_num in data_nums:
                logic_scene_path = os.path.join(task_dir, data_num, "scene_file", "logic_scene_0.json")
                logic_scene = SceneInfo(logic_scene_path)
                data_num = int(data_num)
                print(data_num)

                print(logic_scene_path, sequence, seed_num, data_num)
                current_pose = pose.get_current_pose(data_num + 18)
                for obj in logic_scene.vehicles:
                    obj_index = obj.obj_index
                    if obj_index != 0:
                        continue
                    location = obj.location
                    location = np.array([location])
                    location[:, 2] = location[:, 2] + 20

                    pts_imu = calib_info.lidar_to_imu(location)
                    pts_global = calib_info.imu_to_global(pts_imu, current_pose)
                    loc = calib_info.imu_to_lidar(pts_imu)
                    result_dict[obj_index].append(pts_global[0])
                    frame_dict[obj_index].append(data_num)
            traj_dict = {}
            for k, v in result_dict.items():
                traj_dict[k] = np.array(v)
            for k in traj_dict.keys():
                yaw_arr = []
                for i in range(0, len(traj_dict[k]) - 1):
                    current_point = traj_dict[k][i]
                    next_point = traj_dict[k][i + 1]
                    df = next_point - current_point
                    psi = np.arctan2(df[1], df[0])
                    yaw_arr.append(psi)
                yaw_arr.append(yaw_arr[-1])
                traj_dict[k][:, 2] = yaw_arr

            save_dir2 = save_dir.format(sequence, seed_num)
            if os.path.exists(save_dir2):
                shutil.rmtree(save_dir2)
            if not os.path.exists(save_dir2):
                os.makedirs(save_dir2)

            for k in traj_dict.keys():
                traj = traj_dict[k]
                frame = frame_dict[k]
                start_frame = frame[0]
                end_frame = frame[-1] + 1
                np.save(os.path.join(save_dir2, f"{k}_ID_1_{start_frame}_{end_frame}_-1_1.npy"), traj)
                print("XXXX", os.path.join(save_dir2, f"{k}_ID_1_{start_frame}_{end_frame}_-1_1.npy"))
            assert 1 == 2


def seed_26():
    tasks = natsorted(os.listdir(queue_dir))
    save_dir = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/TrackGen/queue/{}/_seed_pool/{}/npc_insert"
    for task in tasks:
        if "task_0011" not in task:
            continue
        sequence = task.split("_")[1]
        seed_num = task.split("_")[3]
        oxts_base_path = os.path.join(config.common_config.oxts_path, f"{sequence}.txt")
        calib_base_path = os.path.join(config.common_config.tracking_dataset_path, "calib_fill", f"{sequence}.txt")
        calib_info = MyCalibration(calib_base_path)
        pose = Oxts(oxts_base_path)
        task_dir = os.path.join(queue_dir, task)
        if "seed_26" in task:
            result_dict = defaultdict(list)
            frame_dict = defaultdict(list)
            data_nums = natsorted(os.listdir(task_dir))
            for data_num in data_nums:
                if int(data_num) > 72 or int(data_num) < 17:
                    continue
                logic_scene_path = os.path.join(task_dir, data_num, "scene_file", "logic_scene_0.json")
                logic_scene = SceneInfo(logic_scene_path)
                data_num = int(data_num)
                print(data_num)
                print(logic_scene_path, sequence, seed_num, data_num)
                current_pose = pose.get_current_pose(data_num)
                for obj in logic_scene.vehicles:
                    obj_index = obj.obj_index
                    location = obj.location
                    location = np.array([location])
                    location[:, 2] = location[:, 2]

                    pts_imu = calib_info.lidar_to_imu(location)
                    pts_global = calib_info.imu_to_global(pts_imu, current_pose)
                    result_dict[obj_index].append(pts_global[0])
                    frame_dict[obj_index].append(data_num)
            traj_dict = {}
            for k, v in result_dict.items():
                traj_dict[k] = np.array(v)
            for k in traj_dict.keys():
                yaw_arr = []
                for i in range(0, len(traj_dict[k]) - 1):
                    current_point = traj_dict[k][i]
                    next_point = traj_dict[k][i + 1]
                    df = next_point - current_point
                    psi = np.arctan2(df[1], df[0])
                    yaw_arr.append(psi)
                yaw_arr.append(yaw_arr[-1])
                traj_dict[k][:, 2] = yaw_arr

            save_dir2 = save_dir.format(sequence, seed_num)
            if os.path.exists(save_dir2):
                shutil.rmtree(save_dir2)
            if not os.path.exists(save_dir2):
                os.makedirs(save_dir2)

            for k in traj_dict.keys():
                traj = traj_dict[k]
                frame = frame_dict[k]
                start_frame = frame[0]
                end_frame = frame[-1] + 1
                np.save(os.path.join(save_dir2, f"{k}_ID_1_{start_frame}_{end_frame}_-1_1.npy"), traj)
                print("XXXX", os.path.join(save_dir2, f"{k}_ID_1_{start_frame}_{end_frame}_-1_1.npy"))


if __name__ == '__main__':
    seed_26()
