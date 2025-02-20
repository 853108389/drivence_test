import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

current_directory = Path(__file__).parent

sys.path.append(str(current_directory))
from my_utils import (load_background_npcs, load_map, load_inserted_npcs,
                      load_ego_car, \
                      get_ego_trajectory, get_start_end_frame)

from mutation import Mutation


def get_map_dict(data_num):
    resolution = 0.1
    file_path = f"{BASE_PATH}/kitti_map/{data_num:04d}_{resolution}.png"

    map_params = {
        0: {
            "origin": [-80, -10],
            "waypoint": (
                [(-10.2, 5.6), (-15.7, 15.2), (-20.0, 24.2),
                 (-30.5, 29.2), (-40.7, 29.4), (-50.3, 28.8), (-62, 31), (-70, 33)],
                [(-10.2, 5.6), (-11.0, 6.5), (-18, 20), (-19.1, 23.2), (-18, 30.5), (-15.5, 33)]
            ),
            "default_speed_level": 0
        },
        2: {
            "origin": [-10, -80],
            "waypoint": (
                [(10.5, -11.5), (20.7, -22.2), (30.7, -32.6), (40.8, -42.3), (50.2, -50.5), (60.0, -58.1),
                 (70.9, -66.1), (80.2, -72.5)],
            ),
            "oppo_lane_wp": (
                [(65, -75), (40, -55), (0, -15)],
                [(43, -57), (32, -50), (11.7, -31.3), (0, -15)],
                [(54, -67), (40, -55), (32, -50), (11.7, -31.3), (0, -15)],
                [(40, -55), (20, -40), (32, -50), (11.7, -31.3), (0, -15)],
            ),
            "default_speed_level": 2
        },
        5: {
            "origin": [0, 0],
            "waypoint": (
                [(50, 48), (100, 92), (145, 129), (199, 168), (260, 210)],
            ),
            "oppo_lane_wp": (
                [(265, 212), (229, 184), (176, 146), (100, 87), (47, 39)],
                [(241, 193), (229, 184), (176, 146), (100, 87), (47, 39)],
                [(176, 146), (100, 87), (47, 39)],
                [(148, 123), (100, 87), (47, 39)],
                [(100, 87), (47, 39)],
                [(64, 52), (47, 39)],
                [(53, 44), (47, 39)],
            ),
            "default_speed_level": 2
        },
        7: {
            "origin": [0, -185],
            "waypoint": (
                [(13, -23), (13, -50), (14.5, -75), (14.5, -125), (14, -150),
                 (15, -169), (17.8, -175), (25, -180),
                 (64, -176), (64, -178), (74, -161),
                 (74, -120), (74, -100), (74, -75), (73, -50), (72.5, -25), (71.7, -6.4),
                 (73, 6), (81, 8.7), (96, 12), (113, 16), (126, 20), (135, 16)],
            ),
            "default_speed_level": 1,
            "segment": [
                (40, 230),
                (230, 360),
                (360, 604),

            ]
        },
        10: {
            "origin": (-385, 0),
            "waypoint": (
                [(-57.9, 25.4), (-106.5, 50.6), (-154.0, 75.6), (-199.5, 100.6), (-246.3, 125.3), (-305.0, 150.4)],
            ),
            "default_speed_level": 1,
            "oppo_lane_wp": (
                [(-304, 156), (-246, 131), (-199, 107), (-155, 82), (-100, 55), (-51, 31), (-29, 23)],
                [(-246, 131), (-199, 107), (-155, 82), (-100, 55), (-51, 31), (-29, 23)],
                [(-199, 107), (-155, 82), (-100, 55), (-51, 31), (-29, 23)],
                [(-155, 82), (-100, 55), (-51, 31), (-29, 23)],
                [(-130, 72), (-51, 31), (-29, 23)],
                [(-120, 65), (-51, 31), (-29, 23)],
                [(-100, 55), (-51, 31), (-29, 23)],
                [(-75, 42), (-51, 31), (-29, 23)]
            ),
        },
        11: {
            "origin": (-100, 0),
            "waypoint": (
                [(-13.5, 25.2), (-26.5, 50.2), (-39.3, 75.7), (-52.1, 100.7), (-64.7, 125.1), (-77.3, 150.3),
                 (-89.3, 175.5)],
            ),
            "default_speed_level": 2,
            "oppo_lane_wp": (
                [(-75.9, 164.4), (-51.1, 118.2), (-23.2, 62.9), (-5.8, 25.9), (-1.5, 19.1)],
                [(-60.1, 133.5), (-51.1, 118.2), (-23.2, 62.9), (-5.8, 25.9), (-1.5, 19.1)],
                [(-51.1, 118.2), (-23.2, 62.9), (-5.8, 25.9), (-1.5, 19.1)],
                [(-33.7, 81.8), (-23.2, 62.9), (-5.8, 25.9), (-1.5, 19.1)],
                [(-16.3, 48.6), (-5.8, 25.9), (-1.5, 19.1)],
            ),
        },
        14: {
            "origin": (-40, 0),
            "waypoint": [
                [(0.4, 6.0), (0.3, 10.3), (-2.8, 15.0), (-5.8, 16.6), (-10.5, 17.5), (-15.4, 17.9), (-20.1, 18.1),
                 (-25.1, 18.3), (-30.6, 18.5), (-35.5, 18.7), (-38.7, 18.7)],
            ],
            "default_speed_level": 2,
            "oppo_lane_wp": (
                [(-35.5, 21), (-30.6, 20.8), (-25.1, 20.6), (-20.1, 20.4), (-15.4, 20.2), (-10.5, 20.2), (-5.8, 20.2),
                 (0, 20.2), (1.5, 20.2)],
                [(-15.4, 20.2), (-10.5, 20.2), (-5.8, 20.2), (0, 20.2), (1.5, 20.2)],

                [(-5.8, 20.2), (0, 20.2), (1.5, 20.2)],
                [(-30.6, 20.8), (-25.1, 20.6), (-20.1, 20.4), (-15.4, 20.2), (-10.5, 19.8), (-5.8, 18.7),
                 (-2.8, 17), (1.5, 10), (2, 8.5)],
                [(-20.1, 20.4), (-10.5, 19.8), (-5.8, 18.7), (-2.8, 17), (1.5, 10), (2, 8.5)],

            ),
            "segment": [(0, 105)]
        },
        18: {
            "origin": (0, 0),
            "waypoint": (
                [(36.5, 25.7), (69.2, 50.9), (97.7, 75.2), (98.6, 76.0), (124.9, 100.3), (125.4, 100.7), (150.2, 125.3),
                 (150.5, 125.7), (150.8, 126.0), (173.6, 150.5), (174.1, 151.0)],

            ),
            "default_speed_level": 2,
            "oppo_lane_wp": (
                [(182.3, 150.4), (173.1, 139.8), (158.4, 125.6), (139.1, 107.2), (111.5, 80.5), (84.9, 56.2),
                 (66.9, 42.4), (49.5, 29.1), (29.3, 14.0), (15.0, 3.8)],
                [(158.4, 125.6), (139.1, 107.2), (111.5, 80.5), (84.9, 56.2), (66.9, 42.4), (49.5, 29.1),
                 (29.3, 14.0), (15.0, 3.8)],
                [(139.1, 107.2), (111.5, 80.5), (84.9, 56.5), (66.9, 42.4), (49.5, 29.1), (29.3, 14.0),
                 (15.0, 3.8)],
                [(111.5, 80.5), (84.9, 56.5), (66.9, 42.4), (49.5, 29.1), (29.3, 14.0), (15.0, 3.8)],
                [(84.9, 56.5), (66.9, 42.4), (49.5, 29.1), (29.3, 14.0), (15.0, 3.8)],
                [(66.9, 42.4), (49.5, 29.1), (29.3, 14.0), (15.0, 3.8)],
                [(56.8, 34.1), (49.5, 29.1), (29.3, 14.0), (15.0, 3.8)],
                [(49.5, 29.1), (29.3, 14.0), (15.0, 3.8)],
                [(37.5, 19.9), (29.3, 14.0), (15.0, 3.8)],
                [(29.3, 14.0), (15.0, 3.8)],
            ),
        },
    }

    map_dict = map_params[data_num]
    map_dict["file_path"] = file_path
    map_dict["resolution"] = resolution

    return map_dict


def get_car_dict(use_safe=True):
    safety_width_margin = 0.4
    safety_length_margin = 1
    if not use_safe:
        safety_width_margin = 0
        safety_length_margin = 0
    car_dict = {
        "length": 4.8 + safety_length_margin,
        "width": 1.8 + safety_width_margin,
        "ts": 0.1,

    }
    return car_dict


car_dict = get_car_dict()

from config import common_config

BASE_PATH = os.path.join(common_config.project_dirname, "TrackGen")


def calculate_front_rear_coordinates(x, y, yaw, length):
    yaw_rad = math.radians(yaw)

    x_front = x + length / 2 * math.cos(yaw_rad)
    y_front = y + length / 2 * math.sin(yaw_rad)

    x_rear = x - length / 2 * math.cos(yaw_rad)
    y_rear = y - length / 2 * math.sin(yaw_rad)

    return (x_front, y_front), (x_rear, y_rear)


def get_stastic_map(npcs, max_t, map_dict):
    my_map = load_map(map_dict)
    data = my_map.data
    for npc in npcs:
        for t in range(max_t):
            rect = npc.get_npc_obstacle_in_t(t)
            if rect is not None:
                data = rect.update_map_occupy_cells(my_map, 0, data)
    my_map.data = data
    return my_map


def init_data_path(data_num):
    ego_path = f"{BASE_PATH}/queue_RQ1/{data_num:04d}/npc/ego_trajectory.npy"
    npc_path = f"{BASE_PATH}/queue_RQ1/{data_num:04d}/npc/background_npc_trajectory.npy"
    return ego_path, npc_path


def init_seed_path(data_num, seed_num):
    base_path = f"{BASE_PATH}/queue_RQ1/{data_num:04d}/_seed_pool/{seed_num}"
    os.makedirs(base_path, exist_ok=True)
    npc_inserted_dir = f"{base_path}/npc_insert"
    path_planning_result_dir = f"{base_path}/planning_result"
    df_result_dir = f"{base_path}/result"
    mpc_result_dir = f"{base_path}/mpc_result"

    os.makedirs(npc_inserted_dir, exist_ok=True)
    os.makedirs(path_planning_result_dir, exist_ok=True)
    os.makedirs(mpc_result_dir, exist_ok=True)
    os.makedirs(df_result_dir, exist_ok=True)
    return npc_inserted_dir, path_planning_result_dir, mpc_result_dir, df_result_dir


def replay(data_num, seed_num, car_dict=None, is_break=True):
    if car_dict is None:
        car_dict = get_car_dict()
    ego_path, npc_path = init_data_path(data_num)
    map_dict = get_map_dict(data_num)
    my_map = load_map(map_dict)
    background_npcs = load_background_npcs(npc_path)
    ego_trajectory = get_ego_trajectory(ego_path)

    npc_ego = load_ego_car(ego_trajectory, car_dict["length"], car_dict["width"])
    max_frames = len(npc_ego.npc_trajectory_box)

    npc_inserted_dir, path_planning_result_dir, _, _ = init_seed_path(data_num, seed_num)
    start_frame, end_frame = get_start_end_frame(npc_inserted_dir, max_frames)

    inserted_npcs, _ = load_inserted_npcs(npc_inserted_dir, car_dict["length"], car_dict["width"], max_frames)
    npcs = [npc_ego] + background_npcs + inserted_npcs

    my_map.add_npcs(npcs)

    my_map.reset_npc(start_frame)

    print("replay", "start_frame", start_frame, "end_frame", end_frame)

    for i in range(start_frame, end_frame):
        my_map.render_T(i)

        plt.imshow(np.flipud(my_map.data), cmap='gray',
                   extent=[my_map.origin[0], my_map.origin[0] +
                           my_map.width * my_map.resolution,
                           my_map.origin[1], my_map.origin[1] +
                           my_map.height * my_map.resolution], vmin=0.0,
                   vmax=1.0)
        for npc in my_map.npcs:
            npc.t = i
            npc.show()
        plt.title(f'replay {data_num} : {seed_num} frame {i}')

        plt.pause(0.1)

        plt.clf()
        my_map.clear_map()
    if is_break:
        exit(-1)


def cal_yaw(car_trajectory):
    car_trajectory = np.array(car_trajectory)
    car_trajectory[:, [0, 1]] = car_trajectory[:, [1, 0]]
    yaw_arr = []
    for wp_id in range(0, len(car_trajectory) - 1):
        current_wp = np.array(car_trajectory[wp_id])
        next_wp = np.array(car_trajectory[wp_id + 1])
        dif_ahead = next_wp - current_wp
        psi = np.arctan2(dif_ahead[1], dif_ahead[0])
        yaw_arr.append(psi)
    yaw_arr.append(yaw_arr[-1])
    car_trajectory_new = car_trajectory
    yaw_arr = np.array(yaw_arr).reshape(-1, 1)
    print(car_trajectory_new.shape)
    print(yaw_arr.shape)
    car_trajectory_new = np.hstack((car_trajectory_new, yaw_arr))
    return car_trajectory_new


def cal_yaw_changed(car_trajectory):
    yaw_arr = []
    car_trajectory = np.array(car_trajectory)
    car_trajectory = car_trajectory[car_trajectory[:, 0] != 0][:, :2]

    dis_arr = []
    for wp_id in range(1, len(car_trajectory) - 1, 5):
        current_wp = np.array(car_trajectory[wp_id])

        if wp_id + 5 >= len(car_trajectory):
            break
        next_wp = np.array(car_trajectory[wp_id + 5])
        dif_ahead = next_wp - current_wp
        dis = np.linalg.norm(dif_ahead)
        if dis < 0.5:
            continue

        psi = np.arctan2(dif_ahead[1], dif_ahead[0])
        psi = psi % (2 * np.pi)
        yaw_arr.append(psi)
        dis_arr.append(dis)

    if len(yaw_arr) == 0 or len(yaw_arr) == 1:
        return None, None
    yaw_changed_arr = np.diff(yaw_arr)
    yaw_changed_arr = (yaw_changed_arr + np.pi) % (2 * np.pi) - np.pi

    yaw_changed_arr = abs(yaw_changed_arr)

    mean_yaw = np.mean(yaw_changed_arr)
    mean_yaw = np.degrees(mean_yaw)

    return yaw_changed_arr, mean_yaw


def run(data_num, seed_nums, res_dict=None):
    car_dict = get_car_dict(use_safe=False)
    assert len(seed_nums) == 1

    ego_path, npc_path = init_data_path(data_num)
    map_dict = get_map_dict(data_num)

    ego_trajectory = get_ego_trajectory(ego_path)
    max_frames = len(ego_trajectory)

    for seed_num in seed_nums:
        npc_inserted_dir, path_planning_result_dir, _, df_result_dir = init_seed_path(data_num, seed_num)
        print("save res_dict")
        start_frame_all, end_frame_all = 0, max_frames
        start_frame_all = np.random.randint(0, max_frames - 50)
        print(start_frame_all)

        np.random.seed(None)
        print("seed_num:", seed_num, "max_frames", max_frames)
        trajectory = []
        while True:
            for start_frame in tqdm(range(start_frame_all, start_frame_all + 40)):
                print("start_frame", start_frame)
                my_map = load_map(map_dict)

                background_npcs = load_background_npcs(npc_path)
                ego_trajectory = get_ego_trajectory(ego_path)
                npc_ego = load_ego_car(ego_trajectory, car_dict["length"], car_dict["width"])
                max_frames = len(npc_ego.npc_trajectory_box)

                inserted_npcs, count = load_inserted_npcs(npc_inserted_dir, car_dict["length"], car_dict["width"],
                                                          max_frames)
                current_insert_index = count
                npcs = [npc_ego] + background_npcs + inserted_npcs

                for i, npc in enumerate(npcs):
                    npc.id = i

                my_map.add_npcs(npcs)

                my_map.reset_npc(start_frame)
                static_map = get_stastic_map(npcs, max_frames, map_dict)

                mutation = Mutation(my_map, car_dict, map_dict, max_frames, frame_th=20)
                mutation.set_ego_trajectory(ego_trajectory)
                mutation.set_static_map(static_map)
                mutation.set_path_planning_result_dir(path_planning_result_dir)
                if start_frame == start_frame_all:
                    start_pos = None
                else:
                    start_pos = trajectory[-1]

                res = mutation.multitest_sample(start_frame, start_pos)
                if res is None:
                    print(trajectory)
                    print("res", len(trajectory))
                    break
                else:
                    trajectory.append(res)
            if len(trajectory) <= 20:
                continue
            trajectory = cal_yaw(trajectory)
            break

        print(trajectory.shape)

        if len(trajectory) >= 1:
            np.save(
                f"{npc_inserted_dir}/{current_insert_index}_MT_{1}_{0}_{len(trajectory)}_{1}_{1}.npy",
                trajectory)


if __name__ == '__main__':
    replay(0, 1013, car_dict)
