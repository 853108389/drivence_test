from collections import defaultdict

import numpy as np

from TrackGen.main import car_dict, init_data_path, init_seed_path
from TrackGen.my_utils import load_background_npcs, get_ego_trajectory, load_ego_car, load_inserted_npcs


def cal_yaw_changed(car_trajectory):
    if len(car_trajectory) < 10:
        return None

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


def calculate_velocity(position, time_interval):
    return np.diff(position, axis=0) / time_interval


def calculate_acceleration(velocity, time_interval):
    return np.diff(velocity, axis=0) / time_interval


def calculate_jerk(acceleration, time_interval):
    return np.diff(acceleration, axis=0) / time_interval


def cal_length(car_trajectory):
    car_trajectory = car_trajectory[car_trajectory[:, 0] != 0][:, :2]
    return len(car_trajectory)


def cal_jerk(car_trajectory):
    if len(car_trajectory) < 10:
        return None
    time = 0.2
    car_trajectory = np.array(car_trajectory)
    car_trajectory = car_trajectory[car_trajectory[:, 0] != 0][:, :2]

    dis_arr = []
    velocity = calculate_velocity(car_trajectory, time)
    acceleration = calculate_acceleration(velocity, time)
    jerk = calculate_jerk(acceleration, time)
    jerk = np.mean(jerk, axis=0)

    return np.linalg.norm(jerk)


if __name__ == '__main__':

    data_dict = defaultdict(list)
    for data_num in [0, 2, 5, 7, 10, 11, 14, 18]:
        ego_path, npc_path = init_data_path(data_num)
        background_npcs = load_background_npcs(npc_path)

        back_yaw_arr = []
        back_jerk_arr = []
        t_length_arr = []

        for npc in background_npcs:

            if npc.type == "static":
                continue
            jerk = cal_jerk(npc.car_trajectory)

            back_jerk_arr.append(np.mean(jerk))
            t_length = cal_length(npc.car_trajectory)
            _, mean_changed = cal_yaw_changed(npc.car_trajectory)
            if mean_changed is not None:
                back_yaw_arr.append(mean_changed)
                t_length_arr.append(t_length)

        ego_trajectory = get_ego_trajectory(ego_path)
        _, mean_ego_yaw_changed = cal_yaw_changed(ego_trajectory)
        jerk = cal_jerk(ego_trajectory)

        back_yaw_arr.append(np.mean(mean_ego_yaw_changed))
        back_jerk_arr.append(np.mean(jerk))

        mean_yaw_arr2 = []
        t_length_arr2 = []
        mean_jerk_arr2 = []
        npc_ego = load_ego_car(ego_trajectory, car_dict["length"], car_dict["width"])
        max_frames = len(npc_ego.npc_trajectory_box)

        for seed_num in range(0, 30):
            npc_inserted_dir, path_planning_result_dir, _ = init_seed_path(data_num, seed_num)
            inserted_npcs, _ = load_inserted_npcs(npc_inserted_dir, car_dict["length"], car_dict["width"], max_frames)
            for npc in inserted_npcs:
                jerk = cal_jerk(npc.car_trajectory)

                mean_jerk_arr2.append(np.mean(jerk))
                _, mean_insert_chagned = cal_yaw_changed(npc.car_trajectory)
                mean_yaw_arr2.append(mean_insert_chagned)
                t_length_arr2.append(cal_length(npc.car_trajectory))

        data_dict["data_num"].append(data_num)
        data_dict["back_yaw"].append(np.mean(back_yaw_arr))
        data_dict["inserted_yaw"].append(np.mean(mean_yaw_arr2))
        data_dict["back_jerk"].append(np.mean(back_jerk_arr))
        data_dict["inserted_jerk"].append(np.mean(mean_jerk_arr2))
        data_dict["back_length"].append(np.mean(t_length_arr))
        data_dict["inserted_length"].append(np.mean(t_length_arr2))
    import pandas as pd

    print(data_dict)
    pd.DataFrame(data_dict).to_csv("yaw_jerk.csv", index=False)
