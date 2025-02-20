import json
import os.path

import numpy as np
from natsort import natsorted
from scipy.spatial import KDTree

import config.common_config
from Utils_shapenet import filter_cars
from config import common_config
from tracking_uitls.tracking_utils import MyCalibration, Oxts
from utils.Utils_IO import UtilsIO


def cal_rz_degree(insert_track, frame_id, start_frame, pose, calib_info):
    if frame_id - start_frame + 1 == len(insert_track):
        step = -1
    else:
        step = 0

    current_pose = pose.get_current_pose(frame_id)
    current_point_g = insert_track[frame_id - start_frame + step]
    next_point_g = insert_track[frame_id - start_frame + 1 + step]

    pts_global = np.array([current_point_g, next_point_g])
    pts_imu = calib_info.global_to_imu(pts_global, current_pose)
    pts_lidar = calib_info.imu_to_lidar(pts_imu)
    current_point, next_point = pts_lidar
    df = next_point - current_point
    psi = np.arctan2(-df[1], df[0])
    rz_degree = np.rad2deg(psi)

    return current_point, rz_degree


def get_logic_scene_info(vedio_index, data_idx, vehicles_list):
    params_dict = {
        "dataset": "KITTI",
        "task": "tracking",
        "weather": "sunny",
        "sequence": vedio_index,
        "bg_index": data_idx,
        "vehicles": vehicles_list,
        "nums_vehicles": len(vehicles_list)
    }
    return params_dict


def generate_vehicle(obj_idx, obj_name, location, rotation):
    params_dict = {
        "obj_index": obj_idx,
        "obj_name": obj_name,
        "location": location,
        "rotation": rotation,
        "scale": 1
    }
    return params_dict


def get_paths(track_dir):
    path_arr = []
    track_paths = natsorted(os.listdir(track_dir))
    start_frame, end_frame = 0, np.inf
    for temp_path in track_paths:
        params = temp_path.split("_")
        _start_frame, _end_frame = int(params[3]), int(params[4])
        direction = int(params[5])

        start_frame = max(start_frame, _start_frame)
        end_frame = min(end_frame, _end_frame)
        track_path = os.path.join(track_dir, temp_path)

        path_arr.append([track_path, _start_frame, _end_frame, direction])
    assert end_frame > start_frame
    return path_arr, start_frame, end_frame


def split_pc(labels):
    inx_road_arr = []
    inx_other_road_arr = []
    inx_other_ground_arr = []
    inx_no_road_arr = []
    inx_npc_arr = []
    inx_buiding_arr = []

    for i in range(len(labels)):
        lb = labels[i][0]
        if lb == 40:
            inx_road_arr.append(i)
        if lb in (10, 11, 15, 18, 20, 30, 31, 32, 71):
            inx_npc_arr.append(i)
        elif lb == 44:
            inx_other_road_arr.append(i)
        elif lb == 48:
            inx_other_road_arr.append(i)
        elif lb in (49, 70, 72):
            inx_other_ground_arr.append(i)
        elif lb in (50, 80):
            inx_buiding_arr.append(i)
        else:
            inx_no_road_arr.append(i)

    return inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr, inx_npc_arr, inx_buiding_arr


def generate_logic_scene(seq_ix, track_dir, res_dir, back_dir, use_obj_name=None):
    car_type_json_path = os.path.join(config.common_config.project_dirname, "_assets", "car_type.json")
    os.makedirs(res_dir, exist_ok=True)

    sequence = f"{seq_ix:04d}"
    oxts_base_path = os.path.join(config.common_config.oxts_path, f"{sequence}.txt")
    pose = Oxts(oxts_base_path)
    calib_base_path = os.path.join(config.common_config.tracking_dataset_path, "calib_fill", f"{sequence}.txt")
    road_label_dir = os.path.join(config.common_config.tracking_dataset_path, "semantic_label", f"{sequence}")
    road_pc_dir = os.path.join(config.common_config.tracking_dataset_path, "road_pc", f"{sequence}")
    os.makedirs(road_pc_dir, exist_ok=True)
    pc_dir = os.path.join(config.common_config.tracking_dataset_path, "velodyne", f"{sequence}")
    calib_info = MyCalibration(calib_base_path)
    path_arr, start_frame, end_frame = get_paths(track_dir)
    print(start_frame, end_frame)

    obj_idx_arr = list(range(len(path_arr)))
    assert len(path_arr) > 0
    with open(car_type_json_path, "r") as f:
        car_type_dict = dict(json.load(f))
    obj_car_dirs = filter_cars(car_type_dict)
    np.random.seed(None)

    random_ix_arr = np.random.permutation(range(len(obj_car_dirs)))
    for frame_id in range(start_frame, end_frame, 1):
        pc_road_path = os.path.join(road_pc_dir, f"{frame_id:06d}.bin")
        if not os.path.exists(pc_road_path):
            road_label_path = os.path.join(road_label_dir, f"{frame_id:06d}.label")
            pc_path = os.path.join(pc_dir, f"{frame_id:06d}.bin")
            pc_bg = UtilsIO.load_pcd(pc_path)
            labels = np.fromfile(road_label_path, dtype=np.uint32).reshape((-1, 1))
            road_pc_ix, _, _, _, _, _ = split_pc(labels)
            road_pc = pc_bg[road_pc_ix]
            road_pc.tofile(pc_road_path)
        road_pc = np.fromfile(pc_road_path, dtype=np.float32).reshape(-1, 3)
        tree = KDTree(road_pc[:, :2])
        vehicles_list = []
        for oi, (track_path, _, _, direction) in zip(obj_idx_arr, path_arr):

            insert_track = np.load(track_path, allow_pickle=True)
            track_start_frame = int(os.path.basename(track_path).split("_")[3])
            insert_track[:, 2] = 0
            position, rz_degree = cal_rz_degree(insert_track, frame_id, track_start_frame, pose, calib_info)
            if not config.use_height_cal:
                distance, index = tree.query(position[:2], k=1)
                h_final = road_pc[index, 2]
                position[2] = h_final
            else:
                distance, index = tree.query(position[:2], k=30)
                height = road_pc[index, 2]
                median_value = np.median(height)
                distance1, index1 = tree.query(position[:2], k=1)
                height1 = road_pc[index1, 2]
                if distance1 < 0.1 and np.abs(height1 - median_value) <= 0.05:
                    h_final = height1
                else:
                    for h in height:
                        if np.abs(h - median_value) <= 0.05:
                            h_final = h
                            break
            position[2] = h_final

            select_ix = random_ix_arr[oi % len(random_ix_arr)]
            obj_name = obj_car_dirs[select_ix]
            if use_obj_name is not None:
                if isinstance(use_obj_name, list):
                    obj_name = use_obj_name[oi]
                else:
                    obj_name = use_obj_name
            v_info = generate_vehicle(oi, obj_name, list(position), rz_degree)
            vehicles_list.append(v_info)
        logic_scene = get_logic_scene_info(seq_ix, frame_id, vehicles_list)
        res_path = f"{res_dir}/scene_{frame_id}.json"

        json.dump(logic_scene, open(res_path, "w"), indent=4)


if __name__ == '__main__':
    seq_ix = 14

    track_dir = common_config.trakcing_insert_dir
    res_dir = common_config.tracking_logic_scene_dir
    generate_logic_scene(seq_ix, track_dir, res_dir)
