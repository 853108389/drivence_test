import os

import numpy as np
from natsort import natsorted
from scipy.spatial.distance import euclidean

from map import NPC, Map, get_box_center2D


def check_trajectory_intersect(trajectory1, trajectory2):
    for t, (rect1, rect2) in enumerate(zip(trajectory1, trajectory2)):
        if np.sum(rect1) == 0 or np.sum(trajectory2) == 0:
            continue
        if check_box_intersect(rect1, rect2):
            print("intersect time", t, rect1, rect2)
            return True
    return False


def check_box_intersect(rect1, rect2):
    from shapely.geometry import Polygon
    rect1_shape = Polygon(rect1)
    rect2_shape = Polygon(rect2)
    return rect1_shape.intersects(rect2_shape)


def load_map(map_dict):
    file_path, origin, resolution = map_dict["file_path"], map_dict["origin"], map_dict["resolution"]
    my_map = Map(file_path=file_path, origin=origin, resolution=resolution)
    return my_map


def filter_path(x, y, yaw, th=1):
    count = 0
    flag = True
    while flag:
        flag2 = False

        for i in range(len(x) - 1):
            if abs(x[i] - x[i - 1]) < th and abs(y[i] - y[i - 1]) < th:
                flag2 = True
                break
        if flag2:
            count += 1
            x = np.delete(x, i)
            y = np.delete(y, i)
            yaw = np.delete(yaw, i)
        else:
            flag = False

    return x, y, yaw


def get_ego_trajectory(ego_path):
    ego_trajectory = np.load(ego_path)

    wp_xs, wp_ys = ego_trajectory[:, 0], ego_trajectory[:, 1]
    waypoints = list(zip(wp_xs, wp_ys))
    yaw_arr = []
    for wp_id in range(len(waypoints) - 1):
        current_wp = np.array(waypoints[wp_id])
        next_wp = np.array(waypoints[wp_id + 1])
        dif_ahead = next_wp - current_wp

        psi = np.arctan2(dif_ahead[1], dif_ahead[0])
        yaw_arr.append(psi)
    yaw_arr.append(yaw_arr[-1])
    yaw_arr = np.array(yaw_arr)
    ego_trajectory[:, 2] = yaw_arr
    assert len(yaw_arr) == len(waypoints)
    return ego_trajectory


def load_ego_car(ego_trajectory, length, width):
    npc_trajectories_boxes = consturct_npc_track_by_mpc_result(ego_trajectory, len(ego_trajectory), length, width, 0)

    npc = NPC(npc_trajectories_boxes)
    npc.car_trajectory = ego_trajectory
    npc.color = "red"
    npc.is_insert = False
    npc.is_ego = True
    npc.type = "dynamic"
    return npc


def get_start_end_frame(npc_inserted_path, max_frames):
    start_frame = 0
    end_frame = max_frames
    npc_inserted_list = natsorted(os.listdir(npc_inserted_path))
    for npc_path in npc_inserted_list:
        if npc_path == ".DS_Store":
            continue
        params = os.path.splitext(npc_path)[0].split("_")
        is_avalibel = params[6]
        if not int(is_avalibel):
            continue

        _start_frame = int(params[3])
        _end_frame = int(params[4])
        start_frame = max(start_frame, _start_frame)
        end_frame = min(end_frame, _end_frame)
    return start_frame, end_frame


def cal_v_on_trajectory(x, y, Ts):
    points = np.column_stack((x, y))

    distances = [euclidean(points[i], points[i + 1]) for i in range(len(points) - 1)]

    velocities = np.array(distances) / Ts

    velocities = np.append(velocities, velocities[-1])

    return velocities


def load_inserted_npcs(npc_inserted_path, length, width, max_frames):
    npc_arr = []
    count = 0
    npc_inserted_list = natsorted(os.listdir(npc_inserted_path))
    for npc_path in npc_inserted_list:
        if npc_path == ".DS_Store":
            continue
        count += 1
        params = os.path.splitext(npc_path)[0].split("_")
        is_avalibel = params[6]
        if not int(is_avalibel):
            continue
        start_frame = int(params[3])
        npc_path2 = os.path.join(npc_inserted_path, npc_path)
        car_trajectory = np.load(npc_path2)

        npc_trajectories_boxes = consturct_npc_track_by_mpc_result(car_trajectory, max_frames, length, width,
                                                                   start_t=start_frame)

        npc = NPC(npc_trajectories_boxes)
        npc.car_trajectory = car_trajectory
        npc.color = "green"
        npc_arr.append(npc)
        npc.is_insert = True
        npc.file_path = npc_path2
        if "IS" in npc_path:
            npc.type = "static"
        else:
            npc.type = "dynamic"
        npc.direction = 1
        speed_mod = int(params[2])
        npc.speed_mod = speed_mod

    return npc_arr, count


def convert_corners_2_centers(npc_trajectories_box):
    center_arr = []
    for corner in npc_trajectories_box:
        center = get_box_center2D(corner)
        center_arr.append(center)

    center_arr = np.array(center_arr)
    yaw_arr = []
    for i in range(len(center_arr) - 1):
        dif_ahead = center_arr[i + 1] - center_arr[i]

        dist_ahead = np.linalg.norm(dif_ahead, 2)

        psi = np.arctan2(dif_ahead[1], dif_ahead[0])
        yaw_arr.append(psi)
    yaw_arr.append(yaw_arr[-1])
    yaw_arr = np.array(yaw_arr).reshape(-1, 1)
    res_arr = np.concatenate([center_arr, yaw_arr], axis=1)
    return res_arr


def find_distance(arr):
    non_zero_rows = np.all(arr != 0, axis=1)

    first_non_zero_idx = np.argmax(non_zero_rows)
    last_non_zero_idx = len(non_zero_rows) - np.argmax(non_zero_rows[::-1]) - 1

    first_row = arr[first_non_zero_idx]
    last_row = arr[last_non_zero_idx]

    distance = np.linalg.norm(first_row - last_row)

    return distance


def load_background_npcs(npc_path):
    npc_arr = []
    npc_trajectories_boxes = np.load(npc_path)
    npc_trajectories_boxes2d = np.zeros((npc_trajectories_boxes.shape[0], npc_trajectories_boxes.shape[1], 4, 2))

    for nid, npc_trajectories_box in enumerate(npc_trajectories_boxes):
        for tid in range(npc_trajectories_box.shape[0]):
            corner = npc_trajectories_box[tid]
            if (corner == 0).any():
                corner2d = np.zeros((4, 2))
            else:
                corner2d = convert_3dbox_2bevbox(corner)
            npc_trajectories_boxes2d[nid, tid] = corner2d

    for nid in range(npc_trajectories_boxes2d.shape[0]):
        trajectory_box = npc_trajectories_boxes2d[nid]
        trajectory = convert_corners_2_centers(trajectory_box)
        npc = NPC(trajectory_box)
        npc.is_insert = False
        npc.id = nid
        npc.car_trajectory = trajectory
        npc_arr.append(npc)

        distance = find_distance(trajectory[:, :2])
        if distance < 3:
            npc.type = "static"
        else:
            npc.type = "dynamic"

    return npc_arr


def convert_3dbox_2bevbox(corner):
    z_mean = np.mean(corner[:, 2])
    top_points = corner[corner[:, 2] > z_mean]
    bottom_points = corner[corner[:, 2] <= z_mean]
    if len(top_points) != 4 or len(bottom_points) != 4:
        raise ValueError("corner should have 8 points")
    corner_2d = top_points[:, :2]
    return corner_2d


def get_rect_corners(rect):
    transformed_corners = rect.get_patch_transform().transform(
        [(0, 0), (1, 0), (1, 1), (0, 1)])
    return transformed_corners


def consturct_npc_track_by_mpc_result(car_trajectory, max_frames, length, width, start_t=0):
    corners = np.zeros((max_frames, 4, 2))
    flag = False
    for i, (x, y, yaw) in enumerate(car_trajectory):
        assert not (x == 0 and y == 0)
        if start_t + i >= len(corners):
            break
        car = get_car_rect(x, y, yaw, length, width)

        corner = get_rect_corners(car)

        corners[start_t + i] = corner

    return corners


def get_car_rect(x, y, psi, length, width):
    import matplotlib.patches as plt_patches
    CAR = '
    CAR_OUTLINE = '
    cog = (x, y)

    yaw = np.rad2deg(psi)

    car = plt_patches.Rectangle(cog, width=length, height=width,
                                angle=yaw, facecolor=CAR,
                                edgecolor=CAR_OUTLINE)

    car.set_x(car.get_x() - (length / 2 * np.cos(psi) - width / 2 * np.sin(psi)))
    car.set_y(car.get_y() - (width / 2 * np.cos(psi) + length / 2 * np.sin(psi)))
    return car
