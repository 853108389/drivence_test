import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

from map import RectObstacle
from my_utils import get_car_rect, get_rect_corners


def find_indices(matrix, x, y, z1, z2):
    rows, cols = matrix.shape

    x_min, x_max = max(0, x - z2), min(rows - 1, x + z2)
    y_min, y_max = max(0, y - z2), min(cols - 1, y + z2)

    grid_x, grid_y = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1), indexing='ij')

    distances = np.abs(grid_x - x) + np.abs(grid_y - y)

    mask = (distances >= z1) & (distances <= z2)

    filtered_x = grid_x[mask]
    filtered_y = grid_y[mask]

    indices = np.where(matrix[filtered_x, filtered_y] == 1)

    return filtered_x[indices], filtered_y[indices]


def samplling_oppo(my_map, oppo_lpath_arr, ego_trajectory, length, width, T=0):
    ego_xy, ego_yaw = ego_trajectory[:, [1, 0]], ego_trajectory[:, 2]
    current_ego_car_pos = ego_xy[T]

    path_arr = []
    for path in oppo_lpath_arr:
        path = np.array(path)
        select_pos_start = path[0]
        distance_now = euclidean(select_pos_start, current_ego_car_pos)
        distance_after_1 = np.linalg.norm(select_pos_start - ego_xy[T + 5])
        print("Txxxxx", T, distance_now, distance_after_1)

        if distance_now > distance_after_1 + 1 and 50 < distance_now < 150:
            path_arr.append(path)
            continue

    if len(path_arr) > 0:
        return np.random.choice(path_arr)
    else:
        return None


def samplling_start_multitest(my_map, ego_trajectory, length, width, margin=0, T=0, sample_pos1=None):
    length = length + margin
    width = width + margin
    my_map.render_T(T)

    map_data_start = my_map.data

    ego_xy, ego_yaw = ego_trajectory[:, [1, 0]], ego_trajectory[:, 2]
    current_ego_car_pos = ego_xy[T]

    tree = KDTree(ego_xy)

    dit_th = 10
    flag = False
    recs = []

    select_pos_start_global = None
    if sample_pos1 is None:
        sample_pos = current_ego_car_pos[0], current_ego_car_pos[1]
        sample_dis_max = int(15 / my_map.resolution)
        sample_dis_min = int(6 / my_map.resolution)

    for i in range(200):
        if sample_pos1 is not None:
            sample_dis_max = int(4 + (i / 40) / my_map.resolution)
            sample_dis_min = int(1 + (i / 40) / my_map.resolution)
            sample_pos = sample_pos1

        ego_x, ego_y = my_map.w2m(sample_pos[0], sample_pos[1])

        indices = find_indices(map_data_start, ego_y, ego_x, sample_dis_min, sample_dis_max)
        pos2 = np.array(list(zip(*indices)))
        np.random.seed(None)
        selcet_idx = np.random.choice(len(pos2))
        select_pos_start = pos2[selcet_idx]
        _select_pos_start_global = my_map.m2w_strict(select_pos_start[1], select_pos_start[0])

        distance_now = np.linalg.norm(_select_pos_start_global - current_ego_car_pos)
        distance_after_1 = np.linalg.norm(_select_pos_start_global - ego_xy[T + 10])

        if distance_now < distance_after_1 + 1:
            continue

        distance, index = tree.query(_select_pos_start_global)
        start_psi = ego_yaw[index]

        start_car = get_car_rect(_select_pos_start_global[1], _select_pos_start_global[0], start_psi, length, width)
        corner = get_rect_corners(start_car)

        rectob1 = RectObstacle(corner)
        flag = rectob1.check_map_occupy_cells(my_map)

        if flag:
            recs.append(rectob1)
            select_pos_start_global = _select_pos_start_global
            break

    my_map.clear_map()
    if select_pos_start_global is None:
        return None
    return select_pos_start_global


def samplling_start(my_map, ego_trajectory, length, width, margin=0, T=0):
    length = length + margin
    width = width + margin
    my_map.render_T(T)

    map_data_start = my_map.data

    ego_xy, ego_yaw = ego_trajectory[:, [1, 0]], ego_trajectory[:, 2]
    current_ego_car_pos = ego_xy[T]
    sample_dis_max = int(15 / my_map.resolution)
    sample_dis_min = int(6 / my_map.resolution)
    tree = KDTree(ego_xy)

    dit_th = 10
    flag = False
    recs = []

    select_pos_start_global = None

    for i in range(100):
        pos = np.where(map_data_start == 1)
        pos2 = np.array(list(zip(*pos)))
        selcet_idx = np.random.choice(len(pos2))

        ego_x, ego_y = my_map.w2m(current_ego_car_pos[0], current_ego_car_pos[1])

        indices = find_indices(map_data_start, ego_y, ego_x, sample_dis_min, sample_dis_max)
        pos2 = np.array(list(zip(*indices)))
        selcet_idx = np.random.choice(len(pos2))
        select_pos_start = pos2[selcet_idx]
        _select_pos_start_global = my_map.m2w_strict(select_pos_start[1], select_pos_start[0])

        distance_now = np.linalg.norm(_select_pos_start_global - current_ego_car_pos)
        distance_after_1 = np.linalg.norm(_select_pos_start_global - ego_xy[T + 10])

        if distance_now < distance_after_1 + 1:
            continue

        distance, index = tree.query(_select_pos_start_global)
        start_psi = ego_yaw[index]

        start_car = get_car_rect(_select_pos_start_global[1], _select_pos_start_global[0], start_psi, length, width)
        corner = get_rect_corners(start_car)

        rectob1 = RectObstacle(corner)
        flag = rectob1.check_map_occupy_cells(my_map)

        if flag:
            recs.append(rectob1)
            select_pos_start_global = _select_pos_start_global
            break

    my_map.clear_map()
    if select_pos_start_global is None:
        return None
    return select_pos_start_global


def samplling_start_and_goal(my_map, ego_trajectory, length, width, margin=0, T=0):
    length = length + margin
    width = width + margin
    my_map.render_T(T)

    map_data_start = my_map.data

    ego_xy, ego_yaw = ego_trajectory[:, [1, 0]], ego_trajectory[:, 2]
    current_ego_car_pos = ego_xy[T]
    sample_dis_max = int(10 / my_map.resolution)
    sample_dis_min = int(4 / my_map.resolution)
    tree = KDTree(ego_xy)

    dit_th = 10
    flag = False
    recs = []

    select_pos_start_global = None

    for i in range(10000):
        pos = np.where(map_data_start == 1)
        pos2 = np.array(list(zip(*pos)))
        selcet_idx = np.random.choice(len(pos2))
        select_pos_start = pos2[selcet_idx]

        ego_x, ego_y = my_map.w2m(current_ego_car_pos[0], current_ego_car_pos[1])

        indices = find_indices(map_data_start, ego_y, ego_x, sample_dis_min, sample_dis_max)
        pos2 = np.array(list(zip(*indices)))
        selcet_idx = np.random.choice(len(pos2))
        select_pos_start = pos2[selcet_idx]

        _select_pos_start_global = my_map.m2w_strict(select_pos_start[1], select_pos_start[0])
        distance, index = tree.query(_select_pos_start_global)
        start_psi = ego_yaw[index]

        start_car = get_car_rect(_select_pos_start_global[1], _select_pos_start_global[0], start_psi, length, width)
        corner = start_car.get_corners()
        rectob1 = RectObstacle(corner)
        flag = rectob1.check_map_occupy_cells(my_map)

        if flag:
            recs.append(rectob1)
            select_pos_start_global = _select_pos_start_global
            break

    if select_pos_start_global is None:
        return None

    select_pos_final_global = None

    map_data_final = my_map.ori_data.copy()
    map_data_final[:, select_pos_start[1]:] = 0
    pos = np.where(map_data_final == 1)
    pos2 = np.array(list(zip(*pos)))

    for j in range(10000):
        selcet_idx = np.random.choice(len(pos2))
        select_pos_final = pos2[selcet_idx]

        _select_pos_final_global = my_map.m2w_strict(select_pos_final[1], select_pos_final[0])

        distance = euclidean(_select_pos_final_global, select_pos_start_global)
        if distance < dit_th:
            continue

        distance, index = tree.query(_select_pos_final_global)
        goal_psi = ego_yaw[index]

        goal_car = get_car_rect(_select_pos_final_global[1], _select_pos_final_global[0], goal_psi, length, width)
        corner = goal_car.get_corners()
        my_map.clear_map()
        rectob2 = RectObstacle(corner)
        flag = rectob2.check_map_occupy_cells(my_map)
        if flag:
            recs.append(rectob2)
            select_pos_final_global = _select_pos_final_global
            break

    if select_pos_final_global is None:
        return None

    start_point = np.zeros((3,))
    goal_point = np.zeros((3,))

    start_point[:2] = select_pos_start_global
    start_point[2] = start_psi
    goal_point[:2] = select_pos_final_global
    goal_point[2] = goal_psi
    plt.figure(figsize=(12, 8))
    plt.imshow(np.flipud(my_map.data), cmap='gray',
               extent=[my_map.origin[0], my_map.origin[0] +
                       my_map.width * my_map.resolution,
                       my_map.origin[1], my_map.origin[1] +
                       my_map.height * my_map.resolution], vmin=0.0,
               vmax=1.0)

    return start_point, goal_point


def covert_global_2_planning_coord(my_map, point):
    start_point_map = my_map.w2m(point[0], point[1])
    start_yaw = np.pi / 2 - point[2]
    point_new = [start_point_map[0], start_point_map[1], start_yaw]
    return point_new


def path_planning(path_planning_result_dir, current_insert_index, my_map, ego_trajectory, car_dict, map_dict,
                  margin=0.5, max_num=100, T=0):
    save_dir = os.path.join(path_planning_result_dir, str(current_insert_index))
    os.makedirs(save_dir, exist_ok=True)
    result = None
    for i in range(max_num):

        res = samplling_start_and_goal(my_map, ego_trajectory, car_dict["length"],
                                       car_dict["width"], margin=margin, T=T)
        if res is None:
            return None
        start_point_global, goal_point_global = res

        start = covert_global_2_planning_coord(my_map, start_point_global)
        goal = covert_global_2_planning_coord(my_map, goal_point_global)

        print("path planning start:", start, "goal:", goal)
        res = run_path_planning(start, goal, map_dict["file_path"], save_dir)
        print("path planning result:", res)
        if res:
            x = np.load(os.path.join(save_dir, "x.npy"))
            y = np.load(os.path.join(save_dir, "y.npy"))
            yaw = np.load(os.path.join(save_dir, "yaw.npy"))
            result = x, y, yaw
            break

    return result


def run_path_planning(start, goal, map_temp_path, save_dir):
    cmd = (f"python /Users/gaoxinyu/PycharmProjects/PythonRobotics/PathPlanning/HybridAStar/hybrid_a_star.py "
           f"-start {start[0]} {start[1]} {start[2]} "
           f"-goal {goal[0]} {goal[1]} {goal[2]} "
           f"-map_path {map_temp_path} "
           f"-save_dir {save_dir}")
    res = os.system(cmd)

    if res == 0:
        return True
    else:
        return False
