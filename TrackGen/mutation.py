import random

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import euclidean

from TrackGen.path_planning import samplling_start_multitest
from frenet_optimal_trajectory import trajectory_planning, get_refrence_path, sampling_way_points
from map import Map, RectObstacle
from my_utils import get_car_rect, cal_v_on_trajectory, get_rect_corners


class Mutation(object):
    def __init__(self, my_map: Map, car_dict, map_dict, max_frames, frame_th=20):
        self.car_dict = car_dict
        self.car_length = car_dict["length"]
        self.car_width = car_dict["width"]
        self.max_frames = max_frames
        self.map_dict = map_dict
        self.my_map = my_map
        self.default_speed = map_dict["default_speed_level"]
        self.Ts = 0.2
        self.frame_th = frame_th

        ...

        ...
        self.ego_trajectory = None
        self.static_map = None

        self.path_planning_result_dir = None

    def get_mapdata_arr(self):
        my_map = self.my_map
        my_mapdata_arr = []
        for i in range(self.max_frames):
            my_map.render_T(i)
            my_mapdata_arr.append(my_map.data)
            my_map.clear_map()
        return my_mapdata_arr

    def set_path_planning_result_dir(self, path_planning_result_dir):
        self.path_planning_result_dir = path_planning_result_dir

    def set_ego_trajectory(self, ego_trajectory):
        self.ego_trajectory = ego_trajectory

    def set_static_map(self, static_map):
        self.static_map = static_map

    def get_all_mutators(self, insert_num=0):
        if insert_num == 0:
            return {

                "ID": self.insert_dynamic_car,
                "LD": self.leading,
                "FL": self.following,
            }
        else:
            return {

                "ID": self.insert_dynamic_car,
                "SU": self.speed_up,
                "CW": self.change_waypoints,
                "SD": self.speed_down,
                "FL": self.following,
                "LD": self.leading
            }

    def change_waypoints(self, start_frame, end_frame, current_insert_index, sample_size=3):
        select_idx = []
        for i, npc in enumerate(self.my_map.npcs):
            if npc.type == "dynamic" and npc.is_insert:
                select_idx.append(i)
        if len(select_idx) == 0:
            print("warning no npc can adjust waypoints")
            return None
        select_npc_idx = np.random.choice(select_idx)
        select_npc = self.my_map.npcs[select_npc_idx]
        self.my_map.npcs.remove(select_npc)
        my_mapdata_arr = self.get_mapdata_arr()

        x, y, yaw = select_npc.car_trajectory[:, 0], select_npc.car_trajectory[:, 1], select_npc.car_trajectory[:, 2]

        for i in range(5):

            if len(x) > sample_size:
                sample_idxes = random.sample(range(1, len(x) - 2), sample_size)
                sample_idxes = sorted(sample_idxes)
                sample_idxes.insert(0, 0)
                sample_idxes.append(len(x) - 1)
                wx, wy = x[sample_idxes], y[sample_idxes]
            else:
                wx, wy = x, y
            wx, wy = wy, wx

            wx += np.random.normal(0, 0.1, len(wx))
            wy += np.random.normal(0, 0.1, len(wy))

            reference_path = get_refrence_path(wx, wy)
            if reference_path is None:
                continue

            trajectory = trajectory_planning(reference_path, start_frame, end_frame, self.my_map, self.car_dict,
                                             self.max_frames,
                                             my_mapdata_arr, speed_level=self.default_speed, frame_th=self.frame_th)
            self.my_map.npcs.insert(select_npc_idx, select_npc)
            if trajectory is not None:
                end_frame = len(trajectory) + start_frame
                return trajectory, self.default_speed, start_frame, end_frame, select_npc_idx, select_npc.direction
        return None

    def insert_static_car(self, start_frame, end_frame, current_insert_index, sample_num=1000, margin=0.5):

        start = 0
        end = self.max_frames - 1
        trajectory = None

        static_map = self.static_map
        ego_trajectory = self.ego_trajectory
        length = self.car_length + margin
        width = self.car_width + margin
        pos_res, yaw_res = None, None
        for i in range(sample_num):
            map_data = static_map.data

            free_cell = np.where(map_data == 1)
            pos2 = np.array(list(zip(*free_cell)))
            selcet_idx = np.random.choice(len(pos2))
            pos = pos2[selcet_idx]

            pos_global = static_map.m2w_strict(pos[1], pos[0])

            ego_xy, ego_yaw = ego_trajectory[:, [1, 0]], ego_trajectory[:, 2]
            tree = KDTree(ego_xy)
            distance, index = tree.query(pos_global)
            psi = ego_yaw[index]
            random_psi = np.random.uniform(-0.3, 0.3)
            psi = psi + random_psi

            start_car = get_car_rect(pos_global[1], pos_global[0], psi, length, width)
            corner = get_rect_corners(start_car)

            rectob1 = RectObstacle(corner)
            occ_flag = rectob1.check_map_occupy_cells(static_map)
            if occ_flag:
                pos_res = pos_global
                yaw_res = psi
                break

        if pos_res is None:
            return None

        car_waypoints = []
        for i in range(self.max_frames):
            car_waypoints.append([pos_res[1], pos_res[0], yaw_res])
        trajectory = car_waypoints[start_frame:end_frame]

        return trajectory, -1, start, end, None, None

    def multitest_sample(self, T, start_pos):
        car_dict = self.car_dict
        start = samplling_start_multitest(self.my_map, self.ego_trajectory, car_dict["length"], car_dict["width"],
                                          margin=0, T=T, sample_pos1=start_pos)
        start = np.array(start)
        return start

    def insert_dynamic_car(self, start_frame, end_frame, current_insert_index, margin=0.5):
        my_mapdata_arr = self.get_mapdata_arr()
        car_dict = self.car_dict

        while start_frame < end_frame - 20:
            print("ID", start_frame)
            for i in range(5):
                res = sampling_way_points(self.my_map, self.ego_trajectory, car_dict,
                                          self.map_dict, start_frame)
                if res is None:
                    continue
                wx, wy, direction, start_wp = res
                reference_path = get_refrence_path(wx, wy, start_wp)
                if reference_path is None:
                    continue
                trajectory = trajectory_planning(reference_path, start_frame, end_frame, self.my_map, car_dict,
                                                 self.max_frames,
                                                 my_mapdata_arr, speed_level=self.default_speed, frame_th=self.frame_th)
                if trajectory is not None:
                    end_frame = len(trajectory) + start_frame
                    return trajectory, self.default_speed, start_frame, end_frame, None, direction
            start_frame += int(self.max_frames / 10)
        return None

    def _speed_adjust(self, start_frame, end_frame, current_insert_index, step, max_speed_mod=5, min_speed_mod=0):

        select_idx = []
        for i, npc in enumerate(self.my_map.npcs):
            if npc.type == "dynamic" and npc.is_insert and min_speed_mod < npc.speed_mod + step < max_speed_mod:
                select_idx.append(i)
        if len(select_idx) == 0:
            print("warning no npc can adjust speed")
            return None
        select_npc_idx = np.random.choice(select_idx)
        select_npc = self.my_map.npcs[select_npc_idx]
        self.my_map.npcs.remove(select_npc)
        my_mapdata_arr = self.get_mapdata_arr()

        speed_level = select_npc.speed_mod + step
        x, y, yaw = select_npc.car_trajectory[:, 0], select_npc.car_trajectory[:, 1], select_npc.car_trajectory[:, 2]

        car_dict = self.car_dict

        wx, wy, yaw = y, x, np.pi / 2 - yaw

        print("select_npc_idx", select_npc_idx, select_npc.id)
        reference_path = get_refrence_path(wx, wy)
        if reference_path is None:
            return None
        trajectory = trajectory_planning(reference_path, start_frame, end_frame, self.my_map, car_dict, self.max_frames,
                                         my_mapdata_arr, speed_level=speed_level, frame_th=self.frame_th)
        self.my_map.npcs.insert(select_npc_idx, select_npc)
        if trajectory is not None:
            end_frame = len(trajectory) + start_frame
            return trajectory, speed_level, start_frame, end_frame, select_npc_idx, select_npc.direction
        return None

    def speed_up(self, start_frame, end_frame, current_insert_index, *args):
        return self._speed_adjust(start_frame, end_frame, current_insert_index, 1)

    def speed_down(self, start_frame, end_frame, current_insert_index, *args):
        return self._speed_adjust(start_frame, end_frame, current_insert_index, -1)

    def leading(self, start_frame, end_frame, *args, car_dis=10):
        car_dis = np.random.randint(3, car_dis)

        res = self._following_or_leading(start_frame, car_dis)
        if res is None:
            return None
        start_i, start_j, select_npc_idx, min_t, max_t = res
        print("select_npc_idx", select_npc_idx)
        trajectory = self.my_map.npcs[select_npc_idx].car_trajectory

        new_trajectory = trajectory[start_j:max_t]

        x, y, yaw = new_trajectory[:, 0], new_trajectory[:, 1], new_trajectory[:, 2]
        wx, wy, yaw = y, x, np.pi / 2 - yaw
        reference_path = get_refrence_path(wx, wy, num=len(wx))

        static_map = self.my_map

        if reference_path is None:
            return None

        reference_v = cal_v_on_trajectory(trajectory[:, 0], trajectory[:, 1], self.Ts)

        my_mapdata_arr = self.get_mapdata_arr()
        trajectory = trajectory_planning(reference_path, start_frame, end_frame, self.my_map, self.car_dict,
                                         self.max_frames,
                                         my_mapdata_arr, speed_level=self.default_speed, refrence_speed=reference_v,
                                         frame_th=self.frame_th)
        if trajectory is not None:
            end_frame = len(trajectory) + start_frame
            return trajectory, self.default_speed, start_frame, end_frame, None, self.my_map.npcs[
                select_npc_idx].direction
        return None

    def following(self, start_frame, end_frame, *args, car_dis=10):

        car_dis = np.random.randint(3, car_dis)

        except_idx = [0]
        res = self._following_or_leading(start_frame, car_dis, except_idx)
        if res is None:
            return None
        start_i, start_j, select_npc_idx, min_t, max_t = res

        trajectory = self.my_map.npcs[select_npc_idx].car_trajectory
        direction = self.my_map.npcs[select_npc_idx].direction
        new_trajectory = trajectory[start_frame:max_t]

        start_frame = start_j
        self.my_map.reset_npc(start_frame)

        x, y, yaw = new_trajectory[:, 0], new_trajectory[:, 1], new_trajectory[:, 2]
        wx, wy, yaw = y, x, np.pi / 2 - yaw
        reference_path = get_refrence_path(wx, wy)
        if reference_path is None:
            return None

        reference_v = cal_v_on_trajectory(trajectory[:, 0], trajectory[:, 1], self.Ts)

        my_mapdata_arr = self.get_mapdata_arr()
        trajectory = trajectory_planning(reference_path, start_frame, end_frame, self.my_map, self.car_dict,
                                         self.max_frames,
                                         my_mapdata_arr, speed_level=self.default_speed, refrence_speed=reference_v,
                                         frame_th=self.frame_th)
        if trajectory is not None:
            end_frame = len(trajectory) + start_frame
            return trajectory, self.default_speed, start_frame, end_frame, None, direction
        return None

    def _following_or_leading(self, start_frame, car_dis, except_idx=()):
        assert car_dis > 0

        ...

        select_idx = []
        for i, npc in enumerate(self.my_map.npcs):
            if i in except_idx:
                continue
            if not npc.has_trajectory_in_T(start_frame):
                continue
            if npc.type != "dynamic":
                continue
            if npc.is_ego:
                continue

            select_idx.append(i)
        if len(select_idx) == 0:
            print("warning no npc can follow or lead")
            return None

        select_npc_idx = np.random.choice(select_idx)

        select_npc = self.my_map.npcs[select_npc_idx]
        trajectory = select_npc.car_trajectory
        npc_trajectory_box = select_npc.npc_trajectory_box

        min_t = len(trajectory)
        max_t = 0
        for i, (x, y, yaw) in enumerate(trajectory):
            if x == 0 and y == 0:
                continue
            else:
                min_t = min(min_t, i)
                max_t = max(max_t, i)

        start_i = None
        start_j = None

        for i, (x, y, yaw) in enumerate(trajectory):
            if i < start_frame:
                continue
            flag = False
            if x == 0 and y == 0:
                continue
            for j, (x2, y2, yaw2) in enumerate(trajectory[i:]):
                if x2 == 0 and y2 == 0:
                    continue
                if euclidean([x, y], [x2, y2]) > car_dis:
                    start_i = i
                    start_j = i + j
                    flag = True
                    break
            if flag:
                break
        if start_i is None or start_j is None:
            return None
        return start_i, start_j, select_npc_idx, min_t, max_t


def stop():
    ...


def start():
    ...


def overtake():
    ...
