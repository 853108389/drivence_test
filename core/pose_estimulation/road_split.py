import math
import os
import shutil
from typing import Tuple, List

import numpy
import numpy as np
import open3d as o3d

import config
from logger import CLogger
from utils.Format_convert import FormatConverter
from utils.Utils_IO import UtilsIO


class RoadSplit(object):
    def __init__(self):
        self.cmd1 = "cd ../third/CENet"
        self.cmd2 = f" python infer.py -d ./data -l ./result -m ./model/512-594 -s valid/test"
        self.dis_th = 5
        self.road_split_pc_dir = config.common_config.road_split_pc_dir
        self.road_split_label_dir = config.common_config.road_split_label_dir
        self.img_height = config.camera_config.img_height
        self.img_width = config.camera_config.img_width

        self.distance_threshold = config.common_config.distance_threshold
        self.ransac_n = config.common_config.ransac_n
        self.num_iterations = config.common_config.num_iterations
        self.number_of_decimal = config.common_config.number_of_decimal
        self.road_range = config.common_config.road_range

    def get_pc_road_in_img(self, pts_img: numpy.ndarray, pts_rect_depth: numpy.ndarray,
                           points: numpy.ndarray) -> numpy.ndarray:

        assert points.shape[1] == 3
        img_shape = (self.img_height, self.img_width)

        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        pts_fov = points[pts_valid_flag]
        return pts_fov

    def split_pcd_road_by_RANSAC(self, bg_index: int, bg_pc_path: str, save_road_label_dir: str, log_dir: str) -> Tuple[
        numpy.ndarray, numpy.ndarray]:

        number_of_decimal = self.number_of_decimal
        log_file = f"{log_dir}/road_split.log"
        save_road_label_path = f"{save_road_label_dir}/{bg_index:06d}.txt"

        if not os.path.exists(save_road_label_dir):
            os.makedirs(save_road_label_dir, exist_ok=True)
        pc_bg = np.fromfile(bg_pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]

        pcd_bg = o3d.geometry.PointCloud()
        pcd_bg.points = o3d.utility.Vector3dVector(pc_bg)

        road_plane = None
        if os.path.exists(save_road_label_path):

            with open(save_road_label_path, 'r') as file:
                road_plane = [float(param) for param in file.readline().strip().split(" ")]

        else:

            road_plane, _ = pcd_bg.segment_plane(distance_threshold=self.distance_threshold,
                                                 ransac_n=self.ransac_n,
                                                 num_iterations=self.num_iterations)

            with open(save_road_label_path, 'w') as file:
                file.write(" ".join([str(i) for i in road_plane]))

        A = road_plane[0]
        B = road_plane[1]
        C = road_plane[2]
        D = road_plane[3]

        CLogger.debug(
            f"Plane equation: {A:.{number_of_decimal}f}x + {B:.{number_of_decimal}f}y + {C:.{number_of_decimal}f}z + {D:.{number_of_decimal}f} = 0")

        idx = []
        pcd_bg_size = len(pc_bg)
        min_x = min_y = math.inf
        max_x = max_y = -math.inf

        x_behind, x_front, y_left, y_right, abs_threshold = self.road_range
        for i in range(pcd_bg_size):
            setx, sety, setz = pc_bg[i]
            if x_behind <= setx <= x_front and y_left <= sety <= y_right:

                z = -1 * (A * setx + B * sety + D) / (C)
                if abs(setz - z) <= abs_threshold:

                    idx.append(i)

                    if min_x > setx:  min_x = setx

                    if min_y > sety:  min_y = sety

                    if max_x < setx:  max_x = setx

                    if max_y < sety:  max_y = sety
        pc_road = pc_bg[idx]
        pc_non_road = pc_bg[list(set(range(pcd_bg_size)).difference(set(idx)))]

        return pc_road, pc_non_road

    def split_pcd_road_by_CENet(self, bg_index: int, bg_pc_path: str, save_road_label_dir: str, log_dir: str) -> Tuple[
        numpy.ndarray, numpy.ndarray]:

        road_split_pc_dir = self.road_split_pc_dir
        log_file = f"{log_dir}/road_split.log"
        road_split_label_dir = self.road_split_label_dir

        if os.path.exists(save_road_label_dir):
            os.makedirs(save_road_label_dir, exist_ok=True)

        if os.path.exists(road_split_pc_dir):
            shutil.rmtree(road_split_pc_dir)
        os.makedirs(road_split_pc_dir, exist_ok=True)

        pc_path = f"{road_split_pc_dir}/{bg_index:06d}.bin"
        label_path = f"{road_split_label_dir}/{bg_index:06d}.label"
        save_road_label_path = f"{save_road_label_dir}/{bg_index:06d}.label"
        save_road_interpolation_path = f"{save_road_label_dir}/{bg_index:06d}.bin"
        print(save_road_interpolation_path)
        if os.path.exists(save_road_interpolation_path):

            labels = UtilsIO.load_road_split_labels(save_road_label_path)
            pc_road = np.fromfile(save_road_interpolation_path, dtype=np.float32).reshape((-1, 3))
            pc_bg = np.fromfile(bg_pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]
            road_index_arr, inx_no_road_arr = self.split_pcd_road_label(labels)
            if len(road_index_arr) <= 10:
                return None, None, None, None
            _pc_non_road = pc_bg[inx_no_road_arr]
        else:

            print("split road .........")

            shutil.copyfile(bg_pc_path, pc_path)
            print(log_file)
            os.system(f"{self.cmd1} && {self.cmd2} > {log_file} 2>&1")

            shutil.copyfile(label_path, save_road_label_path)

            labels = UtilsIO.load_road_split_labels(save_road_label_path)

            road_index_arr, inx_no_road_arr = self.split_pcd_road_label(labels)
            if len(road_index_arr) <= 10:
                return None, None, None, None

            pc_bg = np.fromfile(bg_pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]
            _pc_road = pc_bg[road_index_arr]
            _pc_non_road = pc_bg[inx_no_road_arr]

            pc_road = self.trunc_road_pc(_pc_road)
            pc_road.astype(np.float32).tofile(save_road_interpolation_path, )

        return pc_road, _pc_non_road

    def trunc_road_pc(self, pc_road: numpy.ndarray) -> numpy.ndarray:

        dis_th = self.dis_th
        pcd_road = FormatConverter.pc_numpy_2_pcd(pc_road)

        cl, ind = pcd_road.remove_radius_outlier(nb_points=7, radius=1)
        pcd_inlier_road = pcd_road.select_by_index(ind)

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_inlier_road, 10)

        pcd_inter = mesh.sample_points_uniformly(number_of_points=50000)

        _pc_inter = np.asarray(pcd_inter.points)
        dis = np.linalg.norm(_pc_inter, axis=1, ord=2)
        _pc_inter_valid = _pc_inter[dis > 4]

        pc_road = _pc_inter_valid.astype(np.float32)
        if dis_th is not None:
            pc_road = pc_road[pc_road[:, 0] > dis_th]
        return pc_road

    def split_pcd_road_label(self, labels: list) -> Tuple[List, List]:

        inx_road_arr = []
        inx_other_road_arr = []
        inx_other_ground_arr = []
        inx_no_road_arr = []

        for i in range(len(labels)):
            inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr = self._split_pc_road_label_detail(
                labels, i,
                inx_road_arr,
                inx_other_road_arr,
                inx_other_ground_arr,
                inx_no_road_arr)
        return inx_road_arr, [*inx_other_road_arr, *inx_other_ground_arr, *inx_no_road_arr]

    def _split_pc_road_label_detail(self, label, index, inx_road_arr, inx_other_road_arr, inx_other_ground_arr,
                                    inx_no_road_arr):

        lb = label[index][0]
        if lb == 40:
            inx_road_arr.append(index)
        elif lb == 44:
            inx_other_road_arr.append(index)
        elif lb == 48:
            inx_other_road_arr.append(index)
        elif lb in (49, 70, 71, 72):
            inx_other_ground_arr.append(index)
        else:
            inx_no_road_arr.append(index)
        return inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr


if __name__ == '__main__':
    ...
