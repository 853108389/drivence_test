import math

import numpy as np
import pandas as pd

from utils import object3d_kitti
from utils.Utils_box import UtilsBox


class UtilsCommon(object):

    @staticmethod
    def get_geometric_info(obj):
        min_xyz = obj.get_min_bound()
        max_xyz = obj.get_max_bound()
        x_min, x_max = min_xyz[0], max_xyz[0]
        y_min, y_max = min_xyz[1], max_xyz[1]
        z_min, z_max = min_xyz[2], max_xyz[2]
        half_diagonal = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2) / 2
        center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        half_height = (max_xyz[2] - min_xyz[2]) / 2

        return half_diagonal, center, half_height

    @staticmethod
    def get_geometric_info2(obj):
        min_xyz = obj.get_min_bound()
        max_xyz = obj.get_max_bound()
        x_min, x_max = min_xyz[0], max_xyz[0]
        y_min, y_max = min_xyz[1], max_xyz[1]
        z_min, z_max = min_xyz[2], max_xyz[2]
        width = (y_max - y_min)
        length = (x_max - x_min)
        height = (z_max - z_min)

        return length, width, height

    @staticmethod
    def get_meshes_distance(meshes):

        distance = []
        for obj in meshes:
            xyz = obj.get_center()

            r = math.sqrt(xyz[0] ** 2 + xyz[1] ** 2)
            print("物体中心距离lidar距离：", r)
            distance.append(r)
        print("--------------------------------")
        return distance

    @staticmethod
    def get_random_bg_index_list(val_txt_path, size):

        with open(val_txt_path, "r") as rf:
            idx_arr = rf.readlines()
        idx_arr = [int(idx.strip()) for idx in idx_arr]

        shuffle_idx = np.random.permutation(list(range(len(idx_arr))))
        idx_arr = np.array(idx_arr)[shuffle_idx]
        random_bg_index_list = idx_arr[:size]

        return random_bg_index_list

    @staticmethod
    def extract_initial_objs_from_bg(calib_info, label_path, ignore=True):
        obj_list = object3d_kitti.get_objects_from_label(label_path)

        if ignore:
            obj_list = [obj for obj in obj_list if obj.cls_type != 'DontCare']

        info = {}
        if len(obj_list) == 0:
            return None
        info['name'] = np.array([obj.cls_type for obj in obj_list])

        num_objects = len(obj_list)
        info['num_objs'] = num_objects

        loc = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
        loc_lidar = calib_info.rect_to_lidar(loc)

        dims = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])

        l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]

        loc_lidar[:, 2] += h[:, 0] / 2

        rots = np.array([obj.ry for obj in obj_list])
        gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)

        info['gt_boxes_lidar'] = gt_boxes_lidar

        corners_lidar = UtilsBox.convert_label_box3d2corner_box(gt_boxes_lidar)
        info['corners_lidar'] = corners_lidar

        return info

    @staticmethod
    def get_initial_box3d_by_corners(initial_corners):
        initial_boxes = []
        objs_half_diagonal = []
        objs_center = []

        for corners in initial_corners:
            x_min, y_min, z_min = np.min(corners, axis=0)
            x_max, y_max, z_max = np.max(corners, axis=0)

            half_diagonal = math.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2) / 2
            center = [(x_min + x_max) / 2, (y_min + y_max) / 2]

            initial_boxes.append(UtilsBox.convert_corner_box2box3d(corners))
            objs_half_diagonal.append(half_diagonal)
            objs_center.append(center)

        return initial_boxes, objs_half_diagonal, objs_center

    @staticmethod
    def get_list_max_index(list_: list, n: int) -> list:

        N_large = pd.DataFrame({'score': list_}).sort_values(by='score', ascending=[False])
        return list(N_large.index)[:n]

    @staticmethod
    def get_mask_from_RGBA(rgb):
        mask = (rgb[:, :, 3] != 0).astype("uint8") * 255
        return mask

    @staticmethod
    def overlapping_area(a, b):
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        if (dx > 0) and (dy > 0):
            return dx * dy
        else:
            return None


if __name__ == '__main__':
    pass
