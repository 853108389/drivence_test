import math
from typing import Tuple, List, Any

import numpy
import numpy as np
import open3d as o3d
from shapely.geometry import box

import config
from utils.Utils_box import UtilsBox
from utils.Utils_common import UtilsCommon
from utils.Utils_mesh import UtilsMesh


class UtilsLabel(object):
    def __init__(self):
        self.occlusion_th = config.common_config.occlusion_th
        self.occ_point_max = config.common_config.occ_point_max

    def update_obj_inserted_labels(self, labels, infos, delete_points_mask):
        labels_updated = []
        for index, (label, info) in enumerate(zip(labels, infos)):
            labels_updated.append(self.update_single_labels(label, info, delete_points_mask))
        return labels_updated

    def update_single_labels(self, label, info, delete_points_mask):
        count = np.sum(delete_points_mask[info[0]: info[0] + info[1]])

        occlusion_ratio = count / info[1]
        if occlusion_ratio is np.NAN:
            label[0] = "DontCare"
            label[2] = -1
        else:
            occlusion_level = self.get_occlusion_level(occlusion_ratio)
            label[2] = str(max(int(label[2]), occlusion_level))
            if occlusion_ratio >= self.occlusion_th:
                label[0] = "DontCare"

        return label

    def update_occ_only_image(self, ori_labels: numpy.ndarray) -> list:

        labels = ori_labels.copy()
        from utils.object3d_kitti import Object3d
        label_objects = [Object3d(label=label) for label in labels]
        _, _, dis_arr, image_box_insert = UtilsMesh.get_objs_attr(label_objects, True)
        index = np.asarray(UtilsCommon.get_list_max_index(dis_arr, len(dis_arr)))
        labels_order = []
        image_box_order = []
        for i in index:
            labels_order.append(list(labels[i]))
            image_box_order.append(image_box_insert[i])
        for i in range(len(image_box_order)):
            if labels_order[i][0] == "DontCare":
                continue
            _box1 = np.array(image_box_order[i])
            _boxes2 = np.array(image_box_order[i + 1:])
            occlusion_ratio = self.get_image_occlusion_ratio(_box1, _boxes2)

            occlusion_level = self.get_occlusion_level(occlusion_ratio)
            labels_order[i][2] = str(max(int(labels_order[i][2]), occlusion_level))
            if occlusion_ratio >= self.occlusion_th:
                labels_order[i][0] = "DontCare"

        labels = [None] * len(labels_order)
        for i in range(len(labels_order)):
            labels[index[i]] = labels_order[i]
        return np.array(labels)

    def get_image_occlusion_ratio(self, box1: numpy.ndarray, boxes2: numpy.ndarray) -> float:

        if len(boxes2) == 0:
            return 0
        box_union = None
        box1 = box(*box1)
        for box2 in boxes2:
            box2 = box(*box2)
            box_inter = box1.intersection(box2)
            if box_union is None:
                box_union = box_inter
            else:
                box_union = box_union.union(box_inter)
        return box_union.area / box1.area

    def get_occlusion_level(self, occlusion_ratio: float) -> int:

        return int(np.clip(occlusion_ratio * 4, 0, 3))

    def update_bg_init_label(self, labels, init_pc, combine_pc, corners):
        label_updated = []
        for label, corner in zip(labels, corners):
            label_updated.append(self.update_single_init_label(label, init_pc, combine_pc, corner))
        return label_updated

    def update_single_init_label(self, label, init_pc, combine_pc, corner):
        box = UtilsBox.convert_corner_box2box3d(corner)
        indexesWithinBox_init = box.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(init_pc))
        indexesWithinBox = box.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(combine_pc))

        if len(indexesWithinBox_init) != 0:
            occlusion_ratio = (len(indexesWithinBox_init) - len(indexesWithinBox)) / len(indexesWithinBox_init)
        else:
            occlusion_ratio = 1

        occlusion_level = self.get_occlusion_level(occlusion_ratio)
        label[2] = str(max(int(label[2]), occlusion_level))
        if occlusion_ratio >= self.occlusion_th or \
                len(indexesWithinBox) < self.occ_point_max:
            label[0] = "DontCare"
        return label

    @staticmethod
    def sort_labels(labels_input):
        labels = labels_input.copy()
        if isinstance(labels, list):
            labels.sort(key=lambda x: x[0])
        else:
            labels = labels[np.argsort(labels[:, 0])]
        bg_labels_care, bg_labels_dont_care = UtilsLabel.get_care_labels(labels)
        labels = bg_labels_care + bg_labels_dont_care

        return np.array(labels)

    @staticmethod
    def sort_labels2(labels_input, total_labels_index):
        labels = labels_input.copy()
        ix = np.argsort(labels[:, 0])
        total_labels_index = total_labels_index.copy()
        labels = labels[ix]
        total_labels_index = total_labels_index[ix]
        bg_labels_care_ix, bg_labels_dont_care_ix = UtilsLabel.get_care_labels_index(labels)

        labels = np.vstack((labels[bg_labels_care_ix], labels[bg_labels_dont_care_ix]))
        total_labels_index = np.vstack(
            (total_labels_index[bg_labels_care_ix], total_labels_index[bg_labels_dont_care_ix]))

        return np.array(labels), np.array(total_labels_index)

    @staticmethod
    def get_care_labels(bg_labels, care_key=None) -> Tuple[List[Any], List[Any]]:

        bg_labels_dont_care = []
        bg_labels_care = []
        for x in bg_labels:
            if "DontCare" in x:
                bg_labels_dont_care.append(x)
            else:
                if care_key is None:
                    bg_labels_care.append(x)
                else:
                    if care_key in x:
                        bg_labels_care.append(x)
                    else:
                        bg_labels_dont_care.append(x)
        return bg_labels_care, bg_labels_dont_care

    @staticmethod
    def get_care_labels_index(bg_labels, care_key=None) -> Tuple[List[Any], List[Any]]:

        bg_labels_dont_care = []
        bg_labels_care = []
        for i, x in enumerate(bg_labels):
            if "DontCare" in x:
                bg_labels_dont_care.append(i)
            else:
                if care_key is None:
                    bg_labels_care.append(i)
                else:
                    if care_key in x:
                        bg_labels_care.append(i)
                    else:
                        bg_labels_dont_care.append(i)
        return bg_labels_care, bg_labels_dont_care

    @staticmethod
    def get_truncation_ratio(box1: List[float], box2: List[float]) -> float:
        insert_xmin, insert_ymin, insert_xmax, insert_ymax = box1
        bg_size = (insert_ymax - insert_ymin) * (insert_xmax - insert_xmin)
        from collections import namedtuple

        Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

        obj = Rectangle(insert_xmin, insert_ymin, insert_xmax, insert_ymax)
        bg = Rectangle(box2[0], box2[1], box2[2], box2[3])

        area = UtilsCommon.overlapping_area(obj, bg)
        if area is not None:
            return (bg_size - area) / bg_size
        else:
            return 1

    @staticmethod
    def get_labels(rz_degree: float, lidar_box: o3d.geometry.OrientedBoundingBox, calib_info: dict,
                   image_box: List[float],
                   truncation_ratio: float) -> List[str]:
        place_holder = -1111
        label_2_prefix = ["Car", "0.00", "0", "-10000"]
        img_xmin, img_ymin, img_xmax, img_ymax = place_holder, place_holder, place_holder, place_holder
        if image_box is not None:
            img_xmin, img_ymin, img_xmax, img_ymax = image_box

        if truncation_ratio is not None:
            label_2_prefix[1] = str(round(truncation_ratio, 2))
        if truncation_ratio == 1:
            label_2_prefix[0] = "DontCare"
        x_, y_, z_ = np.asarray(lidar_box.extent)
        h, w, l = z_, y_, x_

        corners = np.asarray(lidar_box.get_box_points())
        x_min, y_min, z_min = np.min(corners, axis=0)
        x_max, y_max, z_max = np.max(corners, axis=0)
        lidar_bottom_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]).reshape((1, 3))

        rect_bottom_center = calib_info.lidar_to_rect(lidar_bottom_center)[0]
        x, y, z = rect_bottom_center

        r_y = math.radians(rz_degree) + np.pi / 2

        paras = [img_xmin, img_ymin, img_xmax, img_ymax, h, w, l, x, y, z, r_y]
        label_2_suffix = [str(round(para, 2)) for para in paras]
        label_2_prefix.extend(label_2_suffix)
        return label_2_prefix
