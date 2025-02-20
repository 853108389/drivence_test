import copy
from typing import List

import numpy
import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as RR

from utils.Format_convert import FormatConverter


class UtilsBox(object):

    @staticmethod
    def get_box2d_from_image(image_obj: numpy.ndarray, pos_image: List[int]) -> List[float]:

        ymax, xmax, _ = image_obj.shape
        img_xmin = pos_image[0] - 0.5 * xmax
        img_xmax = pos_image[0] + 0.5 * xmax
        img_ymin = pos_image[1] - 0.5 * ymax
        img_ymax = pos_image[1] + 0.5 * ymax
        return [img_xmin, img_ymin, img_xmax, img_ymax]

    @staticmethod
    def get_box2d_from_points(pts: numpy.ndarray) -> List[float]:

        pts = pts[:, :2]
        img_xmin, img_ymin = np.min(pts, axis=0)
        img_xmax, img_ymax = np.max(pts, axis=0)
        return [img_xmin, img_ymin, img_xmax, img_ymax]

    @staticmethod
    def trunc_box2d_from_img(box: List[float], max_x: int, max_y: int) -> List[float]:

        img_xmin, img_ymin, img_xmax, img_ymax = box
        x_clip = np.clip([img_xmin, img_xmax], 0, max_x)
        y_clip = np.clip([img_ymin, img_ymax], 0, max_y)
        return [x_clip[0], y_clip[0], x_clip[1], y_clip[1]]

    @staticmethod
    def get_box2d_center(box: List[float]) -> List[int]:

        img_xmin, img_ymin, img_xmax, img_ymax = box
        return [int(0.5 * (img_xmin + img_xmax)), int(0.5 * (img_ymin + img_ymax))]

    @staticmethod
    def iou_2d(box0: List[float], box1: List[float]) -> float:

        xy_max = np.minimum(box0[2:], box1[2:])
        xy_min = np.maximum(box0[:2], box1[:2])

        inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
        inter = inter[0] * inter[1]

        area_0 = (box0[2] - box0[0]) * (box0[3] - box0[1])
        area_1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        union = area_0 + area_1 - inter
        return inter / union

    @staticmethod
    def convert_label_box3d2corner_box(label_boxes3d: numpy.ndarray) -> numpy.ndarray:

        label_boxes3d, is_numpy = FormatConverter.check_numpy_to_torch(label_boxes3d)

        template = label_boxes3d.new_tensor((
            [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
            [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
        )) / 2

        corners3d = label_boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
        corners3d = UtilsBox.rotate_points_along_z(corners3d.view(-1, 8, 3), label_boxes3d[:, 6]).view(-1, 8, 3)
        corners3d += label_boxes3d[:, None, 0:3]

        return corners3d.numpy() if is_numpy else corners3d

    @staticmethod
    def convert_corner_box2box3d(corner: numpy.ndarray) -> o3d.geometry.OrientedBoundingBox:

        box3d = o3d.geometry.OrientedBoundingBox.create_from_points(points=o3d.utility.Vector3dVector(corner))

        return box3d

    @staticmethod
    def convert_box3d2corner_box(box3d: o3d.geometry.OrientedBoundingBox) -> numpy.ndarray:

        corner_box = numpy.asarray(box3d.get_box_points())
        b = corner_box[:, 2]
        index = np.lexsort((b,))
        return corner_box[index]

    @staticmethod
    def rotate_points_along_z(points: torch.Tensor, angle: torch.Tensor) -> numpy.ndarray:

        points, is_numpy = FormatConverter.check_numpy_to_torch(points)
        angle, _ = FormatConverter.check_numpy_to_torch(angle)

        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa, sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)

        return points_rot.numpy() if is_numpy else points_rot

    @staticmethod
    def change_box3d(box3d):

        sciangle_0, sciangle_1, sciangle_2 = UtilsBox.get_euler_from_rotate_matrix(copy.copy(box3d.R))

        if 2.5 > abs(sciangle_0) > 1.47:
            ...

        if abs(sciangle_0) > 2.5 or abs(sciangle_1) > 3 or abs(sciangle_2) > 3:

            R_return = box3d.get_rotation_matrix_from_zyx([-sciangle_2, -sciangle_1, - sciangle_0])

            box3d.rotate(R_return)
            R_return_1 = box3d.get_rotation_matrix_from_xyz([0, 0, -sciangle_2])
            box3d.rotate(R_return_1)

        else:
            R_return = box3d.get_rotation_matrix_from_zyx([-sciangle_2, -sciangle_1, - sciangle_0])
            box3d.rotate(R_return)
            R_return_1 = box3d.get_rotation_matrix_from_xyz([0, 0, sciangle_2])
            box3d.rotate(R_return_1)

        return box3d, [sciangle_0, sciangle_1, sciangle_2]

    @staticmethod
    def get_euler_from_rotate_matrix(R: numpy.ndarray) -> List[float]:

        euler_type = "XYZ"

        sciangle_0, sciangle_1, sciangle_2 = RR.from_matrix(R).as_euler(euler_type)

        return sciangle_0, sciangle_1, sciangle_2
