import copy

import numpy as np
import scipy
import torch
from scipy.spatial import Delaunay

from utils import common_utils


def get_2d_box_from_image(image_obj, pos_image):
    ymax, xmax, _ = image_obj.shape
    img_xmin = pos_image[0] - 0.5 * xmax
    img_xmax = pos_image[0] + 0.5 * xmax
    img_ymin = pos_image[1] - 0.5 * ymax
    img_ymax = pos_image[1] + 0.5 * ymax
    return [img_xmin, img_ymin, img_xmax, img_ymax]


def get_2d_box_from_points(pts):
    pts = pts[:, :2]
    img_xmin, img_ymin = np.min(pts, axis=0)
    img_xmax, img_ymax = np.max(pts, axis=0)
    return [img_xmin, img_ymin, img_xmax, img_ymax]


def trunc_2d_box(box, max_x, max_y):
    img_xmin, img_ymin, img_xmax, img_ymax = box
    x_clip = np.clip([img_xmin, img_xmax], 0, max_x)
    y_clip = np.clip([img_ymin, img_ymax], 0, max_y)
    return [x_clip[0], y_clip[0], x_clip[1], y_clip[1]]


def get_2d_box_center(box):
    img_xmin, img_ymin, img_xmax, img_ymax = box
    return [int(0.5 * (img_xmin + img_xmax)), int(0.5 * (img_ymin + img_ymax))]


def iou_2d(box0, box1):
    xy_max = np.minimum(box0[2:], box1[2:])
    xy_min = np.maximum(box0[:2], box1[:2])

    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
    inter = inter[0] * inter[1]

    area_0 = (box0[2] - box0[0]) * (box0[3] - box0[1])
    area_1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    union = area_0 + area_1 - inter
    return inter / union


def in_hull(p, hull):
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag


def boxes_to_corners_3d(boxes3d):
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def mask_boxes_outside_range_numpy(boxes, limit_range, min_num_corners=1):
    if boxes.shape[1] > 7:
        boxes = boxes[:, 0:7]
    corners = boxes_to_corners_3d(boxes)
    mask = ((corners >= limit_range[0:3]) & (corners <= limit_range[3:6])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners

    return mask


def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):
    boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
    xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
    l, h, w = boxes3d_camera_copy[:, 3:4], boxes3d_camera_copy[:, 4:5], boxes3d_camera_copy[:, 5:6]

    xyz_lidar = calib.rect_to_lidar(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)


def boxes3d_kitti_fakelidar_to_lidar(boxes3d_lidar):
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    w, l, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    r = boxes3d_lidar_copy[:, 6:7]

    boxes3d_lidar_copy[:, 2] += h[:, 0] / 2
    return np.concatenate([boxes3d_lidar_copy[:, 0:3], l, w, h, -(r + np.pi / 2)], axis=-1)


def boxes3d_kitti_lidar_to_fakelidar(boxes3d_lidar):
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    dx, dy, dz = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    heading = boxes3d_lidar_copy[:, 6:7]

    boxes3d_lidar_copy[:, 2] -= dz[:, 0] / 2
    return np.concatenate([boxes3d_lidar_copy[:, 0:3], dy, dx, dz, -heading - np.pi / 2], axis=-1)


def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    large_boxes3d = boxes3d.clone()

    large_boxes3d[:, 3:6] += boxes3d.new_tensor(extra_width)[None, :]
    return large_boxes3d


def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib):
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    xyz_lidar = boxes3d_lidar_copy[:, 0:3]
    l, w, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    r = boxes3d_lidar_copy[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)

    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)


def get_box_size(corner):
    x_min, y_min, z_min = np.min(corner, axis=0)
    x_max, y_max, z_max = np.max(corner, axis=0)
    return x_max - x_min, y_max - y_min, z_max - z_min


def get_box_center(corner):
    x_min, y_min, z_min = np.min(corner, axis=0)
    x_max, y_max, z_max = np.max(corner, axis=0)
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    return center


def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)
    else:
        y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])
    R_list = np.transpose(rot_list, (2, 0, 1))

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)
    rotated_corners = np.matmul(temp_corners, R_list)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)


def boxes3d_kitti_camera_to_imageboxes(boxes3d, calib, image_shape=None):
    corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
    pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)
    max_uv = np.max(corners_in_image, axis=1)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d_image


def boxes_iou_normal(boxes_a, boxes_b):
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = torch.max(boxes_a[:, 0, None], boxes_b[None, :, 0])
    x_max = torch.min(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = torch.max(boxes_a[:, 1, None], boxes_b[None, :, 1])
    y_max = torch.min(boxes_a[:, 3, None], boxes_b[None, :, 3])
    x_len = torch.clamp_min(x_max - x_min, min=0)
    y_len = torch.clamp_min(y_max - y_min, min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / torch.clamp_min(area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6)
    return iou


def boxes3d_lidar_to_aligned_bev_boxes(boxes3d):
    rot_angle = common_utils.limit_period(boxes3d[:, 6], offset=0.5, period=np.pi).abs()
    choose_dims = torch.where(rot_angle[:, None] < np.pi / 4, boxes3d[:, [3, 4]], boxes3d[:, [4, 3]])
    aligned_bev_boxes = torch.cat((boxes3d[:, 0:2] - choose_dims / 2, boxes3d[:, 0:2] + choose_dims / 2), dim=1)
    return aligned_bev_boxes


def boxes3d_nearest_bev_iou(boxes_a, boxes_b):
    boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a)
    boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b)

    return boxes_iou_normal(boxes_bev_a, boxes_bev_b)
