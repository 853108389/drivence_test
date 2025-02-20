from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pykitti.tracking import KittiTrackingLabels
from tqdm import tqdm

from kitti_object import my_show_lidar_on_image, filter_lidar_on_image_by_semantic_label
from my_utils2 import Oxts
from utils import box_utils
from utils.box_utils import get_box_center


def load_labels(label_path):
    labels = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
    return labels


def get_depth_pt3d(depth):
    pt3d = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            pt3d.append([i, j, depth[i, j]])
    return np.array(pt3d)


def inverse_rigid_trans(Tr):
    inv_Tr = np.zeros_like(Tr)
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


class Pose(object):
    def __init__(self, pose_path):
        poses = np.loadtxt(pose_path)
        self.poses = poses[:, :12]

    def get_current_pose(self, frame_id):
        I2G = self.poses[frame_id]
        return I2G.reshape(3, 4)


class MyCalibration(object):

    def __init__(self, calib_filepath, from_video=False, camera="P2"):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)

        self.P = calibs[camera]
        self.P = np.reshape(self.P, [3, 4])

        self.V2C = calibs["Tr_velo_cam"]
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)

        self.I2V = calibs["Tr_imu_velo"]
        self.I2V = np.reshape(self.I2V, [3, 4])
        self.V2I = inverse_rigid_trans(self.I2V)

        self.R0 = calibs["R_rect"]
        self.R0 = np.reshape(self.R0, [3, 3])

        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)
        self.b_y = self.P[1, 3] / (-self.f_v)

    def lidar_to_imu(self, pts_lidar):
        pts_lidar_hom = self.cart_to_hom(pts_lidar)

        pts_imu = np.dot(pts_lidar_hom, self.V2I.T)
        return pts_imu

    def imu_to_lidar(self, pts_imu):
        pts_imu_hom = self.cart_to_hom(pts_imu)
        pts_lidar = np.dot(pts_imu_hom, self.I2V.T)
        return pts_lidar

    def imu_to_global(self, pts_imu, I2G):
        pts_imu_hom = self.cart_to_hom(pts_imu)

        pts_global = np.dot(pts_imu_hom, I2G.T)

        return pts_global

    def global_to_imu(self, pts_global, I2G):
        pts_global_hom = self.cart_to_hom(pts_global)
        G2I = inverse_rigid_trans(I2G)
        pts_imu = np.dot(pts_global_hom, G2I.T)

        return pts_imu

    def lidar_to_img(self, pts_lidar):

        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def lidar_to_rect(self, pts_lidar):

        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))

        return pts_rect

    def rect_to_img(self, pts_rect):

        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]
        return pts_img, pts_rect_depth

    def cart_to_hom(self, pts):

        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):

        pts_rect_hom = self.cart_to_hom(pts_rect)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def read_calib_file(self, filepath):

        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():

                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)

                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def read_calib_from_video(self, calib_root_dir):

        data = {}
        cam2cam = self.read_calib_file(
            os.path.join(calib_root_dir, "calib_cam_to_cam.txt")
        )
        velo2cam = self.read_calib_file(
            os.path.join(calib_root_dir, "calib_velo_to_cam.txt")
        )
        Tr_velo_to_cam = np.zeros((3, 4))
        Tr_velo_to_cam[0:3, 0:3] = np.reshape(velo2cam["R"], [3, 3])
        Tr_velo_to_cam[:, 3] = velo2cam["T"]
        data["Tr_velo_to_cam"] = np.reshape(Tr_velo_to_cam, [12])
        data["R_rect"] = cam2cam["R_rect_00"]
        data["P2"] = cam2cam["P_rect_02"]
        data["P3"] = cam2cam["P_rect_03"]
        return data

    def cart2hom(self, pts_3d):

        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):

        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):

        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):

        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        res = self.project_ref_to_rect(pts_3d_ref)
        return res

    def project_rect_to_image(self, pts_3d_rect):

        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):

        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        res = self.project_rect_to_image(pts_3d_rect)
        return res

    def project_8p_to_4p(self, pts_2d):
        x0 = np.min(pts_2d[:, 0])
        x1 = np.max(pts_2d[:, 0])
        y0 = np.min(pts_2d[:, 1])
        y1 = np.max(pts_2d[:, 1])
        x0 = max(0, x0)

        y0 = max(0, y0)

        return np.array([x0, y0, x1, y1])

    def project_velo_to_4p(self, pts_3d_velo):

        pts_2d_velo = self.project_velo_to_image(pts_3d_velo)
        return self.project_8p_to_4p(pts_2d_velo)

    def project_image_to_rect(self, uv_depth):

        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

    def project_depth_to_velo(self, depth, constraint_box=True):
        cbox = np.array([[0, 70.4], [-40, 40], [-3, 2]])
        depth_pt3d = get_depth_pt3d(depth)
        depth_UVDepth = np.zeros_like(depth_pt3d)
        depth_UVDepth[:, 0] = depth_pt3d[:, 1]
        depth_UVDepth[:, 1] = depth_pt3d[:, 0]
        depth_UVDepth[:, 2] = depth_pt3d[:, 2]

        depth_pc_velo = self.project_image_to_velo(depth_UVDepth)

        if constraint_box:
            depth_box_fov_inds = (
                    (depth_pc_velo[:, 0] < cbox[0][1])
                    & (depth_pc_velo[:, 0] >= cbox[0][0])
                    & (depth_pc_velo[:, 1] < cbox[1][1])
                    & (depth_pc_velo[:, 1] >= cbox[1][0])
                    & (depth_pc_velo[:, 2] < cbox[2][1])
                    & (depth_pc_velo[:, 2] >= cbox[2][0])
            )
            depth_pc_velo = depth_pc_velo[depth_box_fov_inds]
        return depth_pc_velo


def viz_mayavi(points, vals="distance", col=None):
    import torch
    import mayavi
    points = torch.from_numpy(points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    d = torch.sqrt(x ** 2 + y ** 2)

    if col is None:
        if vals == "height":
            col = z
        else:
            col = d

    fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
    mayavi.mlab.quiver3d(0, 0, 1, figure=fig, color=(0.5, 0.5, 1))
    mayavi.mlab.quiver3d(1, 0, 0, figure=fig, color=(1, 1, 1))
    mayavi.mlab.quiver3d(0, 1, 0, figure=fig)
    mayavi.mlab.points3d(x, y, z,
                         col,
                         mode="point",
                         colormap='spectral',
                         figure=fig,
                         )
    mayavi.mlab.show()


def viz_mayavi2(points, vals="distance", col=None, fig=None):
    import torch
    import mayavi
    points = torch.from_numpy(points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    d = torch.sqrt(x ** 2 + y ** 2)

    if col is None:
        if vals == "height":
            col = z
        else:
            col = d
        mayavi.mlab.points3d(x, y, z,
                             col,
                             mode="point",
                             colormap='spectral',
                             figure=fig,
                             )
    else:
        col = torch.ones_like(z) * col
        mayavi.mlab.points3d(x, y, z,
                             mode="point",
                             colormap='spectral',
                             figure=fig,
                             )


def get_corners(boxes, calib_info):
    loc = boxes[:, 0:3]
    dims = boxes[:, 3:6]
    h, w, l = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
    rots = boxes[:, 6]

    loc_lidar = calib_info.rect_to_lidar(loc)
    loc_lidar[:, 2] += h[:, 0] / 2
    gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
    return corners_lidar


def pc_numpy_2_o3d(xyz):
    import open3d as o3d
    pcd_bg = o3d.geometry.PointCloud()

    pcd_bg.points = o3d.utility.Vector3dVector(xyz)
    return pcd_bg


def crop_obj(pc, bg_box):
    import open3d as o3d
    pcd_bg = pc_numpy_2_o3d(pc)
    crop_box = o3d.geometry.OrientedBoundingBox. \
        create_from_points(o3d.utility.Vector3dVector(bg_box))
    return np.asarray(pcd_bg.crop(crop_box).points)


def create_map():
    MAP_SIZE_Y = 80
    MAP_SIZE_X = 80
    dgm = np.zeros((MAP_SIZE_Y,))
    dgm[:, :, 0] = 1


def split_pc(labels):
    print(np.unique(labels))
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


def get_road(pc, pc_labels):
    inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr, inx_npc_arr, inx_buiding_arr = split_pc(
        pc_labels)
    x = pc_labels[inx_road_arr]

    return pc[inx_road_arr], pc[inx_no_road_arr + inx_other_road_arr + inx_other_ground_arr], pc[inx_npc_arr], pc[
        inx_buiding_arr]


def get_pc_road_and_non_road(pc_ori, pc_lb, img_sematic, calib):
    inx_road_arr, inx_other_road_arr, inx_other_ground_arr, inx_no_road_arr, inx_npc_arr, inx_buiding_arr = split_pc(
        pc_lb)

    label_road = [0]
    label_building = [1]
    label_ground = [3]

    pc_road_by_label = pc_ori[inx_road_arr]
    road_pts_idx = filter_lidar_on_image_by_semantic_label \
        (pc_road_by_label, img_sematic, calib, label_road)
    print("len(road_pts_idx", len(road_pts_idx))
    pc_road = pc_road_by_label[road_pts_idx]

    road_non_pts_idx = filter_lidar_on_image_by_semantic_label \
        (pc_road_by_label, img_sematic, calib, label_ground + label_building)

    indx_may_be_road = set(inx_road_arr) - set(road_non_pts_idx) - set(road_pts_idx)
    indx_may_be_road = list(indx_may_be_road)
    pc_maybe_road = pc_ori[indx_may_be_road]

    pc_non_road_building_by_label = pc_ori[inx_buiding_arr]
    building_pts_idx = filter_lidar_on_image_by_semantic_label \
        (pc_non_road_building_by_label, img_sematic, calib, label_building)
    pc_non_road_building = pc_non_road_building_by_label[building_pts_idx]

    pc_non_road_by_label = pc_ori[inx_other_ground_arr + inx_other_road_arr]
    road_non_pts_idx = filter_lidar_on_image_by_semantic_label(pc_non_road_by_label, img_sematic, calib, label_ground)
    pc_non_road_ground = pc_non_road_by_label[road_non_pts_idx]
    pc_non_road = np.concatenate([pc_non_road_building, pc_non_road_ground], axis=0)

    pc_road = pc_road[pc_road[:, 2] < 0.5]
    pc_non_road = pc_non_road[pc_non_road[:, 2] < 0.5]
    pc_maybe_road = pc_maybe_road[pc_maybe_road[:, 2] < 0.5]
    return pc_road, pc_non_road, pc_maybe_road


def merge_pc2(grid_map, frame_num, lidar_base_path, label_base_path, pose, calib, img_sematic_base_path, frame_step=1):
    for frame_id in tqdm(range(0, frame_num, frame_step)):
        semantic_path = os.path.join(img_sematic_base_path, f"{frame_id:06d}.png")
        img_sematic = np.array(Image.open(semantic_path).convert('L'))

        lidar_path = os.path.join(lidar_base_path, f"{frame_id:06d}.bin")
        pc_ori = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
        pc_lb = load_labels(f"{label_base_path}/{frame_id:06d}.label")

        pc_ori = pc_ori[:, :3]
        pc_road, pc_non_road, pc_maybe_road = get_pc_road_and_non_road(pc_ori, pc_lb, img_sematic, calib)
        road_len = len(pc_road)
        pc_maybe_road_len = len(pc_maybe_road)

        current_pose = pose.get_current_pose(frame_id)
        print(pc_road.shape, pc_maybe_road.shape, pc_non_road.shape)
        example = np.concatenate([pc_road, pc_maybe_road, pc_non_road], axis=0)
        example_imu = calib.lidar_to_imu(example)
        example_global = calib.imu_to_global(example_imu, current_pose)

        pc_road_global = example_global[:road_len]
        pc_maybe_road_global = example_global[road_len:road_len + pc_maybe_road_len]
        pc_non_road_global = example_global[road_len + pc_maybe_road_len:]
        grid_map.update_map(pc_road_global)
        grid_map.update_map_non_road(pc_non_road_global)
        grid_map.update_map_maybe_road(pc_maybe_road_global)

    return grid_map


def extract_objs_tracking(obj_num, frame_num, lidar_base_path, labels, pose, calib, cal_ego_motion=True, frame_step=1):
    obj_globals = {}

    for obj_id in tqdm(range(0, obj_num)):
        obj_global = np.zeros((frame_num, 8, 3))
        for frame_id in range(0, frame_num, frame_step):
            current_pose = pose.get_current_pose(frame_id)

            lidar_path = os.path.join(lidar_base_path, f"{frame_id:06d}.bin")
            pc_ori = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            pc = pc_ori[:, 0:3]

            boxes = labels.get_box(frame_id, obj_id)
            corners = get_corners(boxes, calib)
            if len(corners) == 0:
                continue
            corners = corners[0]

            example_imu = calib.lidar_to_imu(corners)
            example_global = calib.imu_to_global(example_imu, current_pose)
            obj_global[frame_id] = example_global

        if len(obj_global) > 0:
            obj_globals[obj_id] = np.array(obj_global)

    ego_car_location = []
    if cal_ego_motion:
        for frame_id in range(0, frame_num, frame_step):
            current_pose = pose.get_current_pose(frame_id)
            example_imu = calib.lidar_to_imu(np.array([[0, 0, 0]]))

            example_global = calib.imu_to_global(example_imu, current_pose)
            ego_car_location.append(example_global[0])
        obj_globals["ego_car_location"] = np.array(ego_car_location)

    return obj_globals


def get_map_size(res_map):
    max_x_arr = []
    min_x_arr = []
    max_y_arr = []
    min_y_arr = []
    for k, v in res_map.items():
        if k == "ego_car_location":
            continue
        else:

            v = v.reshape(v.shape[0] * v.shape[1], v.shape[2])

        max_x, min_x, max_y, min_y = np.max(v[:, 0]), np.min(v[:, 0]), np.max(v[:, 1]), np.min(v[:, 1])
        max_x_arr.append(max_x)
        min_x_arr.append(min_x)
        max_y_arr.append(max_y)
        min_y_arr.append(min_y)
    max_x, min_x, max_y, min_y = np.max(np.array(max_x_arr)), np.min(np.array(min_x_arr)), np.max(
        np.array(max_y_arr)), np.min(np.array(min_y_arr))

    print(max_x, min_x, max_y, min_y)
    print()


def trans_pc_to_map(resolution):
    seq_ix = 0
    sequence = f"{seq_ix:04d}"

    tracking_base_path = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/MANTRA-CVPR20/KITTI"
    road_label_base_path = f"{tracking_base_path}/semantic_label/{sequence}"
    lidar_base_path = os.path.join(tracking_base_path, "velodyne", f"{sequence}")
    label_base_path = os.path.join(tracking_base_path, "label_02", f"{sequence}.txt")
    calib_base_path = os.path.join(tracking_base_path, "calib", f"{sequence}.txt")
    print(calib_base_path)
    oxts_base_path = os.path.join(tracking_base_path, "oxts", f"{sequence}.txt")

    calib = MyCalibration(calib_base_path)

    pose = Oxts(oxts_base_path)
    labels = KittiTrackingLabels(label_base_path, remove_dontcare=True)

    grid_map = MyGridMap((-40, 50), (-80, 40), resolution)
    obj_num = len(labels.ids)
    frame_num = 1
    ...
    grid_map = merge_pc(grid_map, frame_num, lidar_base_path, road_label_base_path, pose, calib, frame_step=1)


def my_test_trans(resolution):
    seq_ix = 0
    sequence = f"{seq_ix:04d}"

    tracking_base_path = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/MANTRA-CVPR20/KITTI"
    road_label_base_path = f"{tracking_base_path}/semantic_label/{sequence}"
    lidar_base_path = os.path.join(tracking_base_path, "velodyne", f"{sequence}")
    label_base_path = os.path.join(tracking_base_path, "label_02", f"{sequence}.txt")
    calib_base_path = os.path.join(tracking_base_path, "calib", f"{sequence}.txt")
    print(calib_base_path)
    oxts_base_path = os.path.join(tracking_base_path, "oxts", f"{sequence}.txt")
    img_sematic_base_path = os.path.join(tracking_base_path, "panoptic_maps", f"{sequence}")

    calib = MyCalibration(calib_base_path)

    pose = Oxts(oxts_base_path)
    labels = KittiTrackingLabels(label_base_path, remove_dontcare=True)

    obj_globals = []
    pc_labels = []
    ego_car_location = []

    grid_map = MyGridMap((-10, 50), (-80, 40), resolution)
    obj_num = len(labels.ids)
    frame_num = 154

    res_map = extract_objs_tracking(obj_num, frame_num, lidar_base_path, labels, pose, calib, cal_ego_motion=True,
                                    frame_step=1)

    grid_map.set_ego_trajectory(res_map["ego_car_location"])
    res_map.pop("ego_car_location")
    grid_map.set_npc_trajectory(res_map)

    grid_map = merge_pc2(grid_map, frame_num, lidar_base_path, road_label_base_path, pose, calib, img_sematic_base_path,
                         frame_step=1)

    map_path = os.path.join(tracking_base_path, "map", f"{sequence}_{resolution}_test.npz")
    grid_map.save(map_path)
    grid_map = MyGridMap.load(map_path)
    grid_map.show2()


class MyGridMap:

    def __init__(self, map_size_x_range, map_size_y_range, resolution, threshold=0.5):
        self.max_x_min = map_size_x_range[0]
        self.max_x_max = map_size_x_range[1]
        self.max_y_min = map_size_y_range[0]
        self.max_y_max = map_size_y_range[1]
        self.resolution = resolution
        self.threshold = threshold

        self.x_bin = np.arange(self.max_x_min, self.max_x_max, self.resolution)
        self.y_bin = np.arange(self.max_y_min, self.max_y_max, self.resolution)

        self.length = len(self.x_bin) - 1
        self.width = len(self.y_bin) - 1

        self._grids = np.zeros((self.length + 2, self.width + 2))
        self.height_grid = np.zeros((self.length + 2, self.width + 2))

        self.ego_trajectory = None
        self.npc_trajectory = None
        self.npc_ids = None

        self.non_road_grid = np.zeros((self.length + 2, self.width + 2))
        self.maybe_road_grid = np.zeros((self.length + 2, self.width + 2))

    def m2w(self, dx, dy):

        x = (dx - 0.5) * self.resolution + self.max_x_min
        y = (dy - 0.5) * self.resolution + self.max_y_min

        return x, y

    @property
    def grid_map(self):
        return self._grid_without_padding(self._grids)

    def _grid_without_padding(self, grids):
        return grids[1:-1, 1:-1]

    def insert(self, x, y, value):
        x_idx, y_idx = self.get_index(x, y)
        self._grids[x_idx, y_idx] = value

    def get(self, x, y):
        x_idx, y_idx = self.get_index(x, y)
        return self._grids[x_idx, y_idx]

    def add(self, x, y, value):
        x_idx, y_idx = self.get_index(x, y)
        self._grids[x_idx, y_idx] += value

    def set(self, x, y, value):
        x_idx, y_idx = self.get_index(x, y)
        self._grids[x_idx, y_idx] = value

    def get_index(self, x, y):
        x_idx = np.digitize(x, self.x_bin)
        y_idx = np.digitize(y, self.y_bin)
        return x_idx, y_idx

    def get_bins_value(self, x_idx, y_idx):
        return self.x_bin[x_idx], self.y_bin[y_idx]

    def save_map(self, path):
        _grids = self._grids.copy()
        _non_grids = self.non_road_grid.copy()
        maybe_road_grid = self.maybe_road_grid.copy()
        maybe_road_grid = maybe_road_grid / 156
        maybe_road_grid[maybe_road_grid > 0.01] = 1
        _grids[_grids > 0] = 1

        x_idxes, y_idxes = self.get_index(self.ego_trajectory[:, 0], self.ego_trajectory[:, 1])
        x_idx_arr = []
        y_idx_arr = []
        for x_idx, y_idx in zip(x_idxes, y_idxes):
            if _grids[x_idx, y_idx] == 0:
                if maybe_road_grid[x_idx, y_idx] == 1:
                    x_idx_arr.append(x_idx)
                    y_idx_arr.append(y_idx)
        for x_idx, y_idx in zip(x_idx_arr, y_idx_arr):
            step = 40
            _grids[x_idx - step:x_idx + step, y_idx - step:y_idx + step] = \
                maybe_road_grid[x_idx - step:x_idx + step, y_idx - step:y_idx + step]

        plt.imsave(path, _grids, cmap='gray')

    def show2(self, show_ego_trajectory=True):
        import matplotlib.pyplot as plt
        _grids = self._grids.copy()
        _non_grids = self.non_road_grid.copy()
        _grids[_grids > 0] = 1

        if show_ego_trajectory and self.ego_trajectory is not None:
            x_idxes, y_idxes = self.get_index(self.ego_trajectory[:, 0], self.ego_trajectory[:, 1])
            for x_idx, y_idx in zip(x_idxes, y_idxes):
                plt.scatter(y_idx, x_idx, c='r', s=1)

        _grids = self._grid_without_padding(_grids)
        _grids = (_grids * 255).astype(np.int)
        _grids = np.clip(_grids, 0, 255).astype(np.uint8)

        plt.imshow(_grids, cmap="gray")
        plt.gca().invert_yaxis()
        plt.ylabel('x')
        plt.xlabel('y')
        plt.show()

        _non_grids = self._grid_without_padding(_non_grids)
        _grids[_non_grids > 0] = 0
        plt.imshow(_grids, cmap="gray")
        plt.gca().invert_yaxis()
        plt.ylabel('x')
        plt.xlabel('y')
        plt.show()

        maybe_road_grid = self.maybe_road_grid.copy()

        maybe_road_grid = self._grid_without_padding(maybe_road_grid)
        maybe_road_grid = maybe_road_grid / 156

        maybe_road_grid2 = np.zeros_like(maybe_road_grid)
        maybe_road_grid2[_grids == 0] = 0
        maybe_road_grid2[maybe_road_grid > 0.01] = 255 * 0.8
        maybe_road_grid2[_grids > 0] = 255
        maybe_road_grid2 = maybe_road_grid2.astype("uint8")

        plt.imshow(maybe_road_grid2, cmap="gray")

        plt.gca().invert_yaxis()
        plt.ylabel('x')
        plt.xlabel('y')
        plt.show()

    def show(self, show_ego_trajectory=False, show_npc_trajectory=False, frames=156, save_path=None):
        import matplotlib.pyplot as plt
        self.threshold = 0.999999
        _grids = self._grids.copy()
        _non_grids = self.non_road_grid.copy()
        if frames is not None:
            _grids = _grids / frames
            _grids[_grids > self.threshold] = 1
            _grids[_grids <= self.threshold] = 0

        if show_ego_trajectory and self.ego_trajectory is not None:
            x_idxes, y_idxes = self.get_index(self.ego_trajectory[:, 0], self.ego_trajectory[:, 1])
            for x_idx, y_idx in zip(x_idxes, y_idxes):
                _grids[x_idx, y_idx] = 0.5
                _non_grids[x_idx, y_idx] = 0.5
        if show_npc_trajectory and len(self.npc_trajectory) != 0:
            tracks = []
            for trajectory in self.npc_trajectory:
                track = []
                for corner in trajectory:
                    box_center = get_box_center(corner)
                    track.append(box_center)
                tracks.append(track)
            color = 0.3
            for i, track in enumerate(tracks):

                x_idxes, y_idxes = self.get_index(np.array(track)[:, 0], np.array(track)[:, 1])
                for x_idx, y_idx in zip(x_idxes, y_idxes):
                    _grids[x_idx, y_idx] = color
                    _non_grids[x_idx, y_idx] = color
                color -= 0.1

                ...

        _grids = self._grid_without_padding(_grids)
        _non_grids = self._grid_without_padding(_non_grids)

        _grids = (_grids * 255).astype(np.int)
        _grids = np.clip(_grids, 0, 255).astype(np.uint8)

        plt.imshow(_grids, cmap="gray")

        plt.ylabel('x')
        plt.xlabel('y')

        plt.gca().invert_yaxis()

        print(np.max(_grids), np.min(_grids))
        if save_path is not None:
            plt.imsave(save_path, _grids, cmap="gray")
        plt.show()

    def set_ego_trajectory(self, ego_trajectory):
        self.ego_trajectory = ego_trajectory

    def set_npc_trajectory(self, npc_trajectory):
        ids = list(npc_trajectory.keys())
        tracks = list(npc_trajectory.values())
        self.npc_ids = np.array(ids)
        self.npc_trajectory = np.array(tracks)
        print()

    def update_map(self, pc, value=1):
        pc_x = pc[:, 0]
        pc_y = pc[:, 1]
        pc_z = pc[:, 2]
        x_idxes, y_idxes = self.get_index(pc_x, pc_y)
        for x_idx, y_idx in zip(x_idxes, y_idxes):
            self._grids[x_idx, y_idx] += value

    def update_map_maybe_road(self, pc, value=1):
        pc_x = pc[:, 0]
        pc_y = pc[:, 1]
        x_idxes, y_idxes = self.get_index(pc_x, pc_y)
        for x_idx, y_idx in zip(x_idxes, y_idxes):
            self.maybe_road_grid[x_idx, y_idx] += value

    def update_map_non_road(self, pc, value=1):
        pc_x = pc[:, 0]
        pc_y = pc[:, 1]
        x_idxes, y_idxes = self.get_index(pc_x, pc_y)
        for x_idx, y_idx in zip(x_idxes, y_idxes):
            self.non_road_grid[x_idx, y_idx] += value

    def get_params_dict(self):
        return {
            "map_size_x_range": [self.max_x_min, self.max_x_max],
            "map_size_y_range": [self.max_y_min, self.max_y_max],
            "resolution": self.resolution,
            "threshold": self.threshold,
            "_grids": self._grids,
            "non_road_grid": self.non_road_grid,
            "ego_trajectory": self.ego_trajectory,
            "npc_trajectory": self.npc_trajectory,
            "npc_ids": self.npc_ids,
            "maybe_road_grid": self.maybe_road_grid,
        }

    def save(self, path):
        params_dict = self.get_params_dict()
        np.savez(path, **params_dict)

    @staticmethod
    def load(path):
        params_dict = np.load(path, allow_pickle=True)
        grid = MyGridMap(params_dict["map_size_x_range"], params_dict["map_size_y_range"],
                         params_dict["resolution"], params_dict["threshold"])
        grid._grids = params_dict["_grids"]
        grid.non_road_grid = params_dict["non_road_grid"]
        grid.ego_trajectory = params_dict["ego_trajectory"]
        grid.npc_trajectory = params_dict["npc_trajectory"]
        grid.npc_ids = params_dict["npc_ids"]
        grid.maybe_road_grid = params_dict["maybe_road_grid"]
        return grid

    def get_pc_point(self, frames=156, threshold=0.8):

        grids = self._grids.copy()
        _grids = grids / frames

        _grids[_grids > threshold] = 1
        _grids[_grids <= threshold] = 0

        point_arr = []
        for x in range(_grids.shape[0]):
            for y in range(_grids.shape[1]):
                if _grids[x, y] > 0.8:
                    point_arr.append([*grid_map.m2w(x, y), 0])
        return np.array(point_arr)


def map_test():
    MAP_LENGTH_X_MIN = -60
    MAP_LENGTH_X_MAX = 80
    MAP_WIDTH_Y_MIN = -80
    MAP_WIDTH_Y_MAX = 60

    MAP_LENGTH_X = MAP_LENGTH_X_MAX - MAP_LENGTH_X_MIN
    MAP_WIDTH_Y = MAP_WIDTH_Y_MAX - MAP_WIDTH_Y_MIN

    RESOLUTION = 0.1

    x_bin = np.arange(MAP_LENGTH_X_MIN, MAP_LENGTH_X_MAX, RESOLUTION)
    y_bin = np.arange(MAP_WIDTH_Y_MIN, MAP_WIDTH_Y_MAX, RESOLUTION)
    grids = np.zeros((len(x_bin), len(y_bin)))

    print(grids.shape)

    ...


def coord_transform_test():
    tra = np.load(os.path.join(tracking_base_path, "map", f"0000_npc_trajectory.npy"))

    origin = (grid_map.max_x_min, grid_map.max_y_min)

    print(tra.shape)

    print(tra[0][0][0][0], tra[0][1][0][0])

    idx = grid_map.get_index(tra[0][0][0][0], tra[0][1][0][0])
    print(idx)

    ori = grid_map.m2w(idx[0], idx[1])
    print(ori)


CITYSCAPES_COLORMAP = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (0, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32),
    76: (0, 255, 0),
}


def apply_colormap(mask, colormap):
    print(mask.shape, np.unique(mask))

    color_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for label, color in colormap.items():
        color_image[mask == label] = color

    return color_image


def load_and_convert_mask(mask_path):
    color_mask = apply_colormap(mask, CITYSCAPES_COLORMAP)
    return color_mask


def visualize_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


def my_test_project_road_to_pc():
    seq_ix = 0
    sequence = f"{seq_ix:04d}"

    tracking_base_path = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/MANTRA-CVPR20/KITTI"
    road_label_base_path = f"{tracking_base_path}/semantic_label/{sequence}"
    lidar_base_path = os.path.join(tracking_base_path, "velodyne", f"{sequence}")
    label_base_path = os.path.join(tracking_base_path, "label_02", f"{sequence}.txt")
    calib_base_path = os.path.join(tracking_base_path, "calib", f"{sequence}.txt")
    print(calib_base_path)
    oxts_base_path = os.path.join(tracking_base_path, "oxts", f"{sequence}.txt")
    image_semantic_base_path = os.path.join(tracking_base_path, "panoptic_maps", f"{sequence}")

    image_base_path = os.path.join(tracking_base_path, "image_02", f"{sequence}")
    calib = MyCalibration(calib_base_path)

    pose = Oxts(oxts_base_path)
    labels = KittiTrackingLabels(label_base_path, remove_dontcare=True)

    frame_id = 1

    semantic_path = os.path.join(image_semantic_base_path, f"{frame_id:06d}.png")
    img_path = os.path.join(image_base_path, f"{frame_id:06d}.png")
    mask_image_path = Image.open(semantic_path).convert('L')
    mask = np.array(mask_image_path)

    color_mask = apply_colormap(mask, CITYSCAPES_COLORMAP)
    visualize_image(color_mask)

    assert 1 == 2
    lidar_path = os.path.join(tracking_base_path, "velodyne", f"{sequence}", f"{frame_id:06d}.bin")
    img_path = os.path.join(tracking_base_path, "image_02", f"{sequence}", f"{frame_id:06d}.png")
    pc_ori = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    pc = pc_ori[:, 0:3]

    pc_lb = load_labels(f"{road_label_base_path}/{frame_id:06d}.label")
    pc_road2, pc_non_road2, _, _ = get_road(pc, pc_lb)

    import kitti_util2 as utils
    img = utils.load_image(img_path)

    img_height, img_width, _ = img.shape
    img = my_show_lidar_on_image(pc_road2[:, 0:3], img, calib, img_width, img_height)
    pts_idx = my_lter_lidar_on_imagefi(pc_road2, mask, calib, img_width, img_height)
    print(len(pts_idx), len(pc_road2))
    pc_road2_filter = pc_road2[pts_idx]
    img = my_show_lidar_on_image(pc_road2_filter[:, 0:3], img, calib, img_width, img_height)


if __name__ == '__main__':
    tracking_base_path = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/MANTRA-CVPR20/KITTI"
    resolution = 0.1
    map_path = os.path.join(tracking_base_path, "map", f"0000_{resolution}_test.npz")
    map_save_path = os.path.join(tracking_base_path, "map", f"0000_{resolution}_test.png")
    grid_map = MyGridMap.load(map_path)
    print(grid_map.max_x_min, grid_map.max_y_min, grid_map.resolution)
    print(grid_map.ego_trajectory[:10])
    grid_map.save_map(map_save_path)
