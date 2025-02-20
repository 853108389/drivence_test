from __future__ import print_function

import os
from collections import namedtuple

import numpy as np

OxtsPacket = namedtuple('OxtsPacket',
                        'lat, lon, alt, ' +
                        'roll, pitch, yaw, ' +
                        'vn, ve, vf, vl, vu, ' +
                        'ax, ay, az, af, al, au, ' +
                        'wx, wy, wz, wf, wl, wu, ' +
                        'pos_accuracy, vel_accuracy, ' +
                        'navstat, numsats, ' +
                        'posmode, velmode, orimode')

OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')


def rotx(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rotz(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def pose_from_oxts_packet(packet, scale):
    er = 6378137.

    tx = scale * packet.lon * np.pi * er / 180.
    ty = scale * er * \
         np.log(np.tan((90. + packet.lat) * np.pi / 360.))
    tz = packet.alt
    t = np.array([tx, ty, tz])

    Rx = rotx(packet.roll)
    Ry = roty(packet.pitch)
    Rz = rotz(packet.yaw)
    R = Rz.dot(Ry.dot(Rx))

    return R, t


def load_oxts_packets_and_poses(oxts_files):
    scale = None

    origin = None

    oxts = []

    for filename in oxts_files:
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.split()

                line[:-5] = [float(x) for x in line[:-5]]
                line[-5:] = [int(float(x)) for x in line[-5:]]

                packet = OxtsPacket(*line)

                if scale is None:
                    scale = np.cos(packet.lat * np.pi / 180.)

                R, t = pose_from_oxts_packet(packet, scale)

                if origin is None:
                    origin = t

                T_w_imu = transform_from_rot_trans(R, t - origin)

                oxts.append(OxtsData(packet, T_w_imu))

    return oxts


def transform_from_rot_trans(R, t):
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


class Oxts(object):
    def __init__(self, oxts_path):
        oxts = load_oxts_packets_and_poses([oxts_path])
        self.oxts = oxts

    def get_current_pose(self, frame_id):
        I2G = self.oxts[frame_id].T_w_imu[:3, :]
        return I2G


def inverse_rigid_trans(Tr):
    inv_Tr = np.zeros_like(Tr)
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


class MyCalibration(object):

    def __init__(self, calib_filepath, from_video=False, camera="P2"):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)

        self.P = calibs[camera]
        self.P = np.reshape(self.P, [3, 4])
        self.P2 = np.reshape(calibs["P2"], [3, 4])

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
        self.cu = self.c_u
        self.cv = self.c_v
        self.fu = self.f_u
        self.fv = self.f_v
        self.tx = self.b_x
        self.ty = self.b_y

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

    def img_to_rect(self, u, v, depth_rect):

        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):

        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)

        img_pts = np.matmul(corners3d_hom, self.P2.T)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

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
