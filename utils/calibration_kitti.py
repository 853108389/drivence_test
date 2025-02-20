import numpy as np


class Calibration(object):
    def __init__(self, calib_file):
        if not isinstance(calib_file, dict):
            calib = self._get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']
        self.R0 = calib['R0']
        self.V2C = calib['Tr_velo2cam']

        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

        self.lidar2img_transformation_matrix = self._get_lidar2img_transformation_matrix()

    def create_rotation_matrix_z(self, rz):

        rz_rad = np.radians(rz)
        R_z = np.array([
            [np.cos(rz_rad), -np.sin(rz_rad), 0],
            [np.sin(rz_rad), np.cos(rz_rad), 0],
            [0, 0, 1]
        ])
        return R_z

    def get_rz_camera_by_lidar(self, rz_lidar):

        R_lidar_object = self.create_rotation_matrix_z(rz_lidar)

        R_lidar_to_camera = self.V2C[:, :3]

        R_camera_object = np.dot(R_lidar_object, np.dot(R_lidar_to_camera.T, self.R0.T))

        rz_camera_rad = np.arctan2(R_camera_object[1, 0], R_camera_object[0, 0])
        rz_camera_deg = np.degrees(rz_camera_rad)

        return rz_camera_deg

    def _get_lidar2img_transformation_matrix(self):
        P2 = self.P2

        R0_rect = self.R0
        R0_rect = np.concatenate((R0_rect, np.zeros((3, 1))), axis=1)
        R0_rect = np.concatenate((R0_rect, np.asarray([[0, 0, 0, 1]])), axis=0)

        Tr_velo_to_cam = self.V2C
        Tr_velo_to_cam = np.concatenate((Tr_velo_to_cam, np.asarray([[0, 0, 0, 1]])), axis=0)
        return np.dot(P2, np.dot(R0_rect, Tr_velo_to_cam))

    def _get_calib_from_file(self, calib_file):
        with open(calib_file) as f:
            lines = f.readlines()

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32)
        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32)
        obj = lines[4].strip().split(' ')[1:]
        R0 = np.array(obj, dtype=np.float32)
        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.array(obj, dtype=np.float32)

        return {'P2': P2.reshape(3, 4),
                'P3': P3.reshape(3, 4),
                'R0': R0.reshape(3, 3),
                'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}

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

    def lidar_to_img(self, pts_lidar):

        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)

        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):

        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect
