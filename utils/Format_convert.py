import numpy
import numpy as np
import open3d as o3d
import torch


class FormatConverter(object):

    @staticmethod
    def pc_numpy_2_pcd(xyz: numpy.ndarray) -> o3d.geometry.PointCloud:
        pcd_bg = o3d.geometry.PointCloud()
        pcd_bg.points = o3d.utility.Vector3dVector(xyz)
        return pcd_bg

    @staticmethod
    def pcd_2_pc_numpy(pcd_obj: o3d.geometry.PointCloud) -> numpy.ndarray:
        return np.asarray(pcd_obj.points)

    @staticmethod
    def pc_numpy_2_pcr(mixed_pc_three_dims) -> numpy.ndarray:
        assert mixed_pc_three_dims.shape[1] == 3

        point_nums = mixed_pc_three_dims.shape[0]
        b = np.zeros((point_nums, 1))
        mixed_pc = np.concatenate([mixed_pc_three_dims, b], axis=1)

        return mixed_pc

    @staticmethod
    def pcr_numpy_2_pcr_channel(mixed_pc_four_dims) -> numpy.ndarray:
        assert mixed_pc_four_dims.shape[1] == 4

        point_nums = mixed_pc_four_dims.shape[0]

        return None

    @staticmethod
    def check_numpy_to_torch(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float(), True
        return x, False
