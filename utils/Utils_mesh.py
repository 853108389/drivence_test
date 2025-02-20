import numpy as np
import open3d as o3d

import config
from utils.Utils_common import UtilsCommon


class UtilsMesh(object):
    def __init__(self):
        self.mesh_obj_scale = config.common_config.multi_scale

    def load_normalized_mesh_obj(self, obj_mesh_path: str) -> o3d.geometry.TriangleMesh:

        mesh_obj = o3d.io.read_triangle_mesh(obj_mesh_path)
        self.obj_adjust(mesh_obj)
        return mesh_obj

    def load_normalized_mesh_obj2(self, obj_mesh_path: str):

        mesh_obj = o3d.io.read_triangle_mesh(obj_mesh_path)
        scale_ratio = self.obj_adjust(mesh_obj)
        return mesh_obj, scale_ratio

    def obj_adjust(self, init_obj: o3d.geometry.TriangleMesh) -> None:

        init_obj.scale(scale=self.mesh_obj_scale, center=init_obj.get_center())
        R1 = init_obj.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
        init_obj.rotate(R1)
        R2 = init_obj.get_rotation_matrix_from_xyz((0, 0, -np.pi / 2))
        init_obj.rotate(R2)

        scale_ratio = self.mesh_obj_scale

        print("length,width,height", UtilsCommon.get_geometric_info2(init_obj))
        return scale_ratio

    @staticmethod
    def get_objs_attr(obj_list: list, has_image_box: bool = False):

        res = []
        score_arr = []
        dis_arr = []
        image_box_arr = []
        for obj in obj_list:
            x = [(obj.l, obj.w, obj.h), obj.ry, obj.loc]
            res.append(x)
            score_arr.append(obj.score)
            image_box_arr.append(obj.box2d)
            if obj.cls_type == "DontCare":
                dis_arr.append(np.inf)
            else:
                dis_arr.append(obj.dis_to_cam)

        if has_image_box:
            return res, score_arr, dis_arr, np.array(image_box_arr)
        else:
            return res, score_arr, dis_arr
