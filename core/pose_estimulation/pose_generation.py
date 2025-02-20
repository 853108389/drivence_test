import copy
import math
from typing import Tuple, List

import numpy
import numpy as np
import open3d as o3d

import config
from core.pose_estimulation.collision_detection import CollisionDetector
from logger import CLogger
from utils.Format_convert import FormatConverter
from utils.Utils_box import UtilsBox


class PoseGenerator(object):

    def __init__(self):
        self.max_try_pose_num = config.common_config.max_try_pose_num
        self.collision_detector = CollisionDetector()
        self.max_non_road_points_limit = config.lidar_config.max_non_road_points_limit

    def generate_pose(self, init_mesh_obj: o3d.geometry.TriangleMesh, road_pc_input: numpy.ndarray,
                      non_road_pc: numpy.ndarray, init_objs_box3d_corners: numpy.ndarray,
                      objs_box3d_corners: numpy.ndarray) -> Tuple[List[float], float]:

        max_try_pose_num = self.max_try_pose_num
        while max_try_pose_num > 0:
            mesh_obj = copy.deepcopy(init_mesh_obj)
            position, rz_degree = self._generate_pose_detail(mesh_obj, road_pc_input)
            CLogger.debug(f"position:{position},rz_degree:{rz_degree}")
            mesh_obj = PoseGenerator.transform_mesh_by_pose(mesh_obj, position, rz_degree)

            obj_inserted_box3d = mesh_obj.get_minimal_oriented_bounding_box()

            obj_box3d_adjusted, _ = UtilsBox.change_box3d(obj_inserted_box3d)
            obj_box3d_corners = UtilsBox.convert_box3d2corner_box(obj_box3d_adjusted)

            is_on_road_flag = self._is_on_road(mesh_obj, non_road_pc)
            if max_try_pose_num % 10 == 0:
                print(f"There are {max_try_pose_num} chances left,is_on_road_flag:", is_on_road_flag)
                print(f"position:{position},rz_degree:{rz_degree}")
            if not is_on_road_flag:
                max_try_pose_num -= 1
                continue

            is_collision_flag = self.collision_detector.collision_detection(init_objs_box3d_corners,
                                                                            objs_box3d_corners,
                                                                            obj_box3d_corners)
            print("is_collision_flag:", is_collision_flag)
            if is_collision_flag:
                max_try_pose_num -= 1
                continue
            else:
                return position, rz_degree

        return None

    def _generate_pose_detail(self, mesh_obj: o3d.geometry.TriangleMesh, road_pc_input: numpy.ndarray) -> Tuple[
        List[float], float]:

        while (True):
            min_xyz = mesh_obj.get_min_bound()
            max_xyz = mesh_obj.get_max_bound()
            half_height = (max_xyz[2] - min_xyz[2]) / 2

            road_pc = road_pc_input.copy()
            road_pc = road_pc[road_pc[:, 0] > 4]
            road_pc = road_pc[road_pc[:, 2] < 3]
            road_pc = road_pc[road_pc[:, 2] > -3]
            sample_index = np.random.randint(0, len(road_pc))
            x, y, z = road_pc[sample_index][:3]
            if x < 7: continue
            position = [x, y, z + half_height]
            position = [round(i, 2) for i in position]
            rz_degree = np.random.randint(2, 10)

            if y < 0: rz_degree = -rz_degree
            break
        return position, rz_degree

    @staticmethod
    def transform_mesh_by_pose2(mesh_obj: o3d.geometry.TriangleMesh, target_bottom_center: List[float] = None,
                                rotation: float = None):

        max_bound = np.array(mesh_obj.get_max_bound())
        min_bound = np.array(mesh_obj.get_min_bound())
        bottom_center = np.array([(max_bound[0] + min_bound[0]) / 2, (max_bound[1] + min_bound[1]) / 2, min_bound[2]])
        translation_vector = target_bottom_center - bottom_center
        translation_vector[2] = translation_vector[2]
        mesh_obj.translate(translation_vector)

        if rotation is not None:
            rz_radians = math.radians(rotation)
            RZ = mesh_obj.get_rotation_matrix_from_xyz((0, 0, -rz_radians))
            mesh_obj.rotate(RZ)

        return mesh_obj, translation_vector

    @staticmethod
    def transform_mesh_by_pose(mesh_obj: o3d.geometry.TriangleMesh, shift: List[float] = None,
                               rotation: float = None) -> o3d.geometry.TriangleMesh:

        if shift is not None:
            mesh_obj.translate(shift)
        if rotation is not None:
            rz_radians = math.radians(rotation)
            RZ = mesh_obj.get_rotation_matrix_from_xyz((0, 0, -rz_radians))
            mesh_obj.rotate(RZ)

        return mesh_obj

    @staticmethod
    def scale_mesh(mesh_obj: o3d.geometry.TriangleMesh, scale_ratio: float) -> o3d.geometry.TriangleMesh:

        if scale_ratio == 1:
            pass
        else:
            mesh_obj.scale(scale=scale_ratio, center=mesh_obj.get_center())
        return mesh_obj

    def _is_on_road(self, mesh_obj: o3d.geometry.TriangleMesh, non_road_pc: numpy.ndarray) -> bool:

        box = mesh_obj.get_oriented_bounding_box()
        non_road_pcd = FormatConverter.pc_numpy_2_pcd(non_road_pc)
        non_road_pcd_contained = non_road_pcd.crop(box)

        if len(non_road_pcd_contained.points) >= self.max_non_road_points_limit:
            return False
        else:
            return True
