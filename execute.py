import json
import os

import numpy
import numpy as np

import config
from core.operation_info import OperationInfo
from core.pose_estimulation.pose_generation import PoseGenerator
from core.scene_info import SceneInfo
from logger import CLogger
from main_function import main
from utils.Utils_mesh import UtilsMesh
from utils.init_.init_parser import InitParser


def execute(args):
    operation_path = args.operation_path
    logic_scene_path = args.logic_scene_path
    print("logic scene json file", logic_scene_path)
    username = args.username
    bg_index = args.bg_index

    if (operation_path == "" or os.path.exists(operation_path) is False) and (
            logic_scene_path == "" or os.path.exists(logic_scene_path) is False):
        assert 1 == 2, "operation_path and logic_scene_path are empty or not exist"
    elif (operation_path == "" or os.path.exists(operation_path) is False) and os.path.exists(logic_scene_path):
        logic_scene_path = logic_scene_path
        logic_scene = SceneInfo(logic_scene_path)
    elif os.path.exists(operation_path) and (logic_scene_path == "" or os.path.exists(logic_scene_path) is False):
        logic_scene_dir = os.path.join(config.common_config.logic_scene_dir, username, f"{bg_index:06d}")
        os.makedirs(logic_scene_dir, exist_ok=True)
        logic_scene_num = len(os.listdir(logic_scene_dir))
        logic_scene = init_logic_scene(logic_scene_path, operation_path)
    else:
        logic_scene = update_logic_scene(logic_scene_path, operation_path)

    CLogger.info("Start")
    main(logic_scene, args)
    CLogger.info("Done")
    return None


def road_info(args) -> numpy.ndarray:
    bg_index = args.bg_index
    bg_pc_path = config.common_config.road_on_img_dirname
    road_pc_valid = np.fromfile(bg_pc_path + f"{bg_index:06d}.bin", dtype=np.float32).reshape(-1, 3)
    return road_pc_valid


def mesh_info(args):
    bg_index = args.obj_index
    mesh_obj_initial_path = os.path.join(
        config.common_config.obj_dirname, f"Car_{bg_index}", config.common_config
        .obj_filename)
    utils_mesh = UtilsMesh()
    mesh_obj_initial = utils_mesh.load_normalized_mesh_obj(mesh_obj_initial_path)
    return mesh_obj_initial


def update_logic_scene(logic_scene_path, operation_path):
    operation_info = OperationInfo(operation_path)
    scene_info = SceneInfo(logic_scene_path)

    nums_vehicles = scene_info.nums_vehicles
    vehicles = [vehicle.to_json() for vehicle in scene_info.vehicles]

    for operator in operation_info.operators:
        if operator.name == "insert":

            if not operator.is_random:
                location = operator.location
                rotation = operator.rotation
                scale = operator.scale
                vehicle = {"obj_index": operator.obj_index, "location": location, "rotation": rotation, "scale": scale}
                print(vehicle)
            else:
                location, rotation = PoseGenerator()._generate_pose_detail(mesh_info(operator),
                                                                           road_info(operation_info))
                vehicle = {"obj_index": operator.obj_index, "location": [np.float32(i).item() for i in location],
                           "rotation": rotation, "scale": 1.00}

            nums_vehicles += 1
            vehicles.append(vehicle)
        elif operator.name == "delete":
            flag = True
            for vehicle in vehicles:
                if vehicle["obj_index"] == operator.obj_index:
                    vehicles.remove(vehicle)
                    nums_vehicles -= 1
                    print(f"delete vehicle{operator.obj_index:06d} successfully")
                    flag = False
                    break
            if flag:
                assert 1 == 2, f"delete vehicle{operator.obj_index:06d} failed:vehicle not found"
        elif operator.name == "scale":
            flag = True
            for vehicle in vehicles:
                if vehicle["obj_index"] == operator.obj_index:
                    vehicle["scale"] = operator.scale
                    print(operator.scale)
                    utils_mesh = UtilsMesh()
                    mesh_obj_initial_path = os.path.join(
                        config.common_config.obj_dirname, f"Car_{operator.obj_index}", config.common_config
                        .obj_filename)
                    mesh_obj_initial = utils_mesh.load_normalized_mesh_obj(mesh_obj_initial_path)
                    obj_initial_z_height = mesh_obj_initial.get_max_bound()[2] - mesh_obj_initial.get_min_bound()[2]
                    obj_lidar_position = vehicle["location"]
                    if operator.scale > 1:
                        obj_lidar_position[2] = obj_lidar_position[2] + (
                                obj_initial_z_height * (operator.scale - 1)) / 2
                    elif operator.scale < 1:
                        obj_lidar_position[2] = obj_lidar_position[2] - (
                                obj_initial_z_height * (1 - operator.scale)) / 2
                    vehicle["location"] = obj_lidar_position
                    print(f"scale vehicle{operator.obj_index:06d}:{operator.scale} successfully")
                    flag = False
                    break
            if flag:
                assert 1 == 2, f"scale vehicle{operator.obj_index:06d} failed:vehicle not found"
        elif operator.name == "rotate":
            flag = True
            for vehicle in vehicles:
                if vehicle["obj_index"] == operator.obj_index:
                    vehicle["rotation"] = operator.rotation
                    print(f"rotate vehicle{operator.obj_index:06d}:{operator.rotation} successfully")
                    flag = False
                    break
            if flag:
                assert 1 == 2, f"rotate vehicle{operator.obj_index:06d} failed:vehicle not found"
        elif operator.name == "translate":
            flag = True
            for vehicle in vehicles:
                if vehicle["obj_index"] == operator.obj_index:
                    x = operator.shift[0] + vehicle["location"][0]
                    y = operator.shift[1] + vehicle["location"][1]
                    z = operator.shift[2] + vehicle["location"][2]
                    vehicle["location"] = [x, y, z]
                    print(f"shift vehicle{operator.obj_index:06d}:{operator.shift} successfully")
                    flag = False
                    break
            if flag:
                assert 1 == 2, f"shift vehicle{operator.obj_index:06d} failed:vehicle not found"

    scene_info.vehicles = vehicles
    scene_info.nums_vehicles = nums_vehicles
    scene_info = scene_info.to_json_self()
    with open(logic_scene_path, "w", encoding="utf-8") as f:
        json.dump(scene_info, f, indent=4, ensure_ascii=False)
    print("update logic scene json file successfully")
    return SceneInfo(scene_info)


def init_logic_scene(logic_scene_path, operation_path):
    operation_info = OperationInfo(operation_path)
    scene_info = {"dataset": "KITTI", "weather": "sunny", "bg_index": operation_info.bg_index}
    vehicles = []
    nums_vehicles = 0
    for operator in operation_info.operators:
        if operator.name == "insert":
            if not operator.is_random:
                location = operator.location
                rotation = operator.rotation
                scale = operator.scale
                vehicle = {"obj_index": operator.obj_index, "location": location, "rotation": rotation, "scale": scale}
            else:
                location, rotation = PoseGenerator._generate_pose_detail(mesh_info(operator), road_info(operation_info))
                print(location, rotation, "------------------")
                vehicle = {"obj_index": operator.obj_index, "location": [np.float32(i).item() for i in location],
                           "rotation": rotation, "scale": 1.00}
            nums_vehicles += 1
            vehicles.append(vehicle)
        elif operator.name == "delete":
            assert 1 == 2, "when initialing the scene json file, deleting operator is not supported"
        elif operator.name == "scale":
            assert 1 == 2, "when initialing the scene json file, scaling operator is not supported"
        elif operator.name == "rotate":
            assert 1 == 2, "when initialing the scene json file, rotating operator is not supported"
        elif operator.name == "translate":
            assert 1 == 2, "when initialing the scene json file, translating operator is not supported"
    scene_info["vehicles"] = vehicles
    scene_info["nums_vehicles"] = nums_vehicles
    print(logic_scene_path)
    with open(logic_scene_path, "w", encoding="utf-8") as f:
        json.dump(scene_info, f, indent=4, ensure_ascii=False)
    print("init logic scene json file successfully")
    return SceneInfo(scene_info)


if __name__ == "__main__":
    args = InitParser.execute_parser()
    execute(args)
