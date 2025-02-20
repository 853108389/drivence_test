import copy
import json
import os
import time

from data_gen.convert import convert_calib_dec2track

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
from logger import CLogger

import config
import java2python.bin2pcd as bin2pcd

from core.occusion_handing.img_combination import ImgCombination
from core.occusion_handing.pc_combination import PcCombination

from core.sensor_simulation.lidar_simulation import VirtualLidar
from core.sensor_simulation.camera_simulation import VirtualCamera

from core.pose_estimulation.pose_generation import PoseGenerator
from core.pose_estimulation.collision_detection import CollisionDetector
from core.pose_estimulation.road_split import RoadSplit

from utils.data_visualization.visual import Visulazation
from utils.Utils_label import UtilsLabel
from utils.init_.init_dir import InitDir
from utils.calibration_kitti import Calibration
from utils.Utils_mesh import UtilsMesh
from utils.Format_convert import FormatConverter
from utils.Utils_common import UtilsCommon
from utils.Utils_IO import UtilsIO
from utils.Utils_box import UtilsBox


def get_currnet_index(seq):
    seq = '%04d' % seq
    input_root = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/_datasets/KITTI_TRACKING"
    in_training = os.path.join(input_root, 'training')
    tracking_label = os.path.join(in_training, 'label_02', f'{seq}.txt')

    import pandas as pd
    ori_labels_df = pd.read_csv(tracking_label, sep=' ', header=None,
                                index_col=0, skip_blank_lines=True)
    current_idx = ori_labels_df.iloc[:, 0].max()
    return current_idx


def main(logic_scene, args):
    lidar_simulator = VirtualLidar()
    utils_mesh = UtilsMesh()
    pose_generator = PoseGenerator()
    collision_detector = CollisionDetector()
    camera_simulator = VirtualCamera()
    InitDir(args)
    road_split = RoadSplit()
    img_combination = ImgCombination()
    pc_combination = PcCombination()
    utils_label = UtilsLabel()

    seq_id = logic_scene.sequence
    seq = '%04d' % seq_id

    task_name = args.task_name
    road_split_way = args.road_split_way
    virtual_lidar_way = args.virtual_lidar_way
    CLogger.info(f"task_name: {task_name}")
    npc_nums = get_currnet_index(seq_id)

    aug_queue_datasets_sub_dirs = ["", "image_2", "image_2_label", "image_2_score", "image_2_noref", "image_objs",
                                   "velodyne", "label_2",
                                   "log", "scene_file", "image_batch_generated", "depth_image", "image_insert_after_bg"]

    kitti_base_aug_dir = os.path.join(config.common_config.aug_datasets_dirname)

    kitti_aug_dirname = InitDir.init_plain_result_dir_tracking(kitti_base_aug_dir, args.task_name, seq_id)

    debug_log_path = config.common_config.debug_log_path

    bg_dirname = config.common_config.bg_dirname
    pcd_bg_dirname = config.common_config.pcd_bg_dirname
    img_bg_dirname = config.common_config.img_bg_dirname
    label_bg_dirname = config.common_config.label_bg_dirname
    label_index_bg_dirname = config.common_config.label_index_bg_dirname
    calib_bg_dirname = config.common_config.calib_bg_dirname
    pcd_show_dirname = config.common_config.pcd_show_dirname
    road_bg_dirname = config.common_config.road_bg_dir
    img_depth_dirname = config.common_config.img_depth_dirname
    road_bg_dirname += config.common_config.split_name_flag + road_split_way

    obj_dirname = config.common_config.obj_dirname
    obj_filename = config.common_config.obj_filename_new
    obj_car_type_json_path = config.common_config.obj_car_type_json_path
    obj_car_dirnames = os.listdir(config.common_config.obj_dirname)

    modality = config.modality

    objs_inserted_max_num = config.algorithm_config.objs_max_num

    objs_inserted_num = logic_scene.nums_vehicles

    bg_index = logic_scene.bg_index

    CLogger.info(f"select background {bg_index} ")

    save_dir, save_image_dir, save_image_dir_label, save_image_dir_fitness, save_image_dir_noref, \
        save_objs_image_dir, save_pc_dir, save_label_dir, save_log_dir, scene_file_dir, image_batch_generated_dir, depth_image_dir, image_insert_after_bg_dir = \
        InitDir.init_queue_guided_result_dir(config.common_config.aug_queue_datasets_dirname, task_name,
                                             f"{bg_index:06d}", aug_queue_datasets_sub_dirs)

    calib_bg_path = os.path.join(bg_dirname, calib_bg_dirname, f"{bg_index:06d}.txt")
    img_bg_path = os.path.join(bg_dirname, img_bg_dirname, f"{bg_index:06d}.png")
    label_bg_path = os.path.join(bg_dirname, label_bg_dirname, f"{bg_index:06d}.txt")
    label_index_bg_path = os.path.join(bg_dirname, label_index_bg_dirname, f"{bg_index:06d}.txt")
    pc_bg_path = os.path.join(bg_dirname, pcd_bg_dirname, f"{bg_index:06d}.bin")
    img_bg_depth_path = os.path.join(bg_dirname, img_depth_dirname, f"{bg_index:06d}.png")

    assert os.path.exists(img_bg_depth_path)

    aug_calib_path = os.path.join(kitti_aug_dirname, "calib_dec", f"{bg_index:04d}.txt")
    aug_image_path = os.path.join(kitti_aug_dirname, "image_02", seq, f"{bg_index:06d}.png")
    aug_image_label_path = os.path.join(kitti_aug_dirname, "image_02_label", seq, f"{bg_index:06d}.png")
    aug_label_path = os.path.join(kitti_aug_dirname, "label_02_temp", seq, f"{bg_index:04d}.txt")
    detect_label_path = os.path.join(kitti_aug_dirname, "label_02_dec", seq, f"{bg_index:06d}.txt")

    aug_pc_path = os.path.join(kitti_aug_dirname, pcd_bg_dirname, seq, f"{bg_index:06d}.bin")
    aug_pcd_show_path = os.path.join(kitti_aug_dirname, pcd_show_dirname, f"{bg_index:06d}.pcd")

    aug_queue_pc_path = os.path.join(save_pc_dir, f"{bg_index:06d}.bin")
    aug_queue_label_path = os.path.join(save_label_dir, f'{bg_index:06d}.txt')
    aug_queue_score_path = os.path.join(save_dir, config.common_config.mc_score_filename)

    img_bg = UtilsIO.load_img(img_bg_path)
    CLogger.debug("bg size:{}".format(img_bg.shape))

    pc_bg = UtilsIO.load_pcd(pc_bg_path)
    pcd_bg = FormatConverter.pc_numpy_2_pcd(pc_bg)

    bg_labels = UtilsIO.load_labels_2(label_bg_path)
    bg_label_index = UtilsIO.load_labels_2(label_index_bg_path)
    bg_labels_care_idx, bg_labels_dont_care_idx = UtilsLabel.get_care_labels_index(bg_labels)
    bg_labels_care = list(np.array(bg_labels)[bg_labels_care_idx])
    bg_labels_dont_care = list(np.array(bg_labels)[bg_labels_dont_care_idx])

    bg_label_index_care = list(np.array(bg_label_index)[bg_labels_care_idx])
    bg_label_index_dont_care = list(np.array(bg_label_index)[bg_labels_dont_care_idx])

    calib_info = Calibration(calib_bg_path)

    road_pc = None
    non_road_pc = None

    score_pre = 0

    bg_objs_info = UtilsCommon.extract_initial_objs_from_bg(calib_info, label_bg_path)

    i = 0

    objs_index_arr = []
    mesh_objs = []
    objs_box3d_corners = []
    pcd_objs = []
    img_objs = []
    labels_obj_inserted = []
    labels_index_inserted = []

    _obj_camera_positions = []
    _obj_rz_degrees = []
    _car_indexes = []
    _obj_name_arr = []
    _scale_ratio_arr = []

    skip_obj_idx_arr = []
    no_skip_obj_idx_arr = []
    while i < objs_inserted_num:
        current_npc_insert_index = npc_nums + i + 1

        start_time = time.time()
        vehicle = logic_scene.vehicles[i]

        obj_inserted_index = vehicle.obj_index

        obj_inserted_name = vehicle.obj_name

        CLogger.info("trying to insert:{}".format(obj_inserted_name))

        mesh_obj_path = os.path.join(obj_dirname, obj_inserted_name, obj_filename)
        mesh_obj_initial, scale_ratio = utils_mesh.load_normalized_mesh_obj2(mesh_obj_path)

        target_bottom_center, rz_degree = vehicle.location, vehicle.rotation

        mesh_obj, translation_vector = PoseGenerator.transform_mesh_by_pose2(mesh_obj_initial, target_bottom_center,
                                                                             rz_degree)
        obj_lidar_position = translation_vector

        obj_inserted_box3d = mesh_obj.get_minimal_oriented_bounding_box()

        obj_box3d_adjusted, _ = UtilsBox.change_box3d(obj_inserted_box3d)
        obj_box3d_corners = UtilsBox.convert_box3d2corner_box(obj_box3d_adjusted)
        box_2d_8con, box_2d_depth = calib_info.lidar_to_img(
            np.array(obj_box3d_adjusted.get_box_points()))

        xmin, ymin, xmax, ymax, clip_distance = 0, 0, config.camera_config.img_width, config.camera_config.img_height, 1
        fov_inds = (
                (box_2d_8con[:, 0] < xmax)
                & (box_2d_8con[:, 0] >= xmin)
                & (box_2d_8con[:, 1] < ymax)
                & (box_2d_8con[:, 1] >= ymin)
        )
        fov_inds = fov_inds & (box_2d_depth > clip_distance)
        if len(box_2d_8con[fov_inds]) == 0:
            CLogger.warning(f"vehicle {vehicle.obj_index} not in camera field skip")
            skip_obj_idx_arr.append(i)
            i += 1
            continue

        box_image_trunc = None
        truncation_ratio = None
        if modality is "pc" or modality is "multi":

            if virtual_lidar_way == "simulation":
                pcd_obj = lidar_simulator.lidar_simulation_by_mesh_simulation(mesh_obj)
            elif virtual_lidar_way == "sampling":
                pcd_obj = lidar_simulator.lidar_simulation_by_mesh_sampling(mesh_obj)
            if pcd_obj is None or len(pcd_obj.points) == 0:
                CLogger.warning(f"vehicle {vehicle.obj_index} not in lidar field skip")
                skip_obj_idx_arr.append(i)
                i += 1
                continue
            pcd_objs.append(pcd_obj)

        no_skip_obj_idx_arr.append(i)
        _car_indexes.append(obj_inserted_index)
        _obj_name_arr.append(obj_inserted_name)
        _scale_ratio_arr.append(scale_ratio)
        print("物体所在位置", obj_lidar_position, "物体旋转角度", rz_degree)

        if modality is "image" or modality is "multi":

            obj_camera_position = calib_info.lidar_to_rect(np.asarray([obj_lidar_position]))[0]
            _obj_camera_positions.append(obj_camera_position)
            _obj_rz_degrees.append(rz_degree)

            if modality is "image":
                box_2d_8con, box_2d_depth = calib_info.lidar_to_img(
                    np.array(obj_box3d_adjusted.get_box_points()))
                img_pts_fonv = box_2d_8con
                box_2d = UtilsBox.get_box2d_from_points(img_pts_fonv)
            else:
                if pcd_obj is None or len(pcd_obj.points) == 0:
                    box_2d = np.array([-1, -1, -1, -1])
                else:
                    img_pts_fonv, box_2d_depth = calib_info.lidar_to_img(
                        np.asarray(pcd_obj.points))
                    if len(img_pts_fonv) == 0:
                        box_2d = np.array([-1, -1, -1, -1])
                    else:
                        box_2d = UtilsBox.get_box2d_from_points(img_pts_fonv)
            if (box_2d[2] > img_bg.shape[1] and box_2d[0] < 0):
                box_2d = np.ones_like(box_2d) * -1
            box_2d_trunc = UtilsBox.trunc_box2d_from_img(box_2d, img_bg.shape[1],
                                                         img_bg.shape[0])
            center_coordinate_trunc = UtilsBox.get_box2d_center(box_2d_trunc)

            box_image = box_2d
            box_image_trunc = box_2d_trunc

            truncation_ratio = UtilsLabel.get_truncation_ratio(box_image, [0, 0, config.camera_config.img_width,
                                                                           config.camera_config.img_height])

        objs_index_arr.append(obj_inserted_index)
        mesh_objs.append(mesh_obj)

        objs_box3d_corners.append(obj_box3d_corners)

        label_obj_inserted = UtilsLabel.get_labels(rz_degree, obj_box3d_adjusted, calib_info, box_image_trunc,
                                                   truncation_ratio)

        labels_obj_inserted.append(label_obj_inserted)
        labels_index_inserted.append([bg_index, current_npc_insert_index])

        i += 1
        end_time = time.time()
        per_time = (end_time - start_time) / 60
        print("插入point cloud该物体所耗时间：", per_time, " min")

    if len(skip_obj_idx_arr) == objs_inserted_num:
        return

    if modality is "image":

        if len(bg_labels_care) == 0 and len(bg_labels_dont_care) == 0:
            total_labels = labels_obj_inserted
        elif len(bg_labels_dont_care) == 0:
            total_labels = np.concatenate([labels_obj_inserted, bg_labels_care], axis=0)
        elif len(bg_labels_care) == 0:
            total_labels = np.concatenate([labels_obj_inserted, bg_labels_dont_care], axis=0)
        else:
            total_labels = np.concatenate([labels_obj_inserted, bg_labels_care, bg_labels_dont_care], axis=0)

        total_labels = utils_label.update_occ_only_image(
            total_labels.copy())

        mixed_pc = FormatConverter.pc_numpy_2_pcr(pc_bg.copy()).astype(np.float32)

    elif modality is "pc" or modality is "multi":
        pcd_objs = list(filter(None, pcd_objs))
        if len(pcd_objs) != 0:

            combine_pc, labels_insert = pc_combination.pc_combination_with_objs(copy.deepcopy(pc_bg), pcd_objs,
                                                                                mesh_objs,
                                                                                labels_obj_inserted)
        else:
            combine_pc = pc_bg
            labels_insert = []
        if bg_objs_info is not None:
            corners_lidar = bg_objs_info['corners_lidar']
            bg_labels_update = utils_label.update_bg_init_label(bg_labels_care.copy(), pc_bg, combine_pc,
                                                                corners_lidar)
        else:
            bg_labels_update = []

        bg_labels_update = bg_labels_update + bg_labels_dont_care
        print(len(labels_insert), len(bg_labels_update), "labels_insert")
        if len(labels_insert) != 0 and len(bg_labels_update) != 0:
            total_labels = np.concatenate([labels_insert, bg_labels_update], axis=0)
        elif len(labels_insert) == 0 and len(bg_labels_update) == 0:
            total_labels = []
        elif len(labels_insert) == 0:
            total_labels = np.array(bg_labels_update)
        elif len(bg_labels_update) == 0:
            total_labels = np.array(labels_insert)
        else:
            assert 1 == 2

        mixed_pc = FormatConverter.pc_numpy_2_pcr(combine_pc)
        mixed_pc = mixed_pc.astype(np.float32)

    else:
        raise ValueError()
    mixed_pc.tofile(aug_queue_pc_path)

    mixed_img_save_filename = str(i) + "_" + "Car_" + "_".join([str(x) for x in objs_index_arr])
    mixed_img_save_path = f"{save_image_dir}/{mixed_img_save_filename}.png"

    if modality is "image" or modality is "multi":

        logic_scene_dir = os.path.join(scene_file_dir)
        os.makedirs(logic_scene_dir, exist_ok=True)

        logic_scene_num = len(os.listdir(logic_scene_dir))
        logic_scene_path = os.path.join(logic_scene_dir, "logic_scene_{}.json".format(logic_scene_num))

        scene_info = {"dataset": "KITTI", "weather": "sunny", "bg_index": bg_index}
        vehicles = []
        nums_vehicles = 0
        assert len(no_skip_obj_idx_arr) == len(_obj_camera_positions)
        assert len(skip_obj_idx_arr) + len(no_skip_obj_idx_arr) == objs_inserted_num
        assert len(_obj_camera_positions) == len(_obj_rz_degrees) == len(_car_indexes) == len(_obj_name_arr) == len(
            _scale_ratio_arr)
        for i in range(len(_obj_camera_positions)):
            location = _obj_camera_positions[i].tolist()
            rotation = _obj_rz_degrees[i]
            obj_index = _car_indexes[i]
            obj_name = _obj_name_arr[i]
            scale = _scale_ratio_arr[i]
            vehicle = {"obj_index": obj_index, "obj_name": obj_name, "location": location, "rotation": rotation,
                       "scale": scale}

            nums_vehicles += 1
            vehicles.append(vehicle)
        scene_info["vehicles"] = vehicles
        scene_info["nums_vehicles"] = nums_vehicles
        scene_info["sequence"] = seq_id
        print(logic_scene_path)
        with open(logic_scene_path, "w", encoding="utf-8") as f:
            json.dump(scene_info, f, indent=4, ensure_ascii=False)

        print("init logic scene json file successfully")

        camera_simulator.camera_simulation_new(image_batch_generated_dir, depth_image_dir, logic_scene_path,
                                               save_log_dir)

        img_mix_bgr = VirtualCamera.get_image_inserted_obj_after_objs(img_bg_path, img_bg_depth_path,
                                                                      image_batch_generated_dir,
                                                                      depth_image_dir, image_insert_after_bg_dir)
        img_mix_gbr_with_label = Visulazation.show_img_with_labels(img_mix_bgr.copy(), total_labels,
                                                                   is_shown=False)
        mixed_img_save_path_label = f"{save_image_dir_label}/{mixed_img_save_filename}.png"

        cv2.imwrite(mixed_img_save_path_label, img_mix_gbr_with_label)
    elif modality == "pc":
        img_mix_bgr = img_bg.copy()
        img_mix_gbr_with_label = img_bg.copy()
    else:
        raise ValueError()
    cv2.imwrite(mixed_img_save_path, img_mix_bgr)
    total_labels_index = np.array(labels_index_inserted + bg_label_index_care + bg_label_index_dont_care)

    total_labels, total_labels_index = UtilsLabel.sort_labels2(total_labels, total_labels_index)

    UtilsIO.write_labels_2(aug_queue_label_path, total_labels)
    labels_all = np.hstack((total_labels_index, total_labels))

    cv2.imwrite(aug_image_path, img_mix_bgr)
    cv2.imwrite(aug_image_label_path, img_mix_gbr_with_label)
    mixed_pc.tofile(aug_pc_path)

    UtilsIO.write_labels_2(aug_label_path, list(labels_all))
    UtilsIO.write_labels_2(detect_label_path, list(total_labels))

    convert_calib_dec2track(calib_bg_path, aug_calib_path)

    bin2pcd.bin_to_pcd(aug_pc_path, aug_pcd_show_path)
