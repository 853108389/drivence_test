import os
import shutil
import uuid

from natsort import natsorted

import config.common_config
from config import common_config
from core.scene_info import SceneInfo
from data_gen.convert import create_train_sample_data, create_depth_data
from data_gen.generate_logic_scene import generate_logic_scene, get_paths
from data_gen.main_function_new import main
from utils.Utils_IO import UtilsIO
from utils.data_visualization.visual import show_pc_with_labels, Visulazation

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class TaskConfig:
    def __init__(self):
        self.task_name = "task" + str(uuid.uuid4())
        self.road_split_way = "CENet"
        self.virtual_lidar_way = "simulation"
        self.execute_date = "2024-01-01"
        self.username = "dylan"

    def to_dict(self):
        return {
            "task_name-": self.task_name,
            "road_split_way": self.road_split_way,
            "virtual_lidar_way": self.virtual_lidar_way,
            "execute_date": self.execute_date,
            "username": self.username
        }


def create_gif(image_folder, output_gif, duration=200, resize_factor=1.0, color_reduction=256, optimize=True):
    from PIL import Image

    image_files = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))])

    images = []
    for img_file in image_files:
        img = Image.open(os.path.join(image_folder, img_file))

        if resize_factor < 1.0:
            img = img.resize((int(img.width * resize_factor), int(img.height * resize_factor)), Image.ANTIALIAS)

        img = img.convert("P", palette=Image.ADAPTIVE, colors=color_reduction)

        images.append(img)

    if images:
        images[0].save(output_gif, save_all=True, append_images=images[1:], duration=duration, loop=0,
                       optimize=optimize)
        print(f"GIF已成功保存为 {output_gif}")
    else:
        print("没有找到图片文件。")


def perpare(seq_ix, obj_name=None):
    track_dir = common_config.trakcing_insert_dir
    back_dir = common_config.trakcing_background_dir
    res_dir = common_config.tracking_logic_scene_dir
    if os.path.exists(res_dir):
        shutil.rmtree(res_dir)

    generate_logic_scene(seq_ix, track_dir, res_dir, back_dir, use_obj_name=obj_name)

    _, start_frame, end_frame = get_paths(track_dir)

    input_root = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/_datasets/KITTI_TRACKING"
    output_root = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/_datasets/KITTI"
    create_depth_data(input_root, seq_ix, start_frame, end_frame)
    create_train_sample_data(input_root, output_root, seq_ix, start_frame, end_frame, init_or_clear_dirs=True)


def merge_all_label(file_dir, output_file):
    file_list = natsorted(os.listdir(file_dir))
    with open(output_file, 'w') as outfile:

        for _file_path in file_list:
            file_path = os.path.join(file_dir, _file_path)
            with open(file_path, 'r') as infile:
                for line in infile:
                    if len(line.strip()) > 0:
                        outfile.write(line.strip() + "\n")

    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read()

    if content.endswith("\n"):
        content = content[:-1]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)


def run(seq, seed_num):
    tracking_logic_scene_dir = os.path.join(common_config.tracking_logic_scene_dir)
    logic_scene_paths = natsorted(os.listdir(tracking_logic_scene_dir))
    task_name = "task_" + f"{seq:04d}_" + f"seed_{seed_num}_" + "trajectories"

    kitti_aug_base_dir = os.path.join(common_config.aug_datasets_dirname, task_name, "KITTI")
    if os.path.exists(kitti_aug_base_dir):
        shutil.rmtree(kitti_aug_base_dir)

    for logic_scene_path in logic_scene_paths:
        logic_scene_path = os.path.join(tracking_logic_scene_dir, logic_scene_path)
        task_config = TaskConfig()
        logic_scene = SceneInfo(logic_scene_path)

        task_config.task_name = task_name
        main(logic_scene, task_config)

    seq = f"{logic_scene.sequence:04d}"
    base_path = os.path.join(kitti_aug_base_dir, "training")
    merge_all_label(os.path.join(base_path, "label_02_temp", seq), os.path.join(base_path, "label_02", f"{seq}.txt"))

    kitti_aug_dir = os.path.join(base_path, "image_02", f"{logic_scene.sequence:04d}")
    ogif_path = os.path.join(base_path, "gif", f"{logic_scene.sequence}.gif")
    create_gif(kitti_aug_dir, ogif_path, duration=100, resize_factor=0.3)

    kitti_aug_dir = os.path.join(base_path, "image_02_label", f"{logic_scene.sequence:04d}")
    ogif_path = os.path.join(base_path, "gif", f"{logic_scene.sequence}_label.gif")
    create_gif(kitti_aug_dir, ogif_path, duration=100, resize_factor=0.3)

    if os.path.exists(os.path.join(base_path, "calib")):
        shutil.rmtree(os.path.join(base_path, "calib"))

    shutil.copytree(os.path.join(config.common_config.tracking_dataset_path, "calib"),
                    os.path.join(base_path, "calib"))

    scene_path = os.path.join(base_path, "logic_scene")
    shutil.copytree(config.common_config.logic_scene_dir, scene_path)


def vis():
    data_num = 28
    seq_num = 18
    task_name = f"task_{seq_num:04d}_seed_4_trajectories"
    base_path = os.path.join(config.common_config.aug_datasets_dirname, task_name, "KITTI", "training")
    image_path = os.path.join(base_path, "image_02", f"{seq_num:04d}", f"{data_num:06d}")
    label_path = os.path.join(base_path, "label_02_dec", f"{seq_num:04d}", f"{data_num:06d}.txt")
    calib_path = os.path.join(base_path, "calib_dec", f"{data_num:04d}.txt")
    pc_path = os.path.join(base_path, "velodyne", f"{seq_num:04d}", f"{data_num:06d}.bin")
    data_num2 = data_num

    dt_path2 = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/VirConv/output/models/kitti/VirConv-L/default/eval/epoch_no_number/" \
               "val/default/final_result/data/{data_num2:06d}.txt"

    pc_bg = UtilsIO.load_pcd(pc_path)

    show_pc_with_labels(pc_bg, label_path, calib_path, dt_path=None)


def vis_image():
    import cv2
    data_num = 25
    for data_num in range(60, 70):
        task_name = "task_0000_seed_1_trajectories"
        base_path = os.path.join(config.common_config.aug_datasets_dirname, task_name, "KITTI", "training")
        image_path = os.path.join(base_path, "image_02", "0000", f"{data_num:06d}.png")

        img_mix_bgr = cv2.imread(image_path)

        label_path = os.path.join(base_path, "label_02_dec", "0000", f"{data_num:06d}.txt")
        dt_path = f"/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/JMODT/output/txt/{data_num:06d}.txt"
        bg_labels = UtilsIO.load_labels_2(label_path)
        dt_labels = UtilsIO.load_labels_2(dt_path)
        img_mix_gbr_with_label = Visulazation.show_img_with_labels(img_mix_bgr.copy(), bg_labels, dt_labels,
                                                                   is_shown=True)


if __name__ == '__main__':
    vis()
