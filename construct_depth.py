import os
import shutil
import subprocess

from natsort import natsorted

from config import common_config
from kitti_converter import create_train_sample_data, create_test_sample_data
from utils.init_.init_dir import symlink


def construct_kitti_dataset_4_VirObject_ori(seq_ix):
    seq = f"{seq_ix:04d}"
    vir_data_dir = os.path.join("/home/niangao/disk1/_PycharmProjects/PycharmProjects/VirConv/data/kitti")
    input_dir = os.path.join(common_config.project_dirname, "_datasets", "KITTI_TRACKING")
    if os.path.exists(vir_data_dir):
        shutil.rmtree(vir_data_dir)

    train_input_dir = os.path.join(input_dir, "training")
    train_iamge_dir = os.path.join(train_input_dir, "image_02")
    train_label_dir = os.path.join(train_input_dir, "label_02")
    train_calib_dir = os.path.join(train_input_dir, "calib")

    frames = natsorted(os.listdir(os.path.join(train_iamge_dir, seq)))
    min_frame = int(frames[0].split('.')[0])
    max_frame = int(frames[-1].split('.')[0])

    TRAIN_SEQ_ID, VALID_SEQ_ID = [seq], [seq]

    create_train_sample_data(input_dir, vir_data_dir, TRAIN_SEQ_ID, VALID_SEQ_ID, init_or_clear_dirs=True,
                             only_labels=False)
    create_test_sample_data(input_root=input_dir, output_root=vir_data_dir, init_or_clear_dirs=True)


def construct_kitti_dataset_4_VirTracking_ori(seq_ix):
    seq = f"{seq_ix:04d}"
    base_path = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/3D-Multi-Object-Tracker/data"
    data_path = os.path.join(base_path, "kitti_tracking")

    input_dir = os.path.join(common_config.project_dirname, "_datasets", "KITTI_TRACKING")

    if os.path.exists(data_path):
        try:
            shutil.rmtree(data_path)
        except:
            os.remove(data_path)
    symlink(input_dir, data_path)
    train_input_dir = os.path.join(input_dir, "training")
    train_iamge_dir = os.path.join(train_input_dir, "image_02")
    train_label_dir = os.path.join(train_input_dir, "label_02")
    train_calib_dir = os.path.join(train_input_dir, "calib")
    SEQ_MAP = os.path.join(f"{data_path}/training/evaluate_tracking.seqmap.training")
    frames = natsorted(os.listdir(os.path.join(train_iamge_dir, seq)))
    min_frame = int(frames[0].split('.')[0])
    max_frame = int(frames[-1].split('.')[0]) + 1
    with open(SEQ_MAP, "w") as f:
        f.write(f"{seq} empty {min_frame} {max_frame}")
    TRAIN_SEQ_ID, VALID_SEQ_ID = [seq], [seq]

    detection_tracking_path = f"{base_path}/virconv/training/{seq}"
    if os.path.exists(detection_tracking_path):
        shutil.rmtree(detection_tracking_path)
    os.makedirs(detection_tracking_path, exist_ok=True)
    detection_path = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/VirConv/output/models/kitti/VirConv-L/default/eval/epoch_no_number/val/default/final_result/data"
    ix = min_frame
    for p in natsorted(os.listdir(detection_path)):
        p2 = f"{ix:06d}.txt"
        shutil.copyfile(os.path.join(detection_path, p), os.path.join(detection_tracking_path, p2))
        ix += 1

    pose_file = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/3D-Multi-Object-Tracker/data/pose"
    output_pose_dir = os.path.join(data_path, "training", "pose")
    if os.path.exists(output_pose_dir):
        shutil.rmtree(output_pose_dir)
    shutil.copytree(pose_file, output_pose_dir)


def construct_kitti_dataset_4_Vir(seq_ix):
    construct_kitti_dataset_4_VirObject_ori(seq_ix)
    print("Object detection run.")
    p = subprocess.run([f"./eval_vir_depth.sh {str(seq_ix)}"], shell=True)


if __name__ == '__main__':
    for seq_ix in [0, 2, 5, 7, 10, 11, 14, 18]:
        construct_kitti_dataset_4_Vir(seq_ix)
