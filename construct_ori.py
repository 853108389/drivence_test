import os
import shutil
import subprocess

from natsort import natsorted

import config.common_config
from config import common_config
from construct_dataset import inference_d2_results, inference_d3_results, convert_obj_ouput2tracking_output, \
    evaluate_epnet
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
    p = subprocess.run(["./eval_vir_detection.sh"], shell=True)

    construct_kitti_dataset_4_VirTracking_ori(seq_ix)
    p = subprocess.run(["./eval_vir_tracking.sh"], shell=True)
    print("Tracking done.")

    result_dir = os.path.join(config.common_config.result_path, "virmot", "ori", f"{seq_ix:04d}")
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    shutil.copytree(
        "/home/niangao/disk1/_PycharmProjects/PycharmProjects/3D-Multi-Object-Tracker/evaluation/results/sha_key",
        os.path.join(result_dir, "tracking"))
    shutil.copytree(
        "/home/niangao/disk1/_PycharmProjects/PycharmProjects/VirConv/output/models/kitti/VirConv-L/default/eval/epoch_no_number/val/default/final_result/data",
        os.path.join(result_dir, "detection")
    )


def construct_kitti_dataset_4_dfmot_ori(seq_ix):
    seq = f"{seq_ix:04d}"

    DFMOT_DIR = os.path.join(common_config.project_dirname, "DeepFusionMOT", "datasets", "kitti")
    DF_TRAIN_DIR = os.path.join(DFMOT_DIR, "train")
    DF_TEST_DIR = os.path.join(DFMOT_DIR, "test")
    DF_2D_LABEL_DIR = os.path.join(DF_TRAIN_DIR, "2D_rrc_Car_val")
    DF_3D_LABEL_DIR = os.path.join(DF_TRAIN_DIR, "3D_pointrcnn_Car_val")
    DF_EVAL_DATA_DIR = os.path.join(common_config.project_dirname, "DeepFusionMOT", "data", "tracking")
    SEQ_MAP = os.path.join(DF_EVAL_DATA_DIR, "evaluate_tracking.seqmap")
    EVAL_LABEL = os.path.join(DF_EVAL_DATA_DIR, "label_02")

    if os.path.exists(DFMOT_DIR):
        shutil.rmtree(DFMOT_DIR)
    os.makedirs(DF_TRAIN_DIR, exist_ok=True)
    os.makedirs(DF_TEST_DIR, exist_ok=True)
    if os.path.exists(DF_2D_LABEL_DIR):
        shutil.rmtree(DF_2D_LABEL_DIR)
    if os.path.exists(DF_3D_LABEL_DIR):
        shutil.rmtree(DF_3D_LABEL_DIR)
    os.makedirs(DF_2D_LABEL_DIR, exist_ok=True)

    input_dir = os.path.join(common_config.project_dirname, "_datasets", "KITTI_TRACKING")
    train_input_dir = os.path.join(input_dir, "training")
    train_iamge_dir = os.path.join(train_input_dir, "image_02")
    train_label_dir = os.path.join(train_input_dir, "label_02")
    train_calib_dir = os.path.join(train_input_dir, "calib")
    frames = natsorted(os.listdir(os.path.join(train_iamge_dir, seq)))
    min_frame = int(frames[0].split('.')[0])
    max_frame = int(frames[-1].split('.')[0])

    print("2D")
    inference_d2_results(train_iamge_dir, DF_2D_LABEL_DIR, seq_idx=seq_ix)

    TRAIN_SEQ_ID, VALID_SEQ_ID = [seq], [seq]
    PCD_DATSSET_DIR = os.path.join(common_config.project_dirname, "openpcdet", "data", "kitti")

    PCD_OUTPUT_DIR = os.path.join(common_config.project_dirname,
                                  "openpcdet/output/cfgs/kitti_models/voxel_rcnn_car/default/eval/epoch_54/val/default/final_result/data")

    if os.path.exists(PCD_OUTPUT_DIR):
        shutil.rmtree(PCD_OUTPUT_DIR)
    BASH_PATH = "./env.sh"
    if os.path.exists(PCD_DATSSET_DIR):
        shutil.rmtree(PCD_DATSSET_DIR)
    print("create data")
    create_train_sample_data(input_dir, PCD_DATSSET_DIR, TRAIN_SEQ_ID, VALID_SEQ_ID, init_or_clear_dirs=True,
                             only_labels=False, add_dt=True, fill_blank_file=True)
    create_test_sample_data(input_root=input_dir, output_root=PCD_DATSSET_DIR, init_or_clear_dirs=True)
    print("3D")
    inference_d3_results(BASH_PATH)
    seq2sample = os.path.join(PCD_DATSSET_DIR, "training", "seq2sample.txt")
    sample2frame = os.path.join(PCD_DATSSET_DIR, "training", "sample2frame.txt")
    convert_obj_ouput2tracking_output(PCD_OUTPUT_DIR, DF_3D_LABEL_DIR, seq2sample, sample2frame)

    shutil.copytree(train_calib_dir, os.path.join(DF_TRAIN_DIR, "calib_train"))
    target_dir = os.path.join(DF_TRAIN_DIR, "image_02_train")
    if os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
        except:
            os.remove(target_dir)
    os.makedirs(os.path.join(DF_TRAIN_DIR, "image_02_train"), exist_ok=True)
    symlink(os.path.join(train_iamge_dir, seq), os.path.join(DF_TRAIN_DIR, "image_02_train", seq))
    symlink(train_label_dir, os.path.join(DF_TRAIN_DIR, "label_02"))

    with open(SEQ_MAP, "w+") as f:
        f.write(f"{seq} empty {frames[0].split('.')[0]} {frames[-1].split('.')[0]}")

    if os.path.exists(EVAL_LABEL):
        shutil.rmtree(EVAL_LABEL)
    shutil.copytree(train_label_dir, EVAL_LABEL)


def evaluate_dfmot(seq_ix):
    seq = f"{seq_ix:04d}"
    input_result_dir = "./DeepFusionMOT/results/train/data"
    if os.path.exists(input_result_dir):
        shutil.rmtree(input_result_dir)

    cmd1 = "cd ./DeepFusionMOT"
    cmd2 = "python main.py"
    os.system("{} && {}".format(cmd1, cmd2))

    cmd3 = "python my_evaluate.py"
    os.system("{} && {}".format(cmd1, cmd3))

    output_result_dir = "./result/dfmot/ori/{}".format(seq)
    os.makedirs("./result/dfmot/ori", exist_ok=True)
    if os.path.exists(output_result_dir):
        shutil.rmtree(output_result_dir)
    shutil.copytree(input_result_dir, output_result_dir)


def construct_kitti_dataset_4_EPNetOject_ori(seq_ix):
    seq = f"{seq_ix:04d}"
    epnet_data_dir = os.path.join(common_config.project_dirname, "system/EPNet/data/KITTI")
    input_dir = os.path.join(common_config.project_dirname, "_datasets", "KITTI_TRACKING")
    if os.path.exists(epnet_data_dir):
        shutil.rmtree(epnet_data_dir)

    train_input_dir = os.path.join(input_dir, "training")
    train_iamge_dir = os.path.join(train_input_dir, "image_02")
    train_label_dir = os.path.join(train_input_dir, "label_02")
    train_calib_dir = os.path.join(train_input_dir, "calib")

    frames = natsorted(os.listdir(os.path.join(train_iamge_dir, seq)))
    min_frame = int(frames[0].split('.')[0])
    max_frame = int(frames[-1].split('.')[0])

    TRAIN_SEQ_ID, VALID_SEQ_ID = [seq], [seq]

    create_train_sample_data(input_dir, epnet_data_dir, TRAIN_SEQ_ID, VALID_SEQ_ID, init_or_clear_dirs=True,
                             only_labels=False)
    create_test_sample_data(input_root=input_dir, output_root=epnet_data_dir, init_or_clear_dirs=True)

    target_dir = os.path.join(epnet_data_dir, "object")
    shutil.move(os.path.join(epnet_data_dir, "training"), os.path.join(target_dir, "training"))
    shutil.move(os.path.join(epnet_data_dir, "testing"), os.path.join(target_dir, "testing"))


def construct_kitti_dataset_4_EPNetTracking_ori(seq_ix):
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
    detection_path = f"{config.common_config.project_dirname}/system/EPNet/tools/log/Car/models/full_epnet_without_iou_branch/" \
                     f"eval_results/eval/epoch_45/val/eval/final_result/data"
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


def construct_kitti_dataset_4_EPNet(seq_ix):
    construct_kitti_dataset_4_EPNetOject_ori(seq_ix)
    evaluate_epnet()
    print("Object detection done.")
    construct_kitti_dataset_4_EPNetTracking_ori(seq_ix)
    p = subprocess.run(["./eval_vir_tracking.sh"], shell=True)
    print("Tracking done.")

    result_dir = os.path.join(config.common_config.result_path, "epmot", "ori", f"{seq_ix:04d}")
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    shutil.copytree(
        "/home/niangao/disk1/_PycharmProjects/PycharmProjects/3D-Multi-Object-Tracker/evaluation/results/sha_key",
        os.path.join(result_dir, "tracking"))
    shutil.copytree(
        "/home/niangao/disk1/_PycharmProjects/PycharmProjects/VirConv/output/models/kitti/VirConv-L/default/eval/epoch_no_number/val/default/final_result/data",
        os.path.join(result_dir, "detection")
    )


if __name__ == '__main__':
    for seq_ix in [0, 2, 5, 7, 10, 11, 14, 18]:
        construct_kitti_dataset_4_Vir(seq_ix)
