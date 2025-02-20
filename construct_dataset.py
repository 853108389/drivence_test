import os
import shutil
import subprocess

from mmdetection.train_2D import get_d2_model
from natsort import natsorted
from tqdm import tqdm

import config.common_config
from config import common_config
from kitti_converter import create_train_sample_data, create_test_sample_data
from utils.init_.init_dir import symlink


def add_python_path(p):
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = p
    else:
        os.environ["PYTHONPATH"] += ':{}'.format(p)


def remove_python_path(p):
    p = ':{}'.format(p)
    os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"].replace(p, "")


def construct_kitti_dataset_4_jmodt(task_name):
    if task_name == "ori":
        input_result_dir = os.path.join(common_config.project_dirname, "_datasets", "KITTI_TRACKING")
    else:
        input_result_dir = os.path.join(common_config.aug_datasets_dirname, task_name, "KITTI")
    output_result_dir = "./JMODT/data/KITTI/tracking"
    if os.path.exists(output_result_dir):
        os.remove(output_result_dir)
    symlink(input_result_dir, output_result_dir)

    data_root = os.path.join(common_config.project_dirname, "JMODT/data/KITTI")
    cmd1 = "cd ./JMODT"
    if task_name == "ori":
        seqs = "0_2_5_7_10_11_14_18"
        cmd2 = "python tools/kitti_converter.py --data_root  {} --seqs  {}".format(data_root, seqs)
    else:
        cmd2 = "python tools/kitti_converter.py --data_root  {}".format(data_root)
    os.system("{} && {}".format(cmd1, cmd2))


def evaluate_jmodt(task_name):
    input_result_dirs = ["./JMODT/output/feat",
                         "./JMODT/output/mot_data",
                         "./JMODT/output/txt",
                         ]
    for input_result_dir in input_result_dirs:
        if os.path.exists(input_result_dir):
            shutil.rmtree(input_result_dir)

    cmd1 = "cd ./JMODT"

    print(common_config.project_dirname)
    data_root = os.path.join(common_config.project_dirname, "JMODT/data/KITTI")
    cmd2 = "python tools/eval.py --data_root {} --det_output output --ckpt jmodt.pth".format(data_root)
    os.system("{} && {}".format(cmd1, cmd2))

    input_result_dirs = ["./JMODT/output/mot_data",
                         "./JMODT/output/txt", ]
    result_name = task_name
    output_result_dirs = [
        "./result/jmodt/{}/mot_data".format(result_name),
        "./result/jmodt/{}/txt".format(result_name)
    ]
    for input_result_dir, output_result_dir in zip(input_result_dirs, output_result_dirs):
        if os.path.exists(output_result_dir):
            shutil.rmtree(output_result_dir)
        shutil.copytree(input_result_dir, output_result_dir)


def inference_d2_results(base_image_2_dir, txt_dir, seq_idx=None):
    from mmdetection.train_2D import my_inference as d2_inference
    model = get_d2_model()
    python_path = '/home/niangao/PycharmProjects/mmdetection'
    add_python_path(python_path)

    os.makedirs(txt_dir, exist_ok=True)
    sub_dirs = natsorted(os.listdir(base_image_2_dir))

    for sub_ix, sub_dir in enumerate(sub_dirs):
        if seq_idx is not None and sub_ix != seq_idx:
            continue
        image_2_dir = os.path.join(base_image_2_dir, sub_dir)
        txt_path = "{}/{}.txt".format(txt_dir, sub_dir)
        if os.path.exists(txt_path):
            os.remove(txt_path)
        fns = natsorted(os.listdir(image_2_dir))
        with open(txt_path, "w") as f:
            for _, fn in enumerate(tqdm(fns)):
                fid = int(fn.split(".")[0])
                img_path = os.path.join(image_2_dir, fn)
                _, d2_str_results = d2_inference(img_path, model=model, fm="dfmot", fid=fid)
                if len(d2_str_results) <= 1:
                    continue
                else:
                    if fn != len(fns) - 1:
                        d2_str_results = d2_str_results + "\n"
                    f.write(d2_str_results)
    remove_python_path(python_path)


def copy_d2_data4dfmot(input_dir, output_dir):
    inference_d2_results(input_dir, output_dir)


def inference_d3_results(bash_path):
    p = subprocess.run([bash_path], shell=True)


def convert_obj_ouput2tracking_output(input_dir, output_dir, seq2sample, sample2frame):
    os.makedirs(output_dir, exist_ok=True)
    with open(seq2sample, "r") as f:
        indexes = f.readlines()
    import pandas as pd
    df_sample2frame = pd.read_csv(sample2frame, sep=' ', header=None,
                                  index_col=None, skip_blank_lines=True)

    d = {}
    for s in indexes:
        arr = s.split(" ")
        d[arr[0]] = []
        for ix in arr[1:]:
            d[arr[0]].append(ix)

    for k, fn_arr in tqdm(d.items()):

        p = os.path.join(output_dir, "{}.txt".format(k))

        with open(p, "w") as fout:
            for _, fn in enumerate(fn_arr):
                if fn == "\n":
                    continue
                fid = df_sample2frame[
                    (df_sample2frame[1] == int(k)) & (df_sample2frame[0] == int(fn))
                    ][2]

                assert len(fid) > 0

                fid = int(fid.item())
                input_fn = os.path.join(input_dir, "{}.txt".format(fn))
                if os.path.exists(input_fn):
                    with open(input_fn, "r") as f:
                        content = f.readlines()
                        filter_content = []
                        for c in content:
                            if "Car" in c:
                                filter_content.append(c.strip())
                        for c in filter_content:
                            arr = c.split(" ")
                            x1, y1, x2, y2 = arr[4], arr[5], arr[6], arr[7]
                            h, w, l, x, y, z, rot_y = arr[8], arr[9], arr[10], arr[11], arr[12], arr[13], arr[14]
                            new_line = "{},2,{},{},{},{},{},{},{},{},{},{},{},{},{}" \
                                .format(fid, x1, y1, x2, y2, arr[-1], h, w, l, x, y, z, rot_y, arr[3])

                            fout.write(new_line + "\n")
                else:
                    ...


def construct_kitti_dataset_4_dfmot(task_name):
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

    input_dir = os.path.join(common_config.aug_datasets_dirname, task_name, "KITTI")
    train_input_dir = os.path.join(input_dir, "training")
    train_iamge_dir = os.path.join(train_input_dir, "image_02")
    train_label_dir = os.path.join(train_input_dir, "label_02")
    train_calib_dir = os.path.join(train_input_dir, "calib")
    seqs = os.listdir(train_iamge_dir)
    assert len(seqs) == 1
    seq = seqs[0]

    frames = natsorted(os.listdir(os.path.join(train_iamge_dir, seq)))
    min_frame = int(frames[0].split('.')[0])
    max_frame = int(frames[-1].split('.')[0])

    inference_d2_results(train_iamge_dir, DF_2D_LABEL_DIR)

    TRAIN_SEQ_ID, VALID_SEQ_ID = [seq], [seq]
    PCD_DATSSET_DIR = os.path.join(common_config.project_dirname, "openpcdet", "data", "kitti")

    PCD_OUTPUT_DIR = os.path.join(common_config.project_dirname,
                                  "openpcdet/output/cfgs/kitti_models/voxel_rcnn_car/default/eval/epoch_54/val/default/final_result/data")

    if os.path.exists(PCD_OUTPUT_DIR):
        shutil.rmtree(PCD_OUTPUT_DIR)
    BASH_PATH = "./env.sh"
    if os.path.exists(PCD_DATSSET_DIR):
        shutil.rmtree(PCD_DATSSET_DIR)
    create_train_sample_data(input_dir, PCD_DATSSET_DIR, TRAIN_SEQ_ID, VALID_SEQ_ID, init_or_clear_dirs=True,
                             only_labels=False, add_dt=True, fill_blank_file=True)
    create_test_sample_data(input_root=input_dir, output_root=PCD_DATSSET_DIR, init_or_clear_dirs=True)
    inference_d3_results(BASH_PATH)
    seq2sample = os.path.join(PCD_DATSSET_DIR, "training", "seq2sample.txt")
    sample2frame = os.path.join(PCD_DATSSET_DIR, "training", "sample2frame.txt")
    convert_obj_ouput2tracking_output(PCD_OUTPUT_DIR, DF_3D_LABEL_DIR, seq2sample, sample2frame)

    shutil.copytree(train_calib_dir, os.path.join(DF_TRAIN_DIR, "calib_train"))
    symlink(train_iamge_dir, os.path.join(DF_TRAIN_DIR, "image_02_train"))
    symlink(train_label_dir, os.path.join(DF_TRAIN_DIR, "label_02"))

    with open(SEQ_MAP, "w+") as f:
        f.write(f"{seq} empty {frames[0].split('.')[0]} {frames[-1].split('.')[0]}")

    if os.path.exists(EVAL_LABEL):
        shutil.rmtree(EVAL_LABEL)
    shutil.copytree(train_label_dir, EVAL_LABEL)


def evaluate_dfmot(task_name):
    input_result_dir = "./DeepFusionMOT/results/train/data"
    if os.path.exists(input_result_dir):
        shutil.rmtree(input_result_dir)

    cmd1 = "cd ./DeepFusionMOT"
    cmd2 = "python main.py"
    os.system("{} && {}".format(cmd1, cmd2))

    cmd3 = "python my_evaluate.py"
    os.system("{} && {}".format(cmd1, cmd3))

    output_result_dir = "./result/dfmot/{}".format(task_name)
    if os.path.exists(output_result_dir):
        shutil.rmtree(output_result_dir)
    shutil.copytree(input_result_dir, output_result_dir)


def construct_kitti_dataset_4_bi(task_name):
    bi_3d_base_dir = f"/home/niangao/disk1/_PycharmProjects/PycharmProjects/BiTrack/data/kitti/tracking"
    if task_name == "ori":
        input_dir = os.path.join(common_config.project_dirname, "_datasets", "ori", "KITTI")
    else:
        input_dir = os.path.join(common_config.aug_datasets_dirname, task_name, "KITTI")
    if os.path.exists(bi_3d_base_dir):
        shutil.rmtree(bi_3d_base_dir)
    shutil.copytree(input_dir, bi_3d_base_dir)
    train_input_dir = os.path.join(input_dir, "training")
    train_iamge_dir = os.path.join(train_input_dir, "image_02")
    train_label_dir = os.path.join(train_input_dir, "label_02")
    train_calib_dir = os.path.join(train_input_dir, "calib")
    seqs = os.listdir(train_iamge_dir)
    assert len(seqs) == 1
    seq = seqs[0]
    bi_3d_dec_dir = f"{bi_3d_base_dir}/training/det3d_out/voxel_rcnn/{seq}"
    if os.path.exists(bi_3d_dec_dir):
        shutil.rmtree(bi_3d_dec_dir)
    os.makedirs(bi_3d_dec_dir)
    SEQ_MAP = os.path.join(f"{bi_3d_base_dir}/training/evaluate_tracking.seqmap.training")
    frames = natsorted(os.listdir(os.path.join(train_iamge_dir, seq)))
    min_frame = int(frames[0].split('.')[0])
    max_frame = int(frames[-1].split('.')[0])
    with open(SEQ_MAP, "w+") as f:
        f.write(f"{seq} empty {min_frame} {max_frame}")
    TRAIN_SEQ_ID, VALID_SEQ_ID = [seq], [seq]
    PCD_DATSSET_DIR = os.path.join(common_config.project_dirname, "openpcdet", "data", "kitti")
    if os.path.exists(PCD_DATSSET_DIR):
        shutil.rmtree(PCD_DATSSET_DIR)

    PCD_OUTPUT_DIR = os.path.join(common_config.project_dirname,
                                  "openpcdet/output/cfgs/kitti_models/voxel_rcnn_car/default/eval/epoch_54/val/default/final_result/data")

    if os.path.exists(PCD_OUTPUT_DIR):
        shutil.rmtree(PCD_OUTPUT_DIR)

    BASH_PATH = "./env.sh"
    if os.path.exists(PCD_OUTPUT_DIR):
        shutil.rmtree(PCD_OUTPUT_DIR)
    create_train_sample_data(input_dir, PCD_DATSSET_DIR, TRAIN_SEQ_ID, VALID_SEQ_ID, init_or_clear_dirs=True,
                             only_labels=False)
    create_test_sample_data(input_root=input_dir, output_root=PCD_DATSSET_DIR, init_or_clear_dirs=True)
    inference_d3_results(BASH_PATH)

    ix = min_frame
    for p in os.listdir(PCD_OUTPUT_DIR):
        p2 = f"{ix:06d}.txt"
        shutil.copyfile(os.path.join(PCD_OUTPUT_DIR, p), os.path.join(bi_3d_dec_dir, p2))
        ix += 1


def my_test_d3_ori(seq_num):
    PCD_DATSSET_DIR = os.path.join(common_config.project_dirname, "openpcdet", "data", "kitti")
    if os.path.exists(PCD_DATSSET_DIR):
        shutil.rmtree(PCD_DATSSET_DIR)
    construct_kitti_dataset_4_ori()
    ori_dir = os.path.join(common_config.aug_datasets_dirname, "ori", "KITTI")
    symlink(ori_dir, PCD_DATSSET_DIR)
    BASH_PATH = "./env.sh"
    inference_d3_results(BASH_PATH)


def my_test_d3_task(task_name):
    input_dir = os.path.join(common_config.aug_datasets_dirname, task_name, "KITTI")
    PCD_DATSSET_DIR = os.path.join(common_config.project_dirname, "openpcdet", "data", "kitti")
    if os.path.exists(PCD_DATSSET_DIR):
        if os.path.islink(PCD_DATSSET_DIR):
            os.remove(PCD_DATSSET_DIR)
        if os.path.isdir(PCD_DATSSET_DIR):
            shutil.rmtree(PCD_DATSSET_DIR)

    train_input_dir = os.path.join(input_dir, "training")
    train_iamge_dir = os.path.join(train_input_dir, "image_02")
    seqs = os.listdir(train_iamge_dir)
    assert len(seqs) == 1
    seq = seqs[0]
    TRAIN_SEQ_ID, VALID_SEQ_ID = [seq], [seq]
    create_train_sample_data(input_dir, PCD_DATSSET_DIR, TRAIN_SEQ_ID, VALID_SEQ_ID, init_or_clear_dirs=True,
                             only_labels=False)
    create_test_sample_data(input_root=input_dir, output_root=PCD_DATSSET_DIR, init_or_clear_dirs=True)

    BASH_PATH = "./env.sh"
    inference_d3_results(BASH_PATH)


def construct_kitti_dataset_4_ori(seq):
    seq = f"{seq:04d}"
    bi_3d_base_dir = os.path.join(common_config.aug_datasets_dirname, "ori", "KITTI")
    input_dir = os.path.join(common_config.project_dirname, "_datasets", "KITTI_TRACKING")
    if os.path.exists(bi_3d_base_dir):
        shutil.rmtree(bi_3d_base_dir)

    train_input_dir = os.path.join(input_dir, "training")
    train_iamge_dir = os.path.join(train_input_dir, "image_02")
    train_label_dir = os.path.join(train_input_dir, "label_02")
    train_calib_dir = os.path.join(train_input_dir, "calib")

    frames = natsorted(os.listdir(os.path.join(train_iamge_dir, seq)))
    min_frame = int(frames[0].split('.')[0])
    max_frame = int(frames[-1].split('.')[0])

    TRAIN_SEQ_ID, VALID_SEQ_ID = [seq], [seq]

    create_train_sample_data(input_dir, bi_3d_base_dir, TRAIN_SEQ_ID, VALID_SEQ_ID, init_or_clear_dirs=True,
                             only_labels=False)
    create_test_sample_data(input_root=input_dir, output_root=bi_3d_base_dir, init_or_clear_dirs=True)


def construct_kitti_dataset_4_EPNetObject(task_name):
    epnet_data_dir = os.path.join(common_config.project_dirname, "system/EPNet/data/KITTI")
    input_dir = os.path.join(common_config.aug_datasets_dirname, task_name, "KITTI")
    if os.path.exists(epnet_data_dir):
        shutil.rmtree(epnet_data_dir)

    train_input_dir = os.path.join(input_dir, "training")
    train_iamge_dir = os.path.join(train_input_dir, "image_02")
    train_label_dir = os.path.join(train_input_dir, "label_02")
    train_calib_dir = os.path.join(train_input_dir, "calib")
    seqs = os.listdir(train_iamge_dir)
    assert len(seqs) == 1
    seq = seqs[0]

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


def evaluate_epnet(output_log_dir="./"):
    name = "EPNet"
    system_dir = os.path.join(common_config.project_dirname, "system", name)
    output_log_path = os.path.join(output_log_dir, "log.txt")
    result_dir = f"{system_dir}/tools/log/Car/models/full_epnet_without_iou_branch/eval_results"
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)

    cmd1 = f"cd {system_dir}/tools"
    cmd2 = "CUDA_VISIBLE_DEVICES=0 python eval_rcnn.py --cfg_file cfgs/LI_Fusion_with_attention_use_ce_loss.yaml " \
           "--eval_mode rcnn_online    --output_dir ./log/Car/models/full_epnet_without_iou_branch/eval_results/  " \
           "--ckpt ./log/Car/models/full_epnet_without_iou_branch/ckpt/checkpoint_epoch_45.pth --set  " \
           "LI_FUSION.ENABLED True LI_FUSION.ADD_Image_Attention True RCNN.POOL_EXTRA_WIDTH 0.2  RPN.SCORE_THRESH 0.2 " \
           "RCNN.SCORE_THRESH 0.2  USE_IOU_BRANCH False "

    os.system("{} && {} > {} 2>&1".format(cmd1, cmd2, output_log_path))


def construct_kitti_dataset_4_VirObject(task_name):
    vir_data_dir = os.path.join("/home/niangao/disk1/_PycharmProjects/PycharmProjects/VirConv/data/kitti")
    input_dir = os.path.join(common_config.aug_datasets_dirname, task_name, "KITTI")
    if os.path.exists(vir_data_dir):
        shutil.rmtree(vir_data_dir)

    train_input_dir = os.path.join(input_dir, "training")
    train_iamge_dir = os.path.join(train_input_dir, "image_02")
    train_label_dir = os.path.join(train_input_dir, "label_02")
    train_calib_dir = os.path.join(train_input_dir, "calib")
    seqs = os.listdir(train_iamge_dir)
    assert len(seqs) == 1
    seq = seqs[0]

    frames = natsorted(os.listdir(os.path.join(train_iamge_dir, seq)))
    min_frame = int(frames[0].split('.')[0])
    max_frame = int(frames[-1].split('.')[0])

    TRAIN_SEQ_ID, VALID_SEQ_ID = [seq], [seq]

    create_train_sample_data(input_dir, vir_data_dir, TRAIN_SEQ_ID, VALID_SEQ_ID, init_or_clear_dirs=True,
                             only_labels=False)
    create_test_sample_data(input_root=input_dir, output_root=vir_data_dir, init_or_clear_dirs=True)


def construct_kitti_dataset_4_VirTracking(task_name):
    base_path = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/3D-Multi-Object-Tracker/data"
    data_path = os.path.join(base_path, "kitti_tracking")
    input_dir = os.path.join(common_config.aug_datasets_dirname, task_name, "KITTI")

    if os.path.exists(data_path):
        try:
            shutil.rmtree(data_path)
        except:
            os.remove(data_path)

    shutil.copytree(input_dir, data_path)
    train_input_dir = os.path.join(input_dir, "training")
    train_iamge_dir = os.path.join(train_input_dir, "image_02")
    train_label_dir = os.path.join(train_input_dir, "label_02")
    train_calib_dir = os.path.join(train_input_dir, "calib")
    seqs = os.listdir(train_iamge_dir)
    assert len(seqs) == 1
    seq = seqs[0]
    SEQ_MAP = os.path.join(f"{data_path}/training/evaluate_tracking.seqmap.training")
    frames = natsorted(os.listdir(os.path.join(train_iamge_dir, seq)))
    min_frame = int(frames[0].split('.')[0])
    max_frame = int(frames[-1].split('.')[0]) + 1
    with open(SEQ_MAP, "w+") as f:
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
    shutil.copytree(pose_file, output_pose_dir)


def construct_kitti_dataset_4_Vir(task_name):
    construct_kitti_dataset_4_VirObject(task_name)
    p = subprocess.run(["./eval_vir_detection.sh"], shell=True)

    print("Object detection done.")
    construct_kitti_dataset_4_VirTracking(task_name)
    p = subprocess.run(["./eval_vir_tracking.sh"], shell=True)
    print("Tracking done.")

    result_dir = os.path.join(config.common_config.result_path, "virmot", task_name)
    os.makedirs(result_dir, exist_ok=True)
    shutil.copytree(
        "/home/niangao/disk1/_PycharmProjects/PycharmProjects/3D-Multi-Object-Tracker/evaluation/results/sha_key",
        os.path.join(result_dir, "tracking"))
    shutil.copytree(
        "/home/niangao/disk1/_PycharmProjects/PycharmProjects/VirConv/output/models/kitti/VirConv-L/default/eval/epoch_no_number/val/default/final_result/data",
        os.path.join(result_dir, "detection")
    )


def construct_kitti_dataset_4_EPNetTracking(task_name):
    base_path = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/3D-Multi-Object-Tracker/data"
    data_path = os.path.join(base_path, "kitti_tracking")
    input_dir = os.path.join(common_config.aug_datasets_dirname, task_name, "KITTI")

    if os.path.exists(data_path):
        try:
            shutil.rmtree(data_path)
        except Exception as e:
            os.remove(data_path)

    shutil.copytree(input_dir, data_path)
    train_input_dir = os.path.join(input_dir, "training")
    train_iamge_dir = os.path.join(train_input_dir, "image_02")
    train_label_dir = os.path.join(train_input_dir, "label_02")
    train_calib_dir = os.path.join(train_input_dir, "calib")
    seqs = os.listdir(train_iamge_dir)
    assert len(seqs) == 1
    seq = seqs[0]
    SEQ_MAP = os.path.join(f"{data_path}/training/evaluate_tracking.seqmap.training")
    frames = natsorted(os.listdir(os.path.join(train_iamge_dir, seq)))
    min_frame = int(frames[0].split('.')[0])
    max_frame = int(frames[-1].split('.')[0]) + 1
    with open(SEQ_MAP, "w+") as f:
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
    shutil.copytree(pose_file, output_pose_dir)


def construct_kitti_dataset_4_EPNet(task_name):
    construct_kitti_dataset_4_EPNetObject(task_name)
    evaluate_epnet()
    print("Object detection done.")
    construct_kitti_dataset_4_EPNetTracking(task_name)
    p = subprocess.run(["./eval_vir_tracking.sh"], shell=True)
    print("Tracking done.")

    result_dir = os.path.join(config.common_config.result_path, "epmot", task_name)
    os.makedirs(result_dir, exist_ok=True)
    shutil.copytree(
        "/home/niangao/disk1/_PycharmProjects/PycharmProjects/3D-Multi-Object-Tracker/evaluation/results/sha_key",
        os.path.join(result_dir, "tracking"))
    shutil.copytree(
        "/home/niangao/disk1/_PycharmProjects/PycharmProjects/VirConv/output/models/kitti/VirConv-L/default/eval/epoch_no_number/val/default/final_result/data",
        os.path.join(result_dir, "detection")
    )


def construct_kitti_dataset_4_yomot(task_name):
    base_path = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/YONTD-MOT"
    SEQ_MAP = os.path.join(f"{base_path}/evaluation/KITTI/data/gt/evaluate_tracking.seqmap.training")
    data_path = os.path.join(base_path, "data", "KITTI")
    if os.path.exists(data_path):
        try:
            shutil.rmtree(data_path)
        except:
            os.remove(data_path)

    if task_name == "ori":
        input_dir = os.path.join(common_config.project_dirname, "_datasets", "KITTI_TRACKING")
        input_seq_map = os.path.join(input_dir, "evaluate_tracking.seqmap.training")
        if os.path.exists(SEQ_MAP):
            os.remove(SEQ_MAP)
        shutil.copyfile(input_seq_map, SEQ_MAP)
        symlink(input_dir, data_path)
        return
    else:
        input_dir = os.path.join(common_config.aug_datasets_dirname, task_name, "KITTI")

    shutil.copytree(input_dir, data_path)
    train_input_dir = os.path.join(input_dir, "training")
    train_iamge_dir = os.path.join(train_input_dir, "image_02")
    train_label_dir = os.path.join(train_input_dir, "label_02")
    train_calib_dir = os.path.join(train_input_dir, "calib")
    seqs = os.listdir(train_iamge_dir)
    assert len(seqs) == 1
    seq = seqs[0]

    frames = natsorted(os.listdir(os.path.join(train_iamge_dir, seq)))
    min_frame = int(frames[0].split('.')[0])
    max_frame = int(frames[-1].split('.')[0]) + 1
    with open(SEQ_MAP, "w+") as f:
        f.write(f"{seq} empty {min_frame} {max_frame}")


def evaluate_yomot(task_name):
    p = subprocess.run(["./eval_yomot.sh"], shell=True)

    result_dir = os.path.join(config.common_config.result_path, "yomot", task_name)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    shutil.copytree("/home/niangao/disk1/_PycharmProjects/PycharmProjects/YONTD-MOT/output/training/results/data",
                    os.path.join(result_dir, "tracking"))


def construct_ori():
    ...


if __name__ == '__main__':

    task_names = ['task_0011_seed_29_trajectories',
                  'task_0014_seed_6_trajectories',
                  'task_0018_seed_16_trajectories',
                  'task_0018_seed_20_trajectories',
                  'task_0011_seed_26_trajectories',
                  'task_0014_seed_20_trajectories',
                  'task_0018_seed_9_trajectories',
                  'task_0018_seed_26_trajectories',
                  'task_0011_seed_27_trajectories',
                  'task_0014_seed_5_trajectories',
                  'task_0014_seed_11_trajectories',
                  'task_0014_seed_19_trajectories',
                  'task_0014_seed_26_trajectories',
                  'task_0018_seed_22_trajectories',
                  'task_0018_seed_0_trajectories',
                  'task_0018_seed_5_trajectories', ]

    for task_name in tqdm(task_names):
        print("task_name: ", task_name)
        params = task_name.split("_")
        data_num = int(params[1])
        seed_num = int(params[3])

        rsult_path = os.path.join(common_config.result_path, "virmot", task_name)
        if os.path.exists(rsult_path):
            continue
        construct_kitti_dataset_4_Vir(task_name)
