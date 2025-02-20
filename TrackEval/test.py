import os

from natsort import natsorted

from TrackEval.scripts.run_kitti import eval_kitti
from config import common_config


def eval_dfmot_ori():
    name = "dfmot"
    base_dir = common_config.result_path
    dt_dir = os.path.join(base_dir)
    GT_FOLDER = os.path.join(common_config.project_dirname, "_datasets", "KITTI_TRACKING", "training")
    SEQ_MAP_DIR = os.path.join(common_config.project_dirname, "_datasets", "KITTI_TRACKING")
    TRACKERS_FOLDER = os.path.join(dt_dir)
    TRACKERS_TO_EVAL = ["ori"]
    TRACKER_SUB_FOLDER = "dfmot"

    OUTPUT_FOLDER = os.path.join(common_config.stastic_path, "ori", name)

    result = eval_kitti(GT_FOLDER, TRACKERS_FOLDER, SEQ_MAP_DIR, TRACKER_SUB_FOLDER, TRACKERS_TO_EVAL, OUTPUT_FOLDER)


def eval_jmodt_ori():
    name = "jmodt"
    base_dir = common_config.result_path
    dt_dir = os.path.join(base_dir, "ori", name)
    GT_FOLDER = os.path.join(common_config.project_dirname, "_datasets", "KITTI_TRACKING", "training")
    SEQ_MAP_DIR = os.path.join(common_config.project_dirname, "_datasets", "KITTI_TRACKING")
    TRACKERS_FOLDER = os.path.join(dt_dir)
    TRACKERS_TO_EVAL = ["mot_data"]
    TRACKER_SUB_FOLDER = "val"

    OUTPUT_FOLDER = os.path.join(common_config.stastic_path, "ori", name)

    result = eval_kitti(GT_FOLDER, TRACKERS_FOLDER, SEQ_MAP_DIR, TRACKER_SUB_FOLDER, TRACKERS_TO_EVAL, OUTPUT_FOLDER)


def eval_jmodt(task_name, OUTPUT_FOLDER):
    params = task_name.split("_")
    seq = params[1]
    seed = params[3]
    gt_dir = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/_aug_datasets"
    dt_dir = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/result"
    SEQ_MAP_DIR = os.path.join(gt_dir, task_name, "KITTI", "training", "seqmap")
    os.makedirs(SEQ_MAP_DIR, exist_ok=True)
    seq_name = 'evaluate_tracking.seqmap.training'
    SEQ_MAP = os.path.join(SEQ_MAP_DIR, seq_name)
    if not os.path.exists(SEQ_MAP):
        train_input_dir = os.path.join(gt_dir, task_name, "KITTI", "training")
        train_iamge_dir = os.path.join(train_input_dir, "image_02")
        frames = natsorted(os.listdir(os.path.join(train_iamge_dir, seq)))
        min_frame = int(frames[0].split('.')[0])
        max_frame = int(frames[-1].split('.')[0]) + 1
        with open(SEQ_MAP, "w+") as f:
            f.write(f"{seq} empty {min_frame} {max_frame}")
    GT_FOLDER = os.path.join(gt_dir, task_name, "KITTI", "training")
    TRACKERS_FOLDER = os.path.join(dt_dir, "jmodt", task_name)
    TRACKER_SUB_FOLDER = "val"
    TRACKERS_TO_EVAL = ["mot_data"]

    result = eval_kitti(GT_FOLDER, TRACKERS_FOLDER, SEQ_MAP_DIR, TRACKER_SUB_FOLDER, TRACKERS_TO_EVAL, OUTPUT_FOLDER)


def eval_dfmot(task_name, OUTPUT_FOLDER):
    name = "dfmot"
    params = task_name.split("_")
    seq = params[1]
    seed = params[3]
    gt_dir = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/_aug_datasets"
    dt_dir = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/result"
    SEQ_MAP_DIR = os.path.join(gt_dir, task_name, "KITTI", "training", "seqmap")
    os.makedirs(SEQ_MAP_DIR, exist_ok=True)
    seq_name = 'evaluate_tracking.seqmap.training'
    SEQ_MAP = os.path.join(SEQ_MAP_DIR, seq_name)
    if not os.path.exists(SEQ_MAP):
        train_input_dir = os.path.join(gt_dir, task_name, "KITTI", "training")
        train_iamge_dir = os.path.join(train_input_dir, "image_02")
        frames = natsorted(os.listdir(os.path.join(train_iamge_dir, seq)))
        min_frame = int(frames[0].split('.')[0])
        max_frame = int(frames[-1].split('.')[0]) + 1
        with open(SEQ_MAP, "w+") as f:
            f.write(f"{seq} empty {min_frame} {max_frame}")
    GT_FOLDER = os.path.join(gt_dir, task_name, "KITTI", "training")
    TRACKERS_FOLDER = os.path.join(dt_dir, name)
    TRACKER_SUB_FOLDER = task_name
    TRACKERS_TO_EVAL = ["./"]

    result = eval_kitti(GT_FOLDER, TRACKERS_FOLDER, SEQ_MAP_DIR, TRACKER_SUB_FOLDER, TRACKERS_TO_EVAL, OUTPUT_FOLDER)


if __name__ == '__main__':

    task_names = natsorted(os.listdir(common_config.aug_datasets_dirname))

    base_output_path = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/result/"
    for task_name in task_names:
        OUTPUT_FOLDER = f"{base_output_path}/jmodt_hota/{task_name}"
        print(task_name)
        if os.path.exists(OUTPUT_FOLDER):
            continue
        eval_jmodt(task_name, OUTPUT_FOLDER)

    name = "dfmot"
    OUTPUT_FOLDER = f"/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/result/{name}_hota/{task_name}"
    for task_name in task_names:
        OUTPUT_FOLDER = f"{base_output_path}/dfmot_hota/{task_name}"
        if os.path.exists(OUTPUT_FOLDER):
            continue
        eval_dfmot(task_name, OUTPUT_FOLDER)
