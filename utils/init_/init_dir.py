import os
import shutil

import config


def symlink(input_path, output_path):
    if not os.path.exists(input_path):
        raise ValueError("input: ", input_path)

    if os.path.exists(output_path):
        os.remove(output_path)
    os.symlink(input_path, output_path)


class InitDir(object):

    def __init__(self, args):

        os.makedirs(os.path.join(config.common_config.aug_queue_datasets_dirname), exist_ok=True)

    @staticmethod
    def init_plain_result_dir(kitti_aug_result_root, execute_task_name, sub_dirs):

        kitti_aug_plain_result_dir = os.path.join(kitti_aug_result_root, execute_task_name)

        kitti_testing = os.path.join(kitti_aug_plain_result_dir, "testing")

        kitti_training = os.path.join(kitti_aug_plain_result_dir, "training")

        os.makedirs(kitti_testing, exist_ok=True)

        os.makedirs(kitti_training, exist_ok=True)

        for sub_dir in sub_dirs:
            os.makedirs(os.path.join(kitti_training, sub_dir), exist_ok=True)

        return kitti_training

    @staticmethod
    def init_plain_result_dir_tracking(kitti_aug_result_root, execute_task_name, seq):

        seq = '%04d' % seq

        kitti_aug_plain_result_dir = os.path.join(kitti_aug_result_root, execute_task_name, "KITTI")

        kitti_testing = os.path.join(kitti_aug_plain_result_dir, "testing")

        kitti_training = os.path.join(kitti_aug_plain_result_dir, "training")

        os.makedirs(kitti_testing, exist_ok=True)

        os.makedirs(kitti_training, exist_ok=True)

        sub_dirs1 = ["image_02", "velodyne", "result", "scene_file", "image_02_label", "label_02_temp", "label_02_dec",
                     "calib_dec"]
        sub_dirs2 = ["label_02", "calib", "gif"]

        for sub_dir in sub_dirs1:
            os.makedirs(os.path.join(kitti_training, sub_dir, seq), exist_ok=True)
        for sub_dir in sub_dirs2:
            os.makedirs(os.path.join(kitti_training, sub_dir), exist_ok=True)

        return kitti_training

    @staticmethod
    def init_queue_guided_result_dir(kitti_aug_queue_result_root, execute_name, file_index, sub_dirs):

        save_dir = os.path.join(kitti_aug_queue_result_root, execute_name, file_index)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        dirs = []

        for sub_dir in sub_dirs:
            temp_dirname = os.path.join(save_dir, sub_dir)

            dirs.append(temp_dirname)

            os.makedirs(temp_dirname, exist_ok=True)

        return dirs


if __name__ == '__main__':
    ...
