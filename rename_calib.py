import os
import shutil


def rename_calib():
    for p in os.listdir(base_dir):
        calib_dir = os.path.join(base_dir, p, "KITTI", "training", "calib")
        rename_calib_dir = os.path.join(base_dir, p, "KITTI", "training", "calib_dec")
        os.rename(calib_dir, rename_calib_dir)


if __name__ == '__main__':
    base_dir = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/_aug_datasets"
    ori_calib_dir = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/_datasets/KITTI_TRACKING/training/calib"
    for p in os.listdir(base_dir):
        calib_dir = os.path.join(base_dir, p, "KITTI", "training", "calib")
        shutil.copytree(ori_calib_dir, calib_dir)
