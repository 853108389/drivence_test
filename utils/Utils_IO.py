import os

import cv2
import numpy as np
from natsort import natsorted


class UtilsIO(object):
    @staticmethod
    def generate_info_ImageSets(kitti_base_aug_dir: str, system_name: str) -> None:

        kitti_aug_dir = os.path.join(kitti_base_aug_dir, system_name, "training")
        kitti_imagesets_dir = os.path.join(kitti_base_aug_dir, system_name, "ImageSets")
        os.makedirs(kitti_imagesets_dir, exist_ok=True)
        kitti_trainval_txt_file = "trainval.txt"
        kitti_val_txt_file = "val.txt"
        kitti_aug_train_val = os.path.join(kitti_imagesets_dir, kitti_trainval_txt_file)
        kitti_aug_val = os.path.join(kitti_imagesets_dir, kitti_val_txt_file)
        kitti_aug_train = os.path.join(kitti_imagesets_dir, "train.txt")
        kitti_aug_test = os.path.join(kitti_imagesets_dir, "test.txt")

        if os.path.exists(kitti_aug_train_val):
            os.remove(kitti_aug_train_val)
        if os.path.exists(kitti_aug_val):
            os.remove(kitti_aug_val)

        fns = natsorted(os.listdir(os.path.join(kitti_aug_dir, "label_2")))
        dir_seq_arr = []
        for fn_name in fns:
            dir_seq_arr.append(fn_name.split(".")[0])

        with open(kitti_aug_train_val, "a") as f:
            for seq in dir_seq_arr:
                f.writelines(str(seq) + "\n")

        with open(kitti_aug_val, "a") as f:
            for seq in dir_seq_arr:
                f.writelines(str(seq) + "\n")

        with open(kitti_aug_train, "w") as f:
            ...
        with open(kitti_aug_test, "w") as f:
            ...

    @staticmethod
    def load_labels_2(path: str, ignore_type=None) -> list:

        data = []
        with open(path, 'r') as f:
            for line in f:
                data_line = line.strip("\n").split()
                if ignore_type is not None:
                    flag = True
                    for key in ignore_type:
                        if key in data_line:
                            flag = False
                            break
                    if flag:
                        data.append(data_line)
                else:

                    data.append(data_line)
        return data

    @staticmethod
    def load_road_split_labels(label_path: str) -> list:

        labels = np.fromfile(label_path, dtype=np.uint32).reshape((-1, 1))
        return list(labels)

    @staticmethod
    def write_labels_2(path: str, labels: list) -> None:

        with open(path, 'w') as f:
            for label in labels:
                f.writelines(" ".join(map(str, label)) + "\n")

    @staticmethod
    def append_labels_2(path: str, labels: list) -> None:

        with open(path, 'a') as f:
            for label in labels:
                f.writelines(" ".join(label) + "\n")

    @staticmethod
    def load_pcd(path: str, return_pcd=False):
        if path.split(".")[-1] == "npy":
            example = np.load(path).astype(np.float32)
        else:
            example = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        example_xyz = example[:, :3]

        if return_pcd:
            return example
        return example_xyz

    @staticmethod
    def load_img(path: str, option=None):
        if option is None:
            img = cv2.imread(path)
        else:
            img = cv2.imread(path, option)

        return img
