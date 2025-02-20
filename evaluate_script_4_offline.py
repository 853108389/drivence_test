from evaluate_utils import get_config
from kitti_dataset_construct import construct_kitti_dataset_4_epnet, construct_kitti_dataset_4_second
from kitti_depth_dataset_construct import construct_kitti_dataset_4_twise, construct_kitti_dataset_4_mda

from evaluate_script import evaluate_jmodt, evaluate_dfmot, \
    evaluate_twise, evaluate_mda, evaluate_second
from kitti_tracking_dataset_construct import construct_kitti_dataset_4_jmodt, construct_kitti_dataset_4_dfmot


def jmodt(noise_arr):
    name = "jmodt"
    for noise in noise_arr:
        config = get_config(name)
        config["noise"] = noise
        print(config)
        construct_kitti_dataset_4_jmodt(config)
        evaluate_jmodt(config)


def dfmot(noise_arr):
    name = "dfmot"
    for noise in noise_arr:
        config = get_config(name)
        config["noise"] = noise
        print(config)
        print("=" * 20, "construct dataset", "=" * 20)
        construct_kitti_dataset_4_dfmot(config)
        print("=" * 20, "evaluate", "=" * 20)
        evaluate_dfmot(config)


def twise(noise_arr):
    name = "twise"
    for noise in noise_arr:
        config = get_config(name)
        config["noise"] = noise
        print(config)
        construct_kitti_dataset_4_twise(config)
        evaluate_twise(config)


def mda(noise_arr):
    name = "mda"
    for noise in noise_arr:
        config = get_config(name)
        config["noise"] = noise
        print(config)
        construct_kitti_dataset_4_mda(config)
        evaluate_mda(config)


def second(noise_arr):
    name = "second"
    for noise in noise_arr:
        config = get_config(name)
        config["noise"] = noise
        print(config)
        construct_kitti_dataset_4_second(config)
        evaluate_second(config)


if __name__ == '__main__':
    config = get_config("epnet")
    config["noise"] = {"img": None, "lidar": None, "calib": None},
    print(config)
    construct_kitti_dataset_4_epnet(config)
