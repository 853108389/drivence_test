import os

import config


def filter_cars_by_id(obj_car_dirs):
    assets_dir = config.common_config.assets_dirname
    assets_dir = os.path.join(assets_dir, "filter_id")
    broken_id_path = os.path.join(assets_dir, "broken_id.txt")
    low_quality_path = os.path.join(assets_dir, "low_quality.txt")
    small_path = os.path.join(assets_dir, "small.txt")
    with open(broken_id_path, "r") as f:
        ids1 = f.readlines()
    with open(low_quality_path, "r") as f:
        ids2 = f.readlines()
    with open(small_path, "r") as f:
        ids3 = f.readlines()
    skip_model_arr = [str(x).strip() for x in ids1 + ids2 + ids3]
    cnt = 0
    for oid in skip_model_arr:
        if oid in obj_car_dirs:
            obj_car_dirs.remove(oid)
        else:
            cnt += 1

    return obj_car_dirs


def filter_cars_by_type(car_type_dict):
    car_type_arr = ["car", "sedan", "coupe", "sport utility", "SUV", "cab"]
    res_arr = []
    for key in car_type_arr:
        if key in car_type_dict.keys():
            res_arr += car_type_dict[key]
    return res_arr


def filter_cars(car_type_dict):
    obj_car_dirs = filter_cars_by_type(car_type_dict)
    obj_car_dirs = filter_cars_by_id(obj_car_dirs)
    return obj_car_dirs


if __name__ == '__main__':
    assets_dir = config.common_config.assets_dirname
    p = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/Multimodality/_assets/shapenet"
    num = os.listdir(p)
    print(len(num))
