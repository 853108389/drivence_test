import os
import shutil
import subprocess

from evaluate_utils import get_op, get_config, get_tracking_suffix
from mmdetection.train_2D import get_d2_model
from natsort import natsorted
from tqdm import tqdm


def symlink(input_path, output_path):
    if not os.path.exists(input_path):
        raise ValueError(input_path)

    if os.path.exists(output_path):
        os.remove(output_path)
    os.symlink(input_path, output_path)
    print(input_path, "-->", output_path)


def construct_kitti_dataset_detail(config):
    base_path = config["base_path"]
    input_dir = os.path.join(base_path, config["public_dir"])
    output_dir = os.path.join(base_path, config["work_dir"])
    os.makedirs(output_dir, exist_ok=True)

    test_input_path = os.path.join(input_dir, "testing")
    test_output_path = os.path.join(output_dir, "testing")

    symlink(test_input_path, test_output_path)

    train_input_dir = os.path.join(input_dir, "training")
    train_output_dir = os.path.join(output_dir, "training")
    os.makedirs(train_output_dir, exist_ok=True)

    train_label_input_dir = os.path.join(train_input_dir, "label_02")
    train_label_output_dir = os.path.join(train_output_dir, "label_02")

    symlink(train_label_input_dir, train_label_output_dir)

    keys = config["noise"].keys()
    op_arr = []
    suffix_arr = []
    for key, op in config["noise"].items():
        op_arr.append(get_op(op))
        suffix_arr.append(get_tracking_suffix(key))

    for key, suffix, op in zip(keys, suffix_arr, op_arr):
        name = "{}_{}".format(op, suffix)
        train_key_input_dir = os.path.join(train_input_dir, "noise", suffix, name)
        train_key_output_dir = os.path.join(train_output_dir, suffix)
        symlink(train_key_input_dir, train_key_output_dir)


def construct_kitti_dataset_4_jmodt(config):
    construct_kitti_dataset_detail(config)
    input_result_dir = os.path.join(config["base_path"], config["work_dir"])
    output_result_dir = "./JMODT/data/KITTI/tracking"
    if os.path.exists(output_result_dir):
        os.remove(output_result_dir)
    symlink(input_result_dir, output_result_dir)
    data_root = os.path.join(config["project_dir"], "JMODT/data/KITTI")
    cmd1 = "cd ./JMODT"
    cmd2 = "python tools/kitti_converter.py --data_root  {}".format(data_root)
    os.system("{} && {}".format(cmd1, cmd2))


def copy_d2_data4dfmot(config):
    input_dir = os.path.join(config["base_path"], config["public_dir"], "training", "noise", "d2_detection_data",
                             get_op(config["noise"]["img"]))
    output_dir = os.path.join(config["base_path"], config["work_dir"], "train", "2D_rrc_Car_val")
    if not os.path.exists(input_dir):
        inference_d2_results(config, input_dir)
    symlink(input_dir, output_dir)


def copy_d3_data4dfmot(config):
    input_dir = os.path.join(config["base_path"], config["public_dir"], "training", "noise", "d3_detection_data",
                             "{}_velodyne_{}_calib".format(get_op(config["noise"]["lidar"]),
                                                           get_op(config["noise"]["calib"])))
    output_dir = os.path.join(config["base_path"], config["work_dir"], "train", "3D_pointrcnn_Car_val")
    bp = config["project_dir"]
    ori_input_dir = "./openpcdet/output/cfgs/kitti_models/second/default/eval/epoch_7862/val/default/final_result/data"
    if not os.path.exists(input_dir):
        print("=" * 5, "copy to  jmodt", "=" * 5)
        construct_kitti_dataset_4_jmodt(config)
        print("=" * 5, "link to pcd", "=" * 5)

        symlink(os.path.join(bp, "JMODT/data/KITTI/tracking_object/testing/"),
                os.path.join(bp, "openpcdet/data/kitti/testing"))
        symlink(os.path.join(bp, "openpcdet/data/kitti/tracking_ImageSets"),
                os.path.join(bp, "openpcdet/data/kitti/ImageSets"))
        symlink(os.path.join(bp, "JMODT/data/KITTI/tracking_object/training/"),
                os.path.join(bp, "openpcdet/data/kitti/training"))
        print("=" * 5, "run open pcd", "=" * 5)
        inference_d3_results(config, './env.sh')
        print("=" * 5, "convert pcd to dfmot", "=" * 5)
        assert os.path.exists(ori_input_dir)
        convert_obj_ouput2tracking_output(ori_input_dir, input_dir)
    symlink(input_dir, output_dir)


def convert_obj_ouput2tracking_output(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    seq2sample = "./seq2sample.txt"
    with open(seq2sample, "r") as f:
        indexes = f.readlines()

    d = {}
    for s in indexes:
        arr = s.split(" ")
        d[arr[0]] = []
        for ix in arr[1:]:
            d[arr[0]].append(ix)

    for k, fn_arr in tqdm(d.items()):
        p = os.path.join(output_dir, "{}.txt".format(k))
        with open(p, "w") as fout:
            for fid, fn in enumerate(fn_arr):
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


def inference_d3_results(config, bash_path, clear_anno=True):
    os.system('export PATH="~/anaconda3/bin:$PATH"')
    bp = config["project_dir"]
    output_idr = os.path.join(bp, "openpcdet/output/cfgs/kitti_models/second/default/eval/epoch_7862/val/default/")
    if os.path.exists(output_idr):
        shutil.rmtree(output_idr)
    if clear_anno:
        kitti_info = os.path.join(bp, "openpcdet/data/kitti/kitti_infos_val.pkl")
        if os.path.exists(kitti_info):
            os.remove(kitti_info)
    p = subprocess.run([bash_path], shell=True)


def inference_d2_results(config, txt_dir):
    from mmdetection.train_2D import my_inference as d2_inference
    model = get_d2_model()
    python_path = '/home/niangao/PycharmProjects/mmdetection'
    add_python_path(python_path)
    base_image_2_dir = os.path.join(config["base_path"], config["work_dir"], "train", "image_02_train")
    os.makedirs(txt_dir, exist_ok=True)
    sub_dirs = natsorted(os.listdir(base_image_2_dir))
    for sub_ix, sub_dir in enumerate(sub_dirs):
        image_2_dir = os.path.join(base_image_2_dir, sub_dir)
        txt_path = "{}/{}.txt".format(txt_dir, sub_dir)
        if os.path.exists(txt_path):
            os.remove(txt_path)
        fns = natsorted(os.listdir(image_2_dir))
        with open(txt_path, "w") as f:
            for fid, fn in enumerate(tqdm(fns)):
                img_path = os.path.join(image_2_dir, fn)
                _, d2_str_results = d2_inference(img_path, model=model, fm="dfmot", fid=fid)
                if len(d2_str_results) <= 1:
                    continue
                else:
                    if fn != len(fns) - 1:
                        d2_str_results = d2_str_results + "\n"
                    f.write(d2_str_results)
    remove_python_path(python_path)


def construct_kitti_dataset_4_dfmot(config):
    print("=" * 10, "link dataset", "=" * 10)
    base_path = config["base_path"]
    input_dir = os.path.join(base_path, config["public_dir"])
    output_dir = os.path.join(base_path, config["work_dir"])
    os.makedirs(output_dir, exist_ok=True)

    train_input_dir = os.path.join(input_dir, "training")
    train_output_dir = os.path.join(output_dir, "train")
    os.makedirs(train_output_dir, exist_ok=True)

    train_label_input_dir = os.path.join(train_input_dir, "label_02")
    train_label_output_dir = os.path.join(train_output_dir, "label_02")

    symlink(train_label_input_dir, train_label_output_dir)

    keys = list(config["noise"].keys())
    op_arr = []
    suffix_arr = []
    for key, op in config["noise"].items():
        op_arr.append(get_op(op))
        suffix_arr.append(get_tracking_suffix(key, fm=config["name"]))

    for key, suffix, op in zip(keys, suffix_arr, op_arr):
        if key == "lidar":
            continue
        name = "{}_{}".format(op, suffix)
        train_key_input_dir = os.path.join(train_input_dir, "noise", suffix, name)
        train_key_output_dir = os.path.join(train_output_dir, "{}_train".format(suffix))
        symlink(train_key_input_dir, train_key_output_dir)

    print("=" * 10, "d2 detection data", "=" * 10)
    copy_d2_data4dfmot(config)
    print("=" * 10, "d3 detection data", "=" * 10)
    copy_d3_data4dfmot(config)

    print("=" * 10, "copy to dfmot dir", "=" * 10)
    base_input_result_dir = os.path.join(config["base_path"], config["work_dir"], "train")
    base_output_result_dir = "./DeepFusionMOT/datasets/kitti/train"
    sub_dirs = os.listdir(base_input_result_dir)
    for sub in sub_dirs:
        input_result_dir = os.path.join(base_input_result_dir, sub)
        output_result_dir = os.path.join(base_output_result_dir, sub)
        if os.path.exists(output_result_dir):
            os.remove(output_result_dir)
        symlink(input_result_dir, output_result_dir)


def add_python_path(p):
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = p
    else:
        os.environ["PYTHONPATH"] += ':{}'.format(p)


def remove_python_path(p):
    p = ':{}'.format(p)
    os.environ["PYTHONPATH"] = os.environ["PYTHONPATH"].replace(p, "")


if __name__ == '__main__':
    config = get_config(name="dfmot")
    inference_d3_results(config, './env.sh')
