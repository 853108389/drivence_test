from __future__ import print_function

import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "mayavi"))
import kitti_util2 as utils
import argparse

try:
    raw_input
except NameError:
    raw_input = input

cbox = np.array([[0, 70.4], [-40, 40], [-3, 1]])


class kitti_object(object):
    def __init__(self, root_dir, lidardir=None, preddir=None, split="training", args=None):
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == "training":
            self.num_samples = 7481
        elif split == "testing":
            self.num_samples = 7518
        else:
            print("Unknown split: %s" % (split))
            exit(-1)

        lidar_dir = "velodyne"
        depth_dir = "depth"
        pred_dir = "pred"
        d2_pred_dir = "d2_detection_data"

        if args is not None:
            lidar_dir = args.lidar
            depth_dir = args.depthdir
            pred_dir = args.preddir
        if preddir is not None:
            pred_dir = preddir
        if lidardir is not None:
            lidar_dir = lidardir

        self.image_dir = os.path.join(self.split_dir, "image_2")
        self.label_dir = os.path.join(self.split_dir, "label_2")
        self.calib_dir = os.path.join(self.split_dir, "calib")

        self.depthpc_dir = os.path.join(self.split_dir, "depth_pc")
        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        self.depth_dir = os.path.join(self.split_dir, depth_dir)
        self.pred_dir = os.path.join(self.split_dir, pred_dir)
        self.d2_pred_dir = os.path.join(self.split_dir, d2_pred_dir)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.image_dir, "%06d.png" % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        assert idx < self.num_samples
        lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % (idx))
        return utils.load_velo_scan(lidar_filename, dtype, n_vec)

    def get_calibration(self, idx):
        assert idx < self.num_samples
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert idx < self.num_samples and self.split == "training"
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        return utils.read_label(label_filename)

    def get_pred_objects(self, idx):
        assert idx < self.num_samples
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return utils.read_label(pred_filename)
        else:
            raise ValueError("{} not exist".format(pred_filename))

    def get_d2_pred_objects(self, idx):
        assert idx < self.num_samples
        pred_filename = os.path.join(self.d2_pred_dir, "%06d.txt" % (idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return utils.read_label(pred_filename)
        else:
            return None

    def get_depth(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
        return utils.load_depth(img_filename)

    def get_depth_image(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
        return utils.load_depth(img_filename)

    def get_depth_pc(self, idx):
        assert idx < self.num_samples
        lidar_filename = os.path.join(self.depthpc_dir, "%06d.bin" % (idx))
        is_exist = os.path.exists(lidar_filename)
        if is_exist:
            return utils.load_velo_scan(lidar_filename), is_exist
        else:
            return None, is_exist

    def get_top_down(self, idx):
        pass

    def isexist_pred_objects(self, idx):
        assert idx < self.num_samples and self.split == "training"
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        return os.path.exists(pred_filename)

    def isexist_depth(self, idx):
        assert idx < self.num_samples and self.split == "training"
        depth_filename = os.path.join(self.depth_dir, "%06d.txt" % (idx))
        return os.path.exists(depth_filename)

    def get_label_anno(self, idx, obj_interesting=None):
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        return self._get_label_anno(label_filename, obj_interesting)

    def get_pred_anno(self, idx, obj_interesting=None):
        label_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        return self._get_label_anno(label_filename, obj_interesting)

    def _get_label_anno(self, label_path, obj_interesting):
        annotations = {}
        annotations.update({
            'name': [],
            'truncated': [],
            'occluded': [],
            'alpha': [],
            'bbox': [],
            'dimensions': [],
            'location': [],
            'rotation_y': []
        })
        with open(label_path, 'r') as f:
            lines = f.readlines()
        content = [line.strip().split(' ') for line in lines]
        print(content)
        if obj_interesting is not None:
            content = [x for x in content if x[0] == obj_interesting]
        print(content)
        num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
        annotations['name'] = np.array([x[0] for x in content])
        num_gt = len(annotations['name'])
        annotations['truncated'] = np.array([float(x[1]) for x in content])
        annotations['occluded'] = np.array([int(x[2]) for x in content])
        annotations['alpha'] = np.array([float(x[3]) for x in content])
        annotations['bbox'] = np.array(
            [[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
        annotations['dimensions'] = np.array(
            [[float(info) for info in x[8:11]] for x in content]).reshape(
            -1, 3)[:, [2, 0, 1]]
        annotations['location'] = np.array(
            [[float(info) for info in x[11:14]] for x in content]).reshape(-1, 3)
        annotations['rotation_y'] = np.array(
            [float(x[14]) for x in content]).reshape(-1)
        if len(content) != 0 and len(content[0]) == 16:
            annotations['score'] = np.array([float(x[15]) for x in content])
        else:
            annotations['score'] = np.zeros((annotations['bbox'].shape[0],))
        index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
        annotations['index'] = np.array(index, dtype=np.int32)
        annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
        return [annotations]


class kitti_object_video(object):
    def __init__(self, img_dir, lidar_dir, calib_dir):
        self.calib = utils.Calibration(calib_dir, from_video=True)
        self.img_dir = img_dir
        self.lidar_dir = lidar_dir
        self.img_filenames = sorted(
            [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
        )
        self.lidar_filenames = sorted(
            [os.path.join(lidar_dir, filename) for filename in os.listdir(lidar_dir)]
        )
        print(len(self.img_filenames))
        print(len(self.lidar_filenames))
        self.num_samples = len(self.img_filenames)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = self.img_filenames[idx]
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        assert idx < self.num_samples
        lidar_filename = self.lidar_filenames[idx]
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, unused):
        return self.calib


def viz_kitti_video():
    video_path = os.path.join(ROOT_DIR, "dataset/2011_09_26/")
    dataset = kitti_object_video(
        os.path.join(video_path, "2011_09_26_drive_0023_sync/image_02/data"),
        os.path.join(video_path, "2011_09_26_drive_0023_sync/velodyne_points/data"),
        video_path,
    )
    print(len(dataset))
    for _ in range(len(dataset)):
        img = dataset.get_image(0)
        pc = dataset.get_lidar(0)
        cv2.imshow("video", img)
        draw_lidar(pc)
        raw_input()
        pc[:, 0:3] = dataset.get_calibration().project_velo_to_rect(pc[:, 0:3])
        draw_lidar(pc)
        raw_input()
    return


def show_image_with_dt_dt_boxes(img, dt_objects, gt_objects, save_path=None, hull=None):
    import matplotlib.pyplot as plt
    img1 = np.copy(img)
    h, w, c = img.shape
    plt.imshow(img1)
    for objects, color in zip([gt_objects, dt_objects], ['r', 'b']):
        if objects is None:
            continue
        for obj in objects:
            if obj.type == "DontCare":
                continue

            if obj.type == "Car":
                color = (0, 1, 0)
            if obj.type == "Pedestrian":
                color = (1, 1, 1)
            if obj.type == "Cyclist":
                color = (0, 1, 1)
            if obj.conf is not None:
                plt.text(obj.xmin, obj.ymin, str(obj.conf), fontsize=5, color=color)
            plt.gca().add_patch(
                plt.Rectangle((obj.xmin, obj.ymin), obj.xmax - obj.xmin,
                              obj.ymax - obj.ymin, fill=False,
                              edgecolor=color,
                              linewidth=1))

    if hull is not None:
        print(hull.shape)
        print(np.array([hull[0]]).shape)
        hull = np.concatenate([np.array([hull[0]]), hull], axis=0)
        print(hull.shape)
        plt.plot(hull[:, 0], hull[:, 1], 'r--', lw=1)
    if save_path is not None:
        plt.xticks([])
        plt.yticks([])
        plt.savefig(save_path, bbox_inches="tight", pad_inches=-0.05)
    else:
        plt.show()
    plt.close()


def show_image_with_boxes(img, objects, calib, show3d=True, depth=None):
    img1 = np.copy(img)
    img2 = np.copy(img)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        if obj.type == "Car":
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (0, 255, 0),
                2,
            )
        if obj.type == "Pedestrian":
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (255, 255, 0),
                2,
            )
        if obj.type == "Cyclist":
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                (0, 255, 255),
                2,
            )
        box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is None:
            print("something wrong in the 3D box.")
            continue
        if obj.type == "Car":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))
        elif obj.type == "Pedestrian":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(255, 255, 0))
        elif obj.type == "Cyclist":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 255))

    cv2.imshow("2dbox", img1)
    show3d = True
    if show3d:
        cv2.imshow("3dbox", img2)
    if depth is not None:
        cv2.imshow("depth", depth)

    return img1, img2


def show_image_with_boxes_3type(img, objects, calib, objects2d, name, objects_pred):
    img1 = np.copy(img)
    type_list = ["Pedestrian", "Car", "Cyclist"]
    color = (0, 255, 0)
    for obj in objects:
        if obj.type not in type_list:
            continue
        cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            color,
            3,
        )
    startx = 5
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_lables = [obj.type for obj in objects if obj.type in type_list]
    text_lables.insert(0, "Label:")
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
    color = (0, 0, 255)
    for obj in objects2d:
        cv2.rectangle(
            img1,
            (int(obj.box2d[0]), int(obj.box2d[1])),
            (int(obj.box2d[2]), int(obj.box2d[3])),
            color,
            2,
        )
    startx = 85
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_lables = [type_list[obj.typeid - 1] for obj in objects2d]
    text_lables.insert(0, "2D Pred:")
    for n in range(len(text_lables)):
        text_pos = (startx, 25 * (n + 1))
        cv2.putText(img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA)
    if objects_pred is not None:
        color = (255, 0, 0)
        for obj in objects_pred:
            if obj.type not in type_list:
                continue
            cv2.rectangle(
                img1,
                (int(obj.xmin), int(obj.ymin)),
                (int(obj.xmax), int(obj.ymax)),
                color,
                1,
            )
        startx = 165
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_lables = [obj.type for obj in objects_pred if obj.type in type_list]
        text_lables.insert(0, "3D Pred:")
        for n in range(len(text_lables)):
            text_pos = (startx, 25 * (n + 1))
            cv2.putText(
                img1, text_lables[n], text_pos, font, 0.5, color, 0, cv2.LINE_AA
            )

    cv2.imshow("with_bbox", img1)
    cv2.imwrite("imgs/" + str(name) + ".png", img1)


def get_lidar_in_image_fov(
        pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
):
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
            (pts_2d[:, 0] < xmax)
            & (pts_2d[:, 0] >= xmin)
            & (pts_2d[:, 1] < ymax)
            & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def get_lidar_index_in_image_fov(
        pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
):
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
            (pts_2d[:, 0] < xmax)
            & (pts_2d[:, 0] >= xmin)
            & (pts_2d[:, 1] < ymax)
            & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    return fov_inds


def depth_region_pt3d(depth, obj):
    b = obj.box2d
    pt3d = []
    for i in range(int(b[0]), int(b[2])):
        for j in range(int(b[1]), int(b[3])):
            pt3d.append([j, i, depth[j, i]])
    return np.array(pt3d)


def get_depth_pt3d(depth):
    pt3d = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            pt3d.append([i, j, depth[i, j]])
    return np.array(pt3d)


def show_lidar_with_depth(
        pc_velo,
        objects,
        calib,
        fig,
        img_fov=False,
        img_width=None,
        img_height=None,
        objects_pred=None,
        depth=None,
        cam_img=None,
        constraint_box=False,
        pc_label=False,
        save=False,
):
    if "mlab" not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(
            pc_velo[:, :3], calib, 0, 0, img_width, img_height
        )
        pc_velo = pc_velo[pc_velo_index, :]
        print(("FOV point num: ", pc_velo.shape))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig, pc_label=pc_label)

    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))
        print("depth_pc_velo:", depth_pc_velo.shape)
        print("depth_pc_velo:", type(depth_pc_velo))
        print(depth_pc_velo[:5])
        draw_lidar(depth_pc_velo, fig=fig, pts_color=(1, 1, 1))

        if save:
            data_idx = 0
            vely_dir = "data/object/training/depth_pc"
            save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))
            print(save_filename)
            depth_pc_velo = depth_pc_velo.astype(np.float32)
            depth_pc_velo.tofile(save_filename)

    for obj in objects:
        if obj.type == "DontCare":
            continue
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        if obj.type == "Car":
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=(0, 1, 0), label=obj.type)
        elif obj.type == "Pedestrian":
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=(0, 1, 1), label=obj.type)
        elif obj.type == "Cyclist":
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=(1, 1, 0), label=obj.type)

    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    mlab.show(1)


def save_depth0(
        data_idx,
        pc_velo,
        calib,
        img_fov,
        img_width,
        img_height,
        depth,
        constraint_box=False,
):
    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(
            pc_velo[:, :3], calib, 0, 0, img_width, img_height
        )
        pc_velo = pc_velo[pc_velo_index, :]
        type = np.zeros((pc_velo.shape[0], 1))
        pc_velo = np.hstack((pc_velo, type))
        print(("FOV point num: ", pc_velo.shape))
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))

        type = np.ones((depth_pc_velo.shape[0], 1))
        depth_pc_velo = np.hstack((depth_pc_velo, type))
        print("depth_pc_velo:", depth_pc_velo.shape)

        depth_pc = np.concatenate((pc_velo, depth_pc_velo), axis=0)
        print("depth_pc:", depth_pc.shape)

    vely_dir = "data/object/training/depth_pc"
    save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))

    depth_pc = depth_pc.astype(np.float32)
    depth_pc.tofile(save_filename)


def save_depth(
        data_idx,
        pc_velo,
        calib,
        img_fov,
        img_width,
        img_height,
        depth,
        constraint_box=False,
):
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc = np.hstack((depth_pc_velo, indensity))

        print("depth_pc:", depth_pc.shape)

    vely_dir = "data/object/training/depth_pc"
    save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))

    depth_pc = depth_pc.astype(np.float32)
    depth_pc.tofile(save_filename)


def show_lidar_with_boxes(
        pc_velo,
        objects,
        calib,
        img_fov=False,
        img_width=None,
        img_height=None,
        objects_pred=None,
        depth=None,
        cam_img=None,
):
    if "mlab" not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    if img_fov:
        pc_velo = get_lidar_in_image_fov(
            pc_velo[:, 0:3], calib, 0, 0, img_width, img_height
        )
        print(("FOV point num: ", pc_velo.shape[0]))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig)

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)
        box3d_pts_3d_velo2 = box3d_pts_3d_velo + 1
        gts = [box3d_pts_3d_velo]
        draw_gt_boxes3d(gts, fig=fig, color=color)

        if depth is not None:
            depth_pt3d = depth_region_pt3d(depth, obj)
            depth_UVDepth = np.zeros_like(depth_pt3d)
            depth_UVDepth[:, 0] = depth_pt3d[:, 1]
            depth_UVDepth[:, 1] = depth_pt3d[:, 0]
            depth_UVDepth[:, 2] = depth_pt3d[:, 2]
            print("depth_pt3d:", depth_UVDepth)
            dep_pc_velo = calib.project_image_to_velo(depth_UVDepth)
            print("dep_pc_velo:", dep_pc_velo)

            draw_lidar(dep_pc_velo, fig=fig, pts_color=(1, 1, 1))

        # Draw heading arrow
        _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        mlab.plot3d(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color=color,
            tube_radius=None,
            line_width=1,
            figure=fig,
        )
    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    mlab.show()


def my_show_lidar_with_boxes(
        pc_velo,
        objects,
        calib,
        img_fov=False,
        img_width=None,
        img_height=None,
        objects_pred=None,
        depth=None,
        cam_img=None,
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    if "mlab" not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    fig = mlab.figure(
        figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
    )
    if img_fov:
        pc_velo = get_lidar_in_image_fov(
            pc_velo[:, 0:3], calib, 0, 0, img_width, img_height
        )
        print(("FOV point num: ", pc_velo.shape[0]))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig)
    # pc_velo=pc_velo[:,0:3]

    color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)
        box3d_pts_3d_velo2 = box3d_pts_3d_velo + 1
        # gts = [box3d_pts_3d_velo, box3d_pts_3d_velo2]
        gts = [box3d_pts_3d_velo]
        draw_gt_boxes3d(gts, fig=fig, color=color)

        box3d_pts_3d_velo_filter = []
        for ax in [0]:  # , 1, 2
            # print(box3d_pts_3d_velo.shape)
            # p = np.sort(box3d_pts_3d_velo)[0, -1]
            # print(p.shape)
            pp = box3d_pts_3d_velo[np.argsort(box3d_pts_3d_velo[:, ax])]
            box3d_pts_3d_velo_filter.append(pp[0])
            box3d_pts_3d_velo_filter.append(pp[-1])
        box3d_pts_3d_velo_filter = np.array(box3d_pts_3d_velo_filter)
        print("=====")
        print(box3d_pts_3d_velo_filter)
        for gt_point in box3d_pts_3d_velo_filter:
            # gt_point = box3d_pts_3d_velo[0]
            mlab.plot3d(
                [0, gt_point[0]],
                [0, gt_point[1]],
                [0, gt_point[2]],
                color=(0, 0.5, 1),
                tube_radius=None,
                figure=fig,
            )
        # Draw depth
        if depth is not None:
            # import pdb; pdb.set_trace()
            depth_pt3d = depth_region_pt3d(depth, obj)
            depth_UVDepth = np.zeros_like(depth_pt3d)
            depth_UVDepth[:, 0] = depth_pt3d[:, 1]
            depth_UVDepth[:, 1] = depth_pt3d[:, 0]
            depth_UVDepth[:, 2] = depth_pt3d[:, 2]
            print("depth_pt3d:", depth_UVDepth)
            dep_pc_velo = calib.project_image_to_velo(depth_UVDepth)
            print("dep_pc_velo:", dep_pc_velo)

            draw_lidar(dep_pc_velo, fig=fig, pts_color=(1, 1, 1))

        # Draw heading arrow
        _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1, y1, z1 = ori3d_pts_3d_velo[0, :]
        x2, y2, z2 = ori3d_pts_3d_velo[1, :]
        mlab.plot3d(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            color=color,
            tube_radius=None,
            line_width=1,
            figure=fig,
        )
    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    mlab.show()


def box_min_max(box3d):
    box_min = np.min(box3d, axis=0)
    box_max = np.max(box3d, axis=0)
    return box_min, box_max


def get_velo_whl(box3d, pc):
    bmin, bmax = box_min_max(box3d)
    ind = np.where(
        (pc[:, 0] >= bmin[0])
        & (pc[:, 0] <= bmax[0])
        & (pc[:, 1] >= bmin[1])
        & (pc[:, 1] <= bmax[1])
        & (pc[:, 2] >= bmin[2])
        & (pc[:, 2] <= bmax[2])
    )[0]
    # print(pc[ind,:])
    if len(ind) > 0:
        vmin, vmax = box_min_max(pc[ind, :])
        return vmax - vmin
    else:
        return 0, 0, 0, 0


def stat_lidar_with_boxes(pc_velo, objects, calib):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """

    # print(('All point num: ', pc_velo.shape[0]))

    # draw_lidar(pc_velo, fig=fig)
    # color=(0,1,0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        v_l, v_w, v_h, _ = get_velo_whl(box3d_pts_3d_velo, pc_velo)
        print("%.4f %.4f %.4f %s" % (v_w, v_h, v_l, obj.type))


def get_lidar_on_image(pc_velo, calib, img_width, img_height):
    """ Project LiDAR points to image """
    # img = np.copy(img)
    pc_velo = pc_velo[:, 0:3]
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True
    )
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    return imgfov_pc_rect, imgfov_pts_2d
    # import matplotlib.pyplot as plt
    #
    # cmap = plt.cm.get_cmap("hsv", 256)
    # cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    #
    # for i in range(imgfov_pts_2d.shape[0]):
    #     depth = imgfov_pc_rect[i, 2]
    #     ix = int(640.0 / depth)
    #     if ix >= 256:
    #         ix = 255
    #     color = cmap[ix, :]
    #     cv2.circle(
    #         img,
    #         (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
    #         2,
    #         color=tuple(color),
    #         thickness=-1,
    #     )
    # cv2.imshow("projection", img)
    # return img


def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    """ Project LiDAR points to image """
    img = np.copy(img)
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True
    )
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        ix = int(640.0 / depth)
        if ix >= 256:
            ix = 255
        color = cmap[ix, :]
        cv2.circle(
            img,
            (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
            2,
            color=tuple(color),
            thickness=-1,
        )
    # cv2.imshow("projection", img)
    cv2.imwrite("./projectio3n.png", img)
    return img


def show_img_with_labels(image, labels, dt_labels=None):
    """
    :param bg_img_path: 背景图片存储路径
    :param objs_img:    物体图片组
    :param coordinates: 物体图片应插入背景图片的坐标组
    :param objs_center: 物体中心组（x,y）
    :param initial_boxes背景原始物体组的boxes
    :param objs_index:  插入物体标号索引
    :param labels:      插入物体的标签
    :return: 组合图片
    """
    # 背景图片
    # img_bg = cv2.imread(bg_img_path, cv2.IMREAD_UNCHANGED)
    # img_bg = cv2.imread(bg_img_path)
    # print(">>img_bg shape: ", img_bg.shape)

    # img_mix = combine_bg_with_obj(img_bg, objs_img, coordinates, objs_center[len(initial_boxes):])  # 调整大小后的物体收入囊中
    image = image.copy()
    # 显示
    for label in labels:
        # pt1: 左上角坐标
        # pt2: 右下角坐标
        # color: 颜色
        # thickness: 线的粗度， -1代表实心
        cv2.rectangle(image, pt1=(int(float(label[4])), int(float(label[5]))),
                      pt2=(int(float(label[6])), int(float(label[7]))), color=(0, 255, 0), thickness=2)
    if dt_labels is not None:
        for label in dt_labels:
            # pt1: 左上角坐标
            # pt2: 右下角坐标
            # color: 颜色
            # thickness: 线的粗度， -1代表实心
            cv2.rectangle(image, pt1=(int(float(label[4])), int(float(label[5]))),
                          pt2=(int(float(label[6])), int(float(label[7]))), color=(0, 0, 255), thickness=2)
    return image


def filter_lidar_on_image_by_semantic_label(pc_velo, img_sematic, calib, labels):
    img_height, img_width = img_sematic.shape
    pts_2d = calib.project_velo_to_image(pc_velo)
    xmin, ymin, xmax, ymax, clip_distance = 0, 0, img_width, img_height, 1
    fov_inds = (
            (pts_2d[:, 0] < xmax)
            & (pts_2d[:, 0] >= xmin)
            & (pts_2d[:, 1] < ymax)
            & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    idx_list = np.array(list(range(0, pc_velo.shape[0])))
    idx_fov = idx_list[fov_inds]

    road_pts_idx = []
    for i in idx_fov:
        y_index = int(np.round(pts_2d[i, 0]))
        x_index = int(np.round(pts_2d[i, 1]))
        try:
            v = img_sematic[x_index][y_index]
            if v in labels:
                road_pts_idx.append(i)
        except:
            ...

    return road_pts_idx


# def filter_lidar_on_image(pc_velo, img_sematic, calib, img_width, img_height):
#     pts_2d = calib.project_velo_to_image(pc_velo)
#     xmin, ymin, xmax, ymax, clip_distance = 0, 0, img_width, img_height, 1
#     fov_inds = (
#             (pts_2d[:, 0] < xmax)
#             & (pts_2d[:, 0] >= xmin)
#             & (pts_2d[:, 1] < ymax)
#             & (pts_2d[:, 1] >= ymin)
#     )
#     fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
#     idx_list = np.array(list(range(0,pc_velo.shape[0])))
#     idx_fov = idx_list[fov_inds]
#
#     # for i in range(len(pc_velo.shape[0])):
#     #     if i in
#     #     pts_2d[i]
#     road_pts_idx = []
#     road_non_pts_idx = []
#     for i in idx_fov:
#         y_index = int(np.round(pts_2d[i, 0]))
#         x_index = int(np.round(pts_2d[i, 1]))
#         try:
#             v = img_sematic[x_index][y_index]
#             if v == 0:
#                 road_pts_idx.append(i)
#             elif v in [1]: #76  ,2,3
#                 road_non_pts_idx.append(i)
#             else:
#                 if v not in [2,3,4,5,6,76]:
#                     print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv",v) #6 149
#         except:
#             ...
#
#     return road_pts_idx,road_non_pts_idx


# def my_filter_lidar_on_image(pc_velo, img_sematic, calib, img_width, img_height):
#     pts_2d = calib.project_velo_to_image(pc_velo)
#     xmin, ymin, xmax, ymax, clip_distance = 0, 0, img_width, img_height, 1
#     fov_inds = (
#             (pts_2d[:, 0] < xmax)
#             & (pts_2d[:, 0] >= xmin)
#             & (pts_2d[:, 1] < ymax)
#             & (pts_2d[:, 1] >= ymin)
#     )
#     fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
#     idx_list = np.array(list(range(0,pc_velo.shape[0])))
#     idx_fov = idx_list[fov_inds]
#
#     # for i in range(len(pc_velo.shape[0])):
#     #     if i in
#     #     pts_2d[i]
#     road_pts_idx = []
#     for i in idx_fov:
#         y_index = int(np.round(pts_2d[i, 0]))
#         x_index = int(np.round(pts_2d[i, 1]))
#         try:
#             if img_sematic[x_index][y_index] == 0:
#                 road_pts_idx.append(i)
#         except:
#             ...
#
#     return road_pts_idx


def my_show_lidar_on_image(pc_velo, img2, calib, img_width, img_height):
    """ Project LiDAR points to image """
    img = np.ones_like(img2) * 255
    img3 = img2.copy()
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True
    )
    imgfov_pts_2d = pts_2d[fov_inds, :]
    print(len(imgfov_pts_2d))
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    # plt.imshow(img)
    for i in tqdm(range(imgfov_pts_2d.shape[0])):
        depth = imgfov_pc_rect[i, 2]
        ix = int(640.0 / depth)
        if ix >= 256:
            ix = 255
        color = cmap[ix, :]
        # color = [[int(cc) for cc in color]]
        y_index = int(np.round(imgfov_pts_2d[i, 0]))
        x_index = int(np.round(imgfov_pts_2d[i, 1]))
        try:
            img[x_index,
            y_index] = (255, 0, 0)
            img3[x_index,
            y_index] = (255, 0, 0)
        except:
            print("...")
        # img[x_index,
        #     y_index] = img2[x_index, y_index]

        # cv2.circle(
        #     img,
        #     (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
        #     2,
        #     color=tuple(color),
        #     thickness=-1,
        # )
        # cv2.imshow("projection", img)

    # y_min = int(np.min(imgfov_pts_2d[:, 0]))
    # y_max = int(np.max(imgfov_pts_2d[:, 0]))
    # x_min = int(np.min(imgfov_pts_2d[:, 1]))
    # x_max = int(np.max(imgfov_pts_2d[:, 1]))
    #
    # for _x in range(x_min, x_max, 1):
    #     img3[_x, y_min] = (255, 0, 0)
    #     img3[_x, y_max] = (255, 0, 0)

    # for _y in range(y_min, y_max, 1):

    plt.title(img.shape)
    plt.imshow(img)
    plt.show()
    plt.imshow(img3)
    plt.show()

    # print(np.sum(img))
    return img


def my_show_lidar_on_image2(pc_velo, img2, calib, img_width, img_height, shift=0):
    """ Project LiDAR points to image """
    img = np.ones_like(img2) * 255
    img3 = img2.copy()
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True
    )
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pts_2d[:, 0] += shift
    print(len(imgfov_pts_2d))
    if len(imgfov_pts_2d) == 0:
        print("skip len(imgfov_pts_2d)==0")
        return img, img3
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    # plt.imshow(img)
    for i in tqdm(range(imgfov_pts_2d.shape[0])):
        depth = imgfov_pc_rect[i, 2]
        ix = int(640.0 / depth)
        if ix >= 256:
            ix = 255
        color = cmap[ix, :]
        # color = [[int(cc) for cc in color]]
        y_index = int(np.round(imgfov_pts_2d[i, 0]))
        x_index = int(np.round(imgfov_pts_2d[i, 1]))
        try:
            img[x_index,
            y_index] = (0, 0, 0)
            img3[x_index,
            y_index] = (0, 0, 0)
        except:
            print("...")
        # img[x_index,
        #     y_index] = img2[x_index, y_index]

        # cv2.circle(
        #     img,
        #     (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
        #     2,
        #     color=tuple(color),
        #     thickness=-1,
        # )
        # cv2.imshow("projection", img)

    y_min = int(np.min(imgfov_pts_2d[:, 0]))
    y_max = int(np.max(imgfov_pts_2d[:, 0]))
    x_min = int(np.min(imgfov_pts_2d[:, 1]))
    x_max = int(np.max(imgfov_pts_2d[:, 1]))

    for _x in range(x_min, x_max, 1):
        img3[_x, y_min] = (255, 0, 0)
        img3[_x, y_max] = (255, 0, 0)

    for _y in range(y_min, y_max, 1):
        img3[x_min, _y] = (255, 0, 0)
        img3[x_max, _y] = (255, 0, 0)

    plt.title(img.shape)
    plt.imshow(img)
    plt.show()
    plt.imshow(img2)
    plt.show()

    # print(np.sum(img))
    return img, img3


def show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred=None):
    """ top_view image"""
    # print('pc_velo shape: ',pc_velo.shape)
    top_view = utils.lidar_to_top(pc_velo)
    top_image = utils.draw_top_image(top_view)
    print("top_image:", top_image.shape)

    # gt

    def bbox3d(obj):
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objects if obj.type != "DontCare"]
    gt = np.array(boxes3d)
    # print("box2d BV:",boxes3d)
    lines = [obj.type for obj in objects if obj.type != "DontCare"]
    top_image = utils.draw_box3d_on_top(
        top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
    )
    # pred
    if objects_pred is not None:
        boxes3d = [bbox3d(obj) for obj in objects_pred if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in objects_pred if obj.type != "DontCare"]
        top_image = utils.draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
        )

    cv2.imshow("top_image", top_image)
    return top_image


def my_show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred=None):
    """ top_view image"""
    # print('pc_velo shape: ',pc_velo.shape)
    import matplotlib.pyplot as plt
    top_view = utils.lidar_to_top(pc_velo)
    top_image = utils.draw_top_image(top_view)
    plt.imshow(top_image)
    plt.show()
    print("top_image:", top_image.shape)

    # gt

    def bbox3d(obj):
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objects if obj.type != "DontCare"]
    gt = np.array(boxes3d)
    # print("box2d BV:",boxes3d)
    lines = [obj.type for obj in objects if obj.type != "DontCare"]
    top_image = utils.draw_box3d_on_top(
        top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
    )
    # pred
    if objects_pred is not None:
        boxes3d = [bbox3d(obj) for obj in objects_pred if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in objects_pred if obj.type != "DontCare"]
        top_image = utils.draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
        )

    # cv2.imshow("top_image", top_image)
    # cv2.waitKey(0)
    return top_image


def dataset_viz(root_dir, args):
    dataset = kitti_object(root_dir, split=args.split, args=args)
    ## load 2d detection results
    # objects2ds = read_det_file("box2d.list")

    if args.show_lidar_with_depth:
        import mayavi.mlab as mlab

        fig = mlab.figure(
            figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
        )
    for data_idx in range(len(dataset)):
        if args.ind > 0:
            data_idx = args.ind
        # Load data from dataset
        if args.split == "training":
            objects = dataset.get_label_objects(data_idx)
        else:
            objects = []
        # objects2d = objects2ds[data_idx]

        objects_pred = None
        if args.pred:
            # if not dataset.isexist_pred_objects(data_idx):
            #    continue
            objects_pred = dataset.get_pred_objects(data_idx)
            if objects_pred == None:
                continue
        if objects_pred == None:
            print("no pred file")
            # objects_pred[0].print_object()

        n_vec = 4
        if args.pc_label:
            n_vec = 5

        dtype = np.float32
        if args.dtype64:
            dtype = np.float64
        pc_velo = dataset.get_lidar(data_idx, dtype, n_vec)[:, 0:n_vec]
        calib = dataset.get_calibration(data_idx)
        img = dataset.get_image(data_idx)
        img_height, img_width, _ = img.shape
        print(data_idx, "image shape: ", img.shape)
        print(data_idx, "velo  shape: ", pc_velo.shape)
        if args.depth:
            depth, _ = dataset.get_depth(data_idx)
            print(data_idx, "depth shape: ", depth.shape)
        else:
            depth = None

        # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        # depth_height, depth_width, depth_channel = img.shape

        # print(('Image shape: ', img.shape))

        if args.stat:
            stat_lidar_with_boxes(pc_velo, objects, calib)
            continue
        print("======== Objects in Ground Truth ========")
        n_obj = 0
        for obj in objects:
            if obj.type != "DontCare":
                print("=== {} object ===".format(n_obj + 1))
                obj.print_object()
                n_obj += 1

        # Draw 3d box in LiDAR point cloud
        if args.show_lidar_topview_with_boxes:
            # Draw lidar top view
            show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred)

        # show_image_with_boxes_3type(img, objects, calib, objects2d, data_idx, objects_pred)
        if args.show_image_with_boxes:
            # Draw 2d and 3d boxes on image
            show_image_with_boxes(img, objects, calib, True, depth)
        if args.show_lidar_with_depth:
            # Draw 3d box in LiDAR point cloud
            show_lidar_with_depth(
                pc_velo,
                objects,
                calib,
                fig,
                args.img_fov,
                img_width,
                img_height,
                objects_pred,
                depth,
                img,
                constraint_box=args.const_box,
                save=args.save_depth,
                pc_label=args.pc_label,
            )
            # show_lidar_with_boxes(pc_velo, objects, calib, True, img_width, img_height, \
            #    objects_pred, depth, img)
        if args.show_lidar_on_image:
            # Show LiDAR points on image.
            show_lidar_on_image(pc_velo[:, 0:3], img, calib, img_width, img_height)
        input_str = raw_input()

        mlab.clf()
        if input_str == "killall":
            break


def depth_to_lidar_format(root_dir, args):
    dataset = kitti_object(root_dir, split=args.split, args=args)
    for data_idx in range(len(dataset)):
        # Load data from dataset

        pc_velo = dataset.get_lidar(data_idx)[:, 0:4]
        calib = dataset.get_calibration(data_idx)
        depth, _ = dataset.get_depth(data_idx)
        img = dataset.get_image(data_idx)
        img_height, img_width, _ = img.shape
        print(data_idx, "image shape: ", img.shape)
        print(data_idx, "velo  shape: ", pc_velo.shape)
        print(data_idx, "depth shape: ", depth.shape)
        # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        # depth_height, depth_width, depth_channel = img.shape

        # print(('Image shape: ', img.shape))
        save_depth(
            data_idx,
            pc_velo,
            calib,
            args.img_fov,
            img_width,
            img_height,
            depth,
            constraint_box=args.const_box,
        )
        # input_str = raw_input()


def read_det_file(det_filename):
    """ Parse lines in 2D detection output files """
    # det_id2str = {1: "Pedestrian", 2: "Car", 3: "Cyclist"}
    objects = {}
    with open(det_filename, "r") as f:
        for line in f.readlines():
            obj = utils.Object2d(line.rstrip())
            if obj.img_name not in objects.keys():
                objects[obj.img_name] = []
            objects[obj.img_name].append(obj)
        # objects = [utils.Object2d(line.rstrip()) for line in f.readlines()]

    return objects


if __name__ == "__main__":
    import mayavi.mlab as mlab
    from viz_util import draw_lidar, draw_gt_boxes3d

    parser = argparse.ArgumentParser(description="KIITI Object Visualization")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        # default="data/object",
        default="_adv_tempdir/cloc_workplace",
        metavar="N",
        help="input  (default: data/object)",
    )
    parser.add_argument(
        "-i",
        "--ind",
        type=int,
        default=0,
        metavar="N",
        help="input  (default: data/object)",
    )
    parser.add_argument(
        "-p", "--pred", action="store_true", help="show predict results"
    )
    parser.add_argument(
        "-s",
        "--stat",
        action="store_true",
        help=" stat the w/h/l of point cloud in gt bbox",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        help="use training split or testing split (default: training)",
    )
    parser.add_argument(
        "-l",
        "--lidar",
        type=str,
        default="velodyne",
        metavar="N",
        help="velodyne dir  (default: velodyne)",
    )
    parser.add_argument(
        "-e",
        "--depthdir",
        type=str,
        default="depth",
        metavar="N",
        help="depth dir  (default: depth)",
    )
    parser.add_argument(
        "-r",
        "--preddir",
        type=str,
        default="pred",
        metavar="N",
        help="predicted boxes  (default: pred)",
    )
    parser.add_argument("--gen_depth", action="store_true", help="generate depth")
    parser.add_argument("--vis", action="store_true", help="show images")
    parser.add_argument("--depth", action="store_true", help="load depth")
    parser.add_argument("--img_fov", action="store_true", help="front view mapping")
    parser.add_argument("--const_box", action="store_true", help="constraint box")
    parser.add_argument(
        "--save_depth", action="store_true", help="save depth into file"
    )
    parser.add_argument(
        "--pc_label", action="store_true", help="5-verctor lidar, pc with label"
    )
    parser.add_argument(
        "--dtype64", action="store_true", help="for float64 datatype, default float64"
    )

    parser.add_argument(
        "--show_lidar_on_image", action="store_true", help="project lidar on image"
    )
    parser.add_argument(
        "--show_lidar_with_depth",
        action="store_true",
        help="--show_lidar, depth is supported",
    )
    parser.add_argument(
        "--show_image_with_boxes", action="store_true", help="show lidar"
    )
    parser.add_argument(
        "--show_lidar_topview_with_boxes",
        action="store_true",
        help="show lidar topview",
    )
    args = parser.parse_args()
    if args.pred:
        print(args.dir + "/" + args.split + "/pred")
        assert os.path.exists(args.dir + "/" + args.split + "/pred")

    if args.vis:
        dataset_viz(args.dir, args)
    if args.gen_depth:
        depth_to_lidar_format(args.dir, args)
