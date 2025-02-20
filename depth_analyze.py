import cv2
import numpy as np
from PIL import Image

if __name__ == '__main__':
    ...

    bg_img_depth_path = "/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/_datasets/KITTI_TRACKING/training/depth_ori/0000/000018.png"

    bg_img_depth = cv2.imread(bg_img_depth_path, cv2.IMREAD_UNCHANGED)

    img_depth_ori = np.array(Image.open(bg_img_depth_path).convert('L'))
    print(img_depth_ori.shape)
    print(np.unique(img_depth_ori, return_counts=True))
    Image.fromarray(img_depth_ori).save('temp_d.png')
    bg_img_depth = cv2.imread('temp_d.png', cv2.IMREAD_UNCHANGED)
    print(np.unique(bg_img_depth, return_counts=True))
    print(bg_img_depth.shape)
    assert 1 == 2
