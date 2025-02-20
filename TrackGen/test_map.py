import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from src.mpc_backup.reference_path import ReferencePath

from map import Map


def fun1():
    file_path = 'maps/0000.png'

    img = np.array(Image.open(file_path))[:, :, 0]
    print(img.shape)

    ego_trajectory = np.load('maps/0000_ego_trajectory_idx.npy')

    plt.imshow(img, cmap='gray')
    plt.show()


def load_my_map():
    file_path = "maps/0000.png"
    my_map = Map(file_path=file_path, origin=[-80, -10], resolution=0.5)

    path_file_path = "maps/0000_ego_trajectory.npy"
    path_resolution = 1

    wp = np.load(path_file_path)

    wp_x = wp[:, 0]
    wp_y = wp[:, 1]

    reference_path = ReferencePath(my_map, wp_y, wp_x, path_resolution,
                                   smoothing_distance=None, max_width=5,
                                   circular=False, line_difference=False)

    return my_map, reference_path


if __name__ == '__main__':
    import bezier

    nodes = np.asfortranarray([
        [0.0, -1.0, 1.0, -0.75],
        [2.0, 0.0, 1.0, 1.625],
    ])
    curve = bezier.Curve(nodes, degree=3)
    point1 = np.asfortranarray([
        [-0.09375],
        [0.828125],
    ])
    print(curve.locate(point1))
