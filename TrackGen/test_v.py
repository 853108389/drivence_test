import numpy as np

from test_map import load_my_map


def analyze_track():
    path_file_path = "maps/0000_ego_trajectory.npy"
    track = np.load(path_file_path)
    wp_x = track[:, 0]
    wp_y = track[:, 1]
    anlyze_detail(wp_x, wp_y)


def anlyze_detail(wp_x, wp_y):
    v_arr = []
    for i in range(len(wp_x) - 1):
        dis = np.sqrt((wp_x[i + 1] - wp_x[i]) ** 2 +
                      (wp_y[i + 1] - wp_y[i]) ** 2)
        v = dis / 0.1
        v_arr.append(v)
    print("==========", len(v_arr), "==========")
    print(v_arr)


if __name__ == '__main__':
    map, reference_path = load_my_map()
