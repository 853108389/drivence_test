import numpy as np
import traj_dist.distance as tdist
from matplotlib import pyplot as plt

from TrackGen.frenet_optimal_trajectory import get_refrence_path
from TrackGen.main import get_map_dict, init_data_path
from TrackGen.my_utils import load_map, load_background_npcs

if __name__ == '__main__':
    data_num = 0
    map_dict = get_map_dict(0)
    my_map = load_map(map_dict)
    ego_path, npc_path = init_data_path(data_num)
    npc_arr = load_background_npcs(npc_path)

    waypoints_arr = map_dict["waypoint"]
    for current_wp in waypoints_arr:
        plt.imshow(np.flipud(my_map.data), cmap='gray',
                   extent=[my_map.origin[0], my_map.origin[0] +
                           my_map.width * my_map.resolution,
                           my_map.origin[1], my_map.origin[1] +
                           my_map.height * my_map.resolution], vmin=0.0,
                   vmax=1.0)
        length, width = my_map.data.shape
        original = my_map.origin
        l_range = [original[0], original[0] + length]
        w_range = [original[1], original[1] + width]
        traj_list = []
        for npc in npc_arr:
            if npc.type == "static":
                continue
            traj = npc.car_trajectory
            traj = np.array(traj)
            traj[:, 1] = traj[:, 1].clip(min=l_range[0], max=l_range[1])
            traj[:, 0] = traj[:, 0].clip(min=w_range[0], max=w_range[1])
            traj_list.append(traj)
            plt.plot(traj[:, 1], traj[:, 0], "-b")
        for distance in ["sspd", "frechet", "discret_frechet", "hausdorff", "dtw", "lcss", "edr", "erp"]:
            res = tdist.pdist(traj_list, metric=distance)
            print(distance, res)

        wx, wy = zip(*current_wp)
        plt.scatter(wx, wy, c="r")
        tx, ty, _, _, _ = get_refrence_path(wx, wy, start=None, num=None)

        plt.plot(tx, ty, "-r", label="course")
    plt.show()
    print(map_dict)


def is_line(points, threshold=1e-4):
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    m, b = np.polyfit(x, y, 1)

    y_pred = m * x + b
    error = np.mean((y - y_pred) ** 2)

    return error < threshold


def extend_line(points, extension_length=1.0):
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    m, b = np.polyfit(x, y, 1)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = m * x_min + b, m * x_max + b

    x_extension_min = x_min - extension_length
    x_extension_max = x_max + extension_length
    y_extension_min = m * x_extension_min + b
    y_extension_max = m * x_extension_max + b

    return [(x_extension_min, y_extension_min), (x_extension_max, y_extension_max)]


def process_points(points, extension_length=1.0, threshold=1e-4):
    if is_line(points, threshold):
        print("点集为直线，正在延长直线...")
        return extend_line(points, extension_length)
    else:
        print("点集为曲线，无操作。")
        return points
