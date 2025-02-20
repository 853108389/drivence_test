"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
(https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame]
(https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""
import copy
import math
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from map import RectObstacle

sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent))
from CubicSpline import cubic_spline_planner
from QuinticPolynomialsPlanner.quintic_polynomials_planner import QuinticPolynomial

from my_utils import consturct_npc_track_by_mpc_result
from path_planning import samplling_start, samplling_oppo

SIM_LOOP = 500

MAX_ACCEL = 2.0
MAX_CURVATURE = 1.0
MAX_ROAD_WIDTH = 4.0
D_ROAD_W = 1.0
DT = 0.2

SAMPLE_T_ARRAY = np.array([2, 5, 10, 15, 20]) * 0.2

D_T_S = 7.2 / 3.6
SPEED_ARR = np.array([-1, 0, 1, 2, 3, 5, 10, 15, 20])

TARGET_SPEED_ARR = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
N_S_SAMPLE = 1
ROBOT_RADIUS = 2.0

K_J = 1.0
K_T = 1.0
K_D = 0.5
K_LAT = 1.0
K_LON = 1.0

show_animation = False


class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

    def sub_path(self, i):
        self.t = self.t[i:]
        self.d = self.d[i:]
        self.d_d = self.d_d[i:]
        self.d_dd = self.d_dd[i:]
        self.d_ddd = self.d_ddd[i:]
        self.s = self.s[i:]
        self.s_d = self.s_d[i:]
        self.s_dd = self.s_dd[i:]
        self.s_ddd = self.s_ddd[i:]
        self.x = self.x[i:]
        self.y = self.y[i:]
        self.yaw = self.yaw[i:]
        self.ds = self.ds[i:]
        self.c = self.c[i:]


def calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0, TARGET_SPEED):
    frenet_paths = []

    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        for Ti in SAMPLE_T_ARRAY:
            fp = FrenetPath()

            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]

            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            for tv in SPEED_ARR:
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, c_accel, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]

                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))
                Js = sum(np.power(tfp.s_ddd, 2))

                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + -1 * K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + -1 * K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):
    for j, fp in enumerate(fplist):

        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist


def check_collision(car_dict, fp, my_map, start_m, my_mapdata_arr, max_frames):
    x, y, yaw = fp.x, fp.y, fp.yaw
    trajectory = np.array(list(zip(y, x, np.pi / 2 - np.array(yaw))))

    npc_trajectories_boxes2d = consturct_npc_track_by_mpc_result(trajectory, max_frames, car_dict["length"],
                                                                 car_dict["width"], start_m)
    for i in range(start_m, start_m + len(trajectory)):
        if i >= len(npc_trajectories_boxes2d):
            continue
        corner = npc_trajectories_boxes2d[i]
        rectob1 = RectObstacle(corner)

        occ_flag = rectob1.check_map_occupy_cells2(my_map, my_mapdata_arr[i])
        t2 = time.time()

        if not occ_flag:
            return False
    return True


def check_paths(car_dict, fplist, my_map, start_m, my_mapdata_arr, max_frames, TARGET_SPEED):
    ok_ind = []
    count = [0, 0, 0]

    for i, _ in enumerate(fplist):
        if any([v > TARGET_SPEED + 5 for v in fplist[i].s_d]):
            count[0] += 1
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):
            count[1] += 1
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):
            count[2] += 1
            continue
        else:
            t1 = time.time()
            flag = check_collision(car_dict, fplist[i], my_map, start_m, my_mapdata_arr, max_frames)

            if not flag:
                continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(TARGET_SPEED, car_dict, csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, my_map, start_m,
                            my_mapdata_arr, max_frames):
    fplist = calc_frenet_paths(c_speed, c_accel, c_d, c_d_d, c_d_dd, s0, TARGET_SPEED)

    fplist = calc_global_paths(fplist, csp)

    fplist = check_paths(car_dict, fplist, my_map, start_m, my_mapdata_arr, max_frames, TARGET_SPEED)

    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path


def generate_target_course(x, y, num=None):
    csp = cubic_spline_planner.CubicSpline2D(x, y)
    if num is not None:
        s = np.linspace(0, csp.s[-1], num)
        s = s[s < csp.s[-1]]
    else:
        s = np.arange(0, csp.s[-1], 0.1)
    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))
    return rx, ry, ryaw, rk, csp, s


def find_edges_and_internals(matrix):
    rows, cols = matrix.shape

    edges = []
    internals = []

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if matrix[i, j] == 0:

                if (matrix[i - 1, j] == 0 and
                        matrix[i, j - 1] == 0 and matrix[i, j + 1] == 0
                        and matrix[i + 1, j] == 0 == 0):
                    internals.append((i, j))
                else:
                    edges.append((i, j))

    return edges, internals


def get_obstacle_on_map(map):
    map = map.copy()
    _, iternals_idxs = find_edges_and_internals(map)
    indxes_ignore = set(iternals_idxs)
    idx = np.where(map == 0)
    indxes_all = set(zip(idx[0], idx[1]))
    idxes = indxes_all - indxes_ignore
    idxes = np.array(list(idxes))
    ox = np.array(list(idxes[:, 0]))
    oy = np.array(list(idxes[:, 1]))

    return ox, oy


def sampling_way_points(my_map, ego_trajectory, car_dict, map_dict, T):
    along_path_arr = []
    waypoints_arr = map_dict["waypoint"]

    for waypoints in waypoints_arr:
        along_path_arr.append(waypoints)

    oppo_lpath_arr = []
    if "oppo_lane_wp" in map_dict:
        oppo_lane_wp = map_dict["oppo_lane_wp"]
        oppo_lpath_arr = oppo_lane_wp

    for path in oppo_lpath_arr:
        path = np.array(path).astype(np.float64)
        path[:, 0] += np.random.uniform(0, 0.1, len(path))
        path[:, 1] += np.random.uniform(0, 0.1, len(path))

    if np.random.randint(0, 2) == 0:

        current_wp = samplling_oppo(my_map, oppo_lpath_arr, ego_trajectory, car_dict["length"], car_dict["width"], T=0)
        direction = -1
        start = None
    else:
        path_arr = along_path_arr
        direction = 1
        start = samplling_start(my_map, ego_trajectory, car_dict["length"], car_dict["width"], margin=0, T=T)
        if start is None:
            print("warning can not sampling start")
            return None
        current_ix = np.random.randint(len(path_arr))
        current_wp = path_arr[current_ix]
        current_wp = np.array(current_wp)
    if current_wp is None:
        return None
    wx, wy = zip(*current_wp)

    current_wp = np.array(current_wp)

    return wx, wy, direction, start


def get_refrence_path(wx, wy, start=None, num=None):
    wx = np.array(wx)
    wy = np.array(wy)

    try:
        tx, ty, tyaw, tc, csp, s = generate_target_course(wx, wy, num=num)
        if start is not None:
            start = np.array(start)
            wps = csp.s
            points = np.vstack([tx, ty]).T
            distances = np.linalg.norm(points - start, axis=1)
            min_ix = np.argmin(distances)
            start_s = s[min_ix]
            for i, _s in enumerate(wps):
                if _s > start_s + 0.2:
                    break
            target_i = i

            wx = np.insert(wx[target_i:], 0, start[0])
            wy = np.insert(wy[target_i:], 0, start[1])

            tx, ty, tyaw, tc, csp, s = generate_target_course(wx, wy, num=num)
            return tx, ty, tyaw, tc, csp
    except Exception as e:
        print("generate refrence path error", e)
        plt.plot(wx, wy, "-b", label="course")
        plt.scatter(wx, wy, color="b")
        if start is not None:
            plt.scatter(start[0], start[1], color="g")
        plt.savefig(str(time.time()) + "error.png")

        return None

    return tx, ty, tyaw, tc, csp


def trajectory_planning(refrence_path, start_frame, end_frame, my_map, car_dict, max_frames, my_mapdata_arr,
                        speed_level=2,
                        target_speed=None,
                        refrence_speed=None,
                        frame_th=None,
                        ):
    TARGET_SPEED = TARGET_SPEED_ARR[speed_level]
    tx, ty, tyaw, tc, csp = refrence_path

    print("TARGET_SPEED", TARGET_SPEED)

    if refrence_speed is not None:
        c_speed = refrence_speed[0]
        global K_D
        K_D = 1
    else:
        c_speed = TARGET_SPEED
    c_accel = 0.0
    c_d = 0
    c_d_d = 0.0
    c_d_dd = 0.0
    s0 = 0.0

    area = 20.0

    res_x = [tx[0]]
    res_y = [ty[0]]
    res_yaw = [tyaw[0]]
    history_path = list()
    max_capacity = 5
    try:
        for i in tqdm(range(start_frame, end_frame + 1)):
            for npc in my_map.npcs:
                if npc.is_ego:
                    if not npc.has_trajectory():
                        print("ego car track over")
                        break
            my_map.render()
            if refrence_speed is not None:
                TARGET_SPEED = refrence_speed[i]
                print("TARGET_SPEED", TARGET_SPEED)
            path = frenet_optimal_planning(TARGET_SPEED, car_dict,
                                           csp, s0, c_speed, c_accel, c_d, c_d_d, c_d_dd, my_map, i, my_mapdata_arr,
                                           max_frames)
            if path is not None and len(path.s) > 1:
                history_path.insert(0, path)
                if len(history_path) > max_capacity:
                    history_path.pop(-1)

            predx, predy = None, None
            if path is None or len(path.s) == 0:
                raise ValueError("no avaliabel path")

                _x, _y, _yaw = [res_x[-1]] * 4, [res_y[-1]] * 4, [res_yaw[-1]] * 4
                _x, _y, _yaw = _y, _x, np.pi / 2 - np.array(_yaw)
                trajectory = np.array(list(zip(_y, _x, _yaw)))
                npc_trajectories_boxes2d = consturct_npc_track_by_mpc_result(trajectory, max_frames, car_dict["length"],
                                                                             car_dict["width"], i)
                for j in range(5):
                    corner = npc_trajectories_boxes2d[i + j]
                    rectob1 = RectObstacle(corner)

                    occ_flag = rectob1.check_map_occupy_cells2(my_map, my_mapdata_arr[i])
                    if not occ_flag:

                        x, y, yaw = _x[0], _y[0], _yaw[0]
                        break
                    else:
                        raise ValueError("no avaliabel path")
            elif len(path.s) == 1:
                print("warning, on length")
                s0 = path.s[0]
                c_d = path.d[0]
                c_d_d = path.d_d[0]
                c_d_dd = path.d_dd[0]
                c_speed = path.s_d[0]
                c_accel = path.s_dd[0]
                x, y, yaw = path.x[0], path.y[0], path.yaw[0]
            else:
                s0 = path.s[1]
                c_d = path.d[1]
                c_d_d = path.d_d[1]
                c_d_dd = path.d_dd[1]
                c_speed = path.s_d[1]
                c_accel = path.s_dd[1]
                x, y, yaw = path.x[1], path.y[1], path.yaw[1]
                predx = path.x[1:]
                predy = path.y[1:]

            res_x.append(x)
            res_y.append(y)
            res_yaw.append(yaw)

            if np.hypot(x - tx[-1], y - ty[-1]) <= 1.0:
                print("Goal")
                break

            if show_animation:
                plt.cla()

                plt.gcf().canvas.mpl_connect(
                    'key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
                plt.plot(tx, ty)

                plt.imshow(np.flipud(my_map.data), cmap='gray',
                           extent=[my_map.origin[0], my_map.origin[0] +
                                   my_map.width * my_map.resolution,
                                   my_map.origin[1], my_map.origin[1] +
                                   my_map.height * my_map.resolution], vmin=0.0,
                           vmax=1.0)
                if predx is not None:
                    plt.plot(predx, predy, "-or", linewidth=0.2)
                trajectory = np.array([[y, x, np.pi / 2 - np.array(yaw)]])
                npc_trajectories_boxes2d = consturct_npc_track_by_mpc_result(trajectory, max_frames, car_dict["length"],
                                                                             car_dict["width"], i)
                from map import NPC
                npc = NPC(npc_trajectories_boxes2d)
                npc.color = "red"
                npc.t = i
                npc.show()

                plt.plot(x, y, "vc")

                plt.title(str(i) + " , " + "c_accel" + str(c_accel) + "v[m/s]:" + str(c_speed)[0:4])

                for npc in my_map.npcs:
                    npc.show()
                    npc.update()

                plt.pause(0.0001)
                my_map.clear_map()


    except Exception as ex:
        print("出现如下异常%s" % ex)
        import traceback
        traceback.print_exc()
        if len(res_x) >= frame_th and len(res_x) >= (end_frame - start_frame) / 2:
            res_x = res_x[:-10]
            res_y = res_y[:-10]
            res_yaw = res_yaw[:-10]
            print("Success with cut trajectory")
            trajectory = np.array(list(zip(res_y, res_x, np.pi / 2 - np.array(res_yaw))))
            return trajectory

    if len(res_x) >= frame_th:
        print("Success")
        trajectory = np.array(list(zip(res_y, res_x, np.pi / 2 - np.array(res_yaw))))

        return trajectory
    else:
        print("Failed")
        return None
