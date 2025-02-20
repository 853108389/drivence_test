import matplotlib.patches as plt_patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.draw import line_aa
from skimage.morphology import remove_small_holes

OBSTACLE = '


def get_box_center2D(corner):
    x_min, y_min = np.min(corner, axis=0)
    x_max, y_max = np.max(corner, axis=0)
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
    return np.array(center)


class RectObstacle:
    def __init__(self, corner, color="gray"):

        from shapely.geometry import Polygon
        self.corner = corner
        self.polygon = Polygon(self.corner)
        self.color = color

    def show(self):

        x, y = self.polygon.exterior.xy
        ax = plt.gca()

        ax.fill(y, x, color=self.color, alpha=0.5)

    def check_map_occupy_cells(self, map, convert=True):
        data = map.data.copy()
        occupied_cells = set()

        exterior_xy = self.polygon.exterior.xy
        if np.sum(exterior_xy) == 0:
            return None

        for i in range(len(exterior_xy[0]) - 1):
            x1, y1 = exterior_xy[0][i], exterior_xy[1][i]
            x2, y2 = exterior_xy[0][i + 1], exterior_xy[1][i + 1]

            if convert:
                y1, x1 = map.w2m(y1, x1)
                y2, x2 = map.w2m(y2, x2)

            rr, cc, _ = line_aa(int(x1), int(y1), int(x2), int(y2))

            for r, c in zip(rr, cc):
                occupied_cells.add((r, c))

        for cell in occupied_cells:
            if cell[0] >= data.shape[0] or cell[1] >= data.shape[1]:
                return False
            if data[cell[0], cell[1]] <= 0:
                return False
            data[cell[0], cell[1]] = 0

        return True

    def check_map_occupy_cells2(self, map, map_data, convert=True):
        data = map_data
        occupied_cells = set()
        exterior_xy = self.polygon.exterior.xy
        if np.sum(exterior_xy) == 0:
            return None

        for i in range(len(exterior_xy[0]) - 1):
            x1, y1 = exterior_xy[0][i], exterior_xy[1][i]
            x2, y2 = exterior_xy[0][i + 1], exterior_xy[1][i + 1]

            if convert:
                y1, x1 = map.w2m(y1, x1)
                y2, x2 = map.w2m(y2, x2)

            rr, cc, _ = line_aa(int(x1), int(y1), int(x2), int(y2))

            for r, c in zip(rr, cc):
                occupied_cells.add((r, c))

        for cell in occupied_cells:
            if cell[0] >= data.shape[0] or cell[1] >= data.shape[1]:
                return False
            if data[cell[0], cell[1]] <= 0:
                return False
        return True

    def update_map_occupy_cells(self, map, value=0, data=None):
        if data is None:
            data = map.data.copy()
        occupied_cells = set()

        exterior_xy = self.polygon.exterior.xy
        if np.sum(exterior_xy) == 0:
            return data

        for i in range(len(exterior_xy[0]) - 1):
            x1, y1 = exterior_xy[0][i], exterior_xy[1][i]
            x2, y2 = exterior_xy[0][i + 1], exterior_xy[1][i + 1]

            y1, x1 = map.w2m(y1, x1)
            y2, x2 = map.w2m(y2, x2)

            rr, cc, _ = line_aa(int(x1), int(y1), int(x2), int(y2))

            for r, c in zip(rr, cc):
                occupied_cells.add((r, c))

        for cell in occupied_cells:
            if cell[0] < 0 or cell[1] < 0:
                continue
            if cell[0] >= data.shape[0] or cell[1] >= data.shape[1]:
                continue
            data[cell[0], cell[1]] = value
        return data


class Obstacle:
    def __init__(self, cx, cy, radius):

        self.cx = cx
        self.cy = cy
        self.radius = radius

    def show(self):

        circle = plt_patches.Circle(xy=(self.cx, self.cy), radius=
        self.radius, color=OBSTACLE, zorder=20)
        ax = plt.gca()
        ax.add_patch(circle)

    def update_map_occupy_cells(self, map):
        data = map.data.copy()

        radius_px = int(np.ceil(self.radius / map.resolution))

        cx_px, cy_px = map.w2m(self.cx, self.cy)

        y, x = np.ogrid[-radius_px: radius_px, -radius_px: radius_px]
        index = x ** 2 + y ** 2 <= radius_px ** 2
        try:
            data[cy_px - radius_px:cy_px + radius_px, cx_px - radius_px:
                                                      cx_px + radius_px][index] = 0
        except:
            print()
        return data


def get_box_size(corner):
    x_min, y_min, z_min = np.min(corner, axis=0)
    x_max, y_max, z_max = np.max(corner, axis=0)
    return x_max - x_min, y_max - y_min, z_max - z_min


def get_box_center(corner):
    x_min, y_min, z_min = np.min(corner, axis=0)
    x_max, y_max, z_max = np.max(corner, axis=0)
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    return center


class NPC(object):
    def __init__(self, npc_trajectory_box=None):
        self.npc_trajectory_box = npc_trajectory_box
        self.car_trajectory = None
        self.is_insert = False
        self.current_insert = False
        self.type = None
        self.speed_mod = 2
        self.is_ego = False
        self.file_path = None

        self.id = 0
        self.t = 0
        self.centreline_side = None
        self.color = None
        self.direction = 1

    def update(self):
        self.t = self.t + 1
        self.centreline_side = None

    def has_trajectory(self):
        try:
            if self.t >= len(self.npc_trajectory_box):
                return False
            elif (self.npc_trajectory_box[self.t] == 0).all():
                return False
            else:
                return True
        except Exception as e:
            print(self.t, len(self.npc_trajectory_box), self.type, self.file_path, self.is_ego)
            print(e)
            assert 1 == 2

    def has_trajectory_in_T(self, t):
        if t >= len(self.npc_trajectory_box):
            return False
        elif (self.npc_trajectory_box[t] == 0).all():
            return False
        else:
            return True

    def get_current_npc_obstacle(self) -> RectObstacle:
        corner = self.get_current_waypoint_corner()
        if corner is None:
            return None
        else:
            rect = RectObstacle(corner, color=self.color)
        return rect

    def get_npc_obstacle_in_t(self, t) -> RectObstacle:
        if self.has_trajectory():
            corner = self.npc_trajectory_box[t]
            rect = RectObstacle(corner, color=self.color)
            return rect
        else:
            return None

    def show(self):
        if self.has_trajectory():
            self.get_current_npc_obstacle().show()

    def get_current_waypoint_corner(self):
        if self.has_trajectory():
            return self.npc_trajectory_box[self.t]
        else:
            return None

    def get_current_center(self):
        corner = self.get_current_waypoint_corner()
        if corner is not None:
            return get_box_center2D(corner)
        else:
            return None

    def get_center_in_t(self, t):
        corner = self.npc_trajectory_box[t]
        if corner is not None:
            return get_box_center2D(corner)
        else:
            return None


class Map:
    def __init__(self, file_path, origin, resolution, threshold_occupied=100):

        self.threshold_occupied = threshold_occupied

        self.ori_data = self.load_map(file_path)
        self.data = self.ori_data.copy()

        self.height = self.ori_data.shape[0]
        self.width = self.ori_data.shape[1]
        self.resolution = resolution
        self.origin = origin

        self.obstacles = list()
        self.boundaries = list()
        self.npcs = list()

    def w2m_strict(self, x, y, origin=None, resolution=None):
        if origin is None:
            origin = self.origin
        if resolution is None:
            resolution = self.resolution
        dx = (x - origin[0]) / resolution
        dy = (y - origin[1]) / resolution
        return dx, dy

    def m2w_strict(self, dx, dy, origin=None, resolution=None):
        if origin is None:
            origin = self.origin
        if resolution is None:
            resolution = self.resolution
        x = dx * resolution + origin[0]
        y = dy * resolution + origin[1]
        return x, y

    def w2m(self, x, y):

        dx = int(np.floor((x - self.origin[0]) / self.resolution))
        dy = int(np.floor((y - self.origin[1]) / self.resolution))

        return dx, dy

    def m2w(self, dx, dy):

        x = (dx + 0.5) * self.resolution + self.origin[0]
        y = (dy + 0.5) * self.resolution + self.origin[1]

        return x, y

    def load_map(self, file_path, process=True):
        ori_data = np.array(Image.open(file_path).convert("1"))[:, :]

        if process:
            ori_data = self.process_map(ori_data)
        return ori_data

    def process_map(self, data):

        if np.max(data) == 255:
            data = np.where(data >= self.threshold_occupied, 1, 0)

        data = remove_small_holes(data, area_threshold=5,
                                  connectivity=8).astype(np.int8)
        return data

    def add_npcs(self, npcs, process=True):

        self.npcs.extend(npcs)

    def clear_map(self):
        self.data = self.ori_data.copy()
        self.obstacles = list()
        self.boundaries = list()

    def reset_npc(self, t):
        for npc in self.npcs:
            npc.t = t
            npc.centreline_side = None

    def add_obstacles(self, obstacles):

        self.obstacles.extend(obstacles)

    def add_boundary(self, boundaries):

        self.boundaries.extend(boundaries)

    def render_T(self, t):
        for obstacle in self.obstacles:
            self.data = obstacle.update_map_occupy_cells(self)

        for i, npc in enumerate(self.npcs):
            rect = npc.get_npc_obstacle_in_t(t)
            if rect is not None:
                self.data = rect.update_map_occupy_cells(self, -1 * (i + 1))
                self.obstacles.append(rect)

        for boundary in self.boundaries:
            sx = self.w2m(boundary[0][0], boundary[0][1])
            gx = self.w2m(boundary[1][0], boundary[1][1])
            path_x, path_y, _ = line_aa(sx[0], sx[1], gx[0], gx[1])
            for x, y in zip(path_x, path_y):
                self.data[y, x] = 0
        return self.data

    def render(self):

        for obstacle in self.obstacles:
            self.data = obstacle.update_map_occupy_cells(self)

        for i, npc in enumerate(self.npcs):
            rect = npc.get_current_npc_obstacle()
            if rect is not None:
                self.data = rect.update_map_occupy_cells(self, -1 * (i + 1))
                self.obstacles.append(rect)
            npc.centreline_side = None

        for boundary in self.boundaries:
            sx = self.w2m(boundary[0][0], boundary[0][1])
            gx = self.w2m(boundary[1][0], boundary[1][1])
            path_x, path_y, _ = line_aa(sx[0], sx[1], gx[0], gx[1])
            for x, y in zip(path_x, path_y):
                self.data[y, x] = 0


if __name__ == '__main__':
    map = Map('maps/real_map.png')

    plt.imshow(np.flipud(map.data), cmap='gray')
    plt.show()
