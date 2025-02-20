import argparse


class InitParser(object):

    @staticmethod
    def init_parser() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="arg parser")
        parser.add_argument('--system_name', type=str, default='demo')
        parser.add_argument('--select_size', type=int, default=100)
        parser.add_argument('--modality', type=str, default="multi")
        parser.add_argument('--road_split_way', type=str, default="CENet")
        parser.add_argument('--virtual_lidar_way', type=str, default="simulation")

        parser.add_argument('--param_path', type=str,
                            default="D:\Python\pythonHome\multiGen-batch\core\parameter\demo.json")
        args = parser.parse_args()

        return args

    @staticmethod
    def batch_obj_task():
        parser = argparse.ArgumentParser(description="batch_obj_task arg parser")
        parser.add_argument('--username', type=str, default="dylan")
        parser.add_argument('--task_name', type=str, default="demo")
        parser.add_argument('--execute_date', type=str, default="2024-01-01")

        parser.add_argument('--bg_indexes', type=str, default="1")
        parser.add_argument('--obj_indexes', type=str, default="1,2")
        parser.add_argument('--road_split_way', type=str, default="CENet")
        parser.add_argument('--virtual_lidar_way', type=str, default="simulation")
        args = parser.parse_args()

        return args

    @staticmethod
    def execute_parser() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="arg parser")
        parser.add_argument('--operation_path', type=str,
                            default="D:/Python/pythonHome/multiGen-batch/core/parameter/operation.json")

        parser.add_argument('--logic_scene_path', type=str,
                            default="D:/Python/pythonHome/multiGen-batch/scene_workplace/dylan/000004/logic_scene_0.json")

        parser.add_argument('--username', type=str, default="dylan")
        parser.add_argument('--task_name', type=str, default="demo")
        parser.add_argument('--execute_date', type=str, default="2024-01-01")
        parser.add_argument('--bg_index', type=int, default=4)

        parser.add_argument('--select_size', type=int, default=100)
        parser.add_argument('--modality', type=str, default="multi")
        parser.add_argument('--road_split_way', type=str, default="CENet")
        parser.add_argument('--virtual_lidar_way', type=str, default="simulation")
        parser.add_argument("--result_path_dir", type=str,
                            default="E:/web/springboot-pure-admin-ui-5/src/assets/result/")
        args = parser.parse_args()

        return args

    @staticmethod
    def blender_parser() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="arg parser")
        parser.add_argument('-cp', '--bg_calib_path', type=str)
        parser.add_argument('-mp', '--obj_mesh_path', type=str)
        parser.add_argument('-pa', '--obj_camera_position_args', type=str)
        parser.add_argument('-ra', '--rz_degree_args', type=str)
        parser.add_argument('-ip', '--save_img_obj_path', type=str)

        parser.add_argument('-l', '--lens', type=str, default='AUTO')
        parser.add_argument('-fx', '--shift_x', type=str, default='AUTO')
        parser.add_argument('-fy', '--shift_y', type=str, default='AUTO')
        parser.add_argument('-sf', '--sensor_fit', type=str, default='AUTO')
        parser.add_argument('-sh', '--sensor_height', type=int, default=29)
        parser.add_argument('-sw', '--sensor_width', type=int, default=29)
        parser.add_argument('-ih', '--img_height', type=int, default=375)
        parser.add_argument('-iw', '--img_width', type=int, default=1242)

        args = parser.parse_args()

        return args
