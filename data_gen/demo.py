import uuid

from data_gen.main_function_new import main_before

logic_scene_path = r"/home/niangao/disk1/_PycharmProjects/PycharmProjects/multiGen/data_gen/scene_0.json"


class TaskConfig:
    def __init__(self):
        self.task_name = "task" + str(uuid.uuid4())
        self.road_split_way = "CENet"
        self.virtual_lidar_way = "simulation"
        self.execute_date = "2024-01-01"
        self.username = "dylan"

    def to_dict(self):
        return {
            "task_name-": self.task_name,
            "road_split_way": self.road_split_way,
            "virtual_lidar_way": self.virtual_lidar_way,
            "execute_date": self.execute_date,
            "username": self.username
        }


main_before(logic_scene_path, TaskConfig())
