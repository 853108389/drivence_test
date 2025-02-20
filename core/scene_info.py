import json

from core.vehicle import Vehicle


class SceneInfo(object):
    def __init__(self, source_info):
        print(source_info)
        global source
        if isinstance(source_info, str):
            source = self.load_json(source_info)
        elif isinstance(source_info, dict):
            source = source_info
        self.dataset = source["dataset"]
        self.weather = source["weather"]
        self.bg_index = source["bg_index"]
        self.nums_vehicles = source["nums_vehicles"]
        self.vehicles = [Vehicle(info) for info in source["vehicles"]]
        if "sequence" in source:
            self.sequence = source["sequence"]

    def to_json_self(self):

        scene = {"dataset": self.dataset, "weather": self.weather, "bg_index": self.bg_index,
                 "nums_vehicles": self.nums_vehicles, "vehicles": self.vehicles}

        return scene

    def to_json_total(self):

        scene = {"dataset": self.dataset, "": self.weather, "bg_index": self.bg_index,
                 "nums_vehicles": self.nums_vehicles}
        vehicles = [vehicle.to_json() for vehicle in self.vehicles]
        scene["vehicles"] = vehicles

        return scene

    def load_json(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            json_file = json.load(f)
        return json_file
