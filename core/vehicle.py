class Vehicle(object):
    def __init__(self, info: dict):
        if not isinstance(info, dict):
            assert 1 == 2, "info is not a dict"
        self.obj_index = info["obj_index"]
        self.obj_name = info["obj_name"]

        self.location = info["location"]
        self.rotation = info["rotation"]
        self.scale = info["scale"]

    def to_json(self):
        vehicle = {"obj_index": self.obj_index, "location": self.location, "rotation": self.rotation,
                   "scale": self.scale}
        return vehicle
