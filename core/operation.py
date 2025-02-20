class Operation(object):
    def __init__(self, info: dict):
        if not isinstance(info, dict):
            assert 1 == 2, "info is not a dict"
        self.name = info["name"]
        self.obj_index = info["obj_index"]
        self.is_random = info["is_random"]
        self.type = info["type"]
        self.location = info["location"]
        self.rotation = info["rotation"]
        self.shift = info["shift"]
        self.scale = info["scale"]
