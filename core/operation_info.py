import json

from core.operation import Operation


class OperationInfo(object):
    def __init__(self, source_info):

        if not isinstance(source_info, str):
            assert 1 == 2, "source_info is not a path(string)"
        else:
            source = self.load_json(source_info)
        self.bg_index = source["bg_index"]
        self.operators = [Operation(info) for info in source["operators"]]

    def load_json(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            json_file = json.load(f)
        return json_file
