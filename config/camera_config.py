blender_path = '/home/niangao/shapenet/blender-3.3.1-linux-x64/blender'
blender_blank_filename = "blank.blend"

blender_render_filename = "blender_bk.py"
blender_render_filename_new = "blender.py"

img_width = 1242
img_height = 375

lens = "AUTO"
sensor_height = 29
sensor_width = 29
sensor_fit = "AUTO"

shift_x = "AUTO"
shift_y = "AUTO"

location = [0, 0, 0]
coordinate_type = "XYZ"

import numpy as np

light_data = {
    "my_light_data":
        {"energy": 5, "type": "SUN"}
}
lights = {
    'my_light': {"object_data": "my_light_data", "dis": (8, -10, 2), "color": (1, 1, 1, 1), "rotation_mode": 'XYZ',
                 "rotation_euler": (np.pi / 3, np.pi / 3, np.pi / 18)}
}

is_image_refine = False

camera_position = [0, 0, 0.3]
