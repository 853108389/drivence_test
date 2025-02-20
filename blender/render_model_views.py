import math
import os
import sys

import bpy
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

from utils.calibration_kitti import Calibration


def check_mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


K_file = sys.argv[-5]
shape_file = sys.argv[-4]
shape_view_params_position = sys.argv[-3]
shape_view_params_angle = sys.argv[-2]
img_obj_path = sys.argv[-1]

x, y, z = [float(x) for x in shape_view_params_position.split('_')]
rz_degree = float(shape_view_params_angle)

bpy.ops.import_scene.gltf(filepath=shape_file)

for obj in bpy.data.objects:
    if obj.name == 'Camera':

        continue
    else:
        import config

        obj.rotation_mode = 'XYZ'
        rz_degree_rad = math.radians(rz_degree)
        obj.rotation_euler = (np.pi / 2, rz_degree_rad, 0)

        scale_multiple = config.common_config.multi_scale_blender
        obj.scale = (scale_multiple, scale_multiple, scale_multiple)

        obj.location = (obj.location[0] + x, obj.location[1] + y, obj.location[2] + z)
        obj.hide_render = False

scene = bpy.context.scene
scene.display.shading.color_type = 'TEXTURE'
scene.display.render_aa = 'OFF'

light_data = config.camera_config.light_data
lights = config.camera_config.lights

light_datas = {}
for light_data_name in light_data.keys():
    light_data_instance = bpy.data.lights.new(name=light_data_name, type=light_data[light_data_name]["type"])
    light_data_instance.energy = light_data[light_data_name]["energy"]
    light_datas[light_data_name] = light_data_instance

for light_name in lights.keys():
    light = bpy.data.objects.new(name=light_name, object_data=light_datas[lights[light_name]["object_data"]])

    dis_x, dix_y, dis_z = lights[light_name]["dis"]
    light.location = ((x + dis_x), (y + dix_y), (z + dis_z))
    light.color = lights[light_name]["color"]
    light.rotation_mode = lights[light_name]["rotation_mode"]
    light.rotation_euler = lights[light_name]["rotation_euler"]

    bpy.context.scene.collection.objects.link(light)

calib_info = Calibration(K_file)
K = calib_info.P2[:, 0:3]

lens = config.camera_config.lens
shift_x = config.camera_config.shift_x
shift_y = config.camera_config.shift_y
sensor_height = config.camera_config.sensor_height
sensor_width = config.camera_config.sensor_width
sensor_fit = config.camera_config.sensor_fit
image_width = config.camera_config.img_width
image_height = config.camera_config.img_height
location = config.camera_config.location
coordinate_type = config.camera_config.coordinate_type

if lens == "AUTO":
    lens = (K[0, 0] + K[1, 1]) / 2 * sensor_width / image_width
if shift_x == "AUTO":
    shift_x = (image_width / 2 - K[0, 2]) / image_width
if shift_y == "AUTO":
    shift_y = (K[1, 2] - image_height / 2) / image_width

bpy.data.scenes['Scene'].render.film_transparent = True
bpy.data.scenes['Scene'].render.resolution_x = image_width
bpy.data.scenes['Scene'].render.resolution_y = image_height

camObj = bpy.data.objects['Camera']
camObj.location[0] = location[0]
camObj.location[1] = location[1]
camObj.location[2] = location[2]
camObj.rotation_mode = coordinate_type
camObj.rotation_euler = (np.pi, 0, 0)

camObj.data.lens = lens
camObj.data.shift_x = shift_x
camObj.data.shift_y = shift_y
camObj.data.sensor_height = sensor_height
camObj.data.sensor_width = sensor_width
camObj.data.sensor_fit = sensor_fit

bpy.data.scenes['Scene'].render.filepath = img_obj_path
bpy.ops.render.render(write_still=True)
