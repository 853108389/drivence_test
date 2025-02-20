import math
import os
import pathlib
import sys

import numpy

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import config
import os.path as osp
import numpy as np


def check_mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


class Blender(object):
    def __init__(self):

        self.project_dirname = config.common_config.project_dirname
        self.BASE_DIR = config.common_config.blender_root_dirname
        self.blank_filename = config.camera_config.blender_blank_filename
        self.render_filename = config.camera_config.blender_render_filename_new
        self.param_split_flag = config.common_config.split_name_flag

        self.location = config.camera_config.location
        self.coordinate_type = config.camera_config.coordinate_type

        self.light_data = config.camera_config.light_data
        self.lights = config.camera_config.lights

    def run_blender_script(self, save_img_obj_path: str, save_depth_image_dir: str, scene_file_path: str,
                           save_log_dir: str, lens, shift_x, shift_y, sensor_fix,
                           sensor_height, sensor_width, image_height, image_width) -> numpy.ndarray:

        blender_blank_file = os.path.join(self.BASE_DIR, self.blank_filename)

        render_code_path = osp.join(self.BASE_DIR, self.render_filename)

        try:

            render_cmd = '%s %s --background --python %s %s %s %s %s %s %s %s %s %s %s %s 1> %s/blender_log.txt' % (
                config.camera_config.blender_path, blender_blank_file, render_code_path, scene_file_path,
                save_img_obj_path, save_depth_image_dir, lens, shift_x, shift_y, sensor_fix,
                sensor_height, sensor_width, image_height, image_width,
                save_log_dir)

            os.system(render_cmd)
        except:
            print('render failed. render_cmd: %s' % (render_cmd))
        assert os.path.exists(save_img_obj_path)
        import cv2
        img_obj = cv2.imread(save_img_obj_path, cv2.IMREAD_UNCHANGED)
        return img_obj

    def render_gltf_obj(self, args):

        import bpy
        import math
        import numpy as np
        from core.scene_info import SceneInfo
        from utils.calibration_kitti import Calibration

        scene_file_path = args[-11]
        output_img_dir = args[-10]
        output_depth_img_dir = args[-9]
        lens = args[-8]
        shift_x = args[-7]
        shift_y = args[-6]
        sensor_fit = args[-5]
        sensor_height = int(args[-4])
        sensor_width = int(args[-3])
        image_height = int(args[-2])
        image_width = int(args[-1])

        logic_scene = SceneInfo(scene_file_path)
        bg_index = logic_scene.bg_index
        calib_path = os.path.join(config.common_config.bg_dirname, config.common_config.calib_bg_dirname,
                                  f"{bg_index:06d}.txt")

        sun_rotation_euler = (np.pi / 3, np.pi / 2, np.pi / 18)

        bg_image_path = os.path.join(config.common_config.bg_dirname, config.common_config.img_bg_dirname,
                                     f"{bg_index:06d}.png")

        image_batch_generated_num = len(os.listdir(output_img_dir))
        image_batch_generated_filename = os.path.join("blender_scene_{}.png".format(image_batch_generated_num))
        init_scene(output_img_dir, bg_image_path, output_depth_img_dir, image_batch_generated_filename)

        objs = []
        for idx in range(logic_scene.nums_vehicles):
            obj_name = logic_scene.vehicles[idx].obj_name
            obj_gltf_path = os.path.join(config.common_config.obj_dirname, f"{obj_name}",
                                         config.common_config.obj_filename_new)
            bpy.ops.import_scene.obj(filepath=obj_gltf_path)
            assert len(bpy.context.selected_objects) == 1

            bpy.context.selected_objects[0].name = obj_name
            objs.append(bpy.data.objects.get(obj_name))

        plane_y = 0

        selected_objects = bpy.context.selected_objects

        active_object = bpy.context.view_layer.objects.active

        for idx, obj in enumerate(objs):

            obj.rotation_mode = 'XYZ'

            rz_degree_rad = math.radians(logic_scene.vehicles[idx].rotation)
            x, y, z = logic_scene.vehicles[idx].location
            print("--------------------------------0", x, y, z)
            obj.rotation_euler = (0, np.pi - rz_degree_rad, np.pi)

            scale_multiple = logic_scene.vehicles[
                idx].scale
            obj.scale = (scale_multiple, scale_multiple, scale_multiple)
            obj.location = (x, y, z)
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

            bbox_corners_local = [corner[:] for corner in obj.bound_box]

            y_pos_corners = [bbox_corners_local[i] for i in [2, 3, 7, 6]]
            y_values = [corner[1] for corner in y_pos_corners]
            plane_y = np.min(y_values)

            if config.use_enhance_light:
                add_plane(idx)
                plane = bpy.data.objects.get(f"plane_{idx}")
                size = 5
                plane.scale = (size, size, 1)
                plane.rotation_mode = 'XYZ'
                plane.rotation_euler = (np.pi / 2, 0, 0)

                plane.location = (x, plane_y, z)

                dis_x, dix_y, dis_z = (3, -1000, 2)

                bpy.ops.object.light_add(type='SUN', location=((x + dis_x), (y + dix_y), (z + dis_z)))

                light = bpy.context.object
                light.rotation_mode = 'XYZ'
                light.rotation_euler = sun_rotation_euler
                light.data.color = (1, 1, 1)
                light.name = f'light_{idx}'
                light.data.energy = 5
                light_group = bpy.data.collections.new("LightGroup")
                bpy.context.scene.collection.children.link(light_group)

                light_group.objects.link(light)
                light_group.objects.link(plane)
                light_group.objects.link(obj)

        scene = bpy.context.scene
        scene.display.shading.color_type = 'TEXTURE'
        scene.display.render_aa = 'OFF'

        if not config.use_enhance_light:
            x, y, z = logic_scene.vehicles[0].location
            light_data = bpy.data.lights.new(name='my_light_data', type='SUN')
            light_data.energy = 5

            light = bpy.data.objects.new(name='my_light', object_data=light_data)
            light.location = ((x + 8), (y - 10), (z + 2))
            light.color = (1, 1, 1, 1)
            light.rotation_mode = 'XYZ'
            light.rotation_euler = (np.pi / 3, np.pi / 3, np.pi / 18)

            bpy.context.scene.collection.objects.link(light)

        calib_info = Calibration(calib_path)
        K = calib_info.P2[:, 0:3]

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
        camObj.location[0] = self.location[0]
        camObj.location[1] = self.location[1]
        camObj.location[2] = self.location[2]
        camObj.rotation_mode = self.coordinate_type
        camObj.rotation_euler = (np.pi, 0, 0)

        camObj.data.lens = lens
        camObj.data.shift_x = shift_x
        camObj.data.shift_y = shift_y
        camObj.data.sensor_height = sensor_height
        camObj.data.sensor_width = sensor_width
        camObj.data.sensor_fit = sensor_fit

        bpy.ops.render.render(write_still=True)


def get_lat_lon_data(data_num):
    params = {}
    params[0] = [49.011212804408, 8.4228850417969, '2012/09/26 13:57:47']
    if data_num not in params.keys():
        return None
    return params[data_num]


def get_sun(param):
    if param is None:
        return (np.pi / 3, np.pi / 2, np.pi / 18)
    import ephem
    lat, lon, date = param

    observer = ephem.Observer()
    observer.lat = lat
    observer.lon = lon
    observer.date = date

    sun = ephem.Sun(observer)
    sun_alt = sun.alt
    sun_az = sun.az

    rotation_euler = (
        -sun_alt,
        0,
        math.pi / 2 - sun_az
    )
    return rotation_euler


def init_scene(output_img_dir: str, bg_image_path: str, output_depth_img_dir: str, output_file_name: str):
    import bpy

    render_downsample = 1
    render_H = 375
    render_W = 1242
    sample_num = 32
    device = 'GPU'

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

    bpy.context.scene.cycles.device = device

    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    print("preferences.compute_device_type: ", \
          bpy.context.preferences.addons["cycles"].preferences.compute_device_type)

    for dev in bpy.context.preferences.addons["cycles"].preferences.devices:
        print(f"Use Device {dev['name']}: {dev['use']}")

    scene.cycles.samples = sample_num

    scene.render.resolution_x = render_W
    scene.render.resolution_y = render_H
    scene.render.resolution_percentage = 100 // render_downsample

    scene.render.film_transparent = True

    bpy.context.view_layer.use_pass_combined = True
    bpy.context.view_layer.use_pass_z = True
    bpy.context.view_layer.cycles.use_pass_shadow_catcher = True

    scene.use_nodes = True
    node_tree = scene.node_tree
    tree_nodes = node_tree.nodes
    tree_nodes.clear()

    render_node = tree_nodes.new('CompositorNodeRLayers')
    render_node.name = "Render_node"
    render_node.location = -300, 0

    transform_node_for_image = tree_nodes.new(type='CompositorNodeTransform')
    transform_node_for_image.location = 300, 400
    transform_node_for_image.filter_type = 'BILINEAR'
    transform_node_for_image.inputs['Scale'].default_value = 1 / render_downsample

    image_node = tree_nodes.new('CompositorNodeImage')
    image_node.name = "Image_node"
    image_node.location = 0, 400

    image_node.image = bpy.data.images.load(bg_image_path)
    image_node.image.colorspace_settings.name = 'Filmic sRGB'

    RGB_output_node = tree_nodes.new('CompositorNodeOutputFile')
    RGB_output_node.name = 'RGB_output_node'
    RGB_output_node.location = 1500, 200
    RGB_output_node.format.file_format = 'PNG'
    RGB_output_node.format.color_mode = 'RGBA'
    RGB_output_node.base_path = output_img_dir
    RGB_output_node.file_slots[
        0].path = output_file_name
    check_mkdir(output_img_dir)

    depth_output_node = tree_nodes.new('CompositorNodeOutputFile')
    depth_output_node.name = 'Depth_output_node'
    depth_output_node.location = 1500, 0
    depth_output_node.format.file_format = 'OPEN_EXR'
    depth_output_node.format.color_mode = 'RGBA'

    depth_output_node.base_path = output_depth_img_dir
    depth_output_node.file_slots[0].path = "vehicle_and_plane"
    check_mkdir(output_depth_img_dir)

    multiply_node = tree_nodes.new('CompositorNodeMixRGB')
    multiply_node.name = "Multiply_node"
    multiply_node.blend_type = 'MULTIPLY'
    multiply_node.location = 600, 100

    alpha_over_node = tree_nodes.new('CompositorNodeAlphaOver')
    alpha_over_node.name = "Alpha_over_node"
    alpha_over_node.location = 900, 100

    invert_node = tree_nodes.new('CompositorNodeInvert')
    invert_node.name = "Invert_node"
    invert_node.location = 300, -300

    set_alpha_node_1 = tree_nodes.new('CompositorNodeSetAlpha')
    set_alpha_node_1.name = "Set_alpha_node_1"
    set_alpha_node_1.location = 600, -300
    set_alpha_node_1.inputs[0].default_value = (1, 1, 1, 1)

    set_alpha_node_2 = tree_nodes.new('CompositorNodeSetAlpha')
    set_alpha_node_2.name = "Set_alpha_node_2"
    set_alpha_node_2.location = 600, -500
    set_alpha_node_2.inputs[0].default_value = (1, 1, 1, 1)

    add_node = tree_nodes.new('CompositorNodeMixRGB')
    add_node.name = "Add_node"
    add_node.blend_type = 'ADD'
    add_node.location = 900, -300
    add_node.use_clamp = True

    separate_rgba_node = tree_nodes.new(type='CompositorNodeSepRGBA')
    separate_rgba_node.name = "Seperate_RGBA"
    separate_rgba_node.location = 1200, -300

    node_tree.links.clear()
    links = node_tree.links

    links.new(render_node.outputs['Depth'], depth_output_node.inputs[0])

    links.new(render_node.outputs['Image'], alpha_over_node.inputs[2])
    if config.use_shadow:
        links.new(render_node.outputs['Shadow Catcher'],
                  multiply_node.inputs[1])

    links.new(image_node.outputs['Image'], transform_node_for_image.inputs['Image'])
    links.new(transform_node_for_image.outputs['Image'],
              multiply_node.inputs[2])
    links.new(multiply_node.outputs['Image'],
              alpha_over_node.inputs[1])
    links.new(alpha_over_node.outputs['Image'], RGB_output_node.inputs[0])

    links.new(render_node.outputs['Alpha'], set_alpha_node_2.inputs['Alpha'])
    links.new(render_node.outputs['Shadow Catcher'], invert_node.inputs['Color'])
    links.new(invert_node.outputs['Color'], set_alpha_node_1.inputs['Alpha'])
    links.new(set_alpha_node_1.outputs['Image'], add_node.inputs[1])
    links.new(set_alpha_node_2.outputs['Image'], add_node.inputs[2])
    links.new(add_node.outputs['Image'], separate_rgba_node.inputs['Image'])


def add_hdri():
    import bpy
    hdri_path = r"F:\MultiGen-Weather\test_blender\data\waymo_skydome\segment-11379226583756500423_6230_810_6250_810_with_camera_labels\000.exr"
    rotation = None
    C = bpy.context
    scn = C.scene

    node_tree = scn.world.node_tree
    tree_nodes = node_tree.nodes

    tree_nodes.clear()

    node_background = tree_nodes.new(type='ShaderNodeBackground')

    node_environment = tree_nodes.new('ShaderNodeTexEnvironment')

    node_environment.image = bpy.data.images.load(hdri_path)
    node_environment.location = -300, 0

    node_output = tree_nodes.new(type='ShaderNodeOutputWorld')
    node_output.location = 200, 0

    links = node_tree.links
    link = links.new(node_environment.outputs["Color"], node_background.inputs["Color"])
    link = links.new(node_background.outputs["Background"], node_output.inputs["Surface"])

    if rotation is not None:
        node_map = tree_nodes.new('ShaderNodeMapping')
        node_map.location = -500, 0
        node_texcoor = tree_nodes.new('ShaderNodeTexCoord')
        node_texcoor.location = -700, 0
        link = links.new(node_texcoor.outputs['Generated'], node_map.inputs['Vector'])
        link = links.new(node_map.outputs['Vector'], node_environment.inputs['Vector'])

        if isinstance(rotation, list):
            node_map.inputs['Rotation'].default_value = rotation
        elif isinstance(rotation, str):
            if rotation == 'camera_view':
                camera_obj_name = "Camera"
                camera = bpy.data.objects[camera_obj_name]
                camera.rotation_mode = 'XYZ'
                camera_rot_z = camera.rotation_euler.z
                print(camera.rotation_euler)
                node_map.inputs['Rotation'].default_value[2] = -camera_rot_z
                camera.rotation_mode = 'QUATERNION'
            else:
                raise 'This HDRI rotation is not implemented'
        else:
            raise 'This HDRI rotation is not implemented'


def add_plane(idx):
    import bpy
    size = 2770
    bpy.ops.mesh.primitive_plane_add(size=1)

    if hasattr(bpy.context, 'object'):
        plane = bpy.context.object
    else:
        if idx == 0:
            plane = bpy.data.objects["Plane"]
        else:
            plane = bpy.data.objects["Plane.f{idx:03d}"]

    plane.name = f"plane_{idx}"
    plane.is_shadow_catcher = True

    material = bpy.data.materials.new(name="new_plane_material")
    plane.data.materials.append(material)

    material.use_nodes = True
    nodes = material.node_tree.nodes
    BSDF_node = nodes.get("Principled BSDF")

    if BSDF_node:
        BSDF_node.inputs[0].default_value = (0.13, 0.13, 0.13, 1)

        BSDF_node.inputs[9].default_value = 1
        BSDF_node.inputs[
            21].default_value = 1


if __name__ == '__main__':
    ...

    import sys

    args = sys.argv[5:]

    blender_instance = Blender()
    blender_instance.render_gltf_obj(args)
