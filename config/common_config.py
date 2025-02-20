occlusion_th = 0.9
occ_point_max = 20

from pathlib import Path

current_file_path = Path(__file__).resolve()
project_dirname = current_file_path.parent.parent

result_path = "{}/result".format(project_dirname)
stastic_path = "{}/result_stastic".format(project_dirname)

logic_scene_dir = "{}/scene_workplace".format(project_dirname)

debug_log_path = "{}/debug_log.txt".format(project_dirname)

image_harmonization_model_path = "{}/third/S2CRNet/model/S2CRNet_pretrained.pth".format(project_dirname)

road_split_dir = "{}/third/CENet".format(project_dirname)
road_split_pc_dir = "{}/third/CENet/_data/sequences/00/velodyne".format(project_dirname)
road_split_label_dir = "{}/third/CENet/result/sequences/00/predictions".format(project_dirname)

tracking_dataset_path = "{}/_datasets/KITTI_TRACKING/training".format(project_dirname)
tracking_logic_scene_dir = "{}/workplace/logic_scene".format(project_dirname)
trakcing_insert_dir = "{}/workplace/insert_track".format(project_dirname)
trakcing_background_dir = "{}/workplace/background_track".format(project_dirname)
oxts_path = "{}/oxts".format(tracking_dataset_path)

result_root_dirname_web = "E:/multiGen-batch/Generated"
result_root_dirname_local = r"G:\tmp"
aug_datasets_dirname = "{}/_aug_datasets".format(project_dirname)
aug_queue_datasets_dirname = "{}/_queue/KITTI".format(project_dirname)
blender_root_dirname = "{}/blender".format(project_dirname)

bg_root_dirname = "{}/_datasets/KITTI".format(project_dirname)
bg_dirname = "{}/training".format(bg_root_dirname)
pcd_bg_dirname = "velodyne"
img_bg_dirname = "image_2"
label_bg_dirname = "label_2"
label_index_bg_dirname = "label_2_id"
calib_bg_dirname = "calib"
pcd_show_dirname = "pcd"
img_depth_dirname = "depth"
road_on_img_dirname = "{}/road_on_img/".format(bg_dirname)
non_road_dirname = "{}/non_road_pc/".format(bg_dirname)
image_depth_dirname = "{}/image_depth/".format(bg_dirname)
scene_file_dirname = "{}/scene_file".format(bg_root_dirname)
image_batch_generated_dirname = "{}/image_batch_generated".format(bg_root_dirname)

kitti_val_txt_path = "{}/ImageSets/val.txt".format(bg_root_dirname)

road_bg_dir = "{}/road_label".format(bg_dirname)

assets_dirname = "{}/_assets".format(project_dirname)
obj_root_dirname = "{}".format(assets_dirname)
obj_dirname = "{}/shapenet".format(obj_root_dirname)

obj_cp_dir = "{}/copy_paste".format(assets_dirname)

obj_filename_new = "models/model_normalized.obj"
obj_car_type_json_path = "{}/car_type.json".format(obj_root_dirname)
mc_score_filename = "mc_score.txt"
multi_scale = 5.5
multi_scale_blender = 5.5

distance_threshold = 0.01
ransac_n = 3
num_iterations = 1000

number_of_decimal = 6
road_range = [0, 80, -12, 12, 0.35]

split_name_flag = "_"

image_vis_mode = "plt"

max_try_pose_num = 100
