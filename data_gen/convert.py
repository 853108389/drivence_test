import os
import shutil

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.init_.init_dir import symlink


def init_or_clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def _convert_type(_df, column, dest_type, only_from_type=None):
    cond = only_from_type is None or _df[column].dtype == only_from_type
    if cond:
        _df[column] = _df[column].astype(dest_type)


def convert_calib_dec2track(dec_calib, output_file):
    replace_dict = {
        'R0_rect': 'R_rect',
        'Tr_velo_to_cam': 'Tr_velo_cam',
        'Tr_imu_to_velo': 'Tr_imu_velo',

    }
    with open(dec_calib, 'r', encoding='utf-8') as file:
        content = file.read()
        for old_char, new_char in replace_dict.items():
            content = content.replace(old_char, new_char)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(content)


def convert_calib_track2dec(tracking_calib, output_file):
    replace_dict = {
        'R_rect': 'R0_rect',
        'Tr_velo_cam': 'Tr_velo_to_cam',
        'Tr_imu_velo': 'Tr_imu_to_velo',

    }
    with open(tracking_calib, 'r', encoding='utf-8') as file:
        content = file.read()
        for old_char, new_char in replace_dict.items():
            content = content.replace(old_char, new_char)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(content)


def create_depth_data(input_root, seq, start_frame, end_frame):
    seq = '%04d' % seq
    in_training = os.path.join(input_root, 'training')
    input_depth = os.path.join(in_training, 'depth_ori', f"{seq}")
    output_depth = os.path.join(in_training, 'depth_update', f"{seq}")
    os.makedirs(output_depth, exist_ok=True)

    tracking_seg = os.path.join(in_training, 'panoptic_maps', f'{seq}')
    print("create depth data...")
    for frame in tqdm(range(start_frame, end_frame)):
        sample_str = str(frame).zfill(6)

        if not os.path.exists(os.path.join(output_depth, f'{sample_str}.png')):
            depth_path = os.path.join(input_depth, f'{sample_str}.png')
            semantic_path = os.path.join(tracking_seg, f'{sample_str}.png')
            img_sematic = np.array(Image.open(semantic_path).convert('L'))
            img_depth = np.array(Image.open(depth_path).convert('L'))
            img_height, img_width = img_sematic.shape
            img_height2, img_width2 = img_depth.shape
            assert img_height == img_height2 and img_width == img_width2
            for i in range(img_height):
                for j in range(img_width):
                    if img_sematic[i, j] in [0, 3]:
                        img_depth[i, j] = 100000

            Image.fromarray(img_depth).save(os.path.join(output_depth, f'{sample_str}.png'))


def create_train_sample_data(input_root, output_root, seq, start_frame, end_frame, init_or_clear_dirs=True):
    res_training = os.path.join(output_root, 'training')

    res_calib = os.path.join(res_training, 'calib')
    res_image = os.path.join(res_training, 'image_2')
    res_label = os.path.join(res_training, 'label_2')
    res_label_id = os.path.join(res_training, 'label_2_id')
    res_lidar = os.path.join(res_training, 'velodyne')
    res_depth = os.path.join(res_training, 'depth')

    if init_or_clear_dirs:
        init_or_clear_dir(res_calib)
        init_or_clear_dir(res_image)
        init_or_clear_dir(res_label)
        init_or_clear_dir(res_lidar)
        init_or_clear_dir(res_label_id)
        init_or_clear_dir(res_depth)

    in_training = os.path.join(input_root, 'training')

    seq = '%04d' % seq
    tracking_image = os.path.join(in_training, 'image_02', seq)
    tracking_lidar = os.path.join(in_training, 'velodyne', seq)
    tracking_calib = os.path.join(in_training, 'calib_fill', f'{seq}.txt')
    tracking_label = os.path.join(in_training, 'label_02', f'{seq}.txt')
    tracking_depth = os.path.join(in_training, 'depth_update', f'{seq}')

    import pandas as pd
    ori_labels_df = pd.read_csv(tracking_label, sep=' ', header=None,
                                index_col=0, skip_blank_lines=True)
    for c in ori_labels_df.columns:
        _convert_type(ori_labels_df, c, np.float32, np.float64)
        _convert_type(ori_labels_df, c, np.int32, np.int64)

    for frame in range(start_frame, end_frame):

        sample_str = str(frame).zfill(6)
        sample_str2 = str(frame).zfill(4)

        assert os.path.isfile(os.path.join(tracking_image, f'{sample_str}.png'))
        symlink(os.path.join(tracking_image, f'{sample_str}.png'),
                os.path.join(res_image, f'{sample_str}.png'))
        assert os.path.isfile(os.path.join(tracking_lidar, f'{sample_str}.bin'))
        symlink(os.path.join(tracking_lidar, f'{sample_str}.bin'),
                os.path.join(res_lidar, f'{sample_str}.bin'))

        assert os.path.isfile(os.path.join(tracking_depth, f'{sample_str}.png'))
        symlink(os.path.join(tracking_depth, f'{sample_str}.png'),
                os.path.join(res_depth, f'{sample_str}.png'))

        output_file = os.path.join(res_calib, f'{sample_str}.txt')
        convert_calib_track2dec(tracking_calib, output_file)

        olabel_id = os.path.join(res_label_id, f'{sample_str}.txt')
        olabel_main = os.path.join(res_label, f'{sample_str}.txt')

        if frame in ori_labels_df.index:
            labels_df = ori_labels_df.loc[[frame]]
            labels_df_id = labels_df.iloc[:, :1]
            labels_df_main = labels_df.iloc[:, 1:]
            labels_df_id.to_csv(olabel_id, sep=' ', index=True, header=False)
            labels_df_main.to_csv(olabel_main, sep=' ', index=False, header=False)
        else:
            open(olabel_id, 'w').close()
            open(olabel_main, 'w').close()
