from typing import List

import cv2
import numpy
import numpy as np

import config
from utils.Utils_common import UtilsCommon


class ImgCombination(object):

    def img_combination_with_objs(self, img_bg: numpy.ndarray, img_objs: List[numpy.ndarray],
                                  obj_centers_in_img: List[List[int]],
                                  index: numpy.ndarray, refine=True) -> numpy.ndarray:

        objs = []
        for ix in index:
            objs.append(img_objs[ix])
        pos_images = np.asarray(obj_centers_in_img, dtype=object)[index]
        print("插入索引顺序：", index)
        masks_obj = []
        for obj in objs:
            mask = UtilsCommon.get_mask_from_RGBA(obj)
            masks_obj.append(mask)

        bg_no_refine = img_bg.copy()
        bg_refine = img_bg.copy()

        for obj, pos_image in zip(objs, pos_images):
            bg_no_refine = self.img_combination_with_one_obj(pos_image, obj, bg_no_refine)

        if config.camera_config.is_image_refine and refine:
            for obj, pos_image in zip(objs, pos_images):
                mask = UtilsCommon.get_mask_from_RGBA(obj)
                bg_refine = self.combine_bg_with_obj_single_refine(pos_image, obj, bg_refine, mask)
        else:
            bg_refine = bg_no_refine
        return bg_no_refine, bg_refine

    def img_combination_with_one_obj(self, center: numpy.ndarray, obj: numpy.ndarray,
                                     bg: numpy.ndarray) -> numpy.ndarray:

        bg_ymax, bg_xmax, _ = bg.shape
        obg_ymax, obg_xmax, _ = obj.shape
        obj_xmin = center[0] - int(0.5 * obg_xmax)
        obj_ymin = center[1] - int(0.5 * obg_ymax)
        bg_temp = bg.copy()
        for i in range(obj.shape[0]):
            for j in range(obj.shape[1]):
                if not obj[i][j][3] == 0 and 0 <= i + obj_ymin < bg_ymax and 0 <= j + obj_xmin < bg_xmax:
                    bg_temp[i + obj_ymin, j + obj_xmin, :3] = obj[i, j, :3]
                else:
                    pass
        return bg_temp

    def combine_bg_with_obj_single_refine(self, center: numpy.ndarray, obj: numpy.ndarray, bg: numpy.ndarray,
                                          mask: numpy.ndarray) -> numpy.ndarray:

        from core.sensor_simulation.image_refine import ImageHarmonization
        image_refine = ImageHarmonization()
        bg_ymax, bg_xmax, _ = bg.shape
        obg_ymax, obg_xmax, _ = obj.shape
        image_refine.size = (bg_ymax, bg_xmax)
        assert bg_ymax < bg_xmax
        obj_xmin = center[0] - int(0.5 * obg_xmax)
        obj_ymin = center[1] - int(0.5 * obg_ymax)

        obj_rgb = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
        bg_temp2_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        bg_temp2 = image_refine.run(obj_rgb, mask, bg_temp2_rgb, (obj_xmin, obj_ymin))
        bg_temp2 = cv2.cvtColor(bg_temp2, cv2.COLOR_RGB2BGR)
        return bg_temp2
