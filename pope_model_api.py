
import os
import cv2
import torch
import pandas as pd 
import numpy as np
import time
import json
import shutil
from loguru import logger

from  tqdm import tqdm
import torch.nn.functional as F

from numpy.linalg import inv
from src.utils.dataset import (
    read_scannet_gray,
    read_scannet_pose,
    read_scannet_grayv2,
)
from tabulate import tabulate
from loguru import logger

from segment_anything.segment_anything import (
    SamAutomaticMaskGenerator, 
    sam_model_registry
)

from segment_anything.segment_anything import  (
    set_torch_image,
    convert_mask_to_polygon,
    common_resize,
    convert_contour2mask,
    letterbox_image
)
from segment_anything.segment_anything import (
    load_dinov2_model,
    get_cls_token,
    get_cls_token_torch
)

from utils.data_utils import (
    get_image_crop_resize, 
    get_K_crop_resize
)

from utils.base_utils import  (
    project_points,
    transformation_crop
)

from scipy.spatial.transform import Rotation as R
from src.matcher import Matcher, default_cfg
from src.utils.metrics import estimate_pose, relative_pose_error


def recall_object(boxA, boxB, thresholded=0.5):
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)    
    return iou


def _np_to_cv2_kpts(np_kpts):
    cv2_kpts = []
    for np_kpt in np_kpts:
        cur_cv2_kpt = cv2.KeyPoint()
        cur_cv2_kpt.pt = tuple(np_kpt)
        cv2_kpts.append(cur_cv2_kpt)
    return cv2_kpts


def convert_mask_to_polygon(mask):
    contours = None
    if int(cv2.__version__.split('.')[0]) > 3:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    else:
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[1]
    contours = max(contours, key=lambda arr: arr.size)
    if contours.shape.count(1):
        contours = np.squeeze(contours)
    if contours.size < 3 * 2:
        raise Exception('Less then three point have been detected. Can not build a polygon.')
    return contours


def gen_crop_images(masks, image, base_name):
    prefix_name = base_name.split(".")[0]
    res = np.zeros([masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 3])
    # sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    images = []
    for idx, mask in enumerate(masks):
        object_mask = mask["segmentation"]
        x, y, w, h = mask["bbox"]
        object_mask = np.array(255*object_mask, dtype=np.uint8)
        crop_img, crop_pos = crop_tool.crop( image,  mask["bbox"], scale=1.2, out_w=224, out_h=224 )
        torch_image = set_torch_image(crop_img)        
        # cv2.imwrite(f"crop_images/{base_name}-crop-{idx}.jpg", crop_img)
        images.append(torch_image)
    return torch.cat(images, dim = 0)


def get_model_info(type="b"):
    if type == "b":
        sam_checkpoint = "weights/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
    elif type == "l":
        sam_checkpoint = "weights/sam_vit_l_0b3195.pth"
        model_type = "vit_l"
    elif type == "h":
        sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
    else:
        raise NotImplementedError
    return sam_checkpoint, model_type

class CropImage:
    @staticmethod
    def _get_new_box(src_w, src_h, bbox, scale):
        x = bbox[0]
        y = bbox[1]
        box_w = bbox[2]
        box_h = bbox[3]
        scale = min((src_h-1)/box_h, min((src_w-1)/box_w, scale))
        new_width = box_w * scale
        new_height = box_h * scale
        center_x, center_y = box_w/2+x, box_h/2+y
        left_top_x = center_x-new_width/2
        left_top_y = center_y-new_height/2
        right_bottom_x = center_x+new_width/2
        right_bottom_y = center_y+new_height/2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0

        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0

        if right_bottom_x > src_w-1:
            left_top_x -= right_bottom_x-src_w+1
            right_bottom_x = src_w-1

        if right_bottom_y > src_h-1:
            left_top_y -= right_bottom_y-src_h+1
            right_bottom_y = src_h-1

        return int(left_top_x), int(left_top_y),\
               int(right_bottom_x), int(right_bottom_y)

    def crop(self, org_img, bbox, scale, out_w, out_h, crop=True):

        if not crop:
            dst_img = cv2.resize(org_img, (out_w, out_h))
        else:
            src_h, src_w, _ = np.shape(org_img)
            left_top_x, left_top_y, \
                right_bottom_x, right_bottom_y = self._get_new_box(src_w, src_h, bbox, scale)

            img = org_img[left_top_y: right_bottom_y+1,
                          left_top_x: right_bottom_x+1]
            dst_img = cv2.resize(img, (out_w, out_h))
        return dst_img, [left_top_x, left_top_y, right_bottom_x, right_bottom_y ]



from src.utils.metrics import estimate_pose, relative_pose_error


matcher = Matcher(config=default_cfg)
# we set strict to False
matcher.load_state_dict(torch.load("weights/matcher.pth")['state_dict'], strict=False)
matcher = matcher.eval().cuda()
logger.info(f"load Matcher successfully")