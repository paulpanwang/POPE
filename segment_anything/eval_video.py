from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import  (
    set_torch_image,
    convert_mask_to_polygon,
    common_resize,
    convert_contour2mask,
    letterbox_image
)
from segment_anything import (
    load_dinov2_model,
    get_cls_token,
    get_cls_token_torch
)
import sys
sys.path.insert(0,"/mnt/bn/raoqiang-lq-nas/panpanwang/data/OnePose_Plus_Plus/src/utils")
from data_utils import get_image_crop_resize, get_K_crop_resize

sys.path.insert(0, "/mnt/bn/raoqiang-lq-nas/panpanwang/data/Gen6d")
from utils.base_utils import  project_points, transformation_crop

import cv2
import numpy as np 
from loguru import logger

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
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        model_type = "vit_b"
    elif type == "l":
        sam_checkpoint = "sam_vit_l_0b3195.pth"
        model_type = "vit_l"
    elif type == "h":
        sam_checkpoint = "segment_anything/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
    else:
        raise NotImplementedError
    return sam_checkpoint, model_type

import torch
from src.loftr import LoFTR, default_cfg
import os
import pandas as pd 
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import cv2
import json
import shutil

from numpy.linalg import inv
from src.utils.dataset import (
    read_scannet_gray,
    read_scannet_pose,
    read_scannet_grayv2,
)

from tabulate import tabulate
from loguru import logger
from exps.epipolar_util import (
    draw_epiplor_line,
)
import time

from src.utils.metrics import estimate_pose, relative_pose_error
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("weights/indoor_ot.ckpt")['state_dict'], strict = False )
matcher = matcher.eval().cuda()
logger.info(f"load LOFTR model")

ckpt, model_type = get_model_info("h")
sam = sam_model_registry[model_type](checkpoint=ckpt)
DEVICE = "cuda"
sam.to(device=DEVICE)
MASK_GEN = SamAutomaticMaskGenerator(sam)
logger.info(f"load SAM model from {ckpt}")
crop_tool = CropImage()
dinov2_model = load_dinov2_model()
dinov2_model.to("cuda:0")

metrics = dict()
metrics.update({'R_errs': [], 't_errs': [], 'inliers': [] , "identifiers":[] })


id2name_dict = {
    1: "ape",
    2: "benchvise",
    4: "camera",
    5: "can",
    6: "cat",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}

def _np_to_cv2_kpts(np_kpts):
    cv2_kpts = []
    for np_kpt in np_kpts:
        cur_cv2_kpt = cv2.KeyPoint()
        cur_cv2_kpt.pt = tuple(np_kpt)
        cv2_kpts.append(cur_cv2_kpt)
    return cv2_kpts

# 使用LOFTR作为基线方法
matcher = LoFTR(config=default_cfg)
matcher.load_state_dict(torch.load("weights/indoor_ot.ckpt")['state_dict'], strict = False )
matcher = matcher.eval().cuda()

# load model 
ROOT_DIR = "/mnt/bn/raoqiang-lq-nas/panpanwang/data/onepose/"


import random
from  tqdm import tqdm
import torch.nn.functional as F
res_table = []

import json
with open("assets/Onepose-HighTexture-test.json") as f:
    dir_list = json.load(f)

for label_idx , test_dict in enumerate(dir_list):
    print(f"Onepose: {label_idx}")
    metrics = dict()
    metrics.update({'R_errs': [], 't_errs': [], 'inliers': [] , "identifiers":[] })
    sample_data = dir_list[label_idx]["0"][0]
    label = sample_data.split("/")[0]
    name = label.split("-")[1]
    dir_name = os.path.dirname(sample_data)
    FULL_ROOT_DIR = os.path.join(ROOT_DIR, dir_name) 
    recall_image,all_image = 0,0
    for rotation_key, rotation_list in zip(test_dict.keys(), test_dict.values()):
        for pair_idx,pair_name in enumerate(tqdm(rotation_list)):
            all_image = all_image + 1
            base_name = os.path.basename(pair_name)
            idx0_name = base_name.split("-")[0]
            idx1_name = base_name.split("-")[1]
            image0_name = os.path.join( FULL_ROOT_DIR, idx0_name )

            image1_name = os.path.join( FULL_ROOT_DIR.replace("color", "color"),  idx1_name )
            intrinsic_path = image0_name.replace("color", "intrin_ba").replace("png","txt")
            K0 = np.loadtxt(intrinsic_path, delimiter=' ')
            intrinsic_path = image1_name.replace("color", "intrin_ba").replace("png","txt")
            K1 = np.loadtxt(intrinsic_path, delimiter=' ')

            image0 = cv2.imread(image0_name)
            ref_torch_image = set_torch_image(image0, center_crop=True)
            ref_fea = get_cls_token_torch(dinov2_model, ref_torch_image)
            image1 = cv2.imread(image1_name)
            image_h,image_w,_ = image1.shape
            t1 = time.time()
            masks = MASK_GEN.generate(image1)
            t2 = time.time()
            similarity_score, top_images  = np.array([0,0,0],np.float32) , [[],[],[]]
            t3 = time.time()
            compact_percent = 0.3
            for xxx, mask in enumerate(masks):
                object_mask = np.expand_dims(mask["segmentation"], -1)
                x0, y0, w, h = mask["bbox"]
                x1, y1 = x0+w,y0+h
                x0 -= int(w * compact_percent)
                y0 -= int(h * compact_percent)
                x1 += int(w * compact_percent)
                y1 += int(h * compact_percent)
                box = np.array([x0, y0, x1, y1])
                resize_shape = np.array([y1 - y0, x1 - x0])
                K_crop, K_crop_homo = get_K_crop_resize(box, K1, resize_shape)
                image_crop, _ = get_image_crop_resize(image1, box, resize_shape)
                # object_mask,_ = get_image_crop_resize(object_mask, box, resize_shape)
                box_new = np.array([0, 0, x1 - x0, y1 - y0])
                resize_shape = np.array([512, 512])
                K_crop, K_crop_homo = get_K_crop_resize(box_new, K_crop, resize_shape)
                image_crop, _ = get_image_crop_resize(image_crop, box_new, resize_shape)
                crop_tensor = set_torch_image(image_crop, center_crop=True)
                with torch.no_grad():
                    fea = get_cls_token_torch(dinov2_model, crop_tensor)
                score = F.cosine_similarity(ref_fea, fea, dim=1, eps=1e-8)
                if  (score.item() > similarity_score).any():
                    mask["crop_image"] = image_crop
                    mask["K"] = K_crop
                    mask["bbox"] = box
                    min_idx = np.argmin(similarity_score)
                    similarity_score[min_idx] = score.item()
                    top_images[min_idx] = mask.copy()

            img0 =  cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            img0 = torch.from_numpy(img0).float()[None] / 255.
            img0 = img0.unsqueeze(0).cuda()

            matching_score =  [ [0] for _ in range(len(top_images)) ]
            for top_idx in range(len(top_images)):
                if "crop_image" not in top_images[top_idx]:
                    continue
                img1 =  cv2.cvtColor(top_images[top_idx]["crop_image"], cv2.COLOR_BGR2GRAY)
                img1 = torch.from_numpy(img1).float()[None] / 255.
                img1 = img1.unsqueeze(0).cuda()
                batch = {'image0': img0, 'image1': img1}
                with torch.no_grad():
                    matcher(batch)    
                    mkpts0 = batch['mkpts0_f'].cpu().numpy()
                    mkpts1 = batch['mkpts1_f'].cpu().numpy()
                    confidences = batch["mconf"].cpu().numpy()
                conf_mask = np.where(confidences > 0.9)
                matching_score[top_idx] = conf_mask[0].shape[0]
                top_images[top_idx]["mkpts0"] = mkpts0
                top_images[top_idx]["mkpts1"] = mkpts1
                top_images[top_idx]["mconf"] = confidences
   
            t4 = time.time()
            # print(f"t4-t3: object detection:{1000*(t4-t3)} ms")
            pose0_name = image0_name.replace("color", "poses_ba").replace("png","txt")
            pose1_name = image1_name.replace("color", "poses_ba").replace("png","txt")
            pose0 = np.loadtxt(pose0_name)
            pose1 = np.loadtxt(pose1_name)
            # pose0 = np.vstack((pose0, np.array([[0,0,0,1]])))
            # pose1 = np.vstack((pose1, np.array([[0,0,0,1]])))
            relative_pose =  np.matmul(pose1, inv(pose0))
            t = relative_pose[:3,-1].reshape(1,3)
            if "crop_image" not in top_images[top_idx]:
                continue
            max_match_idx = np.argmax(matching_score)
            pre_bbox  = top_images[max_match_idx]["bbox"]
            mkpts0 = top_images[max_match_idx]["mkpts0"]
            mkpts1 = top_images[max_match_idx]["mkpts1"]
            pre_K = top_images[max_match_idx]["K"]

            _3d_bbox = np.loadtxt(f"{os.path.join(ROOT_DIR, label)}/box3d_corners.txt")
            bbox_pts_3d, _ = project_points(_3d_bbox, pose1[:3,:4], K1)
            bbox_pts_3d = bbox_pts_3d.astype(np.int32)
            x0, y0, w, h = cv2.boundingRect(bbox_pts_3d)
            x1,y1 = x0+w, y0+h
            gt_bbox = np.array([x0, y0, x1, y1])
            is_recalled = recall_object(pre_bbox , gt_bbox)
            recall_image = recall_image + int(is_recalled>0.5)
            ret = estimate_pose(mkpts0, mkpts1 , K0 , pre_K , 0.5, 0.99)  
            if ret is  not None:
                Rot, t, inliers = ret 
                t_err, R_err = relative_pose_error(relative_pose, Rot, t, ignore_gt_t_thr=0.0)
                metrics['R_errs'].append(R_err)
                metrics['t_errs'].append(t_err)
            else:
                metrics['R_errs'].append(90)
                metrics['t_errs'].append(90)
            metrics["identifiers"].append( pair_name )

        
    print(f"Acc: {recall_image}/{all_image}")
    import pprint
    from src.utils.metrics import (
        aggregate_metrics
    )
    from loguru import logger
    val_metrics_4tb = aggregate_metrics(metrics, 5e-4)
    val_metrics_4tb["AP50"] = recall_image/all_image
    logger.info('\n' + pprint.pformat(val_metrics_4tb))

    res_table.append( [f"{name }"] +  list(val_metrics_4tb.values())  )

from tabulate import tabulate
headers = ["Category"] + list(val_metrics_4tb.keys())
all_data = np.array(res_table)[:,1:].astype(np.float32)
res_table.append( ["Avg"] + all_data.mean(0).tolist() )
print(tabulate(res_table, \
      headers=headers, tablefmt='fancy_grid'))

