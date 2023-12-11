from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from segment_anything import  set_torch_image
import cv2
import numpy as np 
from loguru import logger

# padding to square
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


def gen_masks(masks, image, base_name):
    if len(masks) == 0:
        return None
    prefix_name = base_name.split(".")[0]
    res = np.zeros([masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 3])
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    for idx, mask in enumerate(sorted_masks):
        object_mask = mask["segmentation"]
        x, y, w, h = mask["bbox"]
        crop_img, crop_pos = crop_tool.crop( image,  mask["bbox"], scale=1.2, out_w=224, out_h=224 )
        torch_image = set_torch_image(crop_img)

    return res


def get_model_info(type="b"):
    if type == "b":
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        model_type = "vit_b"
    elif type == "l":
        sam_checkpoint = "sam_vit_l_0b3195.pth"
        model_type = "vit_l"
    elif type == "h":
        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"
    else:
        raise NotImplementedError
    return sam_checkpoint, model_type

if __name__ == "__main__":
    ckpt, model_type = get_model_info("h")
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    DEVICE = "cuda"
    sam.to(device=DEVICE)
    MASK_GEN = SamAutomaticMaskGenerator(sam)
    logger.info(f"load SAM model from {ckpt}")
    crop_tool = CropImage()
    ROOT_DIR = "JPEGImages"
    import os 
    LINDMOED_DIR = os.listdir(ROOT_DIR)
    for idx, item in enumerate(LINDMOED_DIR):
        full_path = os.path.join(ROOT_DIR, item)
        image = cv2.imread(full_path)
        masks = MASK_GEN.generate(image)
        base_name = os.path.basename(full_path)
        color_mask = gen_masks(masks, image, base_name)
