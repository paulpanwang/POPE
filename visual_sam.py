from pope_model_api import *
import cv2
import numpy as np 
from loguru import logger
import os

def render_mask(masks):
    if len(masks) == 0:
        return None
    res = np.zeros([masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 3])
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    for mask in sorted_masks:
        m = mask["segmentation"]
        res[:, :, 0][m] = np.random.randint(0, 255)
        res[:, :, 1][m] = np.random.randint(0, 255)
        res[:, :, 2][m] = np.random.randint(0, 255)
    res = res.astype(np.uint8)
    return res

if __name__ == "__main__":
    ckpt, model_type = get_model_info("h")
    sam = sam_model_registry[model_type](checkpoint=ckpt)
    DEVICE = "cuda"
    sam.to(device=DEVICE)
    MASK_GEN = SamAutomaticMaskGenerator(sam)
    logger.info(f"load SAM model from {ckpt}")
    
    full_file_name = "data/demos/LINEMOD.png"
    image = cv2.imread(full_file_name)

    masks = MASK_GEN.generate(image)
    color_mask = render_mask(masks)
    thres = 0.75
    render_img = (image * thres + color_mask * (1 - thres)).astype(np.uint8)
    DEST_image = full_file_name.replace(".png", "_mask.png")  
    if render_img is not None:
        cv2.imwrite(DEST_image, render_img)
        logger.info(f"result is saved at: {DEST_image}")