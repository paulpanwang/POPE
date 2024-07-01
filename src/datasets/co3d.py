import os
from os import path as osp
from typing import Dict
from unicodedata import name
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import torch.utils as utils
from numpy.linalg import inv


import glob
from pathlib import Path
import os
import cv2
import numpy as np
import plyfile
from PIL import Image
from skimage.io import imread, imsave

Co3D_ROOT = 'data/co3d'

def mask2bbox(mask):
    if np.sum(mask)==0:
        return np.asarray([0, 0, 0, 0],np.float32)
    ys, xs = np.nonzero(mask)
    x_min = np.min(xs)
    y_min = np.min(ys)
    x_max = np.max(xs)
    y_max = np.max(ys)
    return np.asarray([x_min, y_min, x_max - x_min, y_max - y_min], np.int32)

class Co3DResizeDatabase(BaseDatabase):
    def __init__(self, database_name):
        super(Co3DResizeDatabase, self).__init__(database_name)
        _, self.category, self.sequence, sizes = database_name.split('/')
        self.fg_size, self.bg_size = [int(item) for item in sizes.split('_')]
        self._build_resize_database()

    def _build_resize_database(self):
        annotation_fn = Path(f'{Co3D_ROOT}_{self.fg_size}_{self.bg_size}/{self.category}/{self.sequence}/info.pkl')
        root_dir = annotation_fn.parent
        self.image_root = (root_dir / 'images')
        self.mask_root = (root_dir / 'masks')
        if annotation_fn.exists():
            self.Ks, self.poses, self.img_ids, self.ratios = read_pickle(str(annotation_fn))
        else:
            raise NotImplementedError

    def get_image(self, img_id, ref_mode=False):
        return imread(str(self.image_root / f'{img_id}.jpg'))

    def get_K(self, img_id):
        return self.Ks[img_id].copy()

    def get_pose(self, img_id):
        return self.poses[img_id].copy()

    def get_img_ids(self):
        return self.img_ids

    def get_bbox(self, img_id):
        return mask2bbox(self.get_mask(img_id))

    def get_mask(self, img_id):
        return imread(str(self.mask_root / f'{img_id}.png')) > 0


