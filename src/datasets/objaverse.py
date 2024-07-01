import os
from os import path as osp
from typing import Dict
from unicodedata import name
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import torch.utils as utils
from numpy.linalg import inv

from src.utils.dataset import (
    read_scannet_rgb,
    read_scannet_gray,
    read_scannet_depth,
    read_scannet_pose,
    read_scannet_intrinsic,
    convert_gray,
    read_scannet_grayv2
)
from loguru import logger
import random

from src.utils.metrics import relative_pose_error


class ObjaverseDataset(utils.data.Dataset):
    def __init__(self,
                 root_dir="data/objKV",
                 npz_path = "data1_kv.npy" ,
                 mode='train',
                 min_overlap_score=0.4,
                 augment_fn=None,
                 pose_dir=None,
                 **kwargs):
        """Manage one scene of ScanNet Dataset.
        Args:
            root_dir (str): ScanNet root directory that contains scene folders.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            mode (str): options are ['train', 'val', 'test'].
            augment_fn (callable, optional): augments images with pre-defined visual effects.
            pose_dir (str): ScanNet root directory that contains all poses.
                (we use a separate (optional) pose_dir since we store images and poses separately.)
        """
        super().__init__()
        self.ROOT_DIR = root_dir
        self.class_file = np.loadtxt(npz_path, dtype='str')
        logger.info(f"cls: {self.class_file.shape }")
        self.intrinsic = torch.Tensor([[1120,0,256],[0,840,256],[0,0,1]])
        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        return len(self.class_file)

    def _compute_quaternion(self, relative_pose):
        t = relative_pose[:3,3].reshape(1,3)
        gt_quan = R.from_matrix(relative_pose[:3,:3]).as_quat().reshape(1,4)
        return np.hstack((t,gt_quan))

    def _compute_rel_pose(self, pose0, pose1):
        pose0 = np.vstack((pose0, np.array([0,0,0,1]).reshape(1,4)))
        pose1 = np.vstack((pose1, np.array([0,0,0,1]).reshape(1,4)))

        t_err, R_err = relative_pose_error(pose0, \
                     pose1[:3, :3], pose1[:3,3])  


        relative_pose =  np.matmul(pose1, inv(pose0))  # (4, 4)
        return relative_pose, t_err, R_err
    # todo  
    def __getitem__(self, idx):
        cls_item = self.class_file[idx]
        cls_item = os.path.join(self.ROOT_DIR , cls_item )

        while True:
            idx0, idx1 = random.sample(list(range(11)), 2)
            image_0_path = os.path.join(cls_item , f"{idx0:03d}.png" )
            image_1_path = os.path.join(cls_item , f"{idx1:03d}.png" )
            if (not os.path.exists(image_0_path)) or  (not os.path.exists(image_1_path)):
                continue

            # image0 = read_scannet_rgb(image_0_path)
            # image1 = read_scannet_rgb(image_1_path)

            image0 = read_scannet_grayv2(image_0_path)
            image1 = read_scannet_grayv2(image_1_path)
            
            pose0 = np.load( os.path.join(cls_item , f"{idx0:03d}.npy")  )
            pose1 = np.load( os.path.join(cls_item , f"{idx1:03d}.npy")  )
            depth0 = depth1 = torch.tensor([])
            # read the intrinsic of depthmap
            K_0 = K_1 = self.intrinsic
            t0to1, t_err, R_err = self._compute_rel_pose(pose0, pose1) 
            angular_err = random.sample([30,30,30,60,60,90], 1)
            if int(R_err) <= angular_err[0]:
                break

        t1to0 = inv(t0to1)
        # read and compute relative poses
        T_0to1 = torch.tensor(self._compute_quaternion(t0to1), 
                              dtype=torch.float32)

        # T_1to0 = torch.tensor(self._compute_quaternion(t1to0), 
        #                       dtype=torch.float32)

        # images0 = torch.cat([image0, image1],dim=0)
        # images1 = torch.cat([image1, image0],dim=0)
        # T_0to1 =  torch.cat([T_0to1, T_1to0],dim=0)
    
        data = {
            'image0': image0,  # (1, h, w)
            'depth0': depth0,   # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,   # (4, 4)
            'K0': K_0,  # (3, 3)
            'K1': K_1,
        }
        return data
    
if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    from src.utils.augment import build_augmentor
    aug_fun = build_augmentor(method=None)
    ROOT_DIR = "/mnt/bn/pico-panwangpan-v2/views_release"
    # class_file = np.loadtxt("assets/objaverse_label.txt", dtype='str')
    # datas = []
    # for cls_name in class_file:
    #     print("class_file:", cls_name)
    #     full_path = os.path.join(ROOT_DIR, cls_name )
    #     file_list = os.listdir(full_path)
    #     full_file_list = [ os.path.join(full_path, x ) for x in file_list ]
    #     datas.append(  full_file_list  )
    npz_path = "assets/objaverse_label.txt"
    torch_dataset = ObjaverseDataset(root_dir=ROOT_DIR, \
                                        npz_path = npz_path,\
                                        augment_fn=aug_fun)
    loader = DataLoader(
        dataset=torch_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )

    for step, data in enumerate(loader):
        print(data.keys())