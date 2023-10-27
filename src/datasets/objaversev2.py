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
    read_scannet_gray,
    read_scannet_depth,
    read_scannet_pose,
    read_scannet_intrinsic,
    convert_gray,
)
from loguru import logger
import random

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
        self.root_dir = root_dir
        self.pose_dir = pose_dir if pose_dir is not None else root_dir
        self.mode = mode
        full_path = os.path.join(root_dir, npz_path ) 
        self.data =   np.load(full_path , allow_pickle=True).item() 
        self.values = list(self.data.values())
        self.data_names = self.data.keys()
        logger.info(f"cls: {len(self.data_names)}")
        self.intrinsic = torch.Tensor([[1120,0,256],[0,840,256],[0,0,1]])
        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        return len(self.data_names)

    def _compute_rel_pose(self, pose0, pose1):
        pose0 = np.vstack((pose0, np.array([0,0,0,1]).reshape(1,4)))
        pose1 = np.vstack((pose1, np.array([0,0,0,1]).reshape(1,4)))
        relative_pose =  np.matmul(pose1, inv(pose0))  # (4, 4)
        t = relative_pose[:3,3].reshape(1,3)
        gt_quan = R.from_matrix(relative_pose[:3,:3]).as_quat().reshape(1,4)
        return np.hstack((t,gt_quan))

    def __getitem__(self, idx):
        data_name = self.values[idx]
        item0, item1 = random.sample(list(data_name.values()), 2)
        # read the grayscale image which will be resized to (1, 480, 640)
        img0 = item0["image"]
        img1 = item1["image"]
        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0 = convert_gray(img0,  augment_fn=None)
                                #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1 = convert_gray(img1,  augment_fn=None)
                                #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        pose0 =  item0["pose"]
        pose1 =  item1["pose"]
        depth0 = depth1 = torch.tensor([])
        # read the intrinsic of depthmap
        K_0 = K_1 = self.intrinsic
        # read and compute relative poses
        T_0to1 = torch.tensor(self._compute_rel_pose(pose0, pose1), dtype=torch.float32)
        # T_1to0 = T_0to1.inverse()
        data = {
            'image0': image0,   # (1, h, w)
            'depth0': depth0,   # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,   # (4, 4)
            # 'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'dataset_name': 'objaverse',
            'pair_id': idx,
        }

        return data
    
if __name__ == "__main__":
    
    from torch.utils.data import Dataset, DataLoader
    from src.utils.augment import build_augmentor
    aug_fun = build_augmentor(method=None)

    


    # torch_dataset = ObjaverseDataset(root_dir="assets/", \
    #                                   npz_path = "objaverse_label.txt",\
    #                                   augment_fn=aug_fun)

    # loader = DataLoader(
    #     dataset=torch_dataset,
    #     batch_size=16,
    #     shuffle=True,
    #     num_workers=0,
    # )
    # for step, data in enumerate(loader):
    #     print(data.keys())