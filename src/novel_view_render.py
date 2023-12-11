import os
from select import select
import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from gnt_model_api import model as gnt_model,args
from gnt.data_loaders import dataset_dict
from gnt.projection import Projector
from gnt.sample_ray import RaySamplerSingleImage
from gnt.render_image import render_single_image, render_sample_rgb
from gnt_utils import img_HWC2CHW, colorize, img2psnr, lpips, ssim
np.random.seed(0)
from numpy.linalg import inv
import imageio


LINEMOD_OBJ_NAME = ("ape", "benchvise" , "cam", "can", \
                    "cat", "driller", "duck", "eggbox", \
                    "glue", "holepuncher", "iron", "lamp", \
                    "phone")

if __name__ == "__main__":
    device = torch.device("cuda:0")
    gnt_model.switch_to_eval()
    projector = Projector(device=device)
    file_list = ["0804-lm4-others","0802-lm2-others","0801-lm1-others","0815-lm15-others",
                 "0814-lm14-others","0813-lm13-others","0812-lm12-others",
                 "0810-lm10-others", "0809-lm9-others", "0808-lm8-others", "0806-lm6-others",
                 "0805-lm5-others"]
    for file in file_list:
        name =   file.split("-")[1]
        ROOTDIR = f"data/LM_dataset/{file}/{name}-3/color"
        base_name = os.path.basename(ROOTDIR)
        src_rgbs = []
        src_cameras = []
        d_range = []
        select_id = [102,101,100, 103,104,105,106,107,108,109,110]
        image_list = select_id
        for i in image_list:
            src_rgb_file = os.path.join(ROOTDIR, f"{i}.png")
            src_rgb = imageio.imread(src_rgb_file)
            h,w,_ = src_rgb.shape
            src_rgb = src_rgb.astype(np.float32)/255.
            pose_file = src_rgb_file.replace(base_name,"poses_ba").replace(".png",".txt")
            pose = np.loadtxt(pose_file)
            pose = np.vstack((pose, np.array([[0,0,0,1]])))

            d_range.append(pose[2][3])

            K_file = src_rgb_file.replace(base_name,"intrin_ba").replace(".png",".txt")
            if "full" not in base_name:
                intrinsic = np.loadtxt(K_file)
            else:
                intrinsic = np.array([[572.4114, 0., 325.2611],
                                    [0., 573.57043, 242.04899],
                                    [0., 0., 1.]], dtype=np.float32)

            intrinsic = np.vstack( (intrinsic, np.array([0,0,0]).T ))
            intrinsic = np.hstack( (intrinsic, np.array([ [0],[0],[0],[1] ] ) ) )


            src_rgbs.append(src_rgb)
            src_camera = np.concatenate(
                ([h,w], intrinsic.flatten(), pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)


        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        d_range = np.array(d_range)
        #----------------------------------  load target views ----------------------------------
        for idx in select_id:
            rgb_file = os.path.join(ROOTDIR, f"{idx}.png")
            rgb = imageio.imread(rgb_file)
            h,w,c = rgb.shape
            rgb = rgb.astype(np.float32) / 255.0
            pose_file = rgb_file.replace(base_name,"poses_ba").replace(".png",".txt")
            pose = np.loadtxt(pose_file)
            pose = np.vstack((pose, np.array([[0,0,0,1]])))

            K_file = rgb_file.replace(base_name,"intrin_ba").replace(".png",".txt")
            if "full" not in base_name:
                intrinsic = np.loadtxt(K_file)
            else:
                intrinsic = np.array([[572.4114, 0., 325.2611],
                                    [0., 573.57043, 242.04899],
                                    [0., 0., 1.]], dtype=np.float32)

            intrinsic = np.vstack( (intrinsic, np.array([0,0,0]).T ))
            intrinsic = np.hstack( (intrinsic, np.array([ [0],[0],[0],[1] ] ) ) )

            cameras = []
            camera = np.concatenate(
                    ([h,w], intrinsic.flatten(), pose.flatten())).astype(np.float32)
            cameras.append(camera)
            cameras = np.stack(cameras, axis=0)
            #----------------------------------  load target views ----------------------------------
            data = {
                "rgb": torch.from_numpy(rgb[..., :3]).cuda(),
                "camera": torch.from_numpy( cameras  ).cuda(),
                "rgb_path": rgb_file,
                "depth_range": torch.tensor([[d_range.min()*0.6 , d_range.max()*1.5]] ).cuda(),
                "src_rgbs": torch.from_numpy(src_rgbs[..., :3]).cuda(),
                "src_cameras": torch.from_numpy(src_cameras).cuda(),
            }

            args.render_stride = 1
            args.N_importance = 64
            tmp_ray_sampler = RaySamplerSingleImage(data, device, render_stride=args.render_stride)
            H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
            gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
            target_image = gt_img.detach().cpu().numpy()

            featmaps = gnt_model.feature_net(data["src_rgbs"].squeeze(0).permute(0, 3, 1, 2))

            ray_batch = tmp_ray_sampler.get_all()
            with torch.no_grad():
                ret = render_single_image(
                    ray_sampler=tmp_ray_sampler,
                    ray_batch=ray_batch,
                    model=gnt_model,
                    projector=projector,
                    chunk_size=args.chunk_size,
                    N_samples=args.N_samples,
                    inv_uniform=args.inv_uniform,
                    det=True,
                    N_importance=args.N_importance,
                    white_bkgd=args.white_bkgd,
                    render_stride=args.render_stride,
                    featmaps=featmaps,
                    ret_alpha=False,
                    single_net=args.single_net,
                    update_pose = None
                )


            # xxrgb = ret["outputs_coarse"]["rgb"]
            rgb_coarse = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu())
            rgb_coarse = rgb_coarse.permute(1, 2, 0).detach().cpu().numpy()
            imageio.imwrite("{}_gt_pose_{}.jpg".format(file,idx), rgb_coarse)
            imageio.imwrite("{}_gt_{}.jpg".format(file,idx), rgb)