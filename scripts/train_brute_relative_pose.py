import os
import cv2
import torch
import torch.optim as optim

from torch.utils.data import DataLoader
from src.datasets.objaverse import ObjaverseDataset
from pose_loss import posenet_loss
from datetime import datetime
import numpy as np
from collections import OrderedDict
import random

# import model
from src.utils.comm import to_cuda
from src.logger import Logger
from src.loftr import LoFTR, default_cfg
from brute_force_model import BruteForce

# DDP training
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

def setup_ddp(gpu, args):
    dist.init_process_group(                                   
    	backend='nccl',                                 
   		init_method='env://',     
    	world_size=args.world_size,                              
    	rank=gpu)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

def train(gpu, args):
    """ Test to make sure project transform correctly maps points """

    # coordinate multiple GPUs
    if not args.no_ddp:
        setup_ddp(gpu, args)
    rng = np.random.default_rng(12345)
    random.seed(0)

    thiscuda = 'cuda:%d' % gpu
    map_location = {'cuda:%d' % 0: thiscuda}
    args.map_location = map_location
    if args.no_ddp: 
        args.map_location = ''
        thiscuda = 'cuda:0'

    model = BruteForce(default_cfg)
    model.to(thiscuda)
    model.train()
        
    # unused layers
    for param in model.matcher.parameters():
        param.requires_grad = False

    if not args.no_ddp:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    pct_warmup = args.warmup / args.steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
        args.lr, args.steps, pct_start=pct_warmup, div_factor=25, cycle_momentum=False)

    if args.ckpt is not None:
        print('loading separate checkpoint')

        if args.no_ddp:
            existing_ckpt = torch.load(args.ckpt)
        else:
            existing_ckpt = torch.load(args.ckpt, map_location=map_location)

        model.load_state_dict(existing_ckpt['model'], strict=False)
        optimizer.load_state_dict(existing_ckpt['optimizer'])

        del existing_ckpt
    elif args.existing_ckpt is not None:
        if args.no_ddp:
            existing_ckpt = torch.load(args.existing_ckpt)
            state_dict = OrderedDict([
                (k.replace("module.", ""), v) for (k, v) in existing_ckpt['model'].items()])
            model.load_state_dict(state_dict)
            del state_dict
            optimizer.load_state_dict(existing_ckpt['optimizer'])
            if 'scheduler' in existing_ckpt:
                scheduler.load_state_dict(existing_ckpt['scheduler'])
        else:
            existing_ckpt = torch.load(args.existing_ckpt, map_location=map_location)
            model.load_state_dict(existing_ckpt['model'])
            optimizer.load_state_dict(existing_ckpt['optimizer'])
            if 'scheduler' in existing_ckpt:
                scheduler.load_state_dict(existing_ckpt['scheduler'])
        print('loading existing checkpoint')
        del existing_ckpt

    logger = Logger(args.name, scheduler)
    should_keep_training = True
    subepoch = 0
    train_steps = 0
    epoch_count = 0
    while should_keep_training:
        is_training = True
        train_val = 'train'
        if subepoch == 10:
            is_training = False
            train_val = 'val'
        
        from torch.utils.data import Dataset, DataLoader
        from src.utils.augment import build_augmentor
        aug_fun = build_augmentor(method=None)
        torch_dataset = ObjaverseDataset(root_dir="/mnt/bn/pico-panwangpan-v2/views_release", \
                                        npz_path = "assets/objaverse_label_clean.txt",\
                                        augment_fn=aug_fun)
        
        if not args.no_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                db, shuffle=is_training, num_replicas=args.world_size, rank=gpu)
            train_loader = DataLoader(torch_dataset, batch_size=args.batch, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
        else:
            train_loader = DataLoader(torch_dataset, batch_size=args.batch, num_workers=4,shuffle=False)
        
        model.to("cuda").train()

        if not is_training:
            model.eval()

        with tqdm(train_loader, unit="batch") as tepoch:
            for i_batch, item in enumerate(tepoch):
                optimizer.zero_grad()
                item = to_cuda(item)

                pose_gt = item['T_0to1']
                                   
                metrics = {}
                poses_est = model(item)
                # geo_loss_tr, geo_loss_rot, geo_metrics = geodesic_loss(Ps_out, poses_est, train_val=train_val)
                # loss = args.w_tr * geo_loss_tr + args.w_rot * geo_loss_rot
                loss = posenet_loss(poses_est, pose_gt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step() 
                train_steps += 1      
                if gpu == 0 or args.no_ddp:
                    logger.push(metrics)
                    if i_batch % 20 == 0:
                        torch.set_printoptions(sci_mode=False, linewidth=150)
                        for local_index in range(len(poses_est)):
                            # print('pred number:', local_index)
                            print('\n [E]estimated pose')
                            print(poses_est[local_index,0,:7].cpu().detach())
                            print(' [G]ground truth pose')
                            print(pose_gt[local_index,0,:7].cpu().detach())
                            print(f'loss:{loss.item()}')
                            
                    if (i_batch + 10) % 20 == 0:
                        print('\n metrics:', metrics, '\n')
                    if i_batch % 100 == 0:
                        print('epoch', str(epoch_count))
                        print('subepoch: ', str(subepoch))
                        print('using', train_val, 'set')

                if train_steps % 10000 == 0 and (gpu == 0 or args.no_ddp) and is_training:
                    PATH = 'output/%s/checkpoints/%06d.pth' % (args.name, train_steps)
                    checkpoint = {"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()}
                    torch.save(checkpoint, PATH)

                if train_steps >= args.steps:
                    PATH = 'output/%s/checkpoints/%06d.pth' % (args.name, train_steps)
                    checkpoint = {"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()}
                    torch.save(checkpoint, PATH)
                    should_keep_training = False
                    break
       
        subepoch = (subepoch + 1)
        if subepoch == 11 or (subepoch == 10 and (args.dataset == "interiornet" or args.dataset == "streetlearn")):
            # we follow Cai et al and don't use a val set for interiornet and streetlearn
            subepoch = 0
            epoch_count += 1

    print("finished training!")
    dist.destroy_process_group()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument('--w_tr', type=float, default=10.0)
    parser.add_argument('--w_rot', type=float, default=10.0)
    parser.add_argument('--warmup', type=int, default=10000)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--steps', type=int, default=12000000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--clip', type=float, default=2.5)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_ddp', action="store_true", default=True)
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--ckpt', default=None,  help='checkpoint to restore')
    parser.add_argument('--name', default='loftr_brute', help='name your experiment')
    # data

    parser.add_argument("--image_size", default=[384,512])
    parser.add_argument("--exp")
    parser.add_argument('--use_mini_dataset', action='store_true')
    parser.add_argument('--streetlearn_interiornet_type', default='', choices=('',"T"))
    parser.add_argument('--dataset', default='objverse', choices=("matterport", "interiornet", 'streetlearn'))

    # model
    parser.add_argument('--no_pos_encoding', action='store_true')
    parser.add_argument('--noess', action='store_true')
    parser.add_argument('--cross_features', action='store_true')
    parser.add_argument('--use_single_softmax', action='store_true')  
    parser.add_argument('--l1_pos_encoding', action='store_true')


    args = parser.parse_args()
    
    print(args)

    PATHS = ['output/%s/checkpoints' % (args.name), 'output/%s/runs' % (args.name), 'output/%s/train_output/images' % (args.name)]
    args.existing_ckpt = None

    for PATH in PATHS:
        try:
            os.makedirs(PATH)
        except:
            if 'checkpoints' in PATH:
                ckpts = os.listdir(PATH)

                if len(ckpts) > 0:
                    if 'most_recent_ckpt.pth' in ckpts:
                        existing_ckpt = 'most_recent_ckpt.pth'
                    else:
                        ckpts = [int(i[:-4]) for i in ckpts]
                        ckpts.sort()
                        existing_ckpt = str(ckpts[-1]).zfill(6) +'.pth'
                
                    args.existing_ckpt = os.path.join(PATH, existing_ckpt)
                    print('existing',args.existing_ckpt)
            pass

    
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d_%H-%M")

    with open('output/%s/args_%s.txt' % (args.name, dt_string), 'w') as f:
        for k, v in vars(args).items():
            f.write(str(k) + '  '+ str(v) + '\n')
        
    if args.no_ddp:
        train(args.gpus, args)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        args.world_size = args.gpus
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    
    
    
    
    
    