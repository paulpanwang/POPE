from loguru import logger
import torch
import torch.nn as nn
from src.loftr import LoFTR, default_cfg
from einops import rearrange, repeat
from src.loftr.loftr_module import LocalFeatureTransformer as CrossAtten
import torch

from src.utils.dataset import (
    read_scannet_rgb,
    read_scannet_gray,
    read_scannet_depth,
    read_scannet_pose,
    read_scannet_intrinsic
)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, num_classes=7, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    

class DINOv2Poser(nn.Module):
    def __init__(self, default_cfg, depth=4, heads=8, dim_head = 64, dropout = 0., emb_dropout = 0., mlp_dim=512):
        super().__init__()
        from dinov2.dinov2.models import build_model_from_cfg
        from easydict import EasyDict as edict
        from dinov2.dinov2.utils.config import get_cfg
        from dinov2.dinov2.utils.utils import load_pretrained_weights
        self.cfg = get_cfg("dinov2/dinov2/configs/eval/vitl14_pretrain.yaml")
        model, _, embed_dim = build_model_from_cfg(self.cfg, only_teacher=False)
        load_pretrained_weights(model, 'assets/dinov2_vitl14_pretrain.pth', checkpoint_key="teacher")
        self.dino_model = model
        token_dim = 1024
        self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
        self.transformer = Transformer(token_dim+1, depth, heads, dim_head, mlp_dim, dropout)
        self.cross_attentionAll = CrossAtten(default_cfg['coarse'], token_dim, ['cross']*1 )
        self.MLP = FeedForward( dim = 4096, hidden_dim = 512, num_classes = 1, dropout = 0.1)
        self.regression_head = FeedForward( dim = 1024, hidden_dim = 128, num_classes = 7, dropout = 0.1)
    
    def forward(self, data, only_att_fea=True , use_avg = False):
        if data["image0"].dim()==5:
            _ ,n,c,h,w = data["image0"].shape
            data["image0"] = data["image0"].view(-1,c,h,w)
            
        if data["image1"].dim()==5:
            _ ,n,c,h,w = data["image1"].shape
            data["image1"] = data["image1"].view(-1,c,h,w)

        batch_size = data["image0"].shape[0]
        outA = self.dino_model.forward(data["image0"],is_training=True )
        outB = self.dino_model.forward(data["image1"],is_training=True )
        feaA = outA["x_norm_patchtokens"]
        feaB = outB["x_norm_patchtokens"]
        # 1024*1024
        cls_tokenA = outA["x_norm_clstoken"].view(batch_size,1,1024).expand(batch_size,1024,1024)
        cls_tokenB = outB["x_norm_clstoken"].view(batch_size,1,1024).expand(batch_size,1024,1024)
        # print(clsA.shape, clsB.shape)
        
        #  torch.Size([2, 2048, 1024])
        all_tokensA = torch.cat([feaA, cls_tokenA],dim=1)
        all_tokensB = torch.cat([feaB, cls_tokenB],dim=1)
        
        # torch.Size([2, 2048, 1024])
        cross_A, cross_B = self.cross_attentionAll(all_tokensA, all_tokensB) 
        ensemble = torch.cat((cross_A, cross_B), dim=1)
        # torch.Size([2, 4096, 1024])
        ensemble = ensemble.permute(0,2,1)
        # torch.Size([2, 1024, 4096])
        ensemble = self.MLP(ensemble)
        # torch.Size([2, 1024, 1])
        ensemble = ensemble.permute(0,2,1)
        output = self.regression_head(ensemble)
        # logger.info(f"DINO feature:  A:{output.shape}" )
        return output
        


if __name__ == "__main__":
    curr_path1 = "assets/000.png"
    curr_path2 = "assets/001.png"
    model = DINOv2Poser(default_cfg)
    img0 =  read_scannet_rgb(curr_path1,)
    img1 =  read_scannet_rgb(curr_path2,)
    print(img0.shape, img1.shape)

    model = model.eval()
    # batch = {'image0': img0, 'image1': img1}
    batch = {'image0': torch.ones((2,3,448,448)), 'image1': torch.ones((2,3,448,448))}

    # # batch = {'image0': img0, 'image1': img1} 
    with torch.no_grad():
        pose = model(batch)   
    logger.info(f"pose shape:{pose.shape} [tx, ty , tz , qx, qy, qz, qw]" )

    # import torch
    # dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    # input_tensor = torch.zeros((1,3,224,224))
    # y = model.forward(input_tensor, is_training=False)
    # hid_states = y["x_norm_patchtokens"]
    # print(hid_states.shape)
