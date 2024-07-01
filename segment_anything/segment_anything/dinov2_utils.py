import torch
import numpy
import numpy as np
from pathlib import Path
from PIL import Image
import cv2 
from torchvision import transforms

def plot_pca( pca_image: numpy.ndarray, save_dir: str, last_components_rgb: bool = False,
             save_resized=False, save_prefix: str = ''):
    """
    finding pca of a set of images.
    :param pil_image: The original PIL image.
    :param pca_image: A numpy tensor containing pca components of the image. HxWxn_components
    :param save_dir: if None than show results.
    :param last_components_rgb: If true save last 3 components as RGB image in addition to each component separately.
    :param save_resized: If true save PCA components resized to original resolution.
    :param save_prefix: optional. prefix to saving
    :return: a list of lists containing an image and its principal components.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    h,w = pca_image.shape
    comp = pca_image
    comp_min = comp.min(axis=(0, 1))
    comp_max = comp.max(axis=(0, 1))
    comp_img = (comp - comp_min) / (comp_max - comp_min)
    comp_img = (comp_img * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(comp_img, cv2.COLORMAP_JET)
    heatmap_color = cv2.resize(heatmap_color, (w*14,h*14))
    cv2.imwrite("headmap.jpg", heatmap_color)        

def load_dinov2_model_from_code():
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_vits14.eval()
    return dinov2_vits14

def load_dinov2_model():
    from dinov2.dinov2.models import build_model_from_cfg
    from easydict import EasyDict as edict
    from dinov2.dinov2.utils.config import get_cfg
    from dinov2.dinov2.utils.utils import load_pretrained_weights
    cfg = get_cfg(f"dinov2/dinov2/configs/eval/vits14_pretrain.yaml")
    model, _, embed_dim = build_model_from_cfg(cfg, only_teacher=False)
    load_pretrained_weights(model, f'weights/dinov2_vits14.pth', checkpoint_key="student")
    model.eval()
    return model

def get_dino_cls_token(model, input_tensor):
    input_tensor = input_tensor.cuda()
    ref_fea = model.get_vit_attn_feat(input_tensor)['cls_']
    return ref_fea


def set_torch_image(
        image: np.ndarray,
        image_format: str = "RGB",
        center_crop = False
    ) :
    # Transform the image to the form expected by the model
    if center_crop:
      prep = transforms.Compose([
          transforms.ToPILImage(),
          transforms.Resize((256,256)),
          transforms.CenterCrop((196,196)),
          transforms.ToTensor(),
          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
      ])
    else:
      prep = transforms.Compose([
          transforms.ToPILImage(),
          transforms.Resize((224,224)),
          transforms.ToTensor(),
          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
      ])
    input_tensor = prep(image)[None, ...]
    input_tensor = input_tensor.cuda()
    return input_tensor


def get_cls_token(model, image):
    prep = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    input_tensor = prep(image)[None, ...]
    input_tensor = input_tensor.cuda()
    out = model(input_tensor, is_training=True )
    cls_token = out['x_norm_clstoken']
    # normalize it for computering similarity score
    # norm_cls_token = torch.nn.functional.normalize(cls_token)
    return norm_cls_token

def get_cls_token_torch_from_code(model, input_tensor):
    input_tensor = input_tensor.cuda()
    with torch.no_grad():
        features_dict = model.forward_features(input_tensor)
    cls_token = out['x_norm_clstoken']
    norm_cls_token = torch.nn.functional.normalize(cls_token)
    return norm_cls_token

def get_cls_token_torch(model, input_tensor):
    input_tensor = input_tensor.cuda()
    out = model(input_tensor, is_training=True )
    cls_token = out['x_norm_clstoken']
    # norm_cls_token = torch.nn.functional.normalize(cls_token)
    return cls_token


if __name__ == "__main__":
    from dinov2.dinov2.models import build_model_from_cfg
    from easydict import EasyDict as edict
    from dinov2.dinov2.utils.config import get_cfg
    from dinov2.dinov2.utils.utils import load_pretrained_weights
    cfg = get_cfg("dinov2/dinov2/configs/eval/vitl14_pretrain.yaml")
    model, teacher, embed_dim = build_model_from_cfg(cfg, only_teacher=False)
    load_pretrained_weights(model, 'assets/dinov2_vitl14_pretrain.pth', checkpoint_key="student")
    model.eval()
    
    pil_image = Image.open("ppw.png").convert('RGB')
    prep = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    input_tensor = prep(pil_image)[None, ...]
    from sklearn.decomposition import PCA
    # dict_keys(['x_norm_clstoken', 'x_norm_patchtokens', 'x_prenorm', 'masks'])
    out = model(input_tensor,is_training=True )
    descriptors = out["x_norm_patchtokens"].detach().numpy()
    pca = PCA(n_components=1).fit(descriptors[0])
    img_pca = pca.transform(descriptors[0])
    plot_pca( img_pca.reshape(( 448//14, 448//14 )), save_dir="./")

