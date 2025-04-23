from models.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from models.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
import numpy as np


def get_TransUnet(img_size, n_classes):
    vit_name = 'R50-ViT-B_16'
    config_vit = CONFIGS_ViT_seg[vit_name]
    vit_patches_size = 16
    config_vit.n_skip = 3
    config_vit.n_classes = n_classes
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    transunet = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes)
    transunet.load_from(weights=np.load(config_vit.pretrained_path))
    return transunet


def get_TransUnet_Custom(img_size, n_classes):
    vit_name = 'R50-ViT-B_16'
    config_vit = CONFIGS_ViT_seg[vit_name]
    vit_patches_size = 16
    config_vit.n_skip = 3
    config_vit.n_classes = n_classes
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    transunet = ViT_seg(config_vit, img_size=img_size, num_classes=config_vit.n_classes)
    return transunet
