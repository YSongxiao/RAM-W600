import random, os
import numpy as np
import torch
import torch.nn as nn
from albumentations import Compose, HorizontalFlip, RandomScale, VerticalFlip, Rotate, Resize, ShiftScaleRotate, RandomBrightnessContrast, CenterCrop
from albumentations.pytorch import ToTensorV2
import cv2
import torch.nn.functional as F


def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_transform(split, image_size):
    if split == "train":
        transform = Compose([
            Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            RandomScale(scale_limit=(1.0, 1.1), p=0.5),
            CenterCrop(image_size, image_size),
            ShiftScaleRotate(rotate_limit=10),
            RandomBrightnessContrast(p=0.5),
            ToTensorV2()
        ], additional_targets={'mask': 'gt'})
    elif split == "val" or split == "test":
        transform = Compose([
            Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            ToTensorV2()
        ], additional_targets={'mask': 'gt'})
    else:
        raise NotImplementedError(f"{split} is not implemented.")
    return transform


def get_reg_transform(split, image_size):
    if split == "train":
        transform = Compose([
            Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            # RandomScale(scale_limit=(1.0, 1.1), p=0.5),
            # CenterCrop(image_size, image_size),
            ShiftScaleRotate(rotate_limit=10),
            RandomBrightnessContrast(p=0.5),
            ToTensorV2()
        ])
    elif split == "val" or split == "test":
        transform = Compose([
            Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            ToTensorV2()
        ])
    else:
        raise NotImplementedError(f"{split} is not implemented.")
    return transform


class FocalRegressionLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        """
        Focal Loss for regression tasks.

        Args:
            gamma (float): Focusing parameter. Default is 2.0.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super(FocalRegressionLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        error = torch.abs(pred - target)  # element-wise absolute error
        loss = error ** self.gamma        # apply focal weighting

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # shape: same as input


class FocalLossMultiClass(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Multiclass Focal Loss.

        Args:
            gamma (float): focusing parameter
            alpha (Tensor, optional): class-wise weight tensor [num_classes], e.g., tensor([1.0, 2.0, ...])
            reduction (str): 'mean', 'sum', or 'none'
        """
        super(FocalLossMultiClass, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # shape [num_classes] or None
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] logits
            targets: [B] integer class labels
        """
        log_probs = F.log_softmax(inputs, dim=1)        # [B, C]
        probs = torch.exp(log_probs)                    # [B, C]

        targets_onehot = F.one_hot(targets, num_classes=inputs.size(1)).float()  # [B, C]
        pt = (probs * targets_onehot).sum(1)            # [B]
        log_pt = (log_probs * targets_onehot).sum(1)    # [B]

        focal_term = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)     # [B]
            loss = -alpha_t * focal_term * log_pt
        else:
            loss = -focal_term * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CDW_CELoss(nn.Module):
    def __init__(self, alpha=1.0, class_weights=None):
        """
        alpha: float, 控制距离惩罚的程度
        class_weights: Tensor of shape (C,), 每个类别的权重（可选）
        """
        super(CDW_CELoss, self).__init__()
        self.alpha = alpha
        self.class_weights = class_weights  # torch.tensor 类型或 None

    def forward(self, logits, target):
        """
        logits: Tensor of shape (B, C)
        target: Tensor of shape (B,) with class indices
        """
        B, C = logits.shape
        probs = F.softmax(logits, dim=1).clamp(min=1e-8, max=1.0 - 1e-8)  # 防止 log(0)
        target_onehot = F.one_hot(target, num_classes=C).float()

        # 构造 (B, C) 的 ground-truth class index 矩阵
        target_idx = target.view(-1, 1).expand(-1, C)  # shape: (B, C)
        class_idx = torch.arange(C, device=logits.device).view(1, -1).expand(B, -1)  # shape: (B, C)
        distance_weight = (class_idx - target_idx).abs().float() ** self.alpha  # shape: (B, C)

        # mask 真实类，防止其参与 loss
        mask_non_true = 1.0 - target_onehot  # (B, C), 真实类为0，非真实类为1

        # 如果给定 class_weights，则将其广播到 (B, C)
        if self.class_weights is not None:
            class_weights = self.class_weights.to(logits.device).view(1, -1).expand(B, -1)
        else:
            class_weights = torch.ones_like(probs)

        # 计算 loss
        log_loss = -torch.log(1.0 - probs)
        final_loss = log_loss * distance_weight * mask_non_true * class_weights
        loss = final_loss.sum(dim=1).mean()  # 先对每个样本 sum，再对 batch 平均

        return loss