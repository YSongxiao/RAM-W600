from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch


class CarpalDataset(Dataset):
    def __init__(self, data_root, annotation_path, transform=None):
        coco = COCO(annotation_file=annotation_path)
        self.data_root = data_root
        self.transform = transform
        img_ids = coco.getImgIds()

        # 存储结果，每张图一个 mask 列表
        self.filenames = []
        self.masks = []

        for img_id in img_ids:
            img_info = coco.loadImgs([img_id])[0]
            height, width = img_info['height'], img_info['width']
            filename = img_info['file_name']
            # filename = img_info['file_name'].split("_bmp")[0].replace("-", "!") + ".bmp"

            # 获取 annotation
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)

            # 按 category_id 排序
            anns_sorted = sorted(anns, key=lambda x: x['category_id'])

            # 对每个 annotation 生成 mask
            image_masks = []
            for num, ann in enumerate(anns_sorted):
                seg = ann['segmentation']

                # Polygon 转 RLE
                if isinstance(seg, list):
                    rles = maskUtils.frPyObjects(seg, height, width)
                    rle = maskUtils.merge(rles)
                else:
                    rle = seg

                mask = maskUtils.decode(rle)  # H x W numpy 数组（0/1）
                image_masks.append(mask)
            self.filenames.append(filename)
            tmp_masks = np.stack(image_masks, axis=0)
            self.masks.append(tmp_masks)

    def __getitem__(self, idx):
        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        path = Path(self.data_root) / self.filenames[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.filenames[idx][-4] == "L":
            img = cv2.flip(img, 1)  # Horizontal Flip
        mask = self.masks[idx].transpose(1, 2, 0)
        img = normalization(img[..., np.newaxis]).astype(np.float32)
        dic = self.transform(image=img, mask=mask)
        img = dic['image'].float()
        mask = dic['mask'].permute(2, 0, 1)
        data = {
            "fname": self.filenames[idx],
            "img": img,
            "mask": mask
        }
        return data

    def __len__(self):
        return len(self.filenames)


class CarpalNpyDataset(Dataset):
    def __init__(self, data_root, annotation_path, transform=None):
        self.data_root = data_root
        self.annotation_path = annotation_path
        self.transform = transform

        # 存储结果，每张图一个 mask 列表
        self.filenames = [str(fname.stem) for fname in Path(self.annotation_path).rglob("*.npy")]
        self.masks = []

        for filename in self.filenames:
            tmp_mask = np.load(Path(self.annotation_path) / (filename + ".npy"))
            self.masks.append(tmp_mask)

    def __getitem__(self, idx):
        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        path = Path(self.data_root) / (self.filenames[idx] + ".bmp")
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = self.masks[idx]
        if self.filenames[idx][-1] == "L":
            img = cv2.flip(img, 1)  # Horizontal Flip
            mask = self.masks[idx][:, :, ::-1]
        mask = mask.transpose(1, 2, 0)
        img = normalization(img[..., np.newaxis]).astype(np.float32)
        dic = self.transform(image=img, mask=mask)
        img = dic['image'].float()
        mask = dic['mask'].permute(2, 0, 1)
        data = {
            "fname": self.filenames[idx],
            "img": img,
            "gt": mask
        }
        return data

    def __len__(self):
        return len(self.filenames)


class CarpalRegressionDataset(Dataset):
    def __init__(self, data_root, annotation_path, norm=True, transform=None):
        self.data_root = data_root
        self.annotation = pd.read_excel(annotation_path)
        self.norm = norm
        self.transform = transform

        # 存储结果，每张图一个 mask 列表
        self.filedirs = [p for p in self.data_root.iterdir() if p.is_dir()]
        self.img_links = {"Metacarpal1st": [], "Trapzium": [], "Scaphoid": [], "Lunate": [], "DistalRadius": [], "DistalUlna": []}
        self.gts = {"Metacarpal1st": [], "Trapzium": [], "Scaphoid": [], "Lunate": [], "DistalRadius": [], "DistalUlna": []}

        for dir_path in self.filedirs:
            filename = dir_path.name
            matched_row = self.annotation[self.annotation["Image Link"] == filename]
            for key in matched_row.keys()[1:]:
                self.gts[key].append(matched_row[key].values[0])
                self.img_links[key].append(dir_path / (key + ".bmp"))

    def __getitem__(self, idx):
        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        imgs = []
        gts = []
        for key in self.img_links.keys():
            path = Path(self.data_root) / self.img_links[key][idx]
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            img = normalization(img[..., np.newaxis]).astype(np.float32)
            if self.norm:
                gts.append(self.gts[key][idx] / 5.0)
            else:
                gts.append(self.gts[key][idx])
            if self.img_links[key][idx].stem[-1] == "L":
                img = cv2.flip(img, 1)  # Horizontal Flip
            if self.transform:
                img = self.transform(image=img)["image"]
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0).squeeze()
        gts = torch.FloatTensor(np.array(gts))

        data = {
            "fname": str(self.filedirs[idx]),
            "key": list(self.img_links.keys()),
            "img": imgs,
            "gt": gts,
        }
        return data

    def __len__(self):
        return len(self.filedirs)


class CarpalTotalRegressionDataset(Dataset):
    def __init__(self, data_root, annotation_path, norm=True, transform=None):
        self.data_root = data_root
        self.annotation = pd.read_excel(annotation_path)
        self.norm = norm
        self.transform = transform

        # 存储结果，每张图一个 mask 列表
        self.filedirs = [p for p in self.data_root.iterdir() if p.is_dir()]
        self.img_links = {"Metacarpal1st": [], "Trapzium": [], "Scaphoid": [], "Lunate": [], "DistalRadius": [], "DistalUlna": []}
        self.gts = {"Metacarpal1st": [], "Trapzium": [], "Scaphoid": [], "Lunate": [], "DistalRadius": [], "DistalUlna": []}
        self.total_gts = []

        for dir_path in self.filedirs:
            filename = dir_path.name
            matched_row = self.annotation[self.annotation["Image Link"] == filename+".bmp"]
            tmp_total_gt = 0.0
            for key in matched_row.keys()[1:]:
                self.gts[key].append(matched_row[key].values[0])
                tmp_total_gt += matched_row[key].values[0]
                self.img_links[key].append(dir_path / (key + ".bmp"))
            self.total_gts.append(tmp_total_gt)

    def __getitem__(self, idx):
        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        imgs = []
        for key in self.img_links.keys():
            path = Path(self.data_root) / self.img_links[key][idx]
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            img = normalization(img[..., np.newaxis]).astype(np.float32)
            if self.img_links[key][idx].stem[-1] == "L":
                img = cv2.flip(img, 1)  # Horizontal Flip
            if self.transform:
                img = self.transform(image=img)["image"]
            imgs.append(img)
        imgs = torch.stack(imgs, dim=0).squeeze()

        data = {
            "fname": str(self.filedirs[idx]),
            "key": list(self.img_links.keys()),
            "img": imgs,
            "gt": torch.tensor(self.total_gts[idx], dtype=torch.float32),
        }
        return data

    def __len__(self):
        return len(self.filedirs)


class CarpalRegressionDataset_(Dataset):
    def __init__(self, data_root, annotation_path, norm=True, transform=None):
        self.data_root = data_root
        self.annotation = pd.read_excel(annotation_path)
        self.norm = norm
        self.transform = transform

        # 存储结果，每张图一个 mask 列表
        self.filedirs = [p for p in self.data_root.iterdir() if p.is_dir()]
        self.joint_fnames = []
        self.joint_names = []
        self.img_links = ["Metacarpal1st", "Trapzium", "Scaphoid", "Lunate", "DistalRadius", "DistalUlna"]
        self.gts = []

        for dir_path in self.filedirs:
            filename = dir_path.name
            matched_row = self.annotation[self.annotation["Image Link"] == filename + ".bmp"]
            for key in matched_row.keys()[1:]:
                self.gts.append(matched_row[key].values[0])
                self.joint_names.append(key)
                self.joint_fnames.append(dir_path / (key + ".bmp"))

    def __getitem__(self, idx):
        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        img = cv2.imread(str(self.joint_fnames[idx]), cv2.IMREAD_GRAYSCALE)
        gt = self.gts[idx]
        img = normalization(img[..., np.newaxis]).astype(np.float32)
        if self.norm:
            gt /= 5.0

        if self.joint_fnames[idx].stem[-1] == "L":
            img = cv2.flip(img, 1)  # Horizontal Flip
        if self.transform:
            img = self.transform(image=img)["image"]

        data = {
            "fname": str(self.joint_fnames[idx]),
            "key": self.joint_names[idx],
            "img": img,
            "gt": torch.tensor(gt),
        }
        return data

    def __len__(self):
        return len(self.gts)


class CarpalClassificationDataset(Dataset):
    def __init__(self, data_root, annotation_path, transform=None):
        self.data_root = data_root
        self.annotation = pd.read_excel(annotation_path)
        self.transform = transform

        # 存储结果，每张图一个 mask 列表
        self.filedirs = [p for p in self.data_root.iterdir() if p.is_dir()]
        self.joint_fnames = []
        self.joint_names = []
        self.img_links = ["Metacarpal1st", "Trapzium", "Scaphoid", "Lunate", "DistalRadius", "DistalUlna"]
        self.gts = []

        for dir_path in self.filedirs:
            filename = dir_path.name
            matched_row = self.annotation[self.annotation["Image Link"] == filename + ".bmp"]
            key = matched_row.keys()[2]
            self.gts.append(matched_row[key].values[0] if matched_row[key].values[0] != 5 else 3)
            self.joint_names.append(key)
            self.joint_fnames.append(dir_path / (key + ".bmp"))
            # for key in matched_row.keys()[1:]:
            #     self.gts.append(matched_row[key].values[0] if matched_row[key].values[0] != 5 else 3)
            #     self.joint_names.append(key)
            #     self.joint_fnames.append(dir_path / (key + ".bmp"))

    def __getitem__(self, idx):
        # 归一化
        def normalization(data):
            range = np.max(data) - np.min(data)
            return (data - np.min(data)) / range

        img = cv2.imread(str(self.joint_fnames[idx]), cv2.IMREAD_GRAYSCALE)
        gt = self.gts[idx]
        img = normalization(img[..., np.newaxis]).astype(np.float32)

        if self.joint_fnames[idx].stem[-1] == "L":
            img = cv2.flip(img, 1)  # Horizontal Flip
        if self.transform:
            img = self.transform(image=img)["image"]

        data = {
            "fname": str(self.joint_fnames[idx]),
            "key": self.joint_names[idx],
            "img": img,
            "gt": torch.tensor(gt, dtype=torch.int64),
        }
        return data

    def __len__(self):
        return len(self.gts)


def get_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def get_dataloader_sampler(dataset, batch_size, shuffle=False):
    unique_labels, counts = np.unique(dataset.gts, return_counts=True)
    label_freq = dict(zip(unique_labels, counts))

    # Step 2: 给每个样本分配采样权重（用频率的倒数）
    weights = np.array([1.0 / label_freq[label] for label in dataset.gts])

    # Step 3: 创建 WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),  # 每轮epoch抽多少样本，常设为 len(weights)
        replacement=True  # 有放回采样
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle)


def get_dataloader_sampler_reg(dataset, batch_size, shuffle=False):
    # Step 1: 获取所有标签
    labels = np.array(dataset.total_gts)
    # Step 2: 分桶 共4个区间：[0~1), [1~2.5), [2.5~4), [4~5]
    custom_bins = [0.0, 1.0, 3.0, 10.0]
    digitized = np.digitize(labels, custom_bins)  # 每个样本的桶编号

    # Step 3: 计算每个bin的权重 = 1 / 样本数
    bin_counts = np.bincount(digitized)
    bin_weights = 1. / (bin_counts + 1e-6)  # 避免除0
    sample_weights = bin_weights[digitized]

    # Step 4: 构建WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle)


if __name__ == '__main__':
    carpal_dataset = CarpalNpyDataset(data_root="/mnt/data2/datasx/Carpal/ExportedDataset/V5/", annotation_path="/mnt/data2/datasx/Carpal/ExportedDataset/V5_npy/test/")
    carpal_dataset.__getitem__(0)
