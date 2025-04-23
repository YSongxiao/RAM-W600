import argparse
from utils import *
from trainer import RegTrainer, RegTester
import monai
from pathlib import Path
from datasets.carpal import CarpalRegressionDataset_, CarpalTotalRegressionDataset, CarpalClassificationDataset, get_dataloader, get_dataloader_sampler, get_dataloader_sampler_reg
import torch.optim as optim
from datetime import datetime
from typing import Union
import torchvision
import torch.nn as nn
from models.UnetPlusPlus import UnetPlusPlus
from models.swin_unet.swin_unet import get_SwinUnet_Custom
from models.Seg_UKAN.archs import UKAN
from models.TransUNet.transUnet import get_TransUnet_Custom


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--seed',
        type=int,
        default=3407,
        help='Seed',
    )

    parser.add_argument(
        '--mode',
        type=str,
        default="train",
        choices=["train", "test"],
        help='Mode',
    )

    parser.add_argument(
        '--image_size',
        type=int,
        default=224,
        help='The size of the images.',
    )

    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=16,
        help='Batch size for training.',
    )

    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=2,
        help='Batch size for validating.',
    )

    parser.add_argument(
        '--model',
        type=str,
        default="ResNet",
        choices=["DenseNet", "EfficientNet", "ViT", "Swin", "ResNet", "ViT"],
        help='The name of the model.',
    )

    parser.add_argument(
        '--scheduler',
        type=str,
        default="CosineAnnealing",
        choices=["CosineAnnealing", "Plateau"],
        help='The name of the model.',
    )

    parser.add_argument(
        '--num_classes',
        type=int,
        default=4,
        help='Number of output channel.',
    )

    parser.add_argument(
        '--amp',
        type=bool,
        default=True,
        help='Whether use amp.',
    )

    parser.add_argument(
        '--grad_clip',
        type=Union[None, float],
        default=None,
        help='Whether use grad_clip.',
    )

    parser.add_argument(
        '--data_path',
        type=str,
        default="/mnt/data2/datasx/Carpal/ExportedDataset/Regression/V3",
        help='The path to data.',
    )

    parser.add_argument(
        '--pretrain_path',
        type=str,
        default="./pretrained_weights/",
        help='The path of pretrained weights.',
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default="ckpts/regression_densenet_202504152155/",
        help='The pretrained weight of the model.',
    )

    parser.add_argument(
        '--trial_name',
        type=str,
        default="classification",
        help='The name of the trial.',
    )

    parser.add_argument(
        '--max_epoch',
        type=int,
        default=100,
        help='Number of epochs.',
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-7,
        help='Initial lr',
    )

    parser.add_argument(
        '--save_csv',
        action='store_true',
        default=False,
        help='Whether save csv (Test mode only).',
    )
    args = parser.parse_args()
    return args


args = get_args()
# Seed everything
seed_everything(args.seed)


# initial network
if args.model == "DenseNet":
    net = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=1, out_channels=1, init_features=64,
                                          growth_rate=32, block_config=(6, 12, 24, 16), pretrained=False, progress=False)

elif args.model == "EfficientNet":
    net = monai.networks.nets.EfficientNetBN("efficientnet-b0", pretrained=False, progress=False, spatial_dims=2,
                                             in_channels=1, num_classes=args.num_classes,
                                             norm=('batch', {'eps': 0.001, 'momentum': 0.01}), adv_prop=False)

elif args.model == "ResNet":
    net = monai.networks.nets.ResNet(block="basic", layers=[2, 2, 2, 2], block_inplanes=[16, 32, 64, 128], spatial_dims=2, n_input_channels=1, conv1_t_size=7,
                                     conv1_t_stride=1, no_max_pool=False, shortcut_type='B', widen_factor=1.0,
                                     num_classes=args.num_classes, feed_forward=True, bias_downsample=True,
                                     act=('relu', {'inplace': True}), norm='batch')

elif args.model == "ViT":
    net = monai.networks.nets.ViT(in_channels=1, img_size=args.image_size, patch_size=8, hidden_size=64, mlp_dim=128, num_layers=4,
                            num_heads=4, proj_type='conv', pos_embed_type='learnable', classification=True,
                            num_classes=args.num_classes, dropout_rate=0.2, spatial_dims=2, post_activation='Tanh', qkv_bias=False,
                            save_attn=False)

print(f"Model size: {sum(p.numel() for p in net.parameters())}")

# pretrain_path = os.path.join(args.pretrain_path, args.checkpoint)
# if os.path.exists(pretrain_path):
#     net.load_state_dict(torch.load(pretrain_path, weights_only=True))
#     print("pretrained weight: {} loaded".format(pretrain_path))
# else:
#     print("no pretrained weight for training.")

if args.mode == "train":
    # 获取当前时间
    now = datetime.now()
    time_str = now.strftime('%Y%m%d%H%M')
    ckpt_path = Path("./ckpts") / (args.trial_name + "_" + args.model.lower() + "_" + time_str)
    if not ckpt_path.exists():
        ckpt_path.mkdir(parents=True)
    args.model_save_path = str(ckpt_path)
    transform_tr = get_reg_transform(split="train", image_size=args.image_size)
    transform_val = get_reg_transform(split="val", image_size=args.image_size)
    train_dataset = CarpalClassificationDataset(data_root=Path(args.data_path) / "train",
                                            annotation_path=Path(args.data_path) / "JointBE_SvdH_GT.xlsx",
                                            transform=transform_tr)
    # import matplotlib.pyplot as plt
    # plt.hist(train_dataset.total_gts)
    # plt.show()
    train_loader = get_dataloader_sampler(train_dataset, batch_size=args.train_batch_size, shuffle=False)
    val_dataset = CarpalClassificationDataset(data_root=Path(args.data_path) / "val",
                                          annotation_path=Path(args.data_path) / "JointBE_SvdH_GT.xlsx",
                                          transform=transform_val)
    val_loader = get_dataloader_sampler(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)
    # optimizer = optim.AdamW(net.parameters(), lr=args.lr, amsgrad=True)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr)
    # criterion = FocalRegressionLoss(gamma=2.0, reduction='mean')
    # criterion = nn.MSELoss()
    criterion = CDW_CELoss(class_weights=torch.tensor([2.0, 0.8, 0.8, 1.0], device="cuda:0"))
    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.5, 1.0, 1.0, 1.2], device="cuda:0"))
    # criterion = nn.CrossEntropyLoss()
    # alpha = torch.tensor([3.0, 1.0, 1.0, 1.0, 1.0]).to("cuda:0")  # 类别权重（可选）
    # criterion = FocalLossMultiClass(gamma=2.0, alpha=alpha)
    trainer = RegTrainer(args, net, train_loader, val_loader, criterion, optimizer, device="cuda:0")
    trainer.fit(args)

elif args.mode == "test":
    transform_test = get_reg_transform(split="test", image_size=args.image_size)
    test_dataset = CarpalClassificationDataset(data_root=Path(args.data_path) / "test",
                                           annotation_path=Path(args.data_path) / "JointBE_SvdH_GT.xlsx",
                                           transform=transform_test)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
    if not (Path(args.checkpoint) / "model_best.pth").exists():
        raise KeyError("Test mode is set but checkpoint does not exist.")
    tester = RegTester(args, net, test_loader, device="cuda:0")
    tester.test()
