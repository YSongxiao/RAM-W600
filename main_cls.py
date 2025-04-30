import argparse
from utils import *
from trainer import RegTrainer, RegTester
import monai
from pathlib import Path
from datasets.carpal import CarpalClassificationDataset, get_dataloader
import torch.optim as optim
from datetime import datetime
from typing import Union
import torch.nn as nn
import timm
from models.MedMamba import VSSM as medmamba
from conv_kan_baseline import SimpleConvKAN


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
        choices=["ResNet", "KAN", "MobileNet", "EfficientFormer", "MedMamba", "MobileViT", "LeViT"],
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
        default=2,
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
        default="/path/to/data",
        help='The path to data.',
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default="",
        help='The pretrained weight of the model.',
    )

    parser.add_argument(
        '--trial_name',
        type=str,
        default="Function_Test",
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
        default=1e-6,
        help='Initial lr',
    )

    parser.add_argument(
        '--save_csv',
        action='store_true',
        default=True,
        help='Whether save csv (Test mode only).',
    )
    args = parser.parse_args()
    return args


args = get_args()
# Seed everything
seed_everything(args.seed)


# initial network
if args.model == "EfficientNet":
    net = monai.networks.nets.EfficientNetBN("efficientnet-b0", pretrained=False, progress=False, spatial_dims=2,
                                             in_channels=1, num_classes=args.num_classes,
                                             norm=('batch', {'eps': 0.001, 'momentum': 0.01}), adv_prop=False)

elif args.model == "ResNet":
    net = monai.networks.nets.ResNet(block="basic", layers=[2, 2, 2, 2], block_inplanes=[16, 32, 64, 128], spatial_dims=2, n_input_channels=1, conv1_t_size=7,
                                     conv1_t_stride=1, no_max_pool=False, shortcut_type='B', widen_factor=1.0,
                                     num_classes=args.num_classes, feed_forward=True, bias_downsample=True,
                                     act=('relu', {'inplace': True}), norm='batch')

elif args.model == "KAN":
    net = SimpleConvKAN([8 * 4, 16 * 4, 32 * 4, 64 * 4], num_classes=2, input_channels=1, spline_order=3, groups=1,
                        dropout=0.25, dropout_linear=0.5, l1_penalty=0.00000, degree_out=1)

elif args.model == "MobileNet":
    net = timm.create_model(
                'mobilenetv2_050',    # 你想用的模型
                            pretrained=False,     # 不使用预训练
                            in_chans=1,           # 1 通道图像（灰度图）
                            num_classes=2         # 二分类
                )

elif args.model == "EfficientFormer":
    net = timm.create_model(
                            "efficientformerv2_s0",
                            pretrained=False,
                            in_chans=1,
                            num_classes=2
                )

elif args.model == "MedMamba":
    net = medmamba(in_chans=1, num_classes=2)

elif args.model == "RepViT":
    net = timm.create_model("repvit_m2_3.dist_300e_in1k", pretrained=False, in_chans=1, num_classes=2)
    # net = timm.create_model('repvit_m0_9', pretrained=False, in_chans=1, num_classes=2)

elif args.model == "MobileViT":
    net = timm.create_model('mobilevit_s', pretrained=False, in_chans=1, num_classes=2)

elif args.model == "LeViT":
    net = timm.create_model('levit_128s', pretrained=False, in_chans=1, num_classes=2)


n_params = sum(p.numel() for p in net.parameters())
print(f"Total parameters: {n_params / 1e6:.2f} M ({n_params:,} parameters)")

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
    transform_tr = get_cls_transform(split="train", image_size=args.image_size)
    transform_val = get_cls_transform(split="val", image_size=args.image_size)
    train_dataset = CarpalClassificationDataset(data_root=Path(args.data_path) / "train",
                                            annotation_path=Path(args.data_path) / "JointBE_SvdH_GT_reformatted.json",
                                            transform=transform_tr)
    train_loader = get_dataloader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_dataset = CarpalClassificationDataset(data_root=Path(args.data_path) / "val",
                                          annotation_path=Path(args.data_path) / "JointBE_SvdH_GT_reformatted.json",
                                          transform=transform_val)
    val_loader = get_dataloader(val_dataset, batch_size=args.val_batch_size, shuffle=True)

    # optimizer = optim.Adam(net.parameters(), lr=args.lr, amsgrad=True)
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    # optimizer = optim.SGD(net.parameters(), lr=args.lr)
    # criterion = FocalRegressionLoss(gamma=2.0, reduction='mean')
    # criterion = nn.MSELoss()
    # criterion = CDW_CELoss(class_weights=torch.tensor([2.0, 0.8, 0.8, 1.0], device="cuda:0"))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 6.0], device="cuda:0")) # [1.0, 6.0]
    # criterion = nn.CrossEntropyLoss()
    trainer = RegTrainer(args, net, train_loader, val_loader, criterion, optimizer, device="cuda:0")
    trainer.fit(args)

elif args.mode == "test":
    transform_test = get_cls_transform(split="test", image_size=args.image_size)
    test_dataset = CarpalClassificationDataset(data_root=Path(args.data_path) / "test",
                                           annotation_path=Path(args.data_path) / "JointBE_SvdH_GT_reformatted.json",
                                           transform=transform_test)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
    if not (Path(args.checkpoint) / "model_best.pth").exists():
        raise KeyError("Test mode is set but checkpoint does not exist.")
    tester = RegTester(args, net, test_loader, device="cuda:0")
    tester.test()
