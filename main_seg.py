import argparse
from utils import *
from trainer import SegTrainer, SegTester
import monai
from pathlib import Path
from datasets.carpal import CarpalNpyDataset, get_dataloader
import torch.optim as optim
from datetime import datetime
from typing import Union
from models.UnetPlusPlus import UnetPlusPlus
from models.swin_unet.swintrans import SwinUnet
from models.swin_unet.swin_unet import get_SwinUnet_Custom
from models.Seg_UKAN.archs import UKAN
from models.TransUNet.transUnet import get_TransUnet_Custom
from models.UMamba import get_UMambaBot, get_UMambaEnc
from models.SwinUMamba import get_SwinUMamba
import segmentation_models_pytorch as smp


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
        choices=["train", "test", "infer"],
        help='Mode',
    )

    parser.add_argument(
        '--image_size',
        type=int,
        default=512,
        help='The size of the images.',
    )

    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=4,
        help='Batch size for training.',
    )

    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=1,
        help='Batch size for validating.',
    )

    parser.add_argument(
        '--model',
        type=str,
        default="DeepLabV3+",
        choices=["Unet", "SwinUnet", "SegResNet", "Unet++", "TransUnet", "UKAN", "DeepLabV3", "DeepLabV3+", "PSPNet",
                 "PAN", "DPT", "SegFormer", "FPN", "UMambaBot", "UMambaEnc", "SwinUMamba"],
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
        default="test",
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
        default=1e-4,
        help='Initial lr',
    )

    parser.add_argument(
        '--save_overlay',
        action='store_true',
        default=False,
        help='Whether to save the overlay (Test mode only).'
    )

    parser.add_argument(
        '--save_csv',
        action='store_true',
        default=False,
        help='Whether save csv (Test mode only).',
    )

    parser.add_argument(
        '--save_pred',
        action='store_true',
        default=False,
        help='Whether save pred (Test mode only).',
    )
    args = parser.parse_args()
    return args


args = get_args()
# Seed everything
seed_everything(args.seed)


# initial network
if args.model == "Unet":
    net = monai.networks.nets.DynUNet(spatial_dims=2, in_channels=1, out_channels=14, kernel_size=[3, 3, 3, 3, 3],
                                      norm_name="batch", strides=[1, 2, 2, 2, 2], upsample_kernel_size=[2, 2, 2, 2, 2],
                                      filters=[32, 64, 128, 256, 512], res_block=True)

elif args.model == "SegResNet":
    net = monai.networks.nets.SegResNet(spatial_dims=2, in_channels=1, out_channels=14, upsample_mode="deconv",
                                        act="LeakyReLU", norm="batch", init_filters=16, blocks_down=(1, 2, 2, 4),
                                        blocks_up=(1, 1, 1),)

elif args.model == "Unet++":
    net = UnetPlusPlus(spatial_dims=2, in_channels=1, out_channels=14, features=(32, 32, 64, 128, 256, 32))

elif args.model == "SwinUnet":
    # net = get_SwinUnet_Custom(img_size=args.image_size, n_classes=14)
    net = SwinUnet(
        img_size=args.image_size, patch_size=4, in_chans=1, num_classes=14, embed_dim=128, depths=[2, 2, 18, 2],
        depths_decoder=[2, 2, 2, 2], num_heads=[4, 8, 16, 32], window_size=8, mlp_ratio=4.0, qkv_bias=True,
        qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=torch.nn.LayerNorm,
        ape=False, patch_norm=True, use_checkpoint=False, final_upsample="expand_first"
    )

elif args.model == "UKAN":
    net = UKAN(num_classes=14,  input_channels=3, embed_dims=[128, 160, 256])

elif args.model == "TransUnet":
    net = get_TransUnet_Custom(img_size=args.image_size, n_classes=14)

elif args.model == "DeepLabV3+":
    net = smp.DeepLabV3Plus(in_channels=1, encoder_weights=None, classes=14)

elif args.model == "DeepLabV3":
    net = smp.DeepLabV3(in_channels=1, encoder_weights=None, classes=14)

elif args.model == "PSPNet":
    net = smp.PSPNet(in_channels=1, encoder_weights=None, classes=14)

elif args.model == "PAN":
    net = smp.PAN(in_channels=1, encoder_weights=None, classes=14)

elif args.model == "DPT":
    net = smp.DPT(in_channels=1, encoder_weights=None, classes=14)

elif args.model == "SegFormer":
    net = smp.Segformer(in_channels=1, encoder_weights=None, classes=14)

elif args.model == "FPN":
    net = smp.FPN(in_channels=1, encoder_weights=None, classes=14)

elif args.model == "UMambaBot":
    net = get_UMambaBot(in_channels=1, num_classes=14)

elif args.model == "UMambaEnc":
    net = get_UMambaEnc(in_channels=1, num_classes=14)

elif args.model == "SwinUMamba":
    net = get_SwinUMamba(in_channels=1, num_classes=14)

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
    transform_tr = get_transform(split="train", image_size=args.image_size)
    transform_val = get_transform(split="val", image_size=args.image_size)
    train_dataset = CarpalNpyDataset(data_root=Path(args.data_path) / "image", annotation_path=Path(args.data_path) / "mask" / "train",
                                     transform=transform_tr)
    # train_dataset = CarpalDataset(data_root=args.data_path, annotation_path=Path(args.data_path) / "train.coco.json",
    #                               transform=transform_tr)
    train_loader = get_dataloader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_dataset = CarpalNpyDataset(data_root=Path(args.data_path) / "image", annotation_path=Path(args.data_path) / "mask" / "val",
                                transform=transform_val)
    # val_dataset = CarpalDataset(data_root=args.data_path, annotation_path=Path(args.data_path) / "val.coco.json",
    #                             transform=transform_val)
    val_loader = get_dataloader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    optimizer = optim.AdamW(net.parameters(), lr=args.lr)
    criterion = monai.losses.DiceLoss(sigmoid=False, squared_pred=True, reduction='mean')
    trainer = SegTrainer(args, net, train_loader, val_loader, criterion, optimizer, device="cuda:0")
    trainer.fit(args)

elif args.mode == "test":
    transform_test = get_transform(split="test", image_size=args.image_size)
    test_dataset = CarpalNpyDataset(data_root=Path(args.data_path) / "image", annotation_path=Path(args.data_path) / "mask" / "test",
                                    transform=transform_test)
    # test_dataset = CarpalDataset(data_root=args.data_path, annotation_path=Path(args.data_path) / "test.coco.json",
    #                              transform=transform_test)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
    if not (Path(args.checkpoint) / "model_best.pth").exists():
        raise KeyError("Test mode is set but checkpoint does not exist.")
    tester = SegTester(args, net, test_loader, device="cuda:0")
    tester.test()

elif args.mode == "test":
    transform_test = get_transform(split="test", image_size=args.image_size)
    test_dataset = CarpalNpyDataset(data_root=Path(args.data_path) / "image", annotation_path=Path(args.data_path) / "mask" / "test",
                                    transform=transform_test)
    # test_dataset = CarpalDataset(data_root=args.data_path, annotation_path=Path(args.data_path) / "test.coco.json",
    #                              transform=transform_test)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=False)
    if not (Path(args.checkpoint) / "model_best.pth").exists():
        raise KeyError("Test mode is set but checkpoint does not exist.")
    tester = SegTester(args, net, test_loader, device="cuda:0")
    tester.test()
