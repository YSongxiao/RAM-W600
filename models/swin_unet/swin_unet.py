import torch

from models.swin_unet.networks.vision_transformer import SwinUnet as ViT_seg
# from config import get_config
import argparse
import yaml


def parse_yaml_to_args(yaml_path):
    """读取 YAML 并转换为 argparse.Namespace，支持嵌套结构"""

    # 递归转换字典 -> argparse.Namespace
    def dict_to_namespace(d):
        if isinstance(d, dict):
            namespace = argparse.Namespace()
            for key, value in d.items():
                setattr(namespace, key, dict_to_namespace(value))  # 递归处理嵌套字典
            return namespace
        elif isinstance(d, list):  # 如果是列表，递归处理
            return [dict_to_namespace(item) if isinstance(item, dict) else item for item in d]
        else:
            return d  # 直接返回普通值

    # 读取 YAML 文件
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # 递归转换为 Namespace
    return dict_to_namespace(config)

# parser = argparse.ArgumentParser()
# parser.add_argument('--root_path', type=str,
#                     default='../data/Synapse/train_npz', help='root dir for data')
# parser.add_argument('--dataset', type=str,
#                     default='Synapse', help='experiment_name')
# parser.add_argument('--list_dir', type=str,
#                     default='./lists/lists_Synapse', help='list dir')
# parser.add_argument('--num_classes', type=int,
#                     default=9, help='output channel of network')
# parser.add_argument('--output_dir', type=str, help='output dir')
# parser.add_argument('--max_iterations', type=int,
#                     default=30000, help='maximum epoch number to train')
# parser.add_argument('--max_epochs', type=int,
#                     default=150, help='maximum epoch number to train')
# parser.add_argument('--batch_size', type=int,
#                     default=24, help='batch_size per gpu')
# parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
# parser.add_argument('--deterministic', type=int, default=1,
#                     help='whether use deterministic training')
# parser.add_argument('--base_lr', type=float, default=0.01,
#                     help='segmentation network learning rate')
# parser.add_argument('--img_size', type=int,
#                     default=224, help='input patch size of network input')
# parser.add_argument('--seed', type=int,
#                     default=1234, help='random seed')
# parser.add_argument('--cfg', type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml", metavar="FILE", help='path to config file', )
# parser.add_argument(
#     "--opts",
#     help="Modify config options by adding 'KEY VALUE' pairs. ",
#     default=None,
#     nargs='+',
# )
# parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
# parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
#                     help='no: no cache, '
#                          'full: cache all data, '
#                          'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
# parser.add_argument('--resume', help='resume from checkpoint')
# parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
# parser.add_argument('--use-checkpoint', action='store_true',
#                     help="whether to use gradient checkpointing to save memory")
# parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
#                     help='mixed precision opt level, if O0, no amp is used')
# parser.add_argument('--tag', help='tag of experiment')
# parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
# parser.add_argument('--throughput', action='store_true', help='Test throughput only')
# # parser.add_argument("--dataset_name", default="datasets")
# parser.add_argument("--n_class", default=14, type=int)
# parser.add_argument("--num_workers", default=8, type=int)
# parser.add_argument("--eval_interval", default=1, type=int)
# args = parser.parse_args()

# config = get_config(args)
def get_SwinUnet(n_classes=14):
    config = parse_yaml_to_args("./models/swin_unet/configs/swin_tiny_patch4_window7_224_lite.yaml")
    config.DATA = argparse.Namespace()
    config.DATA.IMG_SIZE = 224  # 直接赋值
    net = ViT_seg(config, img_size=224, num_classes=n_classes).cuda()
    net.load_from(config)
    return net


def get_SwinUnet_Custom(img_size, n_classes=14):
    config = parse_yaml_to_args("./models/swin_unet/configs/swin_tiny_patch4_window7_224_lite_scratch.yaml")
    config.DATA = argparse.Namespace()
    config.DATA.IMG_SIZE = img_size  # 直接赋值
    net = ViT_seg(config, img_size=img_size, num_classes=n_classes).cuda()
    return net


# print(net)
# dummy_inp = torch.rand((4, 3, 224, 224)).cuda()
# out = net(dummy_inp)
# print(out.shape)
