from nnunetv2.nets.UMambaBot import get_umamba_bot_from_plans
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_json
from nnunetv2.nets.UMambaBot import UMambaBot
from nnunetv2.nets.UMambaEnc import UMambaEnc
import torch
import torch.nn as nn
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm
from dynamic_network_architectures.building_blocks.residual import BasicBlockD


def get_UMambaBot(in_channels, num_classes, n_stages=4, features_per_stage=[32, 64, 128, 256],
                  n_conv_per_stage=[2, 2, 2, 2], n_conv_per_stage_decoder=[2, 2, 2],
                  strides=[(1, 1), (2, 2), (2, 2), (2, 2)]):
    model = UMambaBot(
        input_channels=in_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv2d,
        kernel_sizes=[(3, 3)]*n_stages,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=get_matching_instancenorm(nn.Conv2d),
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=False,
        block=BasicBlockD,
        stem_channels=None
        )
    return model


def get_UMambaEnc(in_channels, num_classes, n_stages=4, features_per_stage=[32, 64, 128, 256],
                  n_conv_per_stage=[2, 2, 2, 2], n_conv_per_stage_decoder=[2, 2, 2],
                  strides=[(1, 1), (2, 2), (2, 2), (2, 2)]):
    model = UMambaEnc(
        input_channels=in_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=nn.Conv2d,
        kernel_sizes=[(3, 3)]*n_stages,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_classes,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=get_matching_instancenorm(nn.Conv2d),
        norm_op_kwargs={'eps': 1e-5, 'affine': True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True},
        deep_supervision=False,
        block=BasicBlockD,
        stem_channels=None
        )
    return model


if __name__ == '__main__':
    model = get_UMambaEnc(in_channels=1, num_classes=14)
    model = model.to("cuda:0")
    # 测试输出 shape
    x = torch.randn(1, 1, 512, 512).to("cuda:0")
    y = model(x)
    print('Input shape:', x.shape)
    print('Output shape:', y.shape)  # 期望: [1, 3, 512, 512]
