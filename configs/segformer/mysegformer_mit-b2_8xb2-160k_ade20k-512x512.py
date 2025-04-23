_base_ = ['./segformer_mit-b0_8xb2-160k_ade20k-512x512.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b2_20220624-66e8bf70.pth'  # noqa

# model settings
model = dict(
    type='EncoderDecoder',
    pretrained=None,  # ✅ 如果你加载的是默认结构的预训练，可以不改
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=1,  # ✅ 修改输入通道数
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),  # 可选是否加载预训练
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[64, 128, 320, 512],  # 和 embed_dims 保持一致
        num_classes=14,  # ✅ 修改类别数
    ),
    auxiliary_head=dict(
        in_channels=320,
        num_classes=14,  # ✅ 同样要改
    )
)

    # backbone=dict(
    #     init_cfg=dict(type='Pretrained', checkpoint=checkpoint, ),
    #     embed_dims=64,
    #     num_heads=[1, 2, 5, 8],
    #     num_layers=[3, 4, 6, 3]),
    # decode_head=dict(in_channels=[64, 128, 320, 512]))
