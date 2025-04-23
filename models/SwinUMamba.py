from nnunetv2.nets.SwinUMamba import SwinUMamba, load_pretrained_ckpt


def get_SwinUMamba(in_channels, num_classes, feat_size=[48, 96, 192, 384, 768], hidden_size=768, deep_supervision=False):
    model = SwinUMamba(
        in_chans=in_channels,
        out_chans=num_classes,
        feat_size=feat_size,
        deep_supervision=deep_supervision,
        hidden_size=hidden_size,
    )
    return model