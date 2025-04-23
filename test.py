from models.swin_unet.swintrans import SwinUnet
import torch
# 实例化 SwinUnet（Base 配置）
model = SwinUnet(
    img_size=512,
    patch_size=4,
    in_chans=1,         # 灰度图输入
    num_classes=3,      # 输出类别数
    embed_dim=128,      # Swin-B 的 embed_dim
    depths=[2, 2, 18, 2],
    depths_decoder=[2, 2, 2, 2],  # 反向路径自定义，通常可以简单一些
    num_heads=[4, 8, 16, 32],
    window_size=8,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    norm_layer=torch.nn.LayerNorm,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
    final_upsample="expand_first"
)

# 测试一下输入输出是否一致
import torch

dummy_input = torch.randn(1, 1, 512, 512)  # B, C, H, W
output = model(dummy_input)

print("Input shape: ", dummy_input.shape)
print("Output shape:", output.shape)  # 应该是 [1, 3, 512, 512]