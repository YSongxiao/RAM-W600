import monai
import torch.nn as nn


class UnetPlusPlus(nn.Module):
    def __init__(self, spatial_dims=2, in_channels=1, out_channels=14, features=(32, 32, 64, 128, 256, 32)):
        super().__init__()
        self.model = monai.networks.nets.BasicUNetPlusPlus(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, features=features)

    def forward(self, x):
        outputs = self.model(x)
        return outputs[0]
