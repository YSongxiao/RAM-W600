import torch

ckpt = torch.load('/mnt/data1/songxiao/CarpalData/ckpts/Baseline_v1_unet_202504140442/model_best.pth', map_location='cpu')
for k in ckpt['model'].keys():
    print(k)
