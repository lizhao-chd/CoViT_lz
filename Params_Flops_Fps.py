from thop import profile
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from cnn_finetune import make_model
import torch
import numpy as np
from pytorch_pretrained_vit import ViT
from torch import nn
from test_models.Co_vit_lz import Covit
import argparse
parser = argparse.ArgumentParser(description="model hype")
parser.add_argument('--img_size', type=int)
parser.add_argument('-in_chans', type=int)
parser.add_argument('-stages', type=int)
parser.add_argument('-embed_dims', type=list)
parser.add_argument('-token_dims', type=list)
parser.add_argument('-kernel_size', type=list)
parser.add_argument('-CNN_heads', type=list)
parser.add_argument('-Transformer_heads', type=list)
parser.add_argument('-dilations', type=list)
parser.add_argument('-CNN_op', type=str)
parser.add_argument('-CNN__group', type=list)
parser.add_argument('-Transformer_group', type=list)
parser.add_argument('-Transformer_depth', type=list)
parser.add_argument('-mlp_ratio', type=list)
parser.add_argument('-qkv_bias', type=bool)
parser.add_argument('-qk_scale', type=bool)
args = parser.parse_args()


model = Covit(img_size=args.img_size, in_chans=args.in_chans, stages=args.stages, embed_dims=args.embed_dims, token_dims=args.token_dims,
              downsample_ratios=args.img_size, kernel_size=args.kernel_size, CNN_heads=args.CNN_heads, Transformer_heads=args.Transformer_heads,
              dilations=args.dilations,
              CNN_op=args.CNN_op, CNN__group=args.CNN__group, Transformer_group=args.Transformer_group, Transformer_depth=args.Transformer_depth,
              mlp_ratio=args.mlp_ratio,
              qkv_bias=args.qkv_bias, qk_scale=args.qk_scale, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., num_classes=10,
              window_size=7, relative_pos=False)

random_input = torch.randn(1, 3, 224, 224)
print(model(random_input))

flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print('Flops:  ' + flops)
print('Params: ' + params)

import torch
from torchvision.models.resnet import resnet101

iterations = 1000
device = torch.device("cuda:0")
model.to(device)
random_input = torch.randn(64, 3, 224, 224).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# GPU warm-up
for _ in range(100):
   _ = model(random_input)
# FPS test
times = torch.zeros(iterations)
with torch.no_grad():
   for iter in range(iterations):
      starter.record()
      _ = model(random_input)
      ender.record()
      # Synchronised GPU time
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      times[iter] = curr_time
mean_time = times.mean().item()
print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))
