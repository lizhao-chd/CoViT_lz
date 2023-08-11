from thop import profile
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from cnn_finetune import make_model
import torch
import numpy as np
from pytorch_pretrained_vit import ViT
from torch import nn
from models_lz.lz_models.co_td_vit_lz import creat_co_td_vit
from models_lz.lz_models.co_vit_lz import creat_co_vit
from models_lz.lz_models.td_vit_lz import creat_td_vit
from models_lz.lz_models.vit_lz import creat_small_vit,creat_large_vit
from models_lz.lz_models.swin_transformer_lz import SwimTransformer
from models_lz.lz_models.Efficientnetv2_lz import effnetv2_xl,effnetv2_s,effnetv2_m


#model = make_model('inception_v4', num_classes=8, pretrained=False ,input_size=(224, 224))
                                                                            # resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d,
                                                                            # densenet121, densenet169, densenet201, densenet161, alexnet,
                                                                            # vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
                                                                            # , squeezenet1_0, squeezenet1_1,
                                                                            # inception_v3, googlenet, mobilenet_v2,
                                                                            # shufflenet_v2_x0_5, shufflenet_v2_x1_0, r
                                                                            # esnext101_32x4d, resnext101_64x4d, nasnetalarge, nasnetamobile, inceptionresnetv2,
                                                                            # dpn68, dpn68b, dpn92, dpn98, dpn131, dpn107,
                                                                            # inception_v4, xception, senet154,
                                                                            # se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d, pnasnet5large, polynet



# -------------------》》》》》vit模型预训练
# model = ViT('B_16', pretrained=True)#C:\Users\TTT/.cache\torch\hub\checkpoints\L_16_imagenet1k.pth
# fc_features=model.fc.in_features
# model.fc = nn.Linear(fc_features, 8)
                                                                            # B_16, B_32, L_16, L_32, B_16_imagenet1k, B_32_imagenet1k, L_16_imagenet1k, L_32_imagenet1k



model=creat_co_td_vit(8)

# --------------------》》》》》模型参数量计算
# input = torch.randn(1, 3, 224, 224)
# flops, params = profile(model, inputs=(input,))
# print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
# print("params=", str(params / 1e6) + '{}'.format("M"))
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print('Flops:  ' + flops)
print('Params: ' + params)



device = torch.device("cuda")
model.to(device)
dummy_input = torch.randn(1, 3, 224, 224,dtype=torch.float).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
   _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
  for rep in range(repetitions):
     starter.record()
     _ = model(dummy_input)
     ender.record()
     # WAIT FOR GPU SYNC
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)
     timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
mean_fps = 1000. / mean_syn
print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
print(mean_syn)