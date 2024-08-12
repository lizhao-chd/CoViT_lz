from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import resnet50
import os
import cv2
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import matplotlib.pyplot as plt
import torch
from test_models.mobilenet_v3 import MobileNetV3_Large
from test_models.shuufflenetv2 import ShuffleNetV2
from test_models.Ghostnetv2 import ghostnetv2
from test_models.Co_vit_lz import Covit
from test_models.convnextv2 import convnextv2_atto
from test_models.edgevit import EdgeViT_XXS
from test_models.swiftformer import SwiftFormer_XS
from test_models.efficientvit import EfficientViT



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
#model = MobileNetV3_Large(num_classes=8)
#model=ShuffleNetV2(num_classes=8)
#model = EfficientViT(num_classes=8)
#model = ghostnetv2(num_classes=8)
#model = SwiftFormer_XS(num_classes=8)      #model.network[-1][2]
#model = EdgeViT_XXS(False,num_classes=8)   #model.main_body[44]
#model = convnextv2_atto(num_classes=8)     #model.stages[-1][-1]
# model = make_model('resnet152', num_classes=10, pretrained=True, input_size=(224, 224))

# load model weights
model_weight_path = "out_weight/Covit_3_without_pretrained_OUR_data_batch_size_322023-0811-1846.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()

print(model.layers[-1].NC[1].DCM2)
target_layers = [model.layers[-2].RC.DCM1]


def reshape_transform_SWIN(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)#,reshape_transform=reshape_transform_SWIN



data_dir = r"F:\Desktop\our_models\CoViT\test_img"  # 待分类数据路径
image_list = os.listdir(data_dir)
print("图片共", len(image_list))
i = 0
while True:
    image_path = os.path.join(data_dir, image_list[i])
    i=i+1
    print(image_path)
    if os.path.exists(image_path):
        image_org = cv2.imread(image_path)
        image = cv2.resize(image_org, (224, 224))
        rgb_img = np.float32(image) / 255
        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam =cam(input_tensor=input_tensor,
                                    targets=None,
                                eigen_smooth=False,
                                aug_smooth=False)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam)

        cam_image = cv2.resize(visualization, (image_org.shape[1], image_org.shape[0]))

        if not os.path.exists("img_out/result_out"):
            os.mkdir("img_out/result_out")
        cv2.imshow("vis_out", cam_image)
        cv2.imwrite("img_out/result_out/%smap.jpg" % i, cam_image)
        cv2.imwrite("img_out/result_out/%s.jpg" % i, image_org)
        cv2.waitKey(1)
        # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
