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



# model = Covit(img_size=224, in_chans=3, stages=4, embed_dims=[32, 64, 128, 256], token_dims=[32, 64, 128, 256],    #model.layers[-1].NC[1].norm3  ,model.layers[-1].RC.PCM
#               downsample_ratios=[4, 2, 2, 2], kernel_size=[7, 5, 5, 3], RC_heads=[1, 1, 1, 1],
#               dilations=[[3, 4], [2, 3], [3], [3]],
#               RC_op='sum', RC_group=[1, 32, 64, 128], NC_group=[1, 32, 64, 64], NC_depth=[1, 2, 2, 2], mlp_ratio=1,
#               qkv_bias=True, qk_scale=None, drop_rate=0.,
#               attn_drop_rate=0., drop_path_rate=0., num_classes=8, window_size=7, relative_pos=False)
#model = MobileNetV3_Large(num_classes=8)
#model=ShuffleNetV2(num_classes=8)
#model = EfficientViT(num_classes=8)
#model = ghostnetv2(num_classes=8)
#model = SwiftFormer_XS(num_classes=8)      #model.network[-1][2]
model = EdgeViT_XXS(False,num_classes=8)   #model.main_body[44]
#model = convnextv2_atto(num_classes=8)     #model.stages[-1][-1]
# model = make_model('resnet152', num_classes=10, pretrained=True, input_size=(224, 224))

# load model weights
model_weight_path = "out_weight/EdgeViT_XXS_2_without_pretrained_driver100_data_batch_size_322023-0807-1846.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()

print(model.main_body[44])
target_layers = [model.main_body[44]]


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
