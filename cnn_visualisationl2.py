# copy from https://github.com/jacobgil/pytorch-grad-cam/blob/master/cam.py

import argparse
import cv2
import numpy as np
import torch
from torchvision import models
from cnn_finetune import make_model

from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image


# 如果出现 OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from test_models.Co_vit_lz import Covit
from test_models.Ghostnetv2 import  ghostnetv2
from test_models.swiftformer import SwiftFormer_XS
from test_models.edgevit import EdgeViT_XXS
from test_models.convnextv2 import convnextv2_atto
from test_models.mobilevit import mobile_vit_x_small
from test_models.Co_vit_lz import Covit
from test_models.efficientformer_lz import EfficientFormer
from test_models.efficientvit import EfficientViT
from test_models.shuufflenetv2 import ShuffleNetV2
from test_models.mobilenet_v3 import MobileNetV3_Large
import cv2
from cnn_finetune import make_model
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam',
                                 'ablationcam', 'eigencam', 'eigengradcam'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam'
                             '/ablationcam/eigencam/eigengradcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM}

    device = torch.device("cpu")
    model=MobileNetV3_Large(num_classes=8).to(device)
    # model=ShuffleNetV2(num_classes=10).to(device)
    # model = EfficientViT(num_classes=10).to(device)
    # model = ghostnetv2().to(device)
    # model = SwiftFormer_XS(num_classes=10).to(device)
    # model = EdgeViT_XXS(False).to(device)
    # model = convnextv2_atto(num_classes=10).to(device)
    # model = EfficientFormer(layers=[3, 2, 6, 4],embed_dims=[48, 96, 224, 448],downsamples=[True, True, True, True],vit_num=1,num_classes=10).to(device)
    # model = make_model('resnet152', num_classes=10, pretrained=True, input_size=(224, 224)).to(device)

    # load model weights
    model_weight_path = "out_weight/OUR/MobileNetV3_Large_without_pretrained_our_data_batch_size_322023-0722-1746.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    # Resnet18 and 50: model.layer4[-1]

    print(model)
    target_layer = model.bneck[9]

    cam = methods[args.method](model=model,
                               target_layer=target_layer,
                               use_cuda=False)

    data_dir = "test"  # 待分类数据路径
    image_list = os.listdir(data_dir)
    print("图片共", len(image_list))
    i = 0
    while True:
        image_path = os.path.join(data_dir, image_list[i])
        print(image_path)
        if os.path.exists(image_path):
            rgb_img_org = cv2.imread(image_path, 1)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img_org, (224, 224))
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
            # If None, returns the map for the highest scoring category.
            # Otherwise, targets the requested category.
            targets = None
            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 32

            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                eigen_smooth=args.eigen_smooth,
                                aug_smooth=args.aug_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam)
            cam_image = cv2.resize(cam_image, (rgb_img_org.shape[1], rgb_img_org.shape[0]))

            if not os.path.exists("img_out/result_out"):
                os.mkdir("img_out/result_out")
            cv2.imshow("vis_out", cam_image)
            cv2.imwrite("img_out/result_out/%smap.jpg" % i, cam_image)
            cv2.imwrite("img_out/result_out/%s.jpg" % i, rgb_img_org)
            cv2.waitKey(1)
            # cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
            i = i + 1
