import argparse
import cv2
import numpy as np
import torch
from torch.nn import Linear
import os
from models_lz.lz_models import co_td_vit_lz
from models_lz.lz_models import vit_lz
from models_lz.lz_models import co_td_vit_lz
from pytorch_pretrained_vit import ViT
import timm
from models_lz.lz_models.swin_transformer_lz import SwinTransformer
from cnn_finetune import make_model
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from models_lz.lz_models.vitaev2.vitmodules import ViTAEv2_basic
from models_lz.test_models.Co_vit_lz import Covit
from models_lz.test_models.Ghostnetv2 import  ghostnetv2
from models_lz.test_models.swiftformer import SwiftFormer_XS
from models_lz.test_models.edgevit import EdgeViT_XXS
from models_lz.test_models.convnextv2 import convnextv2_atto
from models_lz.test_models.mobilevit import mobile_vit_x_small
from models_lz.test_models.Co_vit_lz import Covit
from models_lz.test_models.efficientformer_lz import EfficientFormer
from models_lz.test_models.efficientvit import EfficientViT
from models_lz.test_models.shuufflenetv2 import ShuffleNetV2
from models_lz.test_models.mobilenet_v3 import MobileNetV3_Large
import cv2
from cnn_finetune import make_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='eigengradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam/eigengradcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform_VIT(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def reshape_transform_SWIN(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),height, width, tensor.size(2))
    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.
    """
    args = get_args()
    methods ={"gradcam": GradCAM, "scorecam": ScoreCAM,"gradcam++": GradCAMPlusPlus,"ablationcam": AblationCAM,
         "xgradcam": XGradCAM,"eigencam": EigenCAM, "eigengradcam": EigenGradCAM,"layercam": LayerCAM, "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    device = torch.device("cpu")

    model = Covit(img_size=args.img_size, in_chans=args.in_chans, stages=args.stages, embed_dims=args.embed_dims,
                  token_dims=args.token_dims,
                  downsample_ratios=args.img_size, kernel_size=args.kernel_size, CNN_heads=args.CNN_heads,
                  Transformer_heads=args.Transformer_heads,
                  dilations=args.dilations,
                  CNN_op=args.CNN_op, CNN__group=args.CNN__group, Transformer_group=args.Transformer_group,
                  Transformer_depth=args.Transformer_depth,
                  mlp_ratio=args.mlp_ratio,
                  qkv_bias=args.qkv_bias, qk_scale=args.qk_scale, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                  num_classes=10,
                  window_size=7, relative_pos=False)

    # model=MobileNetV3_Large(num_classes=10).to(device)
    # model=ShuffleNetV2(num_classes=10).to(device)
    # model = EfficientViT(num_classes=10).to(device)
    # model = ghostnetv2().to(device)
    # model = SwiftFormer_XS(num_classes=10).to(device)
    # model = EdgeViT_XXS(False).to(device)
    # model = convnextv2_atto(num_classes=10).to(device)
    # model = EfficientFormer(layers=[3, 2, 6, 4],embed_dims=[48, 96, 224, 448],downsamples=[True, True, True, True],vit_num=1,num_classes=10).to(device)
    # model = make_model('resnet152', num_classes=10, pretrained=True, input_size=(224, 224)).to(device)

    # load model weights
    model_weight_path = "out_weight/OUR/EfficientViT_without_pretrained_our_data_batch_size_322023-0722-1746.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model_name = "EfficientViT_OUR"
    model.eval()

    print(model.layers[-1].NC[1].norm3)
    target_layers = [model.layers[-1].NC[1].norm3]
    #print(model)


    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform_SWIN,#针对vit所做的改变
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   use_cuda=args.use_cuda,
                                   reshape_transform=reshape_transform_SWIN)#


    data_dir = "test"  # 待分类数据路径
    image_list = os.listdir(data_dir)
    print("图片共", len(image_list))
    i = 0
    while True:
        image_path = os.path.join(data_dir, image_list[i])
        print(image_path)
        if os.path.exists(image_path):
            rgb_img_org = cv2.imread(image_path)
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
            cam_image = show_cam_on_image(rgb_img , grayscale_cam)
            cam_image=cv2.resize(cam_image,(rgb_img_org.shape[1],rgb_img_org.shape[0]))


            if not os.path.exists("img_out/result_out"):
                os.mkdir("img_out/result_out")
            cv2.imshow("vis_out", cam_image)
            cv2.imwrite("img_out/result_out/%smap.jpg" %  i, cam_image)
            cv2.imwrite("img_out/result_out/%s.jpg" % i, rgb_img_org)
            cv2.waitKey(1)
            #cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
            i=i+1

