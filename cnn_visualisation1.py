import numpy as np
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchcam.methods import GradCAMpp,XGradCAM,GradCAM
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import cv2
from cnn_finetune import make_model
import torch
from PIL import Image
from models_lz.test_models.mobilenet_v3 import MobileNetV3_Large
import cv2

def vis_cam(input_tensor,model,frame_number,class_leibie,img_org):
    out = model(input_tensor.unsqueeze(0))
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    # Visualize the raw CAM
    #plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
    # Display it
    #plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
    result=np.array(result)
    b,g,r = cv2.split(result)
    result = cv2.merge((r,g,b))

    if not os.path.exists("img_out/result_out3"):
        os.mkdir("img_out/result_out3")

    cv2.imshow("vis_out",result)
    cv2.imwrite("img_out/result_out3/%s%smap.jpg"%(class_leibie,frame_number),img_org)
    cv2.imwrite("img_out/result_out3/%s%s.jpg"%(class_leibie,frame_number),result)
    cv2.waitKey(1)


if __name__ == '__main__':
    data_transform = transforms.Compose(
        [transforms.Resize(244),
         transforms.CenterCrop(244),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
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



    cam_extractor = GradCAMpp(model)
    # Get your input
    class_leibie="c8" #C7820
    data_dir = "test"  # 待分类数据路径
    image_list = os.listdir(data_dir)
    print("图片共", len(image_list))
    i = 0
    while True:
        image_path = os.path.join(data_dir, image_list[i])
        print(image_path)
        if os.path.exists(image_path):
            img_org=cv2.imread(image_path)
            img = read_image(image_path)
            input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            vis_cam(input_tensor, model,i,class_leibie,img_org)
            i=i+1
            if i>=1000:
                break

