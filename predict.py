import os
import json
import torch
from PIL import Image
from torchvision import transforms
import xlwt
from test_models.mobilenet_v3 import MobileNetV3_Large
from test_models.shuufflenetv2 import ShuffleNetV2
from test_models.Ghostnetv2 import ghostnetv2
from test_models.Co_vit_lz import Covit
from test_models.convnextv2 import convnextv2_atto
from test_models.edgevit import EdgeViT_XXS
from test_models.swiftformer import SwiftFormer_XS
from test_models.efficientvit import EfficientViT

import cv2
from cnn_finetune import make_model
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


f2 = xlwt.Workbook()
sheet1 = f2.add_sheet(r'out', cell_overwrite_ok=True)
sheet1.write(0, 0, 'true_label')
sheet1.write(0, 1, 'prediction')
sheet1.write(0, 2, 'prob')


data_transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.5, 0.5, 0.5], [0.4, 0.4, 0.4])])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = Covit(img_size=args.img_size, in_chans=args.in_chans, stages=args.stages, embed_dims=args.embed_dims, token_dims=args.token_dims,
              downsample_ratios=args.img_size, kernel_size=args.kernel_size, CNN_heads=args.CNN_heads, Transformer_heads=args.Transformer_heads,
              dilations=args.dilations,
              CNN_op=args.CNN_op, CNN__group=args.CNN__group, Transformer_group=args.Transformer_group, Transformer_depth=args.Transformer_depth,
              mlp_ratio=args.mlp_ratio,
              qkv_bias=args.qkv_bias, qk_scale=args.qk_scale, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., num_classes=10,
              window_size=7, relative_pos=False)
#model=MobileNetV3_Large(num_classes=10).to(device)
#model=ShuffleNetV2(num_classes=10).to(device)
#model = EfficientViT(num_classes=10).to(device)
#model = ghostnetv2(num_classes=10).to(device)
#model = SwiftFormer_XS(num_classes=10) .to(device)
#model = EdgeViT_XXS(False,num_classes=10).to(device)
#model = convnextv2_atto(num_classes=10).to(device)
# model = EfficientFormer(layers=[3, 2, 6, 4],embed_dims=[48, 96, 224, 448],downsamples=[True, True, True, True],vit_num=1,num_classes=10).to(device)
#model = make_model('resnet152', num_classes=10, pretrained=True, input_size=(224, 224)).to(device)

# load model weights
model_weight_path = "Covit_SFD2_batchsize_32_2024-0324-1609.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model_name="Covit_SFD2"
model.eval()



def predict_fun():
    img_number = 1
    for i in range(10):
        data_dir = r"D:\all_data_sets\driver_distraction_public_datasets\state_farm_split_by_drivers\train_split\test_driver\c%s" % i  # 待分类数据路径
        image_list = os.listdir(data_dir)
        for img_name in image_list:
            image_path=r"D:\all_data_sets\driver_distraction_public_datasets\state_farm_split_by_drivers\train_split\test_driver\c%s\%s" % (i, img_name)
            img = Image.open(image_path)

            img = data_transform(img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            # read class_indict
            json_path = 'class_indices 2.json'
            assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

            with open(json_path, "r") as f:
                class_indict = json.load(f)

            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
            print(predict_cla)
            print(predict[int(predict_cla)].numpy())
            #print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],predict[int(predict_cla)].numpy())

            predict_list=predict.tolist()


            sheet1.write(img_number, 0,"c%s" %i)
            sheet1.write(img_number, 1, str(predict_cla))
            sheet1.write(img_number, 2, str(predict[int(predict_cla)].numpy()))
            sheet1.write(img_number,  3, predict_list[i])
            f2.save('out_prediction/predict_%s.xls'%model_name)
            img_number=img_number+1


            # for j in range(10):
            #     if str(print_res)[6:9].strip() == "c%s"%j:
            #         img_out = cv2.imread(image_path)
            #         print("保存")
            #         cv2.imwrite(r"test_small\c%s\%s.jpg" %(j,i) ,img_out)


if __name__ == '__main__':
    predict_fun()
