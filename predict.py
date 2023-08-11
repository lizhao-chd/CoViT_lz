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


# model =  Covit(img_size=224, in_chans=3, stages=4, embed_dims=[32, 64, 128, 256], token_dims=[32, 64, 128, 256],
#                   downsample_ratios=[4, 2, 2, 2], kernel_size=[7, 5, 5, 3],RC_heads=[32,64,128,256], NC_heads=16,dilations=[[3, 4], [2, 3], [1,2], [2]],
#                   RC_op='sum',RC_group=[1,32,64, 128], NC_group=[1, 32, 64, 64], NC_depth=[1, 1, 2,2],  mlp_ratio=[0.2,0.2,0.2,0.2],
#                   qkv_bias=True, qk_scale=None, drop_rate=0.,attn_drop_rate=0., drop_path_rate=0., num_classes=8, window_size=7, relative_pos=False).to(device)


#model=MobileNetV3_Large(num_classes=8).to(device)
#model=ShuffleNetV2(num_classes=8).to(device)
#model = EfficientViT(num_classes=8).to(device)
#model = ghostnetv2(num_classes=8).to(device)
#model = SwiftFormer_XS(num_classes=8) .to(device)
#model = EdgeViT_XXS(False,num_classes=8).to(device)
model = convnextv2_atto(num_classes=8).to(device)
# model = EfficientFormer(layers=[3, 2, 6, 4],embed_dims=[48, 96, 224, 448],downsamples=[True, True, True, True],vit_num=1,num_classes=10).to(device)
#model = make_model('resnet152', num_classes=10, pretrained=True, input_size=(224, 224)).to(device)

# load model weights
model_weight_path = "out_weight/convnextv2_atto_without_pretrained_driver100_data_batch_size_322023-0808-1746.pth"
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model_name="convnextv2_atto_driver100"
model.eval()


def predict_fun():
    data_dir = r"D:\all_data_sets\driver_distraction_public_datasets\driver_100_day\Driver_small\Test"  # 待分类数据路径
    image_list = os.listdir(data_dir)
    print("图片共", len(image_list))
    i = 0
    img_number  = 0  # 初始帧数
    while True:
        image_path = os.path.join(data_dir, image_list[i])
        if os.path.exists(image_path) or img_number<=2089:
            try:
                img_number = img_number + 1
                img = Image.open(image_path)
                name=image_path[-11:]
                begin = name.find("c")
                end = name.rfind("_")
                ss = name[begin+1:end]
                ss.strip()
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
                print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                             predict[predict_cla].numpy())

                predict_list=predict.tolist()

                print("%s" %name, print_res)
                sheet1.write(img_number, 0,"c%s" %ss)
                sheet1.write(img_number, 1, str(print_res)[6:9].strip())
                pro=str(print_res)[-8:]
                begin2 = pro.find(" ")
                sheet1.write(img_number, 2, pro[begin2:])
                sheet1.write(img_number,  3, predict_list[int(ss)])
                f2.save('out_prediction/predict_%s.xls'%model_name)
                i=i+1
            except:
                i = i + 1
                pass

            # for j in range(10):
            #     if str(print_res)[6:9].strip() == "c%s"%j:
            #         img_out = cv2.imread(image_path)
            #         print("保存")
            #         cv2.imwrite(r"test_small\c%s\%s.jpg" %(j,i) ,img_out)


if __name__ == '__main__':
    predict_fun()
