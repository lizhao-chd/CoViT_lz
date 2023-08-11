import torch
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tools import read_split_data, train_one_epoch, evaluate
from my_dataset import MyDataSet
import math
import argparse
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from ptflops import get_model_complexity_info
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
from cnn_finetune import make_model

import xlwt

f1 = xlwt.Workbook()
sheet1 = f1.add_sheet(r'out', cell_overwrite_ok=True)
sheet1.write(0, 0, 'epoch')
sheet1.write(0, 1, 'Train_Loss')
sheet1.write(0, 2, 'Train_Acc')
sheet1.write(0, 3, 'Val_Loss')
sheet1.write(0, 4, 'Val_Acc')
sheet1.write(0, 5, 'lr')
sheet1.write(0, 6, 'Best val Acc')


def main(args):
    # 获取GPU设备
    if torch.cuda.is_available():  # 如果有GPU就用，没有就用CPU
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')


    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path,val_rate=0.2)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),# 数据增强，随机裁剪224*224大小
                                     transforms.RandomHorizontalFlip(),# 数据增强，随机水平翻转
                                     transforms.ToTensor(), # 数据变成tensor类型，像素值归一化，调整维度[h,w,c]==>[c,h,w]
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),# 对每个通道的像素进行标准化，给出每个通道的均值和方差
        "val": transforms.Compose([transforms.Resize(224),
                                   transforms.RandomResizedCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=32,#默认32
                                               shuffle=True,# 打乱每个batch
                                               num_workers=0) #加载数据时的线程数量，windows环境下只能=0

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=32,
                                             shuffle=False,
                                             num_workers=0)
    date="2023-0807-1846"
    model_name = "EdgeViT_XXS_2_without_pretrained_driver100_data_batch_size_32"+date#pretrained_state_farm_data
    if True:#"pretrained"==True:
        #model=creat_covit(10).to(device)
        #model=MobileNetV3_Large(num_classes=10).to(device)
        #model=lightvit_tiny(num_classes=10).to(device)
        #model=build_model().to(device)
        #model=ShuffleNetV2(num_classes=10).to(device)
        #model=DenseNet121().to(device)
        #model=resnet34(pretrained=False).to(device)
        #model = resnet50(pretrained=True,num_classes=2).to(device)
        #model =creat_vit_token_learn(10).to(device)
        #model = ViT_token_learn(image_size=224,num_tokens=99,fuse=False,v11=True,tokenlearner_loc=3,patch_size=16,hidden_size=768,depth=6,heads=16,mlp_dim=2048,dropout=0.1,emb_dropout=0.1).to(device)
        #model=nextvit_small(10).to(device)
        #model = biformer_tiny(num_classes=10).to(device)
        #model =creat_co_td_vit_plus_20230510(10).to(device)
        #model = resnet18(pretrained=False).to(device)
        #model =SwiftFormer_S(num_classes=6).to(device)
        #model = pvt_v2_b0(num_classes=10).to(device)
        #model = EfficientViT(num_classes=10).to(device)
        #model = resnet50(num_classes=10).to(device)
        #model = effnetv2_s(num_classes=10).to(device)
        #model = ghostnetv2().to(device)
        #model = SwiftFormer_XS(num_classes=10).to(device)
        model = EdgeViT_XXS(num_classes=8).to(device)
        #model = convnextv2_atto(num_classes=10).to(device)
        #model = mobile_vit_x_small(num_classes=10).to(device)
        #model = EfficientFormer(layers=[3, 2, 6, 4],embed_dims=[48, 96, 224, 448],downsamples=[True, True, True, True],vit_num=1,num_classes=10).to(device)
        #model = make_model('resnet152', num_classes=10, pretrained=True, input_size=(224, 224)).to(device)

    else:
        model = make_model('resnet152', num_classes=10, pretrained=True, input_size=(224, 224)).to(device)
        #model.load_state_dict(torch.load("out_weight/%s.pth"%model_name))
        model.load_state_dict(torch.load("out_weight/resnet152_out_pretrained_aucd_data_batch_size_322023-0721-1746.pth"))
        # in_features=model.mlp_head[1].in_features
        # model.mlp_head[1]=nn.Linear(in_features, 8)
        model.cuda()

    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)


    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_acc=0.05

    for epoch in range(args.epochs):

        sheet1.write(epoch + 1, 0, epoch + 1)
        sheet1.write(epoch + 1, 5, str(optimizer.state_dict()['param_groups'][0]['lr']))

        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        sheet1.write(epoch + 1, 1, str(train_loss))
        sheet1.write(epoch + 1, 2, str(train_acc))

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        sheet1.write(epoch + 1, 3, str(val_loss))
        sheet1.write(epoch + 1, 4, str(val_acc))


        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "out_weight/%s.pth"%model_name)
            # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))

        sheet1.write(1, 6, str(best_acc))
        f1.save('out_excel/%s.xls'%model_name)
        print("The Best Acc = : {:.4f}".format(best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes',type=int, default=10)
    parser.add_argument('--epochs',type=int, default=250)
    parser.add_argument('--batch_size',type=int, default=32)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--lrf',type=float,default=0.01)
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default=r"D:\all_data_sets\driver_distraction_public_datasets\driver_100_day\Driver_small\Train")#/state-farm-distracted-driver-detection/imgs/train
    parser.add_argument('--model-name', default='my_vit', help='create model name')
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights',type=str,default='vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers',type=bool,default=False)
    parser.add_argument('--device',default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)