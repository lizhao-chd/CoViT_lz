import torch
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from cnn_finetune import make_model
from tools import read_split_data, train_one_epoch, evaluate
from my_dataset import MyDataSet
import math
import argparse
from thop import profile
from ptflops import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch.optim.lr_scheduler as lr_scheduler
from models_lz.cnn_based import DenseNet
import timm
from torch.nn import Linear
from pytorch_pretrained_vit import ViT
import xlwt
from models_lz.lz_models.co_td_vit_lz import creat_co_td_vit
from models_lz.lz_models.co_vit_lz import creat_co_vit
from models_lz.lz_models.td_vit_lz import creat_td_vit
from models_lz.lz_models.vit_lz import creat_large_vit,creat_small_vit
from models_lz.lz_models.swin_transformer_lz import SwinTransformer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context



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

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

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
                                               batch_size=32,
                                               shuffle=True,# 打乱每个batch
                                               num_workers=0) #加载数据时的线程数量，windows环境下只能=0

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=0)

    date="20230602"
    model_name = "efficientnet_b0_with_pretrained_our_driver_gaze_data20230602"
    #model = make_model('inception_v4', num_classes=10, pretrained=True,input_size=(224, 224))
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

    model = timm.create_model('inception_v4', num_classes=2,pretrained=False).to(device)
    # ['adv_inception_v3', 'cspdarknet53', 'cspresnet50', 'cspresnext50', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'densenetblur121d',
    # 'dla34', 'dla46_c', 'dla46x_c', 'dla60', 'dla60_res2net', 'dla60_res2next', 'dla60x', 'dla60x_c', 'dla102', 'dla102x', 'dla102x2', 'dla169', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131',
    # 'ecaresnet50d', 'ecaresnet50d_pruned', 'ecaresnet101d', 'ecaresnet101d_pruned', 'ecaresnetlight', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b1_pruned', 'efficientnet_b2', 'efficientnet_b2_pruned', 'efficientnet_b2a', 'efficientnet_b3', 'efficientnet_b3_pruned', 'efficientnet_b3a', 'efficientnet_em', 'efficientnet_es', 'efficientnet_lite0',
    # 'ens_adv_inception_resnet_v2', 'ese_vovnet19b_dw', 'ese_vovnet39b', 'fbnetc_100',
    # 'gluon_inception_v3', 'gluon_resnet18_v1b', 'gluon_resnet34_v1b', 'gluon_resnet50_v1b', 'gluon_resnet50_v1c', 'gluon_resnet50_v1d', 'gluon_resnet50_v1s', 'gluon_resnet101_v1b', 'gluon_resnet101_v1c', 'gluon_resnet101_v1d', 'gluon_resnet101_v1s', 'gluon_resnet152_v1b', 'gluon_resnet152_v1c', 'gluon_resnet152_v1d', 'gluon_resnet152_v1s', 'gluon_resnext50_32x4d', 'gluon_resnext101_32x4d',
    # 'gluon_resnext101_64x4d', 'gluon_senet154', 'gluon_seresnext50_32x4d', 'gluon_seresnext101_32x4d', 'gluon_seresnext101_64x4d', 'gluon_xception65', 'hrnet_w18', 'hrnet_w18_small', 'hrnet_w18_small_v2', 'hrnet_w30', 'hrnet_w32', 'hrnet_w40', 'hrnet_w44', 'hrnet_w48', 'hrnet_w64', 'ig_resnext101_32x8d', 'ig_resnext101_32x16d', 'ig_resnext101_32x32d', 'ig_resnext101_32x48d', 'inception_resnet_v2', 'inception_v3', 'inception_v4', 'legacy_senet154', 'legacy_seresnet18', 'legacy_seresnet34', 'legacy_seresnet50', 'legacy_seresnet101', 'legacy_seresnet152', 'legacy_seresnext26_32x4d', 'legacy_seresnext50_32x4d', 'legacy_seresnext101_32x4d', 'mixnet_l', 'mixnet_m', 'mixnet_s', 'mixnet_xl', 'mnasnet_100', 'mobilenetv2_100', 'mobilenetv2_110d', 'mobilenetv2_120d', 'mobilenetv2_140', 'mobilenetv3_large_100', 'mobilenetv3_rw', 'nasnetalarge', 'pnasnet5large', 'regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008', 'regnetx_016', 'regnetx_032', 'regnetx_040', 'regnetx_064', 'regnetx_080', 'regnetx_120', 'regnetx_160', 'regnetx_320', 'regnety_002', 'regnety_004', 'regnety_006', 'regnety_008', 'regnety_016', 'regnety_032', 'regnety_040', 'regnety_064', 'regnety_080', 'regnety_120', 'regnety_160', 'regnety_320', 'res2net50_14w_8s', 'res2net50_26w_4s', 'res2net50_26w_6s', 'res2net50_26w_8s', 'res2net50_48w_2s', 'res2net101_26w_4s', 'res2next50', 'resnest14d', 'resnest26d', 'resnest50d', 'resnest50d_1s4x24d', 'resnest50d_4s2x40d', 'resnest101e', 'resnest200e', 'resnest269e', 'resnet18', 'resnet18d', 'resnet26', 'resnet26d', 'resnet34', 'resnet34d', 'resnet50', 'resnet50d', 'resnetblur50', 'resnext50_32x4d', 'resnext50d_32x4d', 'resnext101_32x8d', 'rexnet_100', 'rexnet_130', 'rexnet_150', 'rexnet_200', 'selecsls42b', 'selecsls60', 'selecsls60b', 'semnasnet_100', 'seresnet50', 'seresnext26d_32x4d', 'seresnext26t_32x4d', 'seresnext26tn_32x4d', 'seresnext50_32x4d', 'skresnet18', 'skresnet34', 'skresnext50_32x4d', 'spnasnet_100', 'ssl_resnet18', 'ssl_resnet50', 'ssl_resnext50_32x4d', 'ssl_resnext101_32x4d', 'ssl_resnext101_32x8d', 'ssl_resnext101_32x16d', 'swsl_resnet18', 'swsl_resnet50', 'swsl_resnext50_32x4d', 'swsl_resnext101_32x4d', 'swsl_resnext101_32x8d', 'swsl_resnext101_32x16d', 'tf_efficientnet_b0', 'tf_efficientnet_b0_ap', 'tf_efficientnet_b0_ns', 'tf_efficientnet_b1', 'tf_efficientnet_b1_ap', 'tf_efficientnet_b1_ns', 'tf_efficientnet_b2', 'tf_efficientnet_b2_ap', 'tf_efficientnet_b2_ns', 'tf_efficientnet_b3', 'tf_efficientnet_b3_ap', 'tf_efficientnet_b3_ns', 'tf_efficientnet_b4', 'tf_efficientnet_b4_ap', 'tf_efficientnet_b4_ns', 'tf_efficientnet_b5', 'tf_efficientnet_b5_ap', 'tf_efficientnet_b5_ns', 'tf_efficientnet_b6', 'tf_efficientnet_b6_ap', 'tf_efficientnet_b6_ns', 'tf_efficientnet_b7', 'tf_efficientnet_b7_ap', 'tf_efficientnet_b7_ns', 'tf_efficientnet_b8', 'tf_efficientnet_b8_ap', 'tf_efficientnet_cc_b0_4e', 'tf_efficientnet_cc_b0_8e', 'tf_efficientnet_cc_b1_8e', 'tf_efficientnet_el', 'tf_efficientnet_em', 'tf_efficientnet_es', 'tf_efficientnet_l2_ns', 'tf_efficientnet_l2_ns_475', 'tf_efficientnet_lite0', 'tf_efficientnet_lite1', 'tf_efficientnet_lite2', 'tf_efficientnet_lite3', 'tf_efficientnet_lite4', 'tf_inception_v3', 'tf_mixnet_l', 'tf_mixnet_m', 'tf_mixnet_s', 'tf_mobilenetv3_large_075', 'tf_mobilenetv3_large_100', 'tf_mobilenetv3_large_minimal_100', 'tf_mobilenetv3_small_075', 'tf_mobilenetv3_small_100', 'tf_mobilenetv3_small_minimal_100', 'tresnet_l', 'tresnet_l_448', 'tresnet_m', 'tresnet_m_448', 'tresnet_xl', 'tresnet_xl_448', 'tv_densenet121', 'tv_resnet34', 'tv_resnet50', 'tv_resnet101', 'tv_resnet152', 'tv_resnext50_32x4d', 'vit_base_patch16_224', 'vit_base_patch16_384', 'vit_base_patch32_384', 'vit_large_patch16_224', 'vit_large_patch16_384', 'vit_large_patch32_384', 'vit_small_patch16_224', 'wide_resnet50_2', 'wide_resnet101_2', 'xception', 'xception41', 'xception65', 'xception71']


    #-------------------》》》》》vit模型预训练
    # model = ViT('B_16', pretrained=False)#C:\Users\TTT/.cache\torch\hub\checkpoints\L_16_imagenet1k.pth
    # fc_features=model.fc.in_features
    # model.fc = nn.Linear(fc_features, 8)#B_16, B_32, L_16, L_32, B_16_imagenet1k, B_32_imagenet1k, L_16_imagenet1k, L_32_imagenet1k



    #--------------------》》》》》模型参数量计算
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('Flops:  ' + flops)
    print('Params: ' + params)

    #------------------>>>>尝试加载参数
    try:
        model.load_state_dict(torch.load("out_weight/%s.pth"%model_name))
        model.cuda()
    except:
        pass


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


        sheet1.write(1, 6, str(best_acc))
        f1.save('out_excel/%s.xls'%model_name)
        print("The Best Acc = : {:.4f}".format(best_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str,
                        default=r"D:\all_data_sets\driver_gaze_dataset\face_img\face")#/state-farm-distracted-driver-detection/imgs/train
    parser.add_argument('--model-name', default='my_vit', help='create model name')
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    opt = parser.parse_args()
    main(opt)