from functools import partial
import torch.nn as nn
import numpy as np
from ptflops import get_model_complexity_info
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from torch.nn import init
from einops import rearrange, repeat
import argparse


class External_MSA_Attention(nn.Module):

    def __init__(self, d_model,S=64,HEAD=8,att_inputsize=3136):
        super().__init__()
        self.input_dim=d_model/HEAD
        self.mk=nn.Linear(int(self.input_dim),S,bias=True)
        self.mv=nn.Linear(S,int(self.input_dim),bias=True)
        self.softmax=nn.Softmax(dim=1)
        self.HEAD=HEAD
        # self.att_inputsize = int(att_inputsize ** (1/2))
        # totalpixel = self.att_inputsize * self.att_inputsize
        # gauss_coords_h = torch.arange(totalpixel) - int((totalpixel - 1) / 2)
        # gauss_coords_w = torch.arange(totalpixel) - int((totalpixel - 1) / 2)
        # gauss_x, gauss_y = torch.meshgrid([gauss_coords_h, gauss_coords_w])
        # sigma = 10
        # gauss_pos_index = torch.exp(torch.true_divide(-(gauss_x ** 2 + gauss_y ** 2), (2 * sigma ** 2)))
        # self.register_buffer("gauss_pos_index", gauss_pos_index)

    def forward(self, queries):
        B, N, C = queries.shape
        queries = queries.view(B, N, self.HEAD, C//self.HEAD)
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        #absolute_pos_bias = self.gauss_pos_index.unsqueeze(0)
        #attn = attn + absolute_pos_bias.unsqueeze(0)
        attn=attn/torch.sum(attn,dim=3,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model
        out=out.permute(0,2,1,3)
        out=out.reshape(B,N,C)
        return out



class DCM2_B(nn.Module):
    """
    Local Representation module for SwiftFormer that is implemented by 3*3 depth-wise and point-wise convolutions.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, dim, kernel_size=3, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = self.drop_path(self.layer_scale * x)+input
        else:
            x =  self.drop_path(x)+input
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class LCAM_B(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(LCAM_B, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        # self.conv1 = nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0, groups=inp)
        # self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, groups=inp)
        self.conv3 = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, groups=inp)
        #self.relu = h_swish()

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)

        x = x.reshape(B, H, W, C)
        x = x.permute(0, 3, 1, 2)

        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()

        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        y = y.permute(0, 2, 3, 1)
        y = y.reshape(B, N, C)

        return y



class PatchEmbedding(nn.Module):
    def __init__(self, inter_channel=32, out_channels=48, img_size=None):
        self.img_size = img_size
        self.inter_channel = inter_channel
        self.out_channel = out_channels
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, inter_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channel, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv3(self.conv2(self.conv1(x)))
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return x

    def flops(self, ) -> float:
        flops = 0
        flops += 3 * self.inter_channel * self.img_size[0] * self.img_size[1] // 4 * 9
        flops += self.img_size[0] * self.img_size[1] // 4 * self.inter_channel
        flops += self.inter_channel * self.out_channel * self.img_size[0] * self.img_size[1] // 16 * 9
        flops += self.img_size[0] * self.img_size[1] // 16 * self.out_channel
        flops += self.out_channel * self.out_channel * self.img_size[0] * self.img_size[1] // 16
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MSDC_BLOCK(nn.Module):
    def __init__(self, img_size, kernel_size, downsample_ratio, dilations, in_chans, embed_dim,
                 share_weights, op,NC_group):
        super().__init__()
        self.dilations = dilations
        self.embed_dim = embed_dim
        self.downsample_ratio = downsample_ratio
        self.op = op
        self.kernel_size = kernel_size
        self.stride = downsample_ratio
        self.share_weights = share_weights
        self.outSize = img_size // downsample_ratio

        print(in_chans,embed_dim)

        if share_weights:
            self.convolution = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=self.kernel_size, \
                                         stride=self.stride, padding=3 * dilations[0] // 2, dilation=dilations[0])

        else:
            self.convs = nn.ModuleList()
            for dilation in self.dilations:
                padding = math.ceil(((self.kernel_size - 1) * dilation + 1 - self.stride) / 2)
                self.convs.append(nn.Sequential(
                    *[nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=self.kernel_size, stride=self.stride, padding=padding, dilation=dilation,groups=NC_group),#
                      #nn.BatchNorm2d(embed_dim),
                      nn.GELU()]))

        if self.op == 'sum':
            self.out_chans = embed_dim
        elif op == 'cat':
            self.out_chans = embed_dim * len(self.dilations)

    def forward(self, x):
        B, C, W, H = x.shape
        if self.share_weights:
            padding = math.ceil(((self.kernel_size - 1) * self.dilations[0] + 1 - self.stride) / 2)
            y = nn.functional.conv2d(x, weight=self.convolution.weight, bias=self.convolution.bias, \
                                     stride=self.downsample_ratio, padding=padding,
                                     dilation=self.dilations[0]).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                padding = math.ceil(((self.kernel_size - 1) * self.dilations[i] + 1 - self.stride) / 2)
                _y = nn.functional.conv2d(x, weight=self.convolution.weight, bias=self.convolution.bias, \
                                          stride=self.downsample_ratio, padding=padding,
                                          dilation=self.dilations[i]).unsqueeze(dim=-1)
                y = torch.cat((y, _y), dim=-1)
        else:
            y = self.convs[0](x).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                _y = self.convs[i](x).unsqueeze(dim=-1)
                y = torch.cat((y, _y), dim=-1)
        B, C, W, H, N = y.shape
        if self.op == 'sum':
            y = y.sum(dim=-1).flatten(2).permute(0, 2, 1).contiguous()
        elif self.op == 'cat':
            y = y.permute(0, 4, 1, 2, 3).flatten(3).reshape(B, N * C, W * H).permute(0, 2, 1).contiguous()
        else:
            raise NotImplementedError('no such operation: {} for multi-levels!'.format(self.op))
        return y, (W, H)


class Conv_block(nn.Module):
    def __init__(self, img_size ,in_chans, embed_dims, token_dims, downsample_ratios, kernel_size,
                 num_heads, dilations, share_weights, op, tokens_type, group,NC_group,
                 relative_pos, window_size):
        super().__init__()
        self.img_size = img_size
        self.window_size = window_size
        self.op = op
        self.tokens_type = tokens_type
        self.dilations = dilations
        self.num_heads = num_heads
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.in_chans = in_chans
        self.downsample_ratios = downsample_ratios
        self.kernel_size = kernel_size
        self.outSize = img_size
        self.relative_pos = relative_pos
        PCMStride = []
        residual = downsample_ratios // 2
        for _ in range(3):
            PCMStride.append((residual > 0) + 1)
            residual = residual // 2
        assert residual == 0
        self.pool = None
        self.DCM1 = nn.Sequential(
            nn.Conv2d(in_chans, embed_dims, kernel_size=(3, 3), stride=PCMStride[0], padding=(1, 1), groups=group),
            #nn.Conv2d(in_chans, embed_dims, kernel_size=(1,1), bias=False),
            # the 1st convolution
            nn.BatchNorm2d(embed_dims),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dims, embed_dims, kernel_size=(3, 3), stride=PCMStride[1], padding=(1, 1), groups=group),
            # the 1st convolution
            nn.BatchNorm2d(embed_dims),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dims, token_dims, kernel_size=(3, 3), stride=PCMStride[2], padding=(1, 1), groups=group),#
            nn.BatchNorm2d(token_dims)
            #nn.Conv2d(embed_dims, token_dims, kernel_size=(1,1), bias=False),
            # the 1st convolution
        )

        self.MSDC = MSDC_BLOCK(img_size=img_size, kernel_size=kernel_size, downsample_ratio=downsample_ratios,
                               dilations=self.dilations,
                               in_chans=in_chans, embed_dim=embed_dims, share_weights=share_weights, op=op, NC_group=NC_group)
        self.outSize = self.outSize // downsample_ratios
        self.num_patches = (img_size // 2) * (img_size // 2)  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        if len(x.shape) < 4:
            B, N, C = x.shape
            n = int(np.sqrt(N))
            x = x.view(B, n, n, C).contiguous()
            x = x.permute(0, 3, 1, 2)
        if self.pool is not None:
            x = self.pool(x)
        shortcut = x
        PRM_x, _ = self.MSDC(x)
        convX = self.DCM1(shortcut)  # PCM
        convX = convX.permute(0, 2, 3, 1).view(*PRM_x.shape).contiguous()
        x = PRM_x + convX
        return x


class Transformer_block(nn.Module):
    def __init__(self, dim, num_heads,NC_group, mlp_ratio, drop, drop_path, act_layer, norm_layer,
                 class_token, window_size, r,img_size):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.class_token = class_token
        self.img_size = img_size
        self.window_size = window_size
        self.r=r
        self.attn=LCAM_B(dim, dim)


        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.DCM2 = DCM2_B(dim, kernel_size=3, drop_path=0., use_layer_scale=True)

    def forward(self, x):
        b, n, c = x.shape
        shortcut = x

        x=self.norm1(x)
        x = self.attn(x)

        wh = int(math.sqrt(n))
        convX = self.drop_path2(self.DCM2(shortcut.view(b, wh, wh, c).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().view(b, n, c))

        x = shortcut  + self.drop_path2(x)+ convX
        x = x + self.drop_path2(self.mlp2(self.norm3(x)))

        return x


class BasicLayer(nn.Module):
    def __init__(self, img_size, in_chans, embed_dims, token_dims, downsample_ratios, kernel_size,
                 RC_heads, NC_heads, dilations,
                 RC_op, RC_tokens_type, RC_group, NC_group,
                 NC_depth, dpr,r, mlp_ratio,
                 drop, norm_layer, class_token, window_size,
                 relative_pos):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.embed_dims = embed_dims
        self.token_dims = token_dims
        self.downsample_ratios = downsample_ratios
        self.out_size = self.img_size // self.downsample_ratios
        self.RC_kernel_size = kernel_size
        self.RC_heads = RC_heads
        self.NC_heads = NC_heads
        self.dilations = dilations
        self.RC_op = RC_op
        self.RC_tokens_type = RC_tokens_type
        self.RC_group = RC_group
        self.NC_group = NC_group
        self.NC_depth = NC_depth
        self.r=r
        self.relative_pos = relative_pos
        if RC_tokens_type == 'stem':
            self.RC = PatchEmbedding(inter_channel=token_dims // 2, out_channels=token_dims, img_size=img_size)
        elif downsample_ratios > 1:
            self.RC = Conv_block(img_size, in_chans, embed_dims, token_dims, downsample_ratios, kernel_size,
                                 RC_heads, dilations, op=RC_op, tokens_type=RC_tokens_type, group=RC_group,NC_group=NC_group,
                                 relative_pos=relative_pos)
        else:
            self.RC = nn.Identity()
        self.NC = nn.ModuleList([
            Transformer_block(token_dims, NC_heads,  NC_group,mlp_ratio=mlp_ratio, drop=drop,
                              drop_path=dpr[i] if isinstance(dpr, list) else dpr, norm_layer=norm_layer,
                              class_token=class_token,
                              img_size=img_size // downsample_ratios, window_size=window_size,r=r)
            for i in range(NC_depth)])

    def forward(self, x):
        x = self.RC(x)
        for nc in self.NC:
            x = nc(x)
        return x


class Covit(nn.Module):
    def __init__(self, img_size, in_chans, stages, embed_dims ,token_dims,
                 downsample_ratios, kernel_size,
                 CNN_heads, Transformer_heads, dilations,
                 CNN_op, CNN_tokens_type,
                 Transformer_tokens_type,
                 CNN__group, Transformer_group, Transformer_depth, r, mlp_ratio, qkv_bias,
                 qk_scale, drop_rate,
                 attn_drop_rate, drop_path_rate, norm_layer, num_classes,
                 window_size, relative_pos):
        super().__init__()
        self.num_classes = num_classes
        self.stages = stages
        repeatOrNot = (lambda x, y, z=list: x if isinstance(x, z) else [x for _ in range(y)])
        self.embed_dims = repeatOrNot(embed_dims, stages)
        self.tokens_dims = token_dims if isinstance(token_dims, list) else [token_dims * (2 ** i) for i in
                                                                            range(stages)]
        self.downsample_ratios = repeatOrNot(downsample_ratios, stages)
        self.kernel_size = repeatOrNot(kernel_size, stages)
        self.CNN_heads = repeatOrNot(CNN_heads, stages)
        self.Transformer_heads = repeatOrNot(Transformer_heads, stages)
        self.dilaions = repeatOrNot(dilations, stages)
        self.CNN_op = repeatOrNot(CNN_op, stages)
        self.CNN_tokens_type = repeatOrNot(CNN_tokens_type, stages)
        self.Transformer_tokens_type = repeatOrNot(Transformer_tokens_type, stages)
        self.CNN_group = repeatOrNot(CNN__group, stages)
        self.Transformer_group = repeatOrNot(Transformer_group, stages)
        self.Transformer_depth = repeatOrNot(Transformer_depth, stages)
        self.mlp_ratio = repeatOrNot(mlp_ratio, stages)
        self.qkv_bias = repeatOrNot(qkv_bias, stages)
        self.qk_scale = repeatOrNot(qk_scale, stages)
        self.drop = repeatOrNot(drop_rate, stages)
        self.attn_drop = repeatOrNot(attn_drop_rate, stages)
        self.norm_layer = repeatOrNot(norm_layer, stages)
        self.relative_pos = relative_pos
        self.r = repeatOrNot(r, stages)

        self.pos_drop = nn.Dropout(p=drop_rate)
        depth = np.sum(self.Transformer_depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        Layers = []
        for i in range(stages):
            startDpr = 0 if i == 0 else self.Transformer_depth[i - 1]
            Layers.append(
                BasicLayer(img_size, in_chans, self.embed_dims[i], self.tokens_dims[i], self.downsample_ratios[i],
                           self.kernel_size[i], self.CNN_heads[i], self.Transformer_heads[i], self.dilaions[i], self.CNN_op[i],
                           self.CNN_tokens_type[i], self.Transformer_tokens_type[i], self.CNN_group[i], self.Transformer_group[i],
                           self.Transformer_depth[i], dpr[startDpr:self.Transformer_depth[i] + startDpr],
                           r=self.r[i],
                           mlp_ratio=self.mlp_ratio[i], qkv_bias=self.qkv_bias[i], qk_scale=self.qk_scale[i],
                           drop=self.drop[i], attn_drop=self.attn_drop[i],
                           norm_layer=self.norm_layer[i], window_size=window_size, relative_pos=relative_pos)
            )
            img_size = img_size // self.downsample_ratios[i]
            in_chans = self.tokens_dims[i]
        self.layers = nn.ModuleList(Layers)

        # Classifier head
        self.head = nn.Linear(self.tokens_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.mean(x, 1)
        x = self.head(x)
        return x


