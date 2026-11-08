# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import logging

import torch
import torch.nn as nn
from .DCNv2.dcn_v2 import DCN
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1


class StripConvBlock(nn.Module):
  """
  条形卷积增强模块：
  使用 1×7 和 7×1 卷积增强细长目标的方向性特征表达。
  """
  def __init__(self, channels):
    super(StripConvBlock, self).__init__()
    mid = max(channels // 2, 32)

    self.reduce = nn.Sequential(
      nn.Conv2d(channels, mid, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(mid, momentum=BN_MOMENTUM),
      nn.ReLU(inplace=True)
    )

    self.conv_h = nn.Sequential(
      nn.Conv2d(mid, mid, kernel_size=(1, 7), stride=1, padding=(0, 3), bias=False),
      nn.BatchNorm2d(mid, momentum=BN_MOMENTUM),
      nn.ReLU(inplace=True)
    )

    self.conv_v = nn.Sequential(
      nn.Conv2d(mid, mid, kernel_size=(7, 1), stride=1, padding=(3, 0), bias=False),
      nn.BatchNorm2d(mid, momentum=BN_MOMENTUM),
      nn.ReLU(inplace=True)
    )

    self.fuse = nn.Sequential(
      nn.Conv2d(mid * 2, channels, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(channels, momentum=BN_MOMENTUM)
    )

    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    y = self.reduce(x)
    y_h = self.conv_h(y)
    y_v = self.conv_v(y)
    y = self.fuse(torch.cat([y_h, y_v], dim=1))
    return self.relu(x + y)


class SmallObjectFeaturePyramid(nn.Module):
  """
  小目标特征增强金字塔：
  通过多尺度卷积分支增强小目标特征响应。
  """
  def __init__(self, channels):
    super(SmallObjectFeaturePyramid, self).__init__()

    self.branch1 = nn.Sequential(
      nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
      nn.ReLU(inplace=True)
    )

    self.branch2 = nn.Sequential(
      nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
      nn.ReLU(inplace=True)
    )

    self.branch3 = nn.Sequential(
      nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
      nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
      nn.ReLU(inplace=True)
    )

    self.branch4 = nn.Sequential(
      nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=3, dilation=3, bias=False),
      nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
      nn.ReLU(inplace=True)
    )

    self.fuse = nn.Sequential(
      nn.Conv2d(channels * 4, channels, kernel_size=1, stride=1, padding=0, bias=False),
      nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    y1 = self.branch1(x)
    y2 = self.branch2(x)
    y3 = self.branch3(x)
    y4 = self.branch4(x)
    y = self.fuse(torch.cat([y1, y2, y3, y4], dim=1))
    return x + y


class MicroEvidenceEnhance(nn.Module):
  """
  条形卷积 + 小目标特征增强金字塔组合模块。
  """
  def __init__(self, channels=256):
    super(MicroEvidenceEnhance, self).__init__()
    self.strip = StripConvBlock(channels)
    self.pyramid = SmallObjectFeaturePyramid(channels)
    self.out = nn.Sequential(
      nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
      nn.BatchNorm2d(channels, momentum=BN_MOMENTUM),
      nn.ReLU(inplace=True)
    )

  def forward(self, x):
    x = self.strip(x)
    x = self.pyramid(x)
    x = self.out(x)
    return x



# ================================================================= #
# 创新点一：空间注意力机制 (Spatial Attention Module)
# 用于强化微小目标（毛发、纸片）的边缘特征响应，抑制桌面背景反光噪声
# ================================================================= #
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 使用 7x7 大卷积核，专门感知细长毛发的局部空间分布
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 沿通道维度分别求平均值和最大值，逼出物体轮廓
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 2. 拼接后通过卷积生成 0~1 的权重蒙版 (Mask)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        weight_map = self.sigmoid(self.conv1(x_cat))
        # 3. 将蒙版乘回原特征图：点亮物证，按灭背景
        return x * weight_map
# ================================================================= #
logger = logging.getLogger(__name__)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv):
        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
        )

        # ================================================================= #
        # 创新点一：特征融合层声明 (Feature Fusion Layers)
        # layer1 输出是 64 通道，deconv3 输出是 256 通道。拼在一起是 320 通道。
        # 我们用一个 1x1 卷积把它们重新压回 256 通道，无缝对接 CenterNet 原生的检测头。
        # ================================================================= #
        self.fusion_conv = nn.Conv2d(64 + 64, 64, kernel_size=1, bias=False)
        self.fusion_bn = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.fusion_relu = nn.ReLU(inplace=True)
        self.spatial_attention = SpatialAttention(kernel_size=7)
        # ================================================================= #

        self.micro_enhance = MicroEvidenceEnhance(512)

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                  nn.Conv2d(64, head_conv,
                    kernel_size=3, padding=1, bias=True),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(head_conv, classes, 
                    kernel_size=1, stride=1, 
                    padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(64, classes, 
                  kernel_size=1, stride=1, 
                  padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes, 
                    kernel_size=(3,3), stride=1,
                    padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1, 
            #         padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            layers.append(fc)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            layers.append(up)
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        # 1. 基础特征提取
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 2. 截获浅层特征 (F_low)：保留了毛发的细微边缘，没被严重压缩
        f_low = self.layer1(x)  # 这里的通道数是 64
        
        # 3. 继续深层特征提取
        x = self.layer2(f_low)
        x = self.layer3(x)
        x = self.layer4(x)

        # 4. 生成深层语义特征 (F_high)：先增强再反卷积上采样
        x = self.micro_enhance(x)              # ★ 修复: 先用条形卷积+金字塔增强深层特征
        f_high = self.deconv_layers(x)          # ★ 修复: 再用增强后的x算f_high

        # ================================================================= #
        # 创新点一：多尺度跳跃连接与注意力强化
        # 公式: F_fused = Conv(Concat(F_high, F_low)) ⊗ Attention
        # ================================================================= #
        # 动作 A：强行拼在一起 (Concat) -> 变成 320 通道
        f_fused = torch.cat([f_low, f_high], dim=1) 
        
        # 动作 B：特征降维与融合 (Conv + BN + ReLU) -> 压回 256 通道
        f_fused = self.fusion_conv(f_fused)
        f_fused = self.fusion_bn(f_fused)
        f_fused = self.fusion_relu(f_fused)
        
        # 动作 C：盖上注意力蒙版，抑制桌面划痕噪声
        f_fused = self.spatial_attention(f_fused)
        # ================================================================= #

        # 5. 送入检测头进行最终预测
        ret = {}
        for head in self.heads:
            # 注意：这里把原本的 x 换成了我们千锤百炼的 f_fused
            ret[head] = self.__getattr__(head)(f_fused) 
        return [ret]

    def init_weights(self, num_layers):
        if not os.environ.get('CENTERNET_NO_PRETRAINED'):
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> CENTERNET_NO_PRETRAINED set, skip backbone pretrained '
                  '(train from scratch)')
        print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv=256):
  block_class, layers = resnet_spec[num_layers]

  model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
  model.init_weights(num_layers)
  return model
