import torch
from torch import nn
import torch.nn.functional as f


import math
from nets.dl_norms.instance_norm import PositionwiseNorm,ChannelwiseNorm,PositionwiseNorm2
class EFRBlock(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, ratio=2,stride=1, relu=True):
        super(EFRBlock, self).__init__()
        self.inp = inp
        self.oup = oup
        self.mid= int(inp/2)
        self.group=2
        init_channels = math.ceil(inp / ratio)
        self.first_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2,groups=self.group, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.second_conv = nn.Sequential(
            nn.Conv2d(inp+init_channels, init_channels, kernel_size, stride, kernel_size // 2,groups=self.group, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(inp+init_channels+init_channels,  self.oup, kernel_size, stride, kernel_size // 2, bias=False),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.first_conv(x)
        f1 = torch.cat([x, x1], dim=1)
        f1 = self.channel_shuffle(f1)
        x2 = self.second_conv(f1)
        f2 = torch.cat([x, x1,x2], dim=1)
        f2 = self.channel_shuffle(f2)
        out =  self.final_conv(f2)
        return out

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x
class EFRBackbone(nn.Module):
    def __init__(self):
        super(EFRBackbone, self).__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_first = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = EFRBlock(32, 64)

        self.conv2_0 = EFRBlock(64, 64)

        self.conv3_0 = EFRBlock(64, 128)

        self.conv4_0 = EFRBlock(128, 128)

        self.PN = PositionwiseNorm2()
        self.CN = ChannelwiseNorm(128)
        self.des = nn.Conv2d((64 + 64 + 128 + 128), 128, kernel_size=3, stride=1, padding=1)
        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_1.data.fill_(0.7)
        self.fuse_weight_2.data.fill_(0.3)

    def forward(self, x):
        e0 = self.conv1_first(x)  #
        e0 = self.relu(e0)
        c1 = self.conv1_1(e0)  # 64
        e1 = self.pool(c1)  #

        c2 = self.conv2_0(e1)  # 1/2
        e3 = self.pool(c2)

        c3 = self.conv3_0(e3)# 1/4
        e5 = self.pool(c3)

        c4 = self.conv4_0(e5) # 1/8

        # Descriptor
        des_size = c2.shape[2:]  # 1/2 HxW
        c1 = f.interpolate(c1, des_size, mode='bilinear')
        c3 = f.interpolate(c3, des_size, mode='bilinear')
        c4 = f.interpolate(c4, des_size, mode='bilinear')
        feature = torch.cat((c1, c2, c3, c4), dim=1)
        descriptor = self.des(feature)
        descriptor1 = self.PN(descriptor)
        descriptor2 = self.CN(descriptor)
        descriptor = descriptor1 * (self.fuse_weight_1/(self.fuse_weight_1+self.fuse_weight_2)) + descriptor2 * (self.fuse_weight_2/(self.fuse_weight_1+self.fuse_weight_2))

        return descriptor
