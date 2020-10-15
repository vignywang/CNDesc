import numpy as np
import torch
import tqdm
from torch import nn
import torch.nn.functional as f
from .affine import AffineChannelwise
from .utils import allclose_or_none


class MyInstanceNorm2d(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5, affine=False, track_running_stats=False):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        if affine:
            self.affine = AffineChannelwise(num_features)
        else:
            self.affine = None

        self.track_running_stats = track_running_stats
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        assert len(x.shape) == 4
        b, c, h, w = x.shape

        if self.training or not self.track_running_stats:
            # All dims except for B and C
            mu = x.mean(dim=(2, 3))
            sigma = x.var(dim=(2, 3), unbiased=False)
        else:
            mu, sigma = self.running_mean, self.running_var
            b = 1

        if self.training and self.track_running_stats:
            sigma_unbiased = sigma * ((h * w) / ((h * w) - 1))
            self.running_mean = self.running_mean * (1 - self.momentum) + mu.mean(dim=0) * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + sigma_unbiased.mean(dim=0) * self.momentum

        mu = mu.reshape(b, c, 1, 1)
        sigma = sigma.reshape(b, c, 1, 1)
        result = (x - mu) / torch.sqrt(sigma + self.eps)

        if self.affine is not None:
            result = self.affine(result)

        return result


import torch


# x is the features of shape [B, C, H, W]

# In the Encoder
class PositionwiseNorm(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        #self.conv1=nn.Conv2d(1,1,kernel_size=7,stride=1,padding=3)
        #self.conv2 = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)
    def forward(self,x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True).add(self.epsilon).sqrt()
        output = (x - mean) / std
        map = torch.mean(x,dim=1, keepdim=True)
        #map1=self.conv1(map)
        #map2=self.conv2(map)
        return output #*map1+map2

class PositionwiseNorm2(nn.Module):
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.conv1=nn.Conv2d(1,1,kernel_size=7,stride=1,padding=3)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=7, stride=1, padding=3)
    def forward(self,x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True).add(self.epsilon).sqrt()
        output = (x - mean) / std
        map = torch.mean(x,dim=1, keepdim=True)
        map1=self.conv1(map)
        map2=self.conv2(map)
        return output*map1+map2

# In the Decoder
# one can call MS(x, mean, std)
# with the mean and std are from a PONO in the encoder
def MS(x, beta, gamma):
    return x * gamma + beta

class ChannelwiseNorm(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5, affine=False, track_running_stats=False):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        if affine:
            self.affine = Adaffine(num_features)
        else:
            self.affine = None

        self.track_running_stats = track_running_stats
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        assert len(x.shape) == 4
        b, c, h, w = x.shape

        if self.training or not self.track_running_stats:
            # All dims except for B and C
            mu = x.mean(dim=(2, 3))
            sigma = x.var(dim=(2, 3), unbiased=False)
        else:
            mu, sigma = self.running_mean, self.running_var
            b = 1

        if self.training and self.track_running_stats:
            sigma_unbiased = sigma * ((h * w) / ((h * w) - 1))
            self.running_mean = self.running_mean * (1 - self.momentum) + mu.mean(dim=0) * self.momentum
            self.running_var = self.running_var * (1 - self.momentum) + sigma_unbiased.mean(dim=0) * self.momentum

        mu = mu.reshape(b, c, 1, 1)
        sigma = sigma.reshape(b, c, 1, 1)
        result = (x - mu) / torch.sqrt(sigma + self.eps)

        if self.affine is not None:
            result = self.affine(result)

        return result


class Adaffine(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Linear(128,64)
        self.relu1 = nn.ReLU()
        self.fc2=nn.Linear(64,128)
        self.fc3 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Linear(64, 128)

    def forward(self, result,x):
        avg_out1 = self.fc2(self.relu1(self.fc1(self.avg_pool(x).squeeze(-1).squeeze(-1)))).unsqueeze(-1).unsqueeze(-1)
        avg_out2 = self.fc4(self.relu2(self.fc3(self.avg_pool(x).squeeze(-1).squeeze(-1)))).unsqueeze(-1).unsqueeze(-1)
        return result * avg_out1 +avg_out2

class Adaffine2(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels,64, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(64, num_channels, 1, bias=False)
        self.fc3 = nn.Conv2d(num_channels, 64, 1, bias=False)
        self.relu2 = nn.ReLU()
        self.fc4 = nn.Conv2d(64, num_channels, 1, bias=False)

        #self.register_parameter('weight', nn.Parameter(torch.ones(num_channels)))
        #self.register_parameter('bias', nn.Parameter(torch.zeros(num_channels)))

    def forward(self, result,x):
        avg_out1 = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        avg_out2 = self.fc4(self.relu2(self.fc3(self.avg_pool(x))))

        #param_shape = [1] * len(x.shape)
        #param_shape[1] = self.num_channels
        #print(result.shape)
        #print(*param_shape)
        #exit(0)
        return result * avg_out1 +avg_out2


class Adaffine3(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        k_size=3
        self.num_channels = num_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        #self.register_parameter('weight', nn.Parameter(torch.ones(num_channels)))
        #self.register_parameter('bias', nn.Parameter(torch.zeros(num_channels)))

    def forward(self, result,x):
        avg_out1 =  self.conv1(self.avg_pool(x).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        avg_out2 =  self.conv2(self.avg_pool(x).squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        #param_shape = [1] * len(x.shape)
        #param_shape[1] = self.num_channels
        #print(result.shape)
        #print(*param_shape)
        #exit(0)
        return result * avg_out1.expand_as(x) +avg_out2.expand_as(x)


class Adaffine4(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        k_size=3
        self.num_channels = num_channels
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        #self.register_parameter('weight', nn.Parameter(torch.ones(num_channels)))
        #self.register_parameter('bias', nn.Parameter(torch.zeros(num_channels)))

    def forward(self, result,mu,sigma):
        mu_l= self.conv1(mu.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        sigma_l= self.conv2(sigma.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        sigma_l= f.softplus(sigma_l)
        #param_shape = [1] * len(x.shape)
        #param_shape[1] = self.num_channels
        #print(result.shape)
        #print(*param_shape)
        #exit(0)

        return result * sigma_l.expand_as(result) + mu_l.expand_as(result)

class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)