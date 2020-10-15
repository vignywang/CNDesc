import torch
from torch import nn


class AffineChannelwise(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

        self.register_parameter('weight', nn.Parameter(torch.ones(num_channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(num_channels)))

    def forward(self, x):
        param_shape = [1] * len(x.shape)
        param_shape[1] = self.num_channels
        return x * self.weight.reshape(*param_shape) + self.bias.reshape(*param_shape)
