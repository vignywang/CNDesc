import numpy as np
import torch
import tqdm
from torch import nn

from .affine import AffineChannelwise
from .utils import allclose_or_none


class MyGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        assert num_channels % num_groups == 0
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        if affine:
            self.affine = AffineChannelwise(num_channels)
        else:
            self.affine = None

    def forward(self, x):
        assert len(x.shape) == 4
        b, c, h, w = x.shape
        assert c == self.num_channels  # not really needed unless we use affine
        g = c // self.num_groups

        # All dims except B; in addition, C gets special treatment.
        x = x.reshape(b, self.num_groups, g, h, w)
        mu = x.mean(dim=(2, 3, 4), keepdim=True)
        sigma = x.var(dim=(2, 3, 4), unbiased=False, keepdim=True)

        result = (x - mu) / torch.sqrt(sigma + self.eps)
        result = result.reshape(b, c, h, w)

        if self.affine is not None:
            result = self.affine(result)

        return result

