#
# Created  on 2019/8/14
#
import os
from glob import glob

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as f
class Consistent_loss(object):
    def __init__(self, ):
        self.device = 1
        self.margin = 1
        self.L1=torch.nn.L1Loss()
    def _compute_dist(self, X, Y):
        """
        批量计算两类描述子之间的距离
        :param X: [bt,n,dim]
        :param Y: [bt,m,dim]
        :return: [bt,n,m]
        """
        XTX = torch.pow(X, 2).sum(dim=2)  # [bt,n]
        YTY = torch.pow(Y, 2).sum(dim=2)  # [bt,m]
        XTY = torch.bmm(X, Y.transpose(1, 2))

        dist2 = XTX.unsqueeze(dim=2) - 2 * XTY + YTY.unsqueeze(dim=1)  # [bt,n,m]
        dist = torch.sqrt(torch.clamp(dist2, 1e-5))
        return dist

    def __call__(self, desp_0, desp_1, valid_mask, not_search_mask):
        """
        Args:
            desp_0: [bt,n,dim]
            desp_1: [bt,n,dim]
            valid_mask: [bt,n] 1有效，0无效
            not_search_mask: [bt,n,n]
        Returns:
            loss
        """
        mask = torch.abs(not_search_mask - 1)
        dist0 = self._compute_dist(desp_0, desp_0)
        dist0 = dist0 * mask

        dist1 = self._compute_dist(desp_1, desp_1)
        dist1 = dist1 * mask

        loss = self.L1(dist0,dist1)

        return loss

class DescriptorTripletLoss_E(object):

    def __init__(self, device, margin):
        self.device = device
        self.margin = margin

    def _compute_dist(self, X, Y):
        """
        批量计算两类描述子之间的距离
        :param X: [bt,n,dim]
        :param Y: [bt,m,dim]
        :return: [bt,n,m]
        """
        XTX = torch.pow(X, 2).sum(dim=2)  # [bt,n]
        YTY = torch.pow(Y, 2).sum(dim=2)  # [bt,m]
        XTY = torch.bmm(X, Y.transpose(1, 2))

        dist2 = XTX.unsqueeze(dim=2) - 2 * XTY + YTY.unsqueeze(dim=1)  # [bt,n,m]
        dist = torch.sqrt(torch.clamp(dist2, 1e-5))
        return dist

    def __call__(self, desp_0, desp_1, valid_mask, not_search_mask):
        """
        Args:
            desp_0: [bt,n,dim]
            desp_1: [bt,n,dim]
            valid_mask: [bt,n] 1有效，0无效
            not_search_mask: [bt,n,n]
        Returns:
            loss
        """
        dist = self._compute_dist(desp_0, desp_1)
        positive_pair = torch.diagonal(dist, dim1=1, dim2=2)  # [bt,n]
        dist = dist + 10 * not_search_mask
        hardest_negative_pair, hardest_negative_idx = torch.min(dist, dim=2)  # [bt,n]
        loss_total = torch.relu(self.margin + positive_pair - hardest_negative_pair)
        loss_total *= valid_mask
        valid_num = torch.sum(valid_mask, dim=1)
        loss = torch.mean(torch.sum(loss_total, dim=1) / (valid_num + 1.))
        return loss
    def __init__(self, device,margin):
        self.device = device
        self.margin = margin

    def _compute_dist(self, X, Y):
        """
        批量计算两类描述子之间的距离
        :param X: [bt,n,dim]
        :param Y: [bt,m,dim]
        :return: [bt,n,m]
        """
        XTX = torch.pow(X, 2).sum(dim=2)  # [bt,n]
        YTY = torch.pow(Y, 2).sum(dim=2)  # [bt,m]
        XTY = torch.bmm(X, Y.transpose(1, 2))

        dist2 = XTX.unsqueeze(dim=2) - 2 * XTY + YTY.unsqueeze(dim=1)  # [bt,n,m]
        dist = torch.sqrt(torch.clamp(dist2, 1e-5))
        return dist

    def __call__(self, desp_0, desp_1,valid_mask, not_search_mask):
        """
        Args:
            desp_0: [bt,n,dim]
            desp_1: [bt,n,dim]
            valid_mask: [bt,n] 1有效，0无效
            not_search_mask: [bt,n,n]
        Returns:
            loss
        """
        dist = self._compute_dist(desp_0,desp_1)
        positive_pair = torch.diagonal(dist, dim1=1, dim2=2)  # [bt,n]
        dist = dist + 10*not_search_mask
        hardest_negative_pair, hardest_negative_idx = torch.min(dist, dim=2)  # [bt,n]
        loss_total = torch.relu(self.margin+positive_pair-hardest_negative_pair)
        loss_total *= valid_mask
        valid_num = torch.sum(valid_mask, dim=1)
        loss = torch.mean(torch.sum(loss_total, dim=1)/(valid_num + 1.))
        return loss

def generate_testing_file(folder, prefix="model"):
    models = glob(os.path.join(folder, prefix + "_*.pt"))
    models = sorted(models)
    return models


def compute_batched_dist(x, y, hamming=False):
    # x:[bt,256,n], y:[bt,256,n]
    cos_similarity = torch.matmul(x.transpose(1, 2), y)  # [bt,n,n]
    if hamming is False:
        square_norm_x = (torch.norm(x, dim=1, keepdim=True).transpose(1, 2))**2  # [bt,n,1]
        square_norm_y = (torch.norm(y, dim=1, keepdim=True))**2  # [bt,1,n]
        dist = torch.sqrt((square_norm_x + square_norm_y - 2 * cos_similarity + 1e-4))
        return dist
    else:
        dist = 0.5*(256-cos_similarity)
    return dist


def compute_cos_similarity_general(x, y):
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
    x = x.div(x_norm+1e-4)
    y = y.div(y_norm+1e-4)
    cos_similarity = torch.matmul(x.transpose(1, 2), y)  # [bt,h*w,h*w]
    return cos_similarity


def compute_cos_similarity_binary(x, y, k=256):
    x = x.div(np.sqrt(k))
    y = y.div(np.sqrt(k))
    cos_similarity = torch.matmul(x.transpose(1, 2), y)
    return cos_similarity


def spatial_nms(prob, kernel_size=9):
    padding = int(kernel_size//2)
    pooled = f.max_pool2d(prob, kernel_size=kernel_size, stride=1, padding=padding)
    prob = torch.where(torch.eq(prob, pooled), prob, torch.zeros_like(prob))
    return prob


