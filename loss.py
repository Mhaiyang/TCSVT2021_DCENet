"""
 @Time    : 2021/7/8 09:48
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : TCSVT2021_DCENet
 @File    : loss.py
 @Function: Loss Functions
 
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse

###################################################################
# ########################## iou loss #############################
###################################################################
def _iou(pred, target):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        Iand1 = torch.sum(target[i,:,:,:] * pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:]) - Iand1
        if Ior1:
            IoU1 = Iand1 / Ior1
        else:
            IoU1 = 1
        IoU = IoU + (1-IoU1)
    return IoU / b

class IOU(torch.nn.Module):
    def __init__(self):
        super(IOU, self).__init__()

    def forward(self, pred, target):
        return _iou(pred, target)

###################################################################
# ########################## edge loss ############################
###################################################################
def cross_entropy(logits, labels):
    return torch.mean((1 - labels) * logits + torch.log(1 + torch.exp(-logits)))

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        laplace = torch.FloatTensor([[-1,-1,-1,],[-1,8,-1],[-1,-1,-1]]).view([1,1,3,3])
        # filter shape in Pytorch: out_channel, in_channel, height, width
        self.laplace = nn.Parameter(data=laplace, requires_grad=False)
    def torchLaplace(self, x):
        edge = F.conv2d(x, self.laplace, padding=1)
        edge = torch.abs(torch.tanh(edge))
        return edge
    def forward(self, y_pred, y_true, mode=None):
        y_true_edge = self.torchLaplace(y_true)
        y_pred_edge = self.torchLaplace(y_pred)
        edge_loss = cross_entropy(y_pred_edge, y_true_edge)
        return edge_loss
