'''
@file: bdc_modele.py
@author: Fei Long
@author: Jiaming Lv
Please cite the paper below if you use the code:

Jiangtao Xie, Fei Long, Jiaming Lv, Qilong Wang and Peihua Li. Joint Distribution Matters: Deep Brownian Distance Covariance for Few-Shot Classification. IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR), 2022.

Copyright (C) 2022 Fei Long and Jiaming Lv

All rights reserved.
'''

import torch
import torch.nn as nn

class BDC(nn.Module):
    def __init__(self, is_vec=True, input_dim=640, dimension_reduction=None, activate='relu'):
        super(BDC, self).__init__()
        self.is_vec = is_vec
        self.dr = dimension_reduction
        self.activate = activate
        self.input_dim = input_dim[0]
        if self.dr is not None and self.dr != self.input_dim:
            if activate == 'relu':
                self.act = nn.ReLU(inplace=True)
            elif activate == 'leaky_relu':
                self.act = nn.LeakyReLU(0.1)
            else:
                self.act = nn.ReLU(inplace=True)

            self.conv_dr_block = nn.Sequential(
            nn.Conv2d(self.input_dim, self.dr, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.dr),
            self.act
            )
        output_dim = self.dr if self.dr else self.input_dim
        if self.is_vec:
            self.output_dim = int(output_dim*(output_dim+1)/2)
        else:
            self.output_dim = int(output_dim*output_dim)

        self.temperature = nn.Parameter(torch.log((1. / (2 * input_dim[1]*input_dim[2])) * torch.ones(1,1)), requires_grad=True)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        if self.dr is not None and self.dr != self.input_dim:
            x = self.conv_dr_block(x)
        x = BDCovpool(x, self.temperature)
        if self.is_vec:
            x = Triuvec(x)
        else:
            x = x.reshape(x.shape[0], -1)
        return x

#BDCovpool函数是一个用于计算Bilinear Deep Convolutional Pooling的函数。
# 它接受两个参数：x和t。其中，x是一个四维张量，其形状为(batchSize, dim, h, w)，
# 表示输入特征图的大小。t是一个标量，表示温度参数。
#在函数中，我们首先获取输入特征图的大小，并计算特征图的像素数M。
# 然后，我们将输入特征图x重塑为(batchSize, dim, M)的形状。
# 接着，我们定义了一个单位矩阵I和一个全1矩阵I_M，并将它们移动到GPU上。
#后，我们计算了x的二阶协方差矩阵dcov。
# 具体来说，我们首先计算了x的平方和x_pow2，
# 然后使用I_M和I对x_pow2进行了一些操作，最后得到了dcov。
# 在计算dcov时，我们使用了torch.clamp函数将其限制在0和正无穷之间，
# 并使用torch.exp函数对其进行了指数变换。
# 接着，我们使用torch.sqrt函数对dcov进行开方，并对其加上一个很小的常数，以避免出现除以0的情况。
#最后，我们使用dcov和一些常数计算了温度参数t，并将其返回
def BDCovpool(x, t):
    batchSize, dim, h, w = x.data.shape
    M = h * w
    x = x.reshape(batchSize, dim, M)

    I = torch.eye(dim, dim, device=x.device).view(1, dim, dim).repeat(batchSize, 1, 1).type(x.dtype)
    I_M = torch.ones(batchSize, dim, dim, device=x.device).type(x.dtype)
    x_pow2 = x.bmm(x.transpose(1, 2))
    dcov = I_M.bmm(x_pow2 * I) + (x_pow2 * I).bmm(I_M) - 2 * x_pow2
    
    dcov = torch.clamp(dcov, min=0.0)
    dcov = torch.exp(t)* dcov
    dcov = torch.sqrt(dcov + 1e-5)
    t = dcov - 1. / dim * dcov.bmm(I_M) - 1. / dim * I_M.bmm(dcov) + 1. / (dim * dim) * I_M.bmm(dcov).bmm(I_M)

    return t

#Triuvec函数实现了将上三角矩阵转换为向量的操作。该函数接受一个三维张量x作为输入，其形状为(batchSize, dim, dim)，
# 表示输入的上三角矩阵的大小。在函数中，我们首先获取输入矩阵的大小，并将其重塑为(batchSize, dim * dim)的形状。
# 然后，我们定义了一个单位上三角矩阵I，并将其重塑为一维张量。接着，我们使用nonzero函数获取I中非零元素的索引，并将其保存在index变量中。
# 然后，我们定义了一个全零张量y，其形状为(batchSize, int(dim * (dim + 1) / 2))，用于保存转换后的向量。
# 最后，我们使用index将r中对应的元素提取出来，并将其保存在y中。最终，函数返回转换后的向量y
def Triuvec(x):
    batchSize, dim, dim = x.shape
    r = x.reshape(batchSize, dim * dim)
    I = torch.ones(dim, dim).triu().reshape(dim * dim)
    index = I.nonzero(as_tuple = False)
    y = torch.zeros(batchSize, int(dim * (dim + 1) / 2), device=x.device).type(x.dtype)
    y = r[:, index].squeeze()
    return y