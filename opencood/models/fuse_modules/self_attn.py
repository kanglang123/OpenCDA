# -*- coding: utf-8 -*-
# Author: Hao Xiang <haxiang@g.ucla.edu>, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt     # 可视化
from torch.nn import Parameter
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"缩放点积注意力
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection
          vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection
          vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the
          encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from
          attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the
          encoder outputs.
    """

    def __init__(self, dim):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, query, key, value):
        query = self.q(query)
        key   = self.k(key)
        value = self.v(value)
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim # torch.Size([35200, 3, 3])
        attn = F.softmax(score, -1) # torch.Size([35200, 3, 3]) *  v torch.Size([35200, 3, 64])
        context = torch.bmm(attn, value) #torch.Size([35200, 3, 64])   # 是在这里做的融合

        context = self.gamma*context + value

        return context


class AttFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x, record_len):
        split_x = self.regroup(x, record_len) # 分batch
        C, W, H = split_x[0].shape[1:]
        out = []
        for xx in split_x:
            # # feature_map 
            # feature_map_viewer1 = xx.detach().cpu().numpy() 
            # m1 = feature_map_viewer1.shape[0]
            
            # fig1 = plt.figure()
            # for q in range(m1):
            #   plt.subplot(m1, 1, q+1)
            #   p = feature_map_viewer1[1]
            #   pic = np.sum(p,axis=0)
            #   plt.imshow(pic)
            #   plt.title('feature_map_viewer1[%d]' %q)
            # plt.suptitle('before attention fuse')
            cav_num = xx.shape[0]
            xx = xx.view(cav_num, C, -1).permute(2, 0, 1) # 现在是每个车分别的点云数据，没有融合
            h = self.att(xx, xx, xx)  # torch.Size([35200, 4, 64])
            # h = h.permute(1, 2, 0).view(cav_num, C, W, H)[0, ...]
            h = h.permute(1, 2, 0).view(cav_num, C, W, H)   # torch.Size([4, 64, 100, 352])
            h = h[0, ...]   # torch.Size([64, 100, 352])  切片只选择第一维度的第一个元素
            out.append(h)
        return torch.stack(out) # 对序列数据内部的张量进行扩维拼接

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
