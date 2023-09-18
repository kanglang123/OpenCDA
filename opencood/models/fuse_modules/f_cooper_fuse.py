# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Implementation of F-cooper maxout fusing.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt     

#一个派生类SpatialFusion     格式：class 派生类名(基类名)  其中的基类就是nn.Module
class SpatialFusion(nn.Module):   #空间融合 
    def __init__(self):
        super(SpatialFusion, self).__init__()
    
    #此函数功能是什么？
    def regroup(self, x, record_len):  #重新组合  record_len是什么？ [3,2]
        cum_sum_len = torch.cumsum(record_len, dim=0)  #从第0维按位置累加
        a= cum_sum_len[:-1].cpu()      #d=a[:-1]  #从位置0到位置-1之前的数
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())  #对x沿着第cpu维度进行分块
        return split_x
                                                                                                                                                                                                                                                                                              
    def forward(self, x, record_len):
        # x: B, C, H, W, split x:[(B1, C, W, H), (B2, C, W, H)]
        split_x = self.regroup(x, record_len)
        out = []
        
        #不同的通道共享不同的权重，即特征图中的某些通道对分类/检测的贡献更大，而其他通道则是冗余或不需要的。
        #因此，从所有 128个通道中选择部分通道进行传输，传输部分通道可以进一步减少传输的时间消耗。
        
        for xx in split_x:  #Python for循环可以遍历任何序列的项目，如一个列表或者一个字符串。
            xx = torch.max(xx, dim=0, keepdim=True)[0]
            out.append(xx)
        return torch.cat(out, dim=0)   #按照第0维度，也就是行，往下排，拼接到一起