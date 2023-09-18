# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Implementation of F-cooper maxout fusing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#一个派生类SpatialFusion     格式：class 派生类名(基类名)  其中的基类就是nn.Module
class CrossAttFusion(nn.Module):   #交叉注意力融合 
    def __init__(self,query_dim, context_dim):
        super(CrossAttFusion, self).__init__()

        self.query_dim = query_dim
        self.context_dim = context_dim

        self.linear_q = nn.Linear(query_dim, query_dim)
        self.linear_c = nn.Linear(context_dim, query_dim)

    #此函数功能是将每个batch中的数据分开
    def regroup(self, x, record_len):  #重新组合  record_len是什么？ [3,2]
        cum_sum_len = torch.cumsum(record_len, dim=0)  #从第0维按位置累加
        a= cum_sum_len[:-1].cpu()      #d=a[:-1]  #从位置0到位置-1之前的数
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())  #对x沿着第cpu维度进行分块
        return split_x
                                                                                                                                                                                                                                                                                              
    def forward(self, x, record_len,query, context):
        # x: B, C, H, W, split x:[(CAR_NUM1, 384, W, H), (CAR_NUM2, C, W, H)]
        split_x = self.regroup(x, record_len)
        out = []
        
        #不同的通道共享不同的权重，即特征图中的某些通道对分类/检测的贡献更大，而其他通道则是冗余或不需要的。
        #因此，从所有 128个通道中选择部分通道进行传输，传输部分通道可以进一步减少传输的时间消耗。
        
        for xx in split_x:  #Python for循环可以遍历任何序列的项目，如一个列表或者一个字符串。
            xx = torch.max(xx, dim=0, keepdim=True)[0]
            out.append(xx)
        return torch.cat(out, dim=0)   #按照第0维度，也就是行，往下排，拼接到一起
    
        # Query和Context的维度分别为 [batch_size, query_len, query_dim] 和 [batch_size, context_len, context_dim]
        # 首先将Query和Context分别通过线性变换
        query_proj = self.linear_q(query)  # [batch_size, query_len, query_dim]
        context_proj = self.linear_c(context)  # [batch_size, context_len, query_dim]

        # 计算注意力权重
        attention_weights = torch.bmm(query_proj, context_proj.transpose(1, 2))  # [batch_size, query_len, context_len]
        attention_weights = F.softmax(attention_weights, dim=-1)

        # 对Context序列进行加权求和
        attended_context = torch.bmm(attention_weights, context)  # [batch_size, query_len, context_dim]

        return torch.cat(out, dim=0),attended_context, attention_weights #按照第0维度，也就是行，往下排，拼接到一起


class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim1, feature_dim2):
        super(CrossAttentionFusion, self).__init__()
        self.feature_dim1 = feature_dim1
        self.feature_dim2 = feature_dim2
        self.attention_fc1 = nn.Sequential(
            nn.Linear(feature_dim1, feature_dim1 // 2),
            nn.ReLU(),
            nn.Linear(feature_dim1 // 2, 1)
        )
        self.attention_fc2 = nn.Sequential(
            nn.Linear(feature_dim2, feature_dim2 // 2),
            nn.ReLU(),
            nn.Linear(feature_dim2 // 2, 1)
        )
   
    def forward(self, feature_map1, feature_map2):
        # Compute attention weights based on feature_map2
        attention_weights1 = self.attention_fc1(feature_map2)
        attention_weights1 = F.softmax(attention_weights1, dim=1)
       
        # Compute attention weights based on feature_map1
        attention_weights2 = self.attention_fc2(feature_map1)
        attention_weights2 = F.softmax(attention_weights2, dim=1)
       
        # Weighted sum of feature_map2 using attention weights from feature_map1
        weighted_feature_map2 = torch.matmul(attention_weights1.transpose(1, 2), feature_map2)
       
        # Weighted sum of feature_map1 using attention weights from feature_map2
        weighted_feature_map1 = torch.matmul(attention_weights2.transpose(1, 2), feature_map1)
       
        # Concatenate and return the fused feature maps
        fused_feature_map1 = torch.cat((feature_map1, weighted_feature_map2), dim=-1)
        fused_feature_map2 = torch.cat((feature_map2, weighted_feature_map1), dim=-1)
        return fused_feature_map1, fused_feature_map2

# Example usage:
batch_size = 8
num_points1 = 1024
num_points2 = 512
feature_dim1 = 128
feature_dim2 = 256

# Generate random point cloud features for demonstration purposes
feature_map1 = torch.randn(batch_size, num_points1, feature_dim1)
feature_map2 = torch.randn(batch_size, num_points2, feature_dim2)

# Initialize the cross-attention fusion module
cross_attention_fusion = CrossAttentionFusion(feature_dim1, feature_dim2)

# Perform cross-attention fusion
fused_feature_map1, fused_feature_map2 = cross_attention_fusion(feature_map1, feature_map2)
print(fused_feature_map1.shape)  # Output shape: (batch_size, num_points1, feature_dim1*2)
print(fused_feature_map2.shape)  # Output shape: (batch_size, num_points2, feature_dim2*2)

    
        
        