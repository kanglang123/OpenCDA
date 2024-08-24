"""
Implementation of kanglang fusion.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from opencood.models.fuse_modules.self_attn import ScaledDotProductAttention

# result:
# The Average Precision at IOU 0.3 is 0.86, 
# The Average Precision at IOU 0.5 is 0.84, 
# The Average Precision at IOU 0.7 is 0.72.

class Communication(nn.Module):     # 交流模块
    def __init__(self, args):
        super(Communication, self).__init__()
        # Threshold of objectiveness  物体的阈值
        self.threshold = args['threshold']
        if 'gaussian_smooth' in args:
            # Gaussian Smooth
            self.smooth = True
            kernel_size = args['gaussian_smooth']['k_size']
            c_sigma = args['gaussian_smooth']['c_sigma']
            self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
            self.init_gaussian_filter(kernel_size, c_sigma)
            self.gaussian_filter.requires_grad = False
        else:
            self.smooth = False

    def init_gaussian_filter(self, k_size=5, sigma=1.0):
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))

        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()

    def forward(self, batch_confidence_maps, B):
        """
        Args:
            batch_confidence_maps: [(L1, H, W), (L2, H, W), ...]
        """
        communication_masks = []      
        for b in range(B):
            ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)#?
            if self.smooth:
                communication_maps = self.gaussian_filter(ori_communication_maps)
            else:
                communication_maps = ori_communication_maps
            communication_mask = communication_maps.to(communication_maps.device)
            communication_mask[0] = 1                                # Ego 
            communication_masks.append(communication_mask)
        communication_masks = torch.cat(communication_masks, dim=0)  # 张量拼接
        return communication_masks


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dim)

    def forward(self, x):
        cav_num, C, H, W = x.shape
        x = x.view(cav_num, C, -1).permute(2, 0, 1)       # (H*W, cav_num, C), perform self attention on each pixel
        x = self.att(x, x, x)
        x = x.permute(1, 2, 0).view(cav_num, C, H, W)[0]  # C, W, H before
        return x


class Where2comm(nn.Module):
    def __init__(self, args):
        super(Where2comm, self).__init__()
        self.discrete_ratio = args['voxel_size'][0]
        self.downsample_rate = args['downsample_rate']

        self.fully = args['fully']

        self.multi_scale = args['multi_scale']
        if self.multi_scale:
            layer_nums = args['layer_nums']     # [ 3, 5, 8 ]
            num_filters = args['num_filters']   # [ 64, 128, 256 ]
            self.num_levels = len(layer_nums)
            self.fuse_modules = nn.ModuleList() # 一系列的神经网络
            for idx in range(self.num_levels):  # 遍历每一层 每个pixel使用自注意力机制
                fuse_network = AttentionFusion(num_filters[idx])
                self.fuse_modules.append(fuse_network)
        else:
            self.fuse_modules = AttentionFusion(args['in_channels'])

        self.naive_communication = Communication(args['communication'])

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)  # 按照次序累计，类似累积分布函数的作用
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, x, psm_single, record_len, backbone=None):
        """
        Fusion forwarding.

        Parameters:
            x: Input data, (sum(n_cav), C, H, W).
            record_len: List, (B).
            pairwise_t_matrix: The transformation matrix from each cav to ego, (B, L, L, 4, 4).

        Returns:
            Fused feature.
        """

        B = len(record_len)
        ups = []
        for i in range(self.num_levels):  # 一共三层
            x = backbone.blocks[i](x)
            # 1. Communication (模拟mask the features)
            batch_confidence_maps = self.regroup(psm_single, record_len)
            communication_masks = self.naive_communication(batch_confidence_maps, B)
            # Down/up samples the input to either the given size or the given scale_factor
            if x.shape[-1] != communication_masks.shape[-1]:
                communication_masks = F.interpolate(communication_masks, size=(x.shape[-2], x.shape[-1]),
                                                    mode='bilinear', align_corners=False) # 插值
            
            x = x * communication_masks +x          # 特征增强

            # 2. Split the features
            # split_x:    [(L1, C,   H, W),   (L2, C,   H,   W), ...]   For example [[2, 256, 48, 176], [1, 256, 48, 176], ...]
            batch_node_features = self.regroup(x, record_len)

            # 3. Fusion
            x_fuse = []
            for b in range(B):
                neighbor_feature = batch_node_features[b]
                x_fuse.append(self.fuse_modules[i](neighbor_feature))
            x_fuse = torch.stack(x_fuse)

            # 4. Deconv
            if len(backbone.deblocks) > 0:
                ups.append(backbone.deblocks[i](x_fuse))
            else:
                ups.append(x_fuse)
        if len(ups) > 1:
            x_fuse = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x_fuse = ups[0]

        if len(backbone.deblocks) > self.num_levels:
            x_fuse = backbone.deblocks[-1](x_fuse)
        return x_fuse