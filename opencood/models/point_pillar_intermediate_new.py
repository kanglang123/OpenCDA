# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
# import nbimporter
# from opencood.models.sub_modules.att import AttBEVBackbone
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.fuse_modules.f_cooper_fuse import SpatialFusion
import numpy as np
import matplotlib.pyplot as plt     # 可视化

class PointPillarIntermediateNew(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediateNew, self).__init__()

        # PIllar VFE  划分成体素
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])   #映射成pseudo images
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.fusion_net = SpatialFusion()

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_number'],kernel_size=1)

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        # 封装成字典类型
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        # 网络处理点云数据
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # fusion
        fused_feature = self.fusion_net(spatial_features_2d,record_len)
        
        # 检测头
        psm = self.cls_head(fused_feature)
        rm = self.reg_head(fused_feature)
        
        # 打包检测结果
        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict