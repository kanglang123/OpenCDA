# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
# import nbimporter
# from opencood.models.sub_modules.att import AttBEVBackbone
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
import numpy as np
import matplotlib.pyplot as plt     # 可视化

class PointPillarIntermediate(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediate, self).__init__()

        # PIllar VFE  划分成体素
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])   #映射成pseudo images
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
                                  kernel_size=1)

    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self, data_dict):

        # voxel_features = data_dict['processed_lidar']['voxel_features']
        # voxel_coords = data_dict['processed_lidar']['voxel_coords']
        # voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        voxel_features = data_dict['voxel_features']
        voxel_coords = data_dict['voxel_coords']
        voxel_num_points = data_dict['voxel_num_points']
        record_len = data_dict['record_len']
        # plan_trajectory = data_dict['plan_trajectory']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len,
                    #   'plan_trajectory':plan_trajectory
                        }
        # # feature_map 自注意力之后的可视化
        # feature_map_viewer1 = plan_trajectory[0][0].squeeze().detach().cpu().numpy() 
        # fig1 = plt.figure()
        # plt.imshow(feature_map_viewer1)
        # plt.colorbar()

        # 网络处理点云数据
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        # TODO 按照轨迹添加掩膜
        mask = 0
        spatial_features_2d = batch_dict['spatial_features_2d']
        
        # a = plan_trajectory[0][0].squeeze()*batch_dict['spatial_features'][0][0]
        # fig2 = plt.figure()
        # plt.imshow((a.detach().cpu().numpy()))
        # plt.colorbar()
        # plt.title('feature_map_viewer')
        
        # 检测头
        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)
        # 打包检测结果
        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict