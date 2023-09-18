# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone


class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(args['base_bev_backbone'], 64)

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
                                  kernel_size=1)

    def forward(self, data_dict):

        voxel_features = data_dict['processed_lidar']['voxel_features']     #torch.Size([31227, 32, 4])
        voxel_coords = data_dict['processed_lidar']['voxel_coords']         #torch.Size([31227, 4])
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points'] #torch.Size([31227])

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)  #add pillar_features      torch.Size([28310, 64])  \4
        batch_dict = self.scatter(batch_dict)     #add spatial_features     torch.Size([2, 64, 200, 704]) \5
        batch_dict = self.backbone(batch_dict)    #add spatial_features_2d  torch.Size([2, 384, 100, 352]) \6

        spatial_features_2d = batch_dict['spatial_features_2d']             #torch.Size([2, 384, 100, 352])
        
        # feature_map 可视化
        feature_map_viewer = torch.squeeze(batch_dict['spatial_features'])
        feature_map_viewer = feature_map_viewer.detach().cpu().numpy() 
        #detach(): 返回一个新的Tensor，但返回的结果是没有梯度的。
        #cpu():把gpu上的数据转到cpu上。
        # numpy():将tensor格式转为numpy。
        for i in range(0, 10):
            if i < 10 :
                feature_map = feature_map_viewer[1,i,:,:]
            else:
                feature_map = feature_map_viewer[1,i,:,:]
            plt.subplot(5, 2, i+1)
            plt.imshow(feature_map)
        plt.show()

        psm = self.cls_head(spatial_features_2d)  #torch.Size([2, 2, 100, 352])
        rm = self.reg_head(spatial_features_2d)   #torch.Size([2, 14, 100, 352])

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict