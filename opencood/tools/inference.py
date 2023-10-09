# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
import matplotlib.pyplot as plt
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset

def inference_code(opt,objects,search_nearby_cav_data):
    hypes = yaml_utils.load_yaml(None, opt)
    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)

    if torch.cuda.is_available():    # we assume gpu is necessary
        model.cuda()    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    record_len = len(search_nearby_cav_data)
    lidar_data = []
    processed_features = []
    for id,data in search_nearby_cav_data.items():
        # TODO 
        # 多车数据的处理（多车层级的封装，tensor的统一转换、到设备） 解决了
        # 点云位置的转移需不需要在特征图层面做改动？！！！！有问题需要再改
        processed_lidar = IntermediateFusionDataset.get_item_single_car(opencood_dataset,data) 
        # 前处理，包含了点云的映射到ego车的坐标系下，以及点云的体素化

        # 每个车的点云都是单独处理的，所以需要在这里进行拼接
        processed_features.append(processed_lidar['processed_features'])

    merged_feature_dict = opencood_dataset.merge_features_to_dict(processed_features)   # 将多个车的特征拼接到一个字典中
    processed_lidar_torch_dict = opencood_dataset.pre_processor.collate_batch(merged_feature_dict) # 将字典转换为tensor
    processed_lidar_torch_dict['record_len'] = torch.tensor([record_len])
    processed_lidar_torch_dict = train_utils.to_device(processed_lidar_torch_dict, device)

    output_dict = model(processed_lidar_torch_dict)   # 模型推理

    anchor_box = opencood_dataset.post_processor.generate_anchor_box()
    anchor_box_tensor = torch.from_numpy(anchor_box)  
    anchor_box_tensor = train_utils.to_device(anchor_box_tensor, device)

    # 包装车的信息(多级字典)
    cars_data = { 'ego':{'processed_lidar': processed_lidar_torch_dict,
                        'anchor_box': anchor_box_tensor,
                        'transformation_matrix': train_utils.to_device(torch.from_numpy(np.identity(4,dtype=np.float32)), device),
                        # 'record_len': record_len,
                        # 'plan_trajectory':plan_trajectory}
                        }
                }
    pred_box_tensor, pred_score = opencood_dataset.post_processor.post_process(cars_data, output_dict)
    objects = {'pred_box_tensor':pred_box_tensor, 
               'pred_score':pred_score}
    # TODO：结果的检查和封装
    return objects