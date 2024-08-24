# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib

from tqdm import tqdm
import numpy as np
import torch
import opencood.hypes_yaml.yaml_utils as yaml_utils
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from opencood.visualization.vis_utils import bbx2oabb
from opencda.core.sensing.perception.o3d_lidar_libs import o3d_lidar

def inference_code(opt,objects,perception_data,lidar_sensor):
    hypes = yaml_utils.load_yaml(None, opt)                                 # 读取配置文件
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)    # 构建数据集
    print('opencood_dataset 加载完成 !')

    model = train_utils.create_model(hypes)                                 # 构建模型
    if torch.cuda.is_available():                                           # 将模型转移到cuda上
        model.cuda()    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设置设备
    saved_path = opt.model_dir                                             
    _, model = train_utils.load_saved_model(saved_path, model)              # 加载模型
    model.eval()                                                            # 设置为评估模式
    print('检测模型加载完成 !')

    record_len = torch.tensor([len(perception_data)])                       # 记录感知车的数量
    processed_features = []
    for id,data in perception_data.items():
        # 多车数据的处理（多车层级的封装，tensor的统一转换、到设备） 解决了
        # 点云位置的转移需不需要在特征图层面做改动？ 解决了
        processed_lidar = IntermediateFusionDataset.get_item_single_car(opencood_dataset,data)      # 数据集中的前处理，包含了点云的映射到ego车的坐标系下，以及点云的体素化
        processed_features.append(processed_lidar['processed_features'])                            # 每个车的点云都是单独处理的，所以需要在这里进行拼接

    merged_feature_dict = opencood_dataset.merge_features_to_dict(processed_features)               # 将多个车的特征拼接到一个字典中
    processed_lidar_torch_dict = opencood_dataset.pre_processor.collate_batch(merged_feature_dict)  # 将字典转换为tensor
    processed_lidar_torch_dict['record_len'] = torch.tensor([record_len])                           # 记录车的数量
    processed_lidar_torch_dict['map_mask'] = perception_data['ego']['map_mask']                     # 地图的mask
    processed_lidar_torch_dict = train_utils.to_device(processed_lidar_torch_dict, device)          # 将tensor转移到设备cuda上

    output_dict = model(processed_lidar_torch_dict)                       # 模型推理

    anchor_box = opencood_dataset.post_processor.generate_anchor_box()    # 生成anchor_box
    anchor_box_tensor = torch.from_numpy(anchor_box)                      # 将anchor_box转换为tensor

    # 包装车的信息(多级字典)
    cars_data = { 'ego':{
                        'processed_lidar': processed_lidar_torch_dict,
                        'anchor_box': anchor_box_tensor,
                        'transformation_matrix': torch.from_numpy(np.identity(4,dtype=np.float32)),
                        'record_len': record_len,
                        # 'plan_trajectory':plan_trajectory}
                        }
                }
    cars_data = train_utils.to_device(cars_data, device)          # 将tensor转移到设备cuda上
    
    # 后处理：NMS极大值抑制、锚框微调、去除小尺寸和低于阈值的候选框
    pred_box_tensor, pred_score = opencood_dataset.post_processor.post_process(cars_data, output_dict)  

    objects = o3d_lidar(objects,pred_box_tensor,'hwl',lidar_sensor)  # 结果的检查和封装
    return objects