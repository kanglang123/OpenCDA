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

def inference_code(opt,objects,lidar_data):

    hypes = yaml_utils.load_yaml(None, opt)

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False)
    print(f"{len(opencood_dataset)} samples found.")
    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()
    
    i = 0
    if i == 0:
        processed_lidar = opencood_dataset.pre_processor.preprocess(lidar_data)   # 前处理
        processed_lidar['voxel_coords'] = np.pad(processed_lidar['voxel_coords'], ((0, 0), (1, 0)),mode='constant', constant_values=0)
        processed_lidar_tensor = {k: torch.from_numpy(v) for k, v in processed_lidar.items()}
        
        processed_lidar_tensor = train_utils.to_device(processed_lidar_tensor, device)
        output_dict = model(processed_lidar_tensor)
        a = 1
    else:
        for i, batch_data in tqdm(enumerate(data_loader)):
            # print(i)
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)
                pred_box_tensor, pred_score, gt_box_tensor = inference_utils.inference_early_fusion(batch_data,model,opencood_dataset)