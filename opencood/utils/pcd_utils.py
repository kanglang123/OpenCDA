# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
Utility functions related to point cloud
"""
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import torch.nn.functional as F

def pcd_to_np(pcd_file):
    """
    Read  pcd and return numpy array.

    Parameters
    ----------
    pcd_file : str
        The pcd file that contains the point cloud.

    Returns
    -------
    pcd : o3d.PointCloud
        PointCloud object, used for visualization
    pcd_np : np.ndarray
        The lidar data in numpy format, shape:(n, 4)

    """
    pcd = o3d.io.read_point_cloud(pcd_file)

    xyz = np.asarray(pcd.points)
    # we save the intensity in the first channel
    intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)
    pcd_np = np.hstack((xyz, intensity))

    return np.asarray(pcd_np, dtype=np.float32)


def mask_points_by_range(points, limit_range):
    """
    Remove the lidar points out of the boundary.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    limit_range : list
        [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """

    mask = (points[:, 0] > limit_range[0]) & (points[:, 0] < limit_range[3])\
           & (points[:, 1] > limit_range[1]) & (
                   points[:, 1] < limit_range[4]) \
           & (points[:, 2] > limit_range[2]) & (
                   points[:, 2] < limit_range[5])

    points = points[mask]

    return points

def generate_feasible_regions(planned_path_x, planned_path_y, car_width):
    # 计算车辆规划路径的切线方向
    dx = np.gradient(planned_path_x)
    dy = np.gradient(planned_path_y)
    yaw = np.arctan2(dy, dx)

    # 计算车辆规划路径的法向量方向
    normal_yaw = yaw + np.pi / 2

    # 计算车辆规划路径两侧的点
    offset = car_width / 2
    left_x = planned_path_x + offset * np.cos(normal_yaw)
    left_y = planned_path_y + offset * np.sin(normal_yaw)
    right_x = planned_path_x - offset * np.cos(normal_yaw)
    right_y = planned_path_y - offset * np.sin(normal_yaw)

    # 以车辆规划路径两侧的点构建可行域
    feasible_regions_x = np.stack((left_x, right_x), axis=1)
    feasible_regions_y = np.stack((left_y, right_y), axis=1)

    return feasible_regions_x, feasible_regions_y

def generate_mask(feasible_regions_x, feasible_regions_y, x_range, y_range):
    # 创建多边形路径
    vertices = np.column_stack((feasible_regions_x.flatten(), feasible_regions_y.flatten()))
    path = Path(vertices)

    # 生成掩膜
    x, y = np.meshgrid(np.arange(x_range[0], x_range[1] + 1), np.arange(y_range[0], y_range[1] + 1))
    # x, y = np.meshgrid(np.arange(x_range[0], x_range[1]), np.arange(y_range[0], y_range[1]))
    points = np.column_stack((x.ravel(), y.ravel()))
    mask = path.contains_points(points)
    mask = mask.reshape(x.shape)

    return mask

def is_point_inside_feasible_region(x, y, feasible_regions_x, feasible_regions_y):
    # 创建多边形路径
    path = Path(np.column_stack((feasible_regions_x, feasible_regions_y)))
    # 判断点(x, y)是否在多边形内部
    if not path.contains_point([x, y]):
        return False
    return True

def mask_points_by_plan_trajectory(points, ego_tarj,limit_range):
    """
    Remove the lidar points out of the boundary.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    ego_tarj : list
        [x, y, z, ……]

    limit_range : list
        [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """
    # print('计算可行域，轨迹长度为：',len(ego_tarj))
    
    # 假设车宽为5m
    car_width = 10.0

    # 生成可行域    feasible_regions_x、y的两列分别是可行域的边界
    feasible_regions_x, feasible_regions_y = generate_feasible_regions(ego_tarj[:,0], ego_tarj[:,1], car_width)
    
    # 基于可行域生成的mask_map   shape:[81,283]
    mask_map = generate_mask(np.concatenate((feasible_regions_x[:, 0],(feasible_regions_x[:, 1])[::-1]),axis=0), \
                             np.concatenate((feasible_regions_y[:, 0],(feasible_regions_y[:, 1])[::-1]),axis=0), \
                             (limit_range[0],limit_range[3]), (limit_range[1],limit_range[4]))
    
    # plt.figure(figsize=(8, 6))
    # plt.imshow(mask_map, extent=(limit_range[0],limit_range[3], limit_range[1],limit_range[4]), origin='lower', cmap='gray')
    # plt.show()

    mask_map = torch.tensor(mask_map,requires_grad=False)
    mask_map = mask_map.float()
    mask_map = torch.unsqueeze(mask_map,dim=0)
    mask_map = torch.unsqueeze(mask_map,dim=0)
    communication_masks = F.interpolate(mask_map, size=(100, 352),mode='bilinear', align_corners=False) 
    # plt.figure(figsize=(8, 6))
    # plt.imshow(communication_masks[0][0])
    # plt.show()
    

    # # 基于可行域生成的掩膜 把上下两边界处理成一个闭环，下边界倒序加到上边界的尾巴上
    # x_band = np.concatenate((feasible_regions_x[:, 0],(feasible_regions_x[:, 1])[::-1]))
    # y_band = np.concatenate((feasible_regions_y[:, 0],(feasible_regions_y[:, 1])[::-1]))
    
    # mask = []   # 用于记录每个点是否在可行域内，存储的是数量为点云个数的布尔值
    # for point in points:
    #     mask.append(is_point_inside_feasible_region(point[0], point[1], x_band,y_band))
    
    # #圆形的掩膜
    # radius2 = 49
    # mask = np.zeros(points.shape[0],dtype = bool)
    # for i in range(len(ego_tarj)-1):   
    #     mask = (((points[:, 0]-ego_tarj[i,0])**2 + (points[:, 1]-ego_tarj[i,1])**2 ) < radius2) + mask
    # # print(sum(mask))

    # return points[mask]
    return communication_masks[0][0]


def mask_ego_points(points):
    """
    Remove the lidar points of the ego vehicle itself.

    Parameters
    ----------
    points : np.ndarray
        Lidar points under lidar sensor coordinate system.

    Returns
    -------
    points : np.ndarray
        Filtered lidar points.
    """
    mask = (points[:, 0] >= -1.95) & (points[:, 0] <= 2.95) \
           & (points[:, 1] >= -1.1) & (points[:, 1] <= 1.1)
    points = points[np.logical_not(mask)]

    return points

# 作用是将点云数据点随机排序（洗牌）
def shuffle_points(points):
    shuffle_idx = np.random.permutation(points.shape[0])
    points = points[shuffle_idx]

    return points


def lidar_project(lidar_data, extrinsic):
    """
    Given the extrinsic matrix, project lidar data to another space.

    Parameters
    ----------
    lidar_data : np.ndarray
        Lidar data, shape: (n, 4)

    extrinsic : np.ndarray
        Extrinsic matrix, shape: (4, 4)

    Returns
    -------
    projected_lidar : np.ndarray
        Projected lida data, shape: (n, 4)
    """

    lidar_xyz = lidar_data[:, :3].T
    # (3, n) -> (4, n), homogeneous transformation
    lidar_xyz = np.r_[lidar_xyz, [np.ones(lidar_xyz.shape[1])]]
    lidar_int = lidar_data[:, 3]

    # transform to ego vehicle space, (3, n)
    project_lidar_xyz = np.dot(extrinsic, lidar_xyz)[:3, :]
    # (n, 3)
    project_lidar_xyz = project_lidar_xyz.T
    # concatenate the intensity with xyz, (n, 4)
    projected_lidar = np.hstack((project_lidar_xyz,
                                 np.expand_dims(lidar_int, -1)))

    return projected_lidar


def projected_lidar_stack(projected_lidar_list):
    """
    Stack all projected lidar together.

    Parameters
    ----------
    projected_lidar_list : list
        The list containing all projected lidar.

    Returns
    -------
    stack_lidar : np.ndarray
        Stack all projected lidar data together.
    """
    stack_lidar = []
    for lidar_data in projected_lidar_list:
        stack_lidar.append(lidar_data)

    return np.vstack(stack_lidar)


def downsample_lidar(pcd_np, num):
    """
    Downsample the lidar points to a certain number.

    Parameters
    ----------
    pcd_np : np.ndarray
        The lidar points, (n, 4).

    num : int
        The downsample target number.

    Returns
    -------
    pcd_np : np.ndarray
        The downsampled lidar points.
    """
    assert pcd_np.shape[0] >= num

    selected_index = np.random.choice((pcd_np.shape[0]),
                                      num,
                                      replace=False)
    pcd_np = pcd_np[selected_index]

    return pcd_np


def downsample_lidar_minimum(pcd_np_list):
    """
    Given a list of pcd, find the minimum number and downsample all
    point clouds to the minimum number.

    Parameters
    ----------
    pcd_np_list : list
        A list of pcd numpy array(n, 4).
    Returns
    -------
    pcd_np_list : list
        Downsampled point clouds.
    """
    minimum = np.Inf

    for i in range(len(pcd_np_list)):
        num = pcd_np_list[i].shape[0]
        minimum = num if minimum > num else minimum

    for (i, pcd_np) in enumerate(pcd_np_list):
        pcd_np_list[i] = downsample_lidar(pcd_np, minimum)

    return pcd_np_list
