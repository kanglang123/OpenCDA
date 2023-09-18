import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def cartesian_to_frenet(x, y, ref_x, ref_y, ref_yaw):
    """
    将车辆轨迹从笛卡尔坐标系转换到Frenet坐标系。

    参数:
        x, y: 车辆轨迹在笛卡尔坐标系中的x和y坐标，可以是单个数值或数组。
        ref_x, ref_y: 参考线在笛卡尔坐标系中的x和y坐标，用于计算Frenet坐标系。
        ref_yaw: 参考线的航向角，用于计算Frenet坐标系。

    返回:
        (s, d): Frenet坐标系中的s和d值，如果输入是数组，则返回对应数组。
    """
    dx = x - ref_x
    dy = y - ref_y
    cos_ref_yaw = np.cos(ref_yaw)
    sin_ref_yaw = np.sin(ref_yaw)
    s = dx * cos_ref_yaw + dy * sin_ref_yaw
    d = -dx * sin_ref_yaw + dy * cos_ref_yaw
    return s, d

def frenet_to_cartesian(s, d, ref_x, ref_y, ref_yaw):
    """
    将车辆轨迹从Frenet坐标系转换到笛卡尔坐标系。

    参数:
        s, d: Frenet坐标系中的s和d值，可以是单个数值或数组。
        ref_x, ref_y: 参考线在笛卡尔坐标系中的x和y坐标，用于计算Frenet坐标系。
        ref_yaw: 参考线的航向角，用于计算Frenet坐标系。

    返回:
        (x, y): 车辆轨迹在笛卡尔坐标系中的x和y坐标，如果输入是数组，则返回对应数组。
    """
    x = ref_x + s * np.cos(ref_yaw) - d * np.sin(ref_yaw)
    y = ref_y + s * np.sin(ref_yaw) + d * np.cos(ref_yaw)
    return x, y


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

def main():
    num_points = 100
    x_min, x_max = 0, 200
    y_min, y_max = -30, 30
    x_range = (0, 200)
    y_range = (-30, 30)

    car_x = np.linspace(x_min, x_max, num_points)
    car_y = np.sin(np.linspace(0,np.pi, num_points)) * (y_max - y_min) / 2
    # 假设车辆规划路径是一系列的(x, y)坐标点
    planned_path_x = car_x  # 替换成实际的x坐标数据
    planned_path_y = car_y  # 替换成实际的y坐标数据

    # 假设车宽为5m
    car_width = 5.0

    # 生成可行域
    feasible_regions_x, feasible_regions_y = generate_feasible_regions(planned_path_x, planned_path_y, car_width)

    mask = generate_mask(np.concatenate((feasible_regions_x[:, 0],(feasible_regions_x[:, 1])[::-1]),axis=0), \
                         np.concatenate((feasible_regions_y[:, 0],(feasible_regions_y[:, 1])[::-1]),axis=0), \
                              x_range, y_range)

    # 可视化规划后的车辆轨迹和可行域
    plt.figure(figsize=(8, 6))
    plt.plot(planned_path_x, planned_path_y, 'b', label='plan traj')
    # 将可行域填充为蓝色
    plt.fill(np.concatenate((feasible_regions_x[:, 0],(feasible_regions_x[:, 1])[::-1]),axis=0), \
             np.concatenate((feasible_regions_y[:, 0],(feasible_regions_y[:, 1])[::-1]),axis=0), \
                color='green', alpha=0.3, edgecolor='blue', linewidth=2, linestyle='dotted', closed=False)
    plt.imshow(mask, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), origin='lower', cmap='gray')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('width=5m can go area')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

    # points = np.array([[100, 30],
    #                [50, 20],
    #                [1, 1]])
    
    # x_band = np.concatenate((feasible_regions_x[:, 0],(feasible_regions_x[:, 1])[::-1]))
    # y_band = np.concatenate((feasible_regions_y[:, 0],(feasible_regions_y[:, 1])[::-1]))
    # for point in points:
    #     x, y = point
    #     is_inside = is_point_inside_feasible_region(x, y, x_band,y_band)
    #     print(is_inside)
    # mask =  is_point_inside_feasible_region(points[:, 0], points[:, 1], x_band,y_band)  
    


if __name__ == '__main__':
    main()