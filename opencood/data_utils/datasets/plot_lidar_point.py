import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 假设您的点云数据存储在一个NumPy数组中，格式为 (33814, 4)
# 在这个示例中，我将随机生成一些点作为示例数据
def plot_3d_point_cloud(point_cloud_data):

    # 提取x、y、z坐标
    x = point_cloud_data[:, 0]
    y = point_cloud_data[:, 1]
    plt.scatter(x, y, label='散点图', color='blue', marker='o')
    # 添加标签和标题
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('二维散点图示例')
    plt.axis('equal')
    # # 创建3D图形
    # fig22 = plt.figure()
    # ax = fig22.add_subplot(111, projection='3d')

    # # 绘制点云
    # ax.scatter(x, y, z, c=z, cmap='viridis', s=0.1)  # 这里使用z值作为颜色，可以根据需要更改

    # # 设置坐标轴标签
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    
    # 保存图形
    plt.savefig('p.png')
    plt.show()