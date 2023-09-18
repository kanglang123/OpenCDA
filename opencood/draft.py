import numpy as np
import matplotlib.pyplot as plt

# 假设有两个多边形组成可行域
feasible_regions_x = np.array([[1, 2, 2, 1],
                               [6, 7, 8, 7]])

feasible_regions_y = np.array([[1, 1.5, 2, 2],
                               [2, 3, 3, 2]])

# 可行域数据的形状
print(feasible_regions_x.shape)  # 输出：(2, 4)
print(feasible_regions_y.shape)  # 输出：(2, 4)

# 可视化可行域
plt.figure(figsize=(8, 6))
for i in range(feasible_regions_x.shape[0]):
    plt.plot(feasible_regions_x[i], feasible_regions_y[i], 'b-', lw=2)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('车辆规划路径可行域')
plt.grid()
plt.show()
