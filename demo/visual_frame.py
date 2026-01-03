import matplotlib.pyplot as plt
import numpy as np
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager

# 给定旋转矩阵
R_board2F = np.array([[-0.9195219278335571, 0.24544388055801392, 0.30698010325431824], [-0.2328125685453415, -0.9694108366966248, 0.0777238979935646], [0.3166666626930237, 0.0, 0.9485369324684143]])

# # 提取三个列向量
# v0 = R[:, 0]  # 第0列
# v1 = R[:, 1]  # 第1列
# v2 = R[:, 2]  # 第2列

# # 计算两两点积
# dot_01 = np.dot(v0, v1)
# dot_02 = np.dot(v0, v2)
# dot_12 = np.dot(v1, v2)

# print("v0 · v1 =", dot_01)
# print("v0 · v2 =", dot_02)
# print("v1 · v2 =", dot_12)



# 给定平移向量
t_board2F = np.array([0.1158125102519989, -0.04816935956478119, 0.7749999761581421])/1000  # 转换为米

# 组合为4×4变换矩阵
T_board2F = np.eye(4)
T_board2F[:3, :3] = R_board2F
T_board2F[:3, 3] = t_board2F
print(T_board2F)
R_griper2board = np.array( [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
t_griper2board = np.array( [t_board2F[0], 0, 0])
T_griper2board = np.eye(4)
T_griper2board[:3, :3] = R_griper2board
T_griper2board[:3, 3] = t_griper2board
print(T_griper2board)


T_griper2F = np.array([[7.007705425379671, 1444.6904388666171, 24198.8051510019, 144.13822254575638], [6.422640660645513, 1096.5184616083186, 18361.855394116275, 109.37407312676659], [40.249600535647126, 7559.920429131797, 126686.96370980213, 754.6115666193723], [0.053847324660353024, 10.018563869855079, 167.88486924312906, 1.0]])
print(T_griper2F)
# 创建变换管理器
tm = TransformManager(strict_check=False)


tm.add_transform("griper", "F", T_griper2F)

# 可视化坐标系
ax = tm.plot_frames_in("griper", s=0.1)
ax.set_xlim((-0.5, 0.5))
ax.set_ylim((-0.5, 0.5))
ax.set_zlim((-0.5, 0.5))
plt.show()