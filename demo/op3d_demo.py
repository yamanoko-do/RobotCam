import open3d as o3d
import numpy as np
import os
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
# 创建一个简单的点云
# 随机生成一些点
points = np.random.rand(100, 3)  # 生成100个随机点，每个点有3个坐标值（x, y, z）

# 将点云数据转换为Open3D的PointCloud对象
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# 可选：为点云添加颜色
colors = np.random.rand(100, 3)  # 随机生成颜色
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# 可选：为点云添加法线（用于更高级的可视化）
# 这里只是随机生成法线，实际中需要根据数据计算
normals = np.random.rand(100, 3)
normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # 归一化
point_cloud.normals = o3d.utility.Vector3dVector(normals)

# 可视化点云
o3d.visualization.draw_geometries([point_cloud],
                                  window_name="Open3D Point Cloud Visualization",
                                  width=800, height=600)