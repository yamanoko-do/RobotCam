import open3d as o3d
import numpy as np

# 创建一个简单的点云
points = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])

# 将点云数据转换为 Open3D 的点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 保存点云为 .pcd 文件
o3d.io.write_point_cloud("example1.pcd", pcd)

print("点云已保存到 example.pcd")