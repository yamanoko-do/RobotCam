import open3d as o3d
import numpy as np
import time
import os
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
# 创建一个随机点云
pcd = o3d.geometry.PointCloud()
points = np.random.randn(1000, 3)  # 生成1000个随机点
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(np.random.rand(1000, 3))  # 随机颜色

# 创建可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)

# 循环更新点云数据，实现动态效果
for i in range(200):
    # 为每个点添加一个微小的随机偏移ec
    points = np.asarray(pcd.points)
    points += np.random.normal(scale=0.01, size=points.shape)
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 更新显示
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    #time.sleep(0.05)  # 调整延时控制动画速度

vis.destroy_window()
