import numpy as np
import cv2
import pyrealsense2 as rs

# 初始化RealSense管道
pipeline = rs.pipeline()
config = rs.config()

# 启用深度流和彩色流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动管道
pipeline.start(config)

# 创建点云对象
pc = rs.pointcloud()

try:
    while True:
        # 等待捕获一帧数据
        frames = pipeline.wait_for_frames()

        # 获取深度帧和彩色帧
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # 将深度帧转换为点云
        points = pc.calculate(depth_frame)
        print(type(points))
        pc.map_to(color_frame)

        # 获取点云的顶点和纹理坐标
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # 3D点 (x, y, z)
        texcoords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)  # 纹理坐标 (u, v)

        # 获取彩色图像
        color_image = np.asanyarray(color_frame.get_data())

        # 创建一个空白图像用于渲染点云
        out = np.zeros((480, 640, 3), dtype=np.uint8)

        # 将点云投影到2D图像
        h, w = out.shape[:2]
        proj = (verts[:, :2] * (w, h) + (w // 2, h // 2)).astype(np.int32)  # 简单的正交投影

        # 过滤掉超出图像范围的点
        valid = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
        proj = proj[valid]
        verts = verts[valid]
        texcoords = texcoords[valid]

        # 从彩色图像中提取颜色
        u = (texcoords[:, 0] * (color_image.shape[1] - 1)).astype(np.int32)  # 映射到宽度范围 [0, width-1]
        v = (texcoords[:, 1] * (color_image.shape[0] - 1)).astype(np.int32)  # 映射到高度范围 [0, height-1]

        # 确保 u 和 v 在有效范围内
        u = np.clip(u, 0, color_image.shape[1] - 1)
        v = np.clip(v, 0, color_image.shape[0] - 1)

        # 提取颜色
        colors = color_image[v, u]

        # 将点云绘制到图像上
        out[proj[:, 1], proj[:, 0]] = colors

        # 显示渲染结果
        cv2.imshow("Point Cloud", out)

        # 按下ESC键退出
        if cv2.waitKey(1) == 27:
            break

finally:
    # 停止管道
    pipeline.stop()
    cv2.destroyAllWindows()