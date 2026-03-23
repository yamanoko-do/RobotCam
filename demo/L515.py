import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import cv2
import os
import sys
sys.path.insert(0, './')
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

def main():
    # 1. 配置管道
    pipeline = rs.pipeline()
    config = rs.config()

    # L515 推荐配置
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    print("正在启动 L515...")
    profile = pipeline.start(config)

    # 2. 创建对齐对象（深度对齐到彩色）
    align = rs.align(rs.stream.color)

    # 3. 获取彩色相机内参（用于坐标转换）
    color_profile = profile.get_stream(rs.stream.color)
    color_intr = color_profile.as_video_stream_profile().get_intrinsics()

    # 4. 初始化 Open3D 可视化
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="L515 RGB PointCloud (Optimized)", width=1280, height=720)

    pcd = o3d.geometry.PointCloud()

    # 坐标辅助线（可选）
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(coord)

    # 5. 下采样参数（步长越大点数越少，性能越高，建议 2 或 3）
    step = 1   # 步长2 → 约 1/4 的点数

    first_frame = True
    try:
        while True:
            # 获取并对齐帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # 获取图像数据（已对齐，分辨率相同）
            depth_image = np.asanyarray(depth_frame.get_data())  # 16位深度图，单位毫米
            color_image = np.asanyarray(color_frame.get_data())
            color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # ========== 可视化深度图 ==========
            # 将深度值缩放到 0-255 范围（例如最大深度 3 米）
            depth_mm = depth_image
            depth_vis = cv2.convertScaleAbs(depth_mm, alpha=0.03)  # 1/30 ≈ 0.033，使 3 米映射到 255
            depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            cv2.imshow("Depth Map", depth_colormap)

            # ========== 下采样（减少点云数量） ==========
            # 对深度和彩色图进行等间隔采样
            depth_sampled = depth_image[::step, ::step]
            color_sampled = color_rgb[::step, ::step, :]

            # 获取有效深度值的掩码（过滤掉 0 值）
            mask = depth_sampled > 0
            if not np.any(mask):
                continue

            # 提取有效深度和颜色
            z = depth_sampled[mask] * depth_frame.get_units()   # 实际深度（米）
            colors = color_sampled[mask] / 255.0                # 归一化颜色 [0,1]

            # 获取采样后的像素坐标
            h_sampled, w_sampled = depth_sampled.shape
            u_indices, v_indices = np.meshgrid(np.arange(w_sampled), np.arange(h_sampled))
            u = u_indices[mask] * step   # 恢复原始彩色图像坐标（因为采样步长是 step）
            v = v_indices[mask] * step

            # ========== 使用相机内参计算 3D 坐标 ==========
            # 公式：X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
            fx, fy = color_intr.fx, color_intr.fy
            cx, cy = color_intr.ppx, color_intr.ppy

            x = (u - cx) * z / fx
            y = (v - cy) * z / fy

            # 转换为 Open3D 坐标系（Y 向上，Z 向前）
            # 此处将原相机坐标（X右, Y下, Z前）变换为（X右, Y上, Z前）
            points = np.column_stack((x, -y, z))

            # 更新点云
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            if first_frame:
                vis.add_geometry(pcd)
                first_frame = False
            else:
                vis.update_geometry(pcd)

            # 渲染控制
            if not vis.poll_events():
                break
            vis.update_renderer()

            # 处理键盘输入（按 'q' 或 ESC 退出）
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        pipeline.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()
        print("已关闭相机和可视化窗口")

if __name__ == "__main__":
    main()