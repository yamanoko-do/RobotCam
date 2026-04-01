import sys
sys.path.insert(0, './')
import os
import cv2
import numpy as np
import argparse
import time
import threading
import queue
from collections import deque
import open3d as o3d
import pyrealsense2 as rs

# 确保 Open3D 使用独立显卡与 X11 渲染
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

from hardware.camera.d435 import CameraD435

class AsyncD435:
    """ 异步 D435 相机读取类 """
    def __init__(self, width=1280, height=720, fps=30):
        self.cam = CameraD435()
        self.cam.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.cam.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.frame_queue = queue.Queue(maxsize=2)
        self.stopped = False
        self.thread = None

    def start(self):
        self.cam.start()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            try:
                frames = self.cam.pipeline.wait_for_frames()
                frames = self.cam.align.process(frames)
                
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue

                # 重要：跨线程传递帧必须调用 keep()，否则内存会被 RealSense 内部回收
                color_frame.keep()
                depth_frame.keep()

                data = {
                    'color_np': np.asanyarray(color_frame.get_data()),
                    'depth_np': np.asanyarray(depth_frame.get_data()),
                    'color_rs': color_frame,
                    'depth_rs': depth_frame,
                    'ts': time.time()
                }

                while self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(data)
            except Exception as e:
                print(f"[ERROR] Camera Thread: {e}")

    def get_frame(self):
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        self.cam.stop()

class PointCloudProcessor:
    """ 点云处理线程 """
    def __init__(self, cam_obj):
        self.cam_obj = cam_obj 
        self.input_queue = queue.Queue(maxsize=1)
        self.output_queue = queue.Queue(maxsize=1)
        self.stopped = False
        self.thread = threading.Thread(target=self.process_loop, daemon=True)
        self.thread.start()

    def process_loop(self):
        while not self.stopped:
            try:
                data = self.input_queue.get(timeout=0.01)
                
                # 计算点云
                self.cam_obj.pc.map_to(data['color_rs'])
                points_obj = self.cam_obj.pc.calculate(data['depth_rs'])
                
                # 获取顶点数据
                v = points_obj.get_vertices()
                verts = np.asanyarray(v).view(np.float32).reshape(-1, 3) 
                
                # 获取纹理坐标
                t = points_obj.get_texture_coordinates()
                texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)
                
                # 过滤无效点 (Z=0 的点)
                # 注意：RealSense 默认 Z 是正值，表示前方距离
                mask = (verts[:, 2] > 0.01) & (verts[:, 2] < 10.0) # 过滤0及10米以外的点
                
                if not np.any(mask):
                    continue

                v_filtered = verts[mask].copy()
                t_filtered = texcoords[mask]

                # 坐标系调整 (为了配合 Open3D 的常规视角)
                v_filtered[:, 1] *= -1
                v_filtered[:, 2] *= -1
                
                # 颜色映射逻辑
                color_img = data['color_np'][..., ::-1] # BGR to RGB
                h, w = color_img.shape[:2]
                
                u = np.clip((t_filtered[:, 0] * w).astype(int), 0, w - 1)
                v = np.clip((t_filtered[:, 1] * h).astype(int), 0, h - 1)
                colors = color_img[v, u] / 255.0

                # 打印调试信息（可选）
                # print(f"Generated {len(v_filtered)} points")

                while self.output_queue.full():
                    self.output_queue.get_nowait()
                self.output_queue.put_nowait((v_filtered.astype(np.float64), colors.astype(np.float64)))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] PC Processor: {e}")

    def submit(self, data):
        try:
            if self.input_queue.empty():
                self.input_queue.put_nowait(data)
        except queue.Full:
            pass

    def get_result(self):
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None, None

class O3DVisualizer:
    """ 改进的 Open3D 渲染器 - 更加稳定 """
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.pcd = o3d.geometry.PointCloud()
        self.initialized = False
        self.window_open = False

    def update(self, points, colors):
        if points is None or len(points) == 0:
            return

        if not self.initialized:
            # 创建窗口
            self.vis.create_window(window_name="D435 3D View", width=1280, height=720)
            
            # 先给点云赋值，再添加到渲染器，防止 "0 points" 警告
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            
            self.vis.add_geometry(self.pcd)
            
            opt = self.vis.get_render_option()
            opt.point_size = 3.5
            opt.background_color = np.array([0.1, 0.1, 0.1])
            
            ctr = self.vis.get_view_control()
            ctr.set_front([0, 0, 1])
            ctr.set_up([0, 1, 0])
            
            self.initialized = True
            self.window_open = True
        else:
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.pcd)
            
        self.vis.poll_events()
        self.vis.update_renderer()

    def keep_alive(self):
        """ 在 2D 模式下也要调用，保持窗口响应而不更新点云 """
        if self.initialized:
            self.vis.poll_events()
            self.vis.update_renderer()

    def close(self):
        if self.initialized:
            self.vis.destroy_window()
            self.initialized = False
            self.window_open = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args()

    async_cam = AsyncD435(args.width, args.height).start()
    pc_processor = PointCloudProcessor(async_cam.cam)
    o3d_vis = O3DVisualizer()

    view_mode = '2d' 
    last_time = time.time()

    try:
        while True:
            data = async_cam.get_frame()
            if data is not None:
                # 2D 深度图处理
                depth_vis = cv2.convertScaleAbs(data['depth_np'], alpha=0.05)
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                
                fps = 1.0 / (time.time() - last_time)
                last_time = time.time()
                cv2.putText(depth_color, f"FPS: {fps:.1f} Mode: {view_mode}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow("D435 Preview", depth_color)

                if view_mode == '3d':
                    pc_processor.submit(data)

            if view_mode == '3d':
                pts, clrs = pc_processor.get_result()
                if pts is not None and len(pts) > 0:
                    o3d_vis.update(pts, clrs)
            else:
                o3d_vis.close()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                view_mode = '3d' if view_mode == '2d' else '2d'
                print(f"Switched to {view_mode}")

    finally:
        async_cam.stop()
        pc_processor.stopped = True
        o3d_vis.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()