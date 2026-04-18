import sys
import os
from collections import deque

# Set environment variables first
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import cv2
import numpy as np
import time
import threading
import queue
import argparse

# Add parent path
sys.path.insert(0, './')

from hardware.camera.zed import CameraZED


def create_depth_colormap(depth_raw: np.ndarray, max_depth: float = 10000.0) -> np.ndarray:
    """
    Create color depth map from raw depth data

    Args:
        depth_raw: Raw depth map (millimeters)
        max_depth: Max depth for normalization (millimeters)

    Returns:
        Color depth image
    """
    # Handle NaN/Inf values
    depth_valid = np.nan_to_num(depth_raw, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize depth to 0-255
    depth_normalized = np.clip(depth_valid, 0, max_depth)
    depth_vis = (depth_normalized / max_depth * 255).astype(np.uint8)
    # Invert so closer is brighter (optional but intuitive)
    depth_vis = 255 - depth_vis
    return cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)


class AsyncZED:
    """异步 ZED 相机读取类"""
    def __init__(self, width=1280, height=720, fps=30):
        self.cam = CameraZED()
        self.cam.enable_stream('left', width, height, framerate=fps)
        self.cam.enable_stream('depth', width, height, framerate=fps)
        self.frame_queue = queue.Queue(maxsize=2)
        self.stopped = False
        self.thread = None
        self.frame_id = 0

    def start(self):
        self.cam.start()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            try:
                frame_dict = self.cam.get_frame()
                if frame_dict:
                    self.frame_id += 1
                    data = {
                        'color_np': frame_dict.get('left'),
                        'depth_np': frame_dict.get('depth'),
                        'depth_view': frame_dict.get('depth_view'),
                        'frame_id': self.frame_id,
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
    """点云处理线程 - 使用 ZED 原生点云"""
    def __init__(self, cam_obj, zfar: float = 10000.0):
        """
        Initialize point cloud processor

        Args:
            cam_obj: CameraZED object
            zfar: Max depth to include (millimeters)
        """
        self.cam_obj = cam_obj
        self.zfar = zfar
        self.input_queue = queue.Queue(maxsize=2)
        self.output_queue = queue.Queue(maxsize=2)
        self.stopped = False
        self.thread = threading.Thread(target=self.process_loop, daemon=True)
        self.thread.start()

    def process_loop(self):
        """Point cloud processing thread loop"""
        while not self.stopped:
            try:
                frame_id = self.input_queue.get(timeout=0.01)
                # Get point cloud directly from ZED
                verts, colors = self.cam_obj.get_point_and_color()

                if verts is not None and len(verts) > 0:
                    # Filter by zfar
                    mask = verts[:, 2] < self.zfar
                    verts = verts[mask]
                    colors = colors[mask]

                    # Convert millimeters to meters for Open3D
                    verts = verts / 1000.0

                    try:
                        while self.output_queue.full():
                            self.output_queue.get_nowait()
                        self.output_queue.put_nowait((verts, colors, frame_id))
                    except queue.Full:
                        pass
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] PointCloudProcessor error: {e}")

    def submit(self, frame_id: int):
        """Submit point cloud computation task"""
        try:
            while self.input_queue.full():
                self.input_queue.get_nowait()
            self.input_queue.put_nowait(frame_id)
        except queue.Full:
            pass

    def get_result(self):
        """Get latest point cloud result"""
        latest = None
        try:
            while True:
                latest = self.output_queue.get_nowait()
        except queue.Empty:
            pass
        return latest if latest else (None, None, None)

    def stop(self):
        """Stop processing thread"""
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)


try:
    import open3d as o3d

    class O3DVisualizer:
        """
        Open3D real-time visualization manager
        """
        def __init__(self, window_name: str = "ZED 3D Point Cloud View",
                     width: int = 1280, height: int = 720,
                     background_color: tuple = (0.2, 0.2, 0.2),
                     point_size: float = 4.5):
            self.vis = None
            self.pcd = None
            self.is_initialized = False
            self.geometry_added = False
            self.data_updated = False
            self.window_name = window_name
            self.width = width
            self.height = height
            self.background_color = np.asarray(background_color)
            self.point_size = point_size

        def init_window(self):
            """Initialize visualization window"""
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name=self.window_name, width=self.width, height=self.height)
            self.pcd = o3d.geometry.PointCloud()
            self.is_initialized = True

        def set_pointcloud_data(self, points: np.ndarray, colors: np.ndarray):
            """Set point cloud data"""
            if not self.is_initialized:
                self.init_window()
            if points is not None and colors is not None and len(points) > 0:
                self.pcd.points = o3d.utility.Vector3dVector(points)
                self.pcd.colors = o3d.utility.Vector3dVector(colors)
                self.data_updated = True

        def spin_once(self) -> bool:
            """Update visualization"""
            if not self.is_initialized:
                return False

            if self.data_updated:
                if not self.geometry_added:
                    self.vis.add_geometry(self.pcd)
                    opt = self.vis.get_render_option()
                    opt.background_color = self.background_color
                    opt.point_size = self.point_size

                    ctr = self.vis.get_view_control()
                    ctr.set_front([0, 0, -1])
                    ctr.set_up([0, -1, 0])
                    self.geometry_added = True
                else:
                    self.vis.update_geometry(self.pcd)

                self.data_updated = False

            if not self.vis.poll_events():
                self.close()
                return False

            self.vis.update_renderer()
            return True

        def close(self):
            """Close visualization window"""
            if self.is_initialized:
                self.vis.destroy_window()
                self.is_initialized = False
                self.geometry_added = False
                self.data_updated = False

except ImportError:
    print("[WARNING] Open3D not available, 3D point cloud visualization disabled")

    class O3DVisualizer:
        """Fallback O3DVisualizer when Open3D is not available"""
        def __init__(self, *args, **kwargs):
            print("[WARNING] Open3D not installed, 3D visualization disabled")
            self.is_initialized = False

        def init_window(self):
            pass

        def set_pointcloud_data(self, *args):
            pass

        def spin_once(self) -> bool:
            return False

        def close(self):
            pass


def display_overlay(img: np.ndarray, frame_id: int, fps: float, mode_str: str) -> np.ndarray:
    """Display overlay information on image"""
    overlay = img.copy()
    h, w = overlay.shape[:2]

    # Background rectangle
    cv2.rectangle(overlay, (10, 10), (350, 90), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Text
    cv2.putText(img, f"Frame: {frame_id}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"FPS: {fps:.1f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f"Mode: {mode_str}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=str, default='HD720',
                        choices=['HD2K', 'HD1080', 'HD720', 'VGA'],
                        help='Resolution: HD2K (2208x1242), HD1080 (1920x1080), HD720 (1280x720), VGA (672x376)')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--zfar', type=float, default=10.0,
                        help='Max depth in meters for point cloud')
    args = parser.parse_args()

    # Resolution map
    res_map = {
        'HD2K': (2208, 1242),
        'HD1080': (1920, 1080),
        'HD720': (1280, 720),
        'VGA': (672, 376)
    }
    args.width, args.height = res_map[args.resolution]

    # Convert zfar to millimeters
    zfar_mm = args.zfar * 1000.0

    print("[INFO] Starting ZED camera...")
    async_cam = AsyncZED(args.width, args.height, args.fps).start()
    pc_processor = PointCloudProcessor(async_cam.cam, zfar=zfar_mm)
    o3d_vis = O3DVisualizer()

    view_mode = '2d'
    fps_counter = deque(maxlen=30)
    last_time = time.time()
    frame_count = 0

    print(f"[INFO] Display mode: OpenCV/Open3D")
    print(f"  - 3D Point Cloud: YES (press v to show)")
    print(f"[INFO] Press 'q' to quit, 'v' to toggle 2D/3D")

    try:
        while True:
            # Get latest frame
            data = async_cam.get_frame()

            if data is not None:
                frame_count += 1
                current_time = time.time()
                fps = 1.0 / (current_time - last_time)
                fps_counter.append(fps)
                avg_fps = sum(fps_counter) / len(fps_counter)
                last_time = current_time

                # Create depth color map
                if data['depth_np'] is not None:
                    depth_color = create_depth_colormap(data['depth_np'], max_depth=zfar_mm)
                else:
                    depth_color = data['depth_view']

                # Display overlay
                mode_str = "3D PointCloud" if view_mode == '3d' else "2D Depth"
                depth_color = display_overlay(depth_color, data['frame_id'], avg_fps, mode_str)

                # Show 2D view
                cv2.imshow("ZED Depth View", depth_color)

                # Submit to point cloud processor if in 3D mode
                if view_mode == '3d':
                    pc_processor.submit(data['frame_id'])

            # Process point cloud results
            if view_mode == '3d':
                points, colors, pc_frame_id = pc_processor.get_result()
                if points is not None and colors is not None:
                    o3d_vis.set_pointcloud_data(points, colors)

            # Update 3D view
            if view_mode == '3d':
                if not o3d_vis.is_initialized:
                    o3d_vis.init_window()
                is_active = o3d_vis.spin_once()
                if not is_active:
                    view_mode = '2d'
            elif view_mode == '2d' and o3d_vis.is_initialized:
                o3d_vis.close()

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                view_mode = '3d' if view_mode == '2d' else '2d'
                print(f"[INFO] View mode: {'[3D PointCloud]' if view_mode == '3d' else '[2D Depth]'}")

            # Small sleep
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n[INFO] User interrupted")
    finally:
        # Cleanup
        async_cam.stop()
        pc_processor.stop()
        o3d_vis.close()
        cv2.destroyAllWindows()

        if fps_counter:
            print(f"\nFinal average FPS: {sum(fps_counter)/len(fps_counter):.2f}")


if __name__ == "__main__":
    main()
