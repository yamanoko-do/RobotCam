import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
from typing import Tuple
import os
import glob

class CameraD405:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.color_frame_enable = False
        self.depth_frame_enable = False
        self.infra1_enable = False
        self.infra2_enable = False
        self.pc = rs.pointcloud()
        self.align = rs.align(rs.stream.color)
        self._started = False

    def enable_stream(self, stream_type, *args):
        """
        启用流
        两种调用方式:
        - 旧方式: enable_stream(stream_type, width, height, format, framerate)
        - 新方式: enable_stream(stream_type, stream_index, width, height, format, framerate)
        """
        if self._started:
            raise RuntimeError("Cannot enable stream after pipeline started. Call enable_stream() before start().")

        if len(args) == 4:
            # 旧方式: stream_type, width, height, format, framerate
            width, height, format, framerate = args
            stream_index = -1
            self.config.enable_stream(stream_type, width, height, format, framerate)
        elif len(args) == 5:
            # 新方式: stream_type, stream_index, width, height, format, framerate
            stream_index, width, height, format, framerate = args
            self.config.enable_stream(stream_type, stream_index, width, height, format, framerate)
        else:
            raise ValueError("Invalid number of arguments. Use either: "
                             "enable_stream(stream_type, width, height, format, framerate) or "
                             "enable_stream(stream_type, stream_index, width, height, format, framerate)")

        if str(stream_type) == "stream.color":
            self.color_frame_enable = True
        elif str(stream_type) == "stream.depth":
            self.depth_frame_enable = True
        elif str(stream_type) == "stream.infrared":
            if stream_index == 1 or (stream_index == -1 and not self.infra1_enable):
                self.infra1_enable = True
            elif stream_index == 2 or (stream_index == -1 and self.infra1_enable):
                self.infra2_enable = True


    def start(self):
        if self._started:
            print("Pipeline already started.")
            return
        self.pipeline.start(self.config)
        self._started = True


    def stop(self):
        if self._started:
            self.pipeline.stop()
            self._started = False

    def get_frame(self,format2numpy=True)-> dict:
        frames = self.pipeline.wait_for_frames()
        if self.color_frame_enable and self.depth_frame_enable:
            frames = self.align.process(frames)
        frame_dict={}
        if self.color_frame_enable:
            color_frame = frames.get_color_frame()
            if format2numpy:
                color_frame = np.asanyarray(color_frame.get_data())
            frame_dict["color"] = color_frame
        if self.depth_frame_enable:
            depth_frame = frames.get_depth_frame()
            if format2numpy:
                depth_frame = np.asanyarray(depth_frame.get_data())
            frame_dict["depth"] = depth_frame

        if frame_dict=={}:
            return None

        return frame_dict

    def get_stereo_frame(self, format2numpy=True, mode='infra') -> dict:
        """
        获取双目图像
        Args:
            format2numpy (bool): 是否转换为numpy数组
            mode (str): 双目模式:
                - 'infra': 双红外 (左红外+右红外，默认)
                - 'color_infra1': RGB+左红外
                - 'color_infra2': RGB+右红外
        Returns:
            dict: {'left': img_left, 'right': img_right, 'binocular': frame}
        """
        frames = self.pipeline.wait_for_frames()

        # 获取各种帧
        color_frame = None
        infra1_frame = None
        infra2_frame = None

        for f in frames:
            if f.profile.stream_type() == rs.stream.color:
                color_frame = f
            elif f.profile.stream_type() == rs.stream.infrared:
                if f.profile.stream_index() == 1:
                    infra1_frame = f
                elif f.profile.stream_index() == 2:
                    infra2_frame = f

        # 根据模式选择左右图像
        left_frame = None
        right_frame = None

        if mode == 'infra':
            if infra1_frame is None or infra2_frame is None:
                raise RuntimeError("未获取到左右红外图像，请确保已启用两个红外流")
            left_frame = infra1_frame
            right_frame = infra2_frame
        elif mode == 'color_infra1':
            if color_frame is None or infra1_frame is None:
                raise RuntimeError("未获取到RGB或左红外图像，请确保已启用相应流")
            left_frame = color_frame
            right_frame = infra1_frame
        elif mode == 'color_infra2':
            if color_frame is None or infra2_frame is None:
                raise RuntimeError("未获取到RGB或右红外图像，请确保已启用相应流")
            left_frame = color_frame
            right_frame = infra2_frame
        else:
            raise ValueError(f"未知的模式: {mode}，可选值: 'infra', 'color_infra1', 'color_infra2'")

        if format2numpy:
            left_frame = np.asanyarray(left_frame.get_data())
            right_frame = np.asanyarray(right_frame.get_data())

        # 如果图像通道数不同，统一转换为3通道
        if len(left_frame.shape) != len(right_frame.shape):
            if len(left_frame.shape) == 2:
                left_frame = cv2.cvtColor(left_frame, cv2.COLOR_GRAY2BGR)
            if len(right_frame.shape) == 2:
                right_frame = cv2.cvtColor(right_frame, cv2.COLOR_GRAY2BGR)

        # 拼接双目图像
        binocular = np.hstack((left_frame, right_frame))

        return {
            'left': left_frame,
            'right': right_frame,
            'binocular': binocular
        }


    def get_point_and_color(self):
        """
        获取点和颜色,用于open3d渲染
        Returns:
            verts(np.array): 点的三维坐标,shape为(n,3),单位?
            colors(np.array): 点的颜色,范围[0,1]
        """
        if not(self.color_frame_enable and self.depth_frame_enable):
            print("获取点云需要同时启用depth流与rgb流")
        else:
            frame_dict = self.get_frame(format2numpy=False)
            if not all(key in frame_dict for key in ["depth","color"]):
                print("帧数据不完整")
                return None
            depth_frame = frame_dict["depth"]
            color_frame = frame_dict["color"]

            self.pc.map_to(color_frame)
            points = self.pc.calculate(depth_frame)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            texcoords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

            verts[:, 1] *= -1
            verts[:, 2] *= -1

            verts = verts.astype(np.float64)

            color_image = np.asanyarray(color_frame.get_data())[..., ::-1]

            u = (texcoords[:, 0] * (color_image.shape[1] - 1)).astype(int)
            v = (texcoords[:, 1] * (color_image.shape[0] - 1)).astype(int)
            u = np.clip(u, 0, color_image.shape[1] - 1)
            v = np.clip(v, 0, color_image.shape[0] - 1)
            colors = color_image[v, u] / 255.0
            return verts,colors

    def take_photo(self, save_dir="./data/chessboard_images/monocular"):
        """
        启动实时预览，按 's' 保存彩色图像，按 'q' 或 ESC 退出。
        Args:
            save_dir (str): 保存图像的目录路径
        """
        os.makedirs(save_dir, exist_ok=True)

        photo_count = 1
        existing_files = glob.glob(os.path.join(save_dir, "image_*.jpg"))
        if existing_files:
            max_num = 0
            for file_path in existing_files:
                try:
                    filename = os.path.splitext(os.path.basename(file_path))[0]
                    number = int(filename.split('_')[-1])
                    if number > max_num:
                        max_num = number
                except (ValueError, IndexError):
                    continue
            photo_count = max_num + 1

        print("D405相机预览中... 按 's' 保存图像，按 'q' 或 ESC 退出。")
        try:
            while True:
                frame_dict = self.get_frame()
                if "color" not in frame_dict:
                    print("未获取到彩色帧")
                    continue
                color_frame = frame_dict["color"]

                cv2.imshow('D405 Camera - Press s to save, q/ESC to quit', color_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    save_path = os.path.join(save_dir, f"image_{photo_count}.jpg")
                    cv2.imwrite(save_path, color_frame)
                    print(f"已保存: {save_path}")
                    photo_count += 1

                elif key in (ord('q'), 27):
                    break

        finally:
            cv2.destroyAllWindows()

    @staticmethod
    def get_support_config():
        pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)

        for sensor in pipeline_profile.get_device().sensors:
            for stream_profile in sensor.get_stream_profiles():
                v_profile = stream_profile.as_video_stream_profile()
                print(f"rs.{stream_profile.stream_type()}, {v_profile.width()},{v_profile.height()}, rs.{v_profile.format()}, {v_profile.fps()}")

    @staticmethod
    def get_intrinsics()->Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        获取内参和畸变,不同分辨率参数的流对应的相机内参不同，记得修改
        Returns:
            tuple: 包含四个元素的元组，分别是：
                - color_intrinsics (np.ndarray): rgb内参
                - color_coeffs (np.ndarray): rgb畸变
                - depth_intrinsics (np.ndarray): depth内参
                - depth_coeffs (np.ndarray): depth畸变
        """
        pipeline = rs.pipeline()
        config = rs.config()

        profile = pipeline.get_active_profile()

        color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        color_intrinsics_=np.array([[color_intrinsics.fx,0,color_intrinsics.ppx],
                                   [0,color_intrinsics.fy,color_intrinsics.ppy],
                                   [0,0,1]]
                                   )
        depth_intrinsics_=np.array([[depth_intrinsics.fx,0,depth_intrinsics.ppx],
                                   [0,depth_intrinsics.fy,depth_intrinsics.ppy],
                                   [0,0,1]]
                                   )

        pipeline.stop()
        return color_intrinsics_,np.array(color_intrinsics.coeffs),depth_intrinsics_,np.array(depth_intrinsics.coeffs)


    @staticmethod
    def get_extrinsics_depth2rgb():
        """
        获取外参
        """
        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)

        pipeline.start(config)
        profile = pipeline.get_active_profile()

        depth_to_color_extrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(
            profile.get_stream(rs.stream.color)
        )

        print("=== Depth to Color Camera Extrinsics ===")

        rotmatrix=np.array([depth_to_color_extrinsics.rotation[0:3],depth_to_color_extrinsics.rotation[3:6],depth_to_color_extrinsics.rotation[6:9]])
        print(f"Rotation Matrix(m): {rotmatrix.tolist()}")
        print(f"Translation Vector(m): {depth_to_color_extrinsics.translation}")

        pipeline.stop()


if __name__=="__main__":
    """
    获取支持的流配置参数
    """
    CameraD405.get_support_config()

    """
    输出rgb流
    """
    # cam=CameraD405()
    # cam.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
    # cam.start()
    # try:
    #     while True:
    #         frame_dict=cam.get_frame()
    #         color_frame = frame_dict['color']

    #         # 显示画面
    #         cv2.imshow('RealSense Color Stream', color_frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    # finally:
    #         cam.stop()
    #         cv2.destroyAllWindows()
    """
    输出depth流
    """
    # cam=CameraD405()
    # cam.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
    # cam.start()
    # try:
    #     while True:
    #         frame_dict=cam.get_frame()
    #         depth_frame = frame_dict['depth']
    #
    #         # 显示画面
    #         cv2.imshow('RealSense depth Stream', depth_frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    # finally:
    #         cam.stop()
    #         cv2.destroyAllWindows()

    """
    输出双目图像 - 模式选择:
    - mode='infra': 双红外 (左红外+右红外)
    - mode='color_infra1': RGB+左红外
    - mode='color_infra2': RGB+右红外
    """
    mode = 'color_infra1'  # 修改这里切换模式

    cam=CameraD405()
    if mode == 'infra':
        # 双红外模式
        cam.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
        cam.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
        window_title = 'D405 Stereo - Infra1 + Infra2'
    elif mode == 'color_infra1':
        # RGB + 左红外模式
        cam.enable_stream(rs.stream.color, 0, 1280, 720, rs.format.bgr8, 30)
        cam.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
        window_title = 'D405 Stereo - RGB + Infra1'
    elif mode == 'color_infra2':
        # RGB + 右红外模式
        cam.enable_stream(rs.stream.color, 0, 1280, 720, rs.format.bgr8, 30)
        cam.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)
        window_title = 'D405 Stereo - RGB + Infra2'

    cam.start()
    try:
        while True:
            frame_dict = cam.get_stereo_frame(mode=mode)
            binocular_frame = frame_dict['binocular']

            # 显示画面
            cv2.imshow(window_title, binocular_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
            cam.stop()
            cv2.destroyAllWindows()

    """
    使用open3d渲染实时点云
    """
    # os.environ["XDG_SESSION_TYPE"] = "x11"
    # os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    # os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    # cam=CameraD405()
    # cam.enable_stream(rs.stream.depth, 1280,720, rs.format.z16, 30)
    # cam.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # cam.start()
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # pcd = o3d.geometry.PointCloud()
    # first_iter = True
    # try:
    #     while True:
    #         start_time = time.time()
    #         points,colors=cam.get_point_and_color()
    #         pcd.points = o3d.utility.Vector3dVector(points)
    #         pcd.colors = o3d.utility.Vector3dVector(colors)
    #         if first_iter:
    #             vis.add_geometry(pcd)
    #             first_iter = False
    #         else:
    #             vis.update_geometry(pcd)
    #
    #         vis.poll_events()
    #         vis.update_renderer()
    #
    #         fps = 1 / (time.time() - start_time)
    #         print(f"FPS: {fps:.2f}", end='\r')
    #
    #         if cv2.waitKey(1) == 27:
    #             break
    # finally:
    #     cam.stop()
    #     vis.destroy_window()
    #     cv2.destroyAllWindows()
