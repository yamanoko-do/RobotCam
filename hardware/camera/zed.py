import sys
sys.path.insert(0, './')
import pyzed.sl as sl
import numpy as np
import cv2
import open3d as o3d
import time
from typing import Tuple
import os
import glob


class CameraZED:
    def __init__(self):
        self.camera = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.coordinate_units = sl.UNIT.MILLIMETER
        self.init_params.sdk_verbose = 0

        self.left_enabled = False
        self.right_enabled = False
        self.depth_enabled = False
        self._started = False

        self.runtime_params = sl.RuntimeParameters()

        # 用于存储图像和深度数据的Mat对象
        self.left_mat = sl.Mat()
        self.right_mat = sl.Mat()
        self.depth_image_mat = sl.Mat()
        self.depth_measure_mat = sl.Mat()
        self.point_cloud_mat = sl.Mat()

        # 分辨率映射
        self.resolution_map = {
            (2208, 1242): sl.RESOLUTION.HD2K,
            (1920, 1080): sl.RESOLUTION.HD1080,
            (1280, 720): sl.RESOLUTION.HD720,
            (960, 600): sl.RESOLUTION.SVGA,
            (672, 376): sl.RESOLUTION.VGA
        }

        self.current_resolution = sl.RESOLUTION.HD720
        self.current_fps = 30

    def enable_stream(self, stream_type, width, height, format=None, framerate=None):
        """
        启用流
        Args:
            stream_type: 流类型 ('left', 'right', 'depth') 或 sl.VIEW
            width: 宽度
            height: 高度
            format: 格式（ZED相机自动处理，此参数保留用于兼容）
            framerate: 帧率（可选，仅在首次设置分辨率时需要）
        """
        if self._started:
            raise RuntimeError("Cannot enable stream after camera started. Call enable_stream() before start().")

        # 处理流类型
        if isinstance(stream_type, str):
            stream_type = stream_type.lower()
            if stream_type == 'left':
                self.left_enabled = True
            elif stream_type == 'right':
                self.right_enabled = True
            elif stream_type == 'depth':
                self.depth_enabled = True
        else:
            # 假设是sl.VIEW或其他类型
            stream_str = str(stream_type)
            if 'left' in stream_str.lower():
                self.left_enabled = True
            elif 'right' in stream_str.lower():
                self.right_enabled = True
            elif 'depth' in stream_str.lower():
                self.depth_enabled = True

        # 设置分辨率（仅当尚未设置或分辨率不同时）
        res_key = (width, height)
        new_resolution = self.resolution_map.get(res_key, sl.RESOLUTION.HD720)

        # 如果是新分辨率或者提供了framerate，则更新设置
        if new_resolution != self.current_resolution or framerate is not None:
            if res_key not in self.resolution_map:
                print(f"Warning: Resolution {width}x{height} not found, using HD720")
            self.current_resolution = new_resolution
            self.init_params.camera_resolution = self.current_resolution

            if framerate is not None:
                self.current_fps = framerate
                self.init_params.camera_fps = framerate

    def start(self):
        """启动相机"""
        if self._started:
            print("Camera already started.")
            return

        # 确保至少启用了一个流
        if not (self.left_enabled or self.right_enabled or self.depth_enabled):
            self.left_enabled = True
            self.depth_enabled = True

        err = self.camera.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        self._started = True
        print(f"ZED camera started at {self.get_resolution_text()} {self.current_fps}fps")

    def stop(self):
        """停止相机"""
        if self._started:
            self.camera.close()
            self._started = False
            print("ZED camera stopped")

    def get_frame(self, format2numpy=True) -> dict:
        """
        获取帧数据
        Returns:
            dict: 包含左图、右图、深度图的字典
        """
        if not self._started:
            raise RuntimeError("Camera not started. Call start() first.")

        frame_dict = {}

        if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            if self.left_enabled:
                self.camera.retrieve_image(self.left_mat, sl.VIEW.LEFT)
                if format2numpy:
                    left_data = self.left_mat.get_data()
                    # BGRA to BGR
                    if left_data.shape[2] == 4:
                        left_data = left_data[:, :, :3]
                    frame_dict["left"] = left_data
                else:
                    frame_dict["left"] = self.left_mat

            if self.right_enabled:
                self.camera.retrieve_image(self.right_mat, sl.VIEW.RIGHT)
                if format2numpy:
                    right_data = self.right_mat.get_data()
                    if right_data.shape[2] == 4:
                        right_data = right_data[:, :, :3]
                    frame_dict["right"] = right_data
                else:
                    frame_dict["right"] = self.right_mat

            if self.depth_enabled:
                self.camera.retrieve_image(self.depth_image_mat, sl.VIEW.DEPTH)
                self.camera.retrieve_measure(self.depth_measure_mat, sl.MEASURE.DEPTH)
                if format2numpy:
                    depth_image = self.depth_image_mat.get_data()
                    if depth_image.shape[2] == 4:
                        depth_image = depth_image[:, :, :3]
                    frame_dict["depth_view"] = depth_image
                    frame_dict["depth"] = self.depth_measure_mat.get_data()
                else:
                    frame_dict["depth_view"] = self.depth_image_mat
                    frame_dict["depth"] = self.depth_measure_mat

        return frame_dict if frame_dict else None

    def get_point_and_color(self):
        """
        获取点云数据和颜色，用于Open3D渲染
        Returns:
            verts(np.array): 点的三维坐标，shape为(n,3)
            colors(np.array): 点的颜色，范围[0,1]
        """
        if not (self.left_enabled and self.depth_enabled):
            print("获取点云需要同时启用left流和depth流")
            return None

        if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
            self.camera.retrieve_image(self.left_mat, sl.VIEW.LEFT)
            self.camera.retrieve_measure(self.point_cloud_mat, sl.MEASURE.XYZRGBA)

            # 获取点云数据
            point_cloud_data = self.point_cloud_mat.get_data()

            # 重塑为 (n, 4) 格式 - XYZRGBA
            points = point_cloud_data.reshape(-1, 4)

            # 分离XYZ和RGB
            verts = points[:, :3].astype(np.float64)
            rgba = points[:, 3].view(np.uint32)

            # 解压RGBA
            r = ((rgba >> 16) & 0xFF) / 255.0
            g = ((rgba >> 8) & 0xFF) / 255.0
            b = (rgba & 0xFF) / 255.0
            colors = np.stack([r, g, b], axis=1).astype(np.float64)

            # 过滤无效点（NaN或Inf）
            mask = np.isfinite(verts).all(axis=1)
            verts = verts[mask]
            colors = colors[mask]

            # 调整坐标系以匹配Open3D视角
            verts[:, 1] *= -1
            verts[:, 2] *= -1

            # 转换单位：毫米 -> 米（如果需要）
            # verts /= 1000.0

            return verts, colors

        return None

    def get_calibration_parameters(self):
        """获取校准参数（内参和外参）"""
        if not self._started:
            raise RuntimeError("Camera not started. Call start() first.")

        cam_info = self.camera.get_camera_information()
        calib = cam_info.camera_configuration.calibration_parameters
        return calib

    def get_intrinsics(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        获取左右相机的内参和畸变系数
        Returns:
            tuple: (left_intrinsics, left_coeffs, right_intrinsics, right_coeffs)
        """
        calib = self.get_calibration_parameters()

        left_cam = calib.left_cam
        right_cam = calib.right_cam

        # 左相机内参矩阵
        left_intrinsics = np.array([
            [left_cam.fx, 0, left_cam.cx],
            [0, left_cam.fy, left_cam.cy],
            [0, 0, 1]
        ])

        # 右相机内参矩阵
        right_intrinsics = np.array([
            [right_cam.fx, 0, right_cam.cx],
            [0, right_cam.fy, right_cam.cy],
            [0, 0, 1]
        ])

        # 畸变系数（取前5个）
        left_coeffs = np.array(left_cam.disto[:5])
        right_coeffs = np.array(right_cam.disto[:5])

        return left_intrinsics, left_coeffs, right_intrinsics, right_coeffs

    def get_extrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取右相机相对于左相机的外参
        Returns:
            tuple: (rotation_matrix, translation_vector)
        """
        calib = self.get_calibration_parameters()

        # 获取立体变换 (4x4 变换矩阵)
        stereo_transform = calib.stereo_transform
        transform_matrix = stereo_transform.m  # 4x4 numpy array

        # 旋转矩阵 (3x3) - 从变换矩阵中提取
        rotation = transform_matrix[:3, :3]

        # 平移向量 (3,) - 从变换矩阵中提取
        translation = transform_matrix[:3, 3]

        return rotation, translation

    def get_baseline(self) -> float:
        """获取基线距离（毫米）"""
        calib = self.get_calibration_parameters()
        return calib.get_camera_baseline()#毫米

    def get_resolution_text(self) -> str:
        """获取当前分辨率的文字描述"""
        res_map = {
            sl.RESOLUTION.HD2K: "HD2K (2208x1242)",
            sl.RESOLUTION.HD1080: "HD1080 (1920x1080)",
            sl.RESOLUTION.HD720: "HD720 (1280x720)",
            sl.RESOLUTION.SVGA: "SVGA (960x600)",
            sl.RESOLUTION.VGA: "VGA (672x376)"
        }
        return res_map.get(self.current_resolution, "Unknown")

    def print_camera_info(self):
        """打印相机信息"""
        cam_info = self.camera.get_camera_information()
        print(f"相机型号: {cam_info.camera_model}")
        print(f"序列号: {cam_info.serial_number}")
        print(f"分辨率: {self.get_resolution_text()}")
        print(f"基线: {self.get_baseline():.2f} mm")

    def export_rectify_maps(self, output_dir="./data/zed_calib", resolution=None, return_rect_params=False):
        """
        导出立体校正映射表，供 zed_uvc.py 使用
        使用 calibration_parameters_raw (未校正图像原始参数) 生成

        Args:
            output_dir: 保存映射表的目录
            resolution: 分辨率名称 ('HD2K', 'HD1080', 'HD720', 'VGA')，
                       如果为 None 则使用当前分辨率
            return_rect_params: 是否返回校正后的参数 (K_rect, baseline)
        """
        if not self._started:
            raise RuntimeError("Camera not started. Call start() first.")

        import cv2
        import os

        # 分辨率映射
        res_dim_map = {
            'HD2K': (2208, 1242),
            'HD1080': (1920, 1080),
            'HD720': (1280, 720),
            'SVGA': (960, 600),
            'VGA': (672, 376)
        }

        # 获取分辨率
        if resolution is None:
            res_text = self.get_resolution_text()
            for key in res_dim_map:
                if key in res_text:
                    resolution = key
                    break
            if resolution is None:
                resolution = 'HD720'

        if resolution not in res_dim_map:
            raise ValueError(f"不支持的分辨率: {resolution}")

        width, height = res_dim_map[resolution]
        image_size = (width, height)

        print(f"生成 {resolution} ({width}x{height}) 的校正映射表...")

        # 获取相机信息和校准参数
        cam_info = self.camera.get_camera_information()
        cam_config = cam_info.camera_configuration

        # 关键：使用原始参数 (未校正图像) 和校正后参数
        calib_raw = cam_config.calibration_parameters_raw
        calib_rect = cam_config.calibration_parameters

        # 原始相机参数 (未校正图像)
        left_cam_raw = calib_raw.left_cam
        right_cam_raw = calib_raw.right_cam

        # 校正后相机参数
        left_cam_rect = calib_rect.left_cam
        right_cam_rect = calib_rect.right_cam

        # 原始内参矩阵
        K1_raw = np.array([[left_cam_raw.fx, 0, left_cam_raw.cx],
                          [0, left_cam_raw.fy, left_cam_raw.cy],
                          [0, 0, 1]])
        K2_raw = np.array([[right_cam_raw.fx, 0, right_cam_raw.cx],
                          [0, right_cam_raw.fy, right_cam_raw.cy],
                          [0, 0, 1]])

        # 原始畸变系数
        D1_raw = np.array(left_cam_raw.disto[:5])
        D2_raw = np.array(right_cam_raw.disto[:5])

        # 校正后内参矩阵
        K1_rect = np.array([[left_cam_rect.fx, 0, left_cam_rect.cx],
                           [0, left_cam_rect.fy, left_cam_rect.cy],
                           [0, 0, 1]])
        K2_rect = np.array([[right_cam_rect.fx, 0, right_cam_rect.cx],
                           [0, right_cam_rect.fy, right_cam_rect.cy],
                           [0, 0, 1]])

        # 校正后的统一内参 (左右相机校正后内参几乎相同，取左相机即可)
        K_rect = K1_rect

        # 基线距离 (毫米)
        baseline = calib_rect.get_camera_baseline()

        # 外参 (从 raw 参数中获取)
        transform_matrix = calib_raw.stereo_transform.m
        R = transform_matrix[:3, :3]
        T = transform_matrix[:3, 3]

        # 计算立体校正
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K1_raw, D1_raw, K2_raw, D2_raw, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )

        # 生成映射表 - 从原始(未校正)图像映射到校正后图像
        map1x, map1y = cv2.initUndistortRectifyMap(
            K1_raw, D1_raw, R1, K1_rect, image_size, cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            K2_raw, D2_raw, R2, K2_rect, image_size, cv2.CV_32FC1
        )

        # 验证映射表（用 SDK 的图像测试）
        lu_img, lr_img, ru_img, rr_img = None, None, None, None
        diff_l, diff_r = -1, -1

        for _ in range(5):
            if self.camera.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                m_lu, m_lr, m_ru, m_rr = sl.Mat(), sl.Mat(), sl.Mat(), sl.Mat()
                self.camera.retrieve_image(m_lu, sl.VIEW.LEFT_UNRECTIFIED)
                self.camera.retrieve_image(m_lr, sl.VIEW.LEFT)
                self.camera.retrieve_image(m_ru, sl.VIEW.RIGHT_UNRECTIFIED)
                self.camera.retrieve_image(m_rr, sl.VIEW.RIGHT)
                lu_img, lr_img = m_lu.get_data(), m_lr.get_data()
                ru_img, rr_img = m_ru.get_data(), m_rr.get_data()

        if lu_img is not None:
            # 测试映射
            test_l = cv2.remap(lu_img, map1x, map1y, cv2.INTER_LINEAR)
            test_r = cv2.remap(ru_img, map2x, map2y, cv2.INTER_LINEAR)

            # 计算与 SDK 结果的差异
            def to_bgr(img):
                return img[:, :, :3] if img.shape[2] == 4 else img

            diff_l = np.abs(to_bgr(test_l).astype(float) - to_bgr(lr_img).astype(float)).mean()
            diff_r = np.abs(to_bgr(test_r).astype(float) - to_bgr(rr_img).astype(float)).mean()

            print(f"  左图与 SDK 差异: {diff_l:.3f} 像素")
            print(f"  右图与 SDK 差异: {diff_r:.3f} 像素")

            if diff_l < 1.0 and diff_r < 1.0:
                print("  ✓ 映射表质量良好")
            else:
                print("  ⚠️  映射表差异稍大，但仍可用")

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "map1x.npy"), map1x)
        np.save(os.path.join(output_dir, "map1y.npy"), map1y)
        np.save(os.path.join(output_dir, "map2x.npy"), map2x)
        np.save(os.path.join(output_dir, "map2y.npy"), map2y)

        # 保存完整参数供参考
        np.savez(os.path.join(output_dir, "calib_params.npz"),
                 K1_raw=K1_raw, D1_raw=D1_raw,
                 K2_raw=K2_raw, D2_raw=D2_raw,
                 K1_rect=K1_rect, K2_rect=K2_rect,
                 R=R, T=T, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
                 resolution=resolution, image_size=image_size,
                 diff_left=diff_l, diff_right=diff_r)

        # 保存 OpenStereo 格式的 K.txt (内参矩阵 + 基线)
        k_txt_path = os.path.join(output_dir, "K.txt")
        with open(k_txt_path, 'w') as f:
            # 第一行：内参矩阵 9 个元素 (按行展开)
            f.write(f"{K_rect[0,0]:.10f} {K_rect[0,1]:.10f} {K_rect[0,2]:.10f} "
                   f"{K_rect[1,0]:.10f} {K_rect[1,1]:.10f} {K_rect[1,2]:.10f} "
                   f"{K_rect[2,0]:.10f} {K_rect[2,1]:.10f} {K_rect[2,2]:.10f}\n")
            # 第二行：基线 (毫米)
            f.write(f"{baseline:.10f}\n")
        print(f"✓ 内参文件已保存到: {k_txt_path}")

        # 保存验证图像
        if lu_img is not None:
            cv2.imwrite(os.path.join(output_dir, "verify_left.png"),
                        np.hstack([to_bgr(lu_img), to_bgr(test_l), to_bgr(lr_img)]))

        print(f"✓ 校正映射表已保存到: {output_dir}")

        if return_rect_params:
            return output_dir, K_rect, baseline
        return output_dir

    def take_photo(self, save_dir="./data/chessboard_images/monocular", resolution='HD720'):
        """
        启动实时预览，按 's' 保存图像，按 'q' 或 ESC 退出
        Args:
            save_dir (str): 保存图像的目录路径
            resolution: 分辨率选项 ('HD2K', 'HD1080', 'HD720', 'SVGA', 'VGA')
        """
        if not self._started:
            # 分辨率配置
            res_config = {
                'HD2K': (2208, 1242, 15),
                'HD1080': (1920, 1080, 30),
                'HD720': (1280, 720, 60),
                'SVGA': (960, 600, 100),
                'VGA': (672, 376, 100)
            }
            if resolution not in res_config:
                print(f"未知分辨率: {resolution}, 使用 HD720")
                resolution = 'HD720'
            width, height, max_fps = res_config[resolution]
            self.enable_stream('left', width, height, framerate=max_fps)
            self.start()

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 确定起始编号
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

        print("ZED相机预览中... 按 's' 保存图像，按 'q' 或 ESC 退出。")
        try:
            while True:
                frame_dict = self.get_frame()
                if frame_dict is None or "left" not in frame_dict:
                    continue

                left_frame = frame_dict["left"]
                cv2.imshow('ZED Camera - Press s to save, q/ESC to quit', left_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    save_path = os.path.join(save_dir, f"image_{photo_count}.jpg")
                    cv2.imwrite(save_path, left_frame)
                    print(f"已保存: {save_path}")
                    photo_count += 1
                elif key in (ord('q'), 27):
                    break

        finally:
            cv2.destroyAllWindows()

    @staticmethod
    def get_support_config():
        """打印支持的分辨率配置"""
        print("ZED相机支持的分辨率:")
        print("  HD2K: 2208x1242")
        print("  HD1080: 1920x1080")
        print("  HD720: 1280x720")
        print("  SVGA: 960x600")
        print("  VGA: 672x376")
        print("支持的流类型: left, right, depth")


def test_generate_rectify_maps():
    """测试：生成所有分辨率的校正映射表"""
    print("=== 生成 ZED 所有分辨率校正映射表 ===\n")

    # 分辨率配置: (名称, 宽度, 高度, 最高FPS)
    resolutions = [
        ('HD2K', 2208, 1242, 15),
        ('HD1080', 1920, 1080, 30),
        ('HD720', 1280, 720, 60),
        ('SVGA', 960, 600, 100),
        ('VGA', 672, 376, 100)
    ]

    for res_name, width, height, max_fps in resolutions:
        print(f"\n{'='*60}")
        print(f"处理分辨率: {res_name} ({width}x{height}) @ {max_fps}fps")
        print(f"{'='*60}")

        cam = CameraZED()
        cam.enable_stream('left', width, height, framerate=max_fps)
        cam.enable_stream('right', width, height)

        try:
            cam.start()
            output_dir = f"./data/zed_calib/{res_name.lower()}"
            _, K_rect, baseline = cam.export_rectify_maps(output_dir, return_rect_params=True)

            print(f"\n--- {res_name} 校正后的统一内参和基线 ---")
            print(f"统一内参矩阵 K_rect:")
            print(f"  [ {K_rect[0,0]:.4f}, {K_rect[0,1]:.4f}, {K_rect[0,2]:.4f} ]")
            print(f"  [ {K_rect[1,0]:.4f}, {K_rect[1,1]:.4f}, {K_rect[1,2]:.4f} ]")
            print(f"  [ {K_rect[2,0]:.4f}, {K_rect[2,1]:.4f}, {K_rect[2,2]:.4f} ]")
            print(f"焦距 fx = {K_rect[0,0]:.4f}, fy = {K_rect[1,1]:.4f}")
            print(f"光心 cx = {K_rect[0,2]:.4f}, cy = {K_rect[1,2]:.4f}")
            print(f"基线: {baseline:.4f} mm")
            print(f"----------------------------------------")

            print(f"\n✓ {res_name} 映射表已生成到: {output_dir}")
        except Exception as e:
            print(f"\n✗ {res_name} 失败: {e}")
        finally:
            cam.stop()

    print(f"\n{'='*60}")
    print("所有分辨率处理完成！")
    print(f"{'='*60}")
    print("\n在 zed_uvc.py 中使用:")
    print("  from zed_uvc import ZEDUVC")
    print("  cam = ZEDUVC(map_dir='./data/zed_calib/hd720', resolution='HD720')")
    print("  # 或")
    print("  cam = ZEDUVC(map_dir='./data/zed_calib/hd1080', resolution='HD1080')")


def test_basic_camera(resolution='HD720'):
    """测试：基本相机功能 - 显示左右图和深度图

    Args:
        resolution: 分辨率选项 ('HD2K', 'HD1080', 'HD720', 'SVGA', 'VGA')
    """
    print("=== 测试 ZED 基本相机功能 ===\n")

    # 分辨率配置
    res_config = {
        'HD2K': (2208, 1242, 15),
        'HD1080': (1920, 1080, 30),
        'HD720': (1280, 720, 60),
        'SVGA': (960, 600, 100),
        'VGA': (672, 376, 100)
    }

    if resolution not in res_config:
        print(f"未知分辨率: {resolution}, 使用 HD720")
        resolution = 'HD720'

    width, height, max_fps = res_config[resolution]
    print(f"使用分辨率: {resolution} ({width}x{height}), 最大FPS: {max_fps}")

    cam = CameraZED()
    cam.enable_stream('left', width, height, framerate=max_fps)
    cam.enable_stream('right', width, height, framerate=max_fps)
    cam.enable_stream('depth', width, height, framerate=max_fps)
    cam.start()

    try:
        cam.print_camera_info()

        # 获取内参
        print("\n相机内参:")
        left_intr, left_coeff, right_intr, right_coeff = cam.get_intrinsics()
        print("左相机内参:")
        print(left_intr)
        print("左相机畸变:", left_coeff)

        # 获取外参
        print("\n相机外参:")
        rot, trans = cam.get_extrinsics()
        print("旋转矩阵:")
        print(rot)
        print("平移向量:", trans)
        print(f"基线: {cam.get_baseline():.2f} mm")

        # 测试获取帧
        print(f"\n测试获取帧 (按 'q' 退出), 当前分辨率: {resolution} ({width}x{height})")

        # FPS统计
        frame_count = 0
        fps_start_time = time.time()
        current_fps = 0

        while True:
            frame_dict = cam.get_frame()

            if frame_dict:
                # 计算FPS
                frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    current_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start_time = time.time()

                # 在图像上显示FPS
                if "left" in frame_dict:
                    left_img = frame_dict["left"].copy()
                    cv2.putText(left_img, f"FPS: {current_fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Left Image", left_img)
                if "right" in frame_dict:
                    cv2.imshow("Right Image", frame_dict["right"])
                if "depth_view" in frame_dict:
                    cv2.imshow("Depth View", frame_dict["depth_view"])

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()


def test_point_cloud(resolution='HD720'):
    """测试：使用 Open3D 渲染点云

    Args:
        resolution: 分辨率选项 ('HD2K', 'HD1080', 'HD720', 'SVGA', 'VGA')
    """
    print("=== 测试 ZED 点云渲染 ===\n")

    # 分辨率配置
    res_config = {
        'HD2K': (2208, 1242, 15),
        'HD1080': (1920, 1080, 30),
        'HD720': (1280, 720, 60),
        'SVGA': (960, 600, 100),
        'VGA': (672, 376, 100)
    }

    if resolution not in res_config:
        print(f"未知分辨率: {resolution}, 使用 HD720")
        resolution = 'HD720'

    width, height, max_fps = res_config[resolution]
    print(f"使用分辨率: {resolution} ({width}x{height}), 最大FPS: {max_fps}")

    os.environ["XDG_SESSION_TYPE"] = "x11"
    os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

    cam = CameraZED()
    cam.enable_stream('left', width, height, framerate=max_fps)
    cam.enable_stream('depth', width, height)
    cam.start()

    # 创建Open3D可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    first_iter = True

    try:
        while True:
            start_time = time.time()
            points, colors = cam.get_point_and_color()

            if points is not None:
                # 更新点云数据
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)

                # 首次迭代设置视角
                if first_iter:
                    vis.add_geometry(pcd)
                    first_iter = False
                else:
                    vis.update_geometry(pcd)

                # 更新渲染
                vis.poll_events()
                vis.update_renderer()

                # 计算并显示帧率
                fps = 1 / (time.time() - start_time)
                print(f"FPS: {fps:.2f}", end='\r')

            # 按ESC退出
            if cv2.waitKey(1) == 27:
                break
    finally:
        cam.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()


def test_take_photo(resolution='HD720'):
    """测试：拍摄单目照片

    Args:
        resolution: 分辨率选项 ('HD2K', 'HD1080', 'HD720', 'SVGA', 'VGA')
    """
    print("=== 测试 ZED 拍摄照片 ===\n")

    cam = CameraZED()
    cam.take_photo(save_dir="./data/chessboard_images/monocular", resolution=resolution)


def test_rgb_only(resolution='HD720'):
    """测试：仅RGB左图性能测试（无深度计算，最大FPS）

    Args:
        resolution: 分辨率选项 ('HD2K', 'HD1080', 'HD720', 'SVGA', 'VGA')
    """
    print("=== 测试 ZED RGB-only 性能 ===\n")

    # 分辨率配置
    res_config = {
        'HD2K': (2208, 1242, 15),
        'HD1080': (1920, 1080, 30),
        'HD720': (1280, 720, 60),
        'SVGA': (960, 600, 100),
        'VGA': (672, 376, 100)
    }

    if resolution not in res_config:
        print(f"未知分辨率: {resolution}, 使用 HD720")
        resolution = 'HD720'

    width, height, max_fps = res_config[resolution]
    print(f"使用分辨率: {resolution} ({width}x{height}), 理论最大FPS: {max_fps}")

    cam = CameraZED()

    # 仅启用左图，不启用深度和右图
    cam.enable_stream('left', width, height, framerate=max_fps)

    # 覆盖深度模式为性能模式（虽然不启用深度，但确保初始化参数合理）
    cam.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    cam.start()

    try:
        cam.print_camera_info()

        # 测试1: 纯 grab 性能（不 retrieve 图像）
        print(f"\n{'='*60}")
        print("测试1: 纯 grab 性能（不检索图像）")
        print(f"{'='*60}")
        frame_count = 0
        warmup_frames = 5
        pure_grab_fps = 0
        timeout = 15.0
        debug_print_interval = 50
        warmed_up = False
        stats_start_time = 0

        while True:
            loop_start = time.time()
            err = cam.camera.grab(cam.runtime_params)

            if err == sl.ERROR_CODE.SUCCESS:
                frame_count += 1

                if not warmed_up:
                    if frame_count >= warmup_frames:
                        warmed_up = True
                        frame_count = 0
                        stats_start_time = time.time()
                        print(f"预热完成 ({warmup_frames}帧)，开始统计...")
                    continue

                if frame_count % debug_print_interval == 0:
                    elapsed = time.time() - stats_start_time
                    inst_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"  已获取 {frame_count} 帧... 当前: {inst_fps:.1f} fps")

                elapsed = time.time() - stats_start_time
                if elapsed >= 2.0:
                    pure_grab_fps = frame_count / elapsed
                    print(f"纯 grab FPS: {pure_grab_fps:.1f} (目标: {max_fps})")
                    print(f"达到理论值的: {pure_grab_fps/max_fps*100:.1f}%")
                    break
            else:
                print(f"grab 失败: {err}")
                time.sleep(0.1)

            if time.time() - loop_start > timeout:
                print(f"超时！已获取 {frame_count} 帧")
                if frame_count > 0 and warmed_up:
                    elapsed = time.time() - stats_start_time
                    pure_grab_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"估算 FPS: {pure_grab_fps:.1f}")
                break

        # 测试2: grab + retrieve 左图
        print(f"\n{'='*60}")
        print("测试2: grab + retrieve 左图（不显示）")
        print(f"{'='*60}")
        frame_count = 0
        retrieve_fps = 0
        left_mat = sl.Mat()
        test2_timeout = 15.0
        warmed_up = False
        stats_start_time = 0

        while True:
            loop_start = time.time()
            if cam.camera.grab(cam.runtime_params) == sl.ERROR_CODE.SUCCESS:
                cam.camera.retrieve_image(left_mat, sl.VIEW.LEFT)
                # 只获取数据不做处理
                _ = left_mat.get_data()
                frame_count += 1

                if not warmed_up:
                    if frame_count >= warmup_frames:
                        warmed_up = True
                        frame_count = 0
                        stats_start_time = time.time()
                        print(f"预热完成 ({warmup_frames}帧)，开始统计...")
                    continue

                if frame_count % debug_print_interval == 0:
                    elapsed = time.time() - stats_start_time
                    inst_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"  已获取 {frame_count} 帧... 当前: {inst_fps:.1f} fps")

                elapsed = time.time() - stats_start_time
                if elapsed >= 2.0:
                    retrieve_fps = frame_count / elapsed
                    print(f"grab+retrieve FPS: {retrieve_fps:.1f} (目标: {max_fps})")
                    print(f"达到理论值的: {retrieve_fps/max_fps*100:.1f}%")
                    break

            if time.time() - loop_start > test2_timeout:
                print(f"超时！已获取 {frame_count} 帧")
                if frame_count > 0 and warmed_up:
                    elapsed = time.time() - stats_start_time
                    retrieve_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"估算 FPS: {retrieve_fps:.1f}")
                break

        # 测试3: 完整显示
        print(f"\n{'='*60}")
        print("测试3: 完整流程（含显示）按 'q' 退出")
        print(f"{'='*60}")
        frame_count = 0
        fps_start_time = time.time()
        current_fps = 0

        while True:
            frame_dict = cam.get_frame()

            if frame_dict and "left" in frame_dict:
                frame_count += 1
                elapsed = time.time() - fps_start_time
                if elapsed >= 0.5:
                    current_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start_time = time.time()

                left_img = frame_dict["left"].copy()
                cv2.putText(left_img, f"FPS: {current_fps:.1f} / {max_fps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(left_img, f"Res: {resolution}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("RGB Only", left_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        print(f"\n{'='*60}")
        print("性能总结:")
        print(f"{'='*60}")
        print(f"分辨率: {resolution} ({width}x{height})")
        print(f"理论最大 FPS: {max_fps}")
        print(f"纯 grab FPS: {pure_grab_fps:.1f}")
        print(f"grab+retrieve FPS: {retrieve_fps:.1f}")
        print(f"{'='*60}")

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    """
    测试1: 生成校正映射表 (供 zed_uvc.py 使用)
    """
    # test_generate_rectify_maps()

    """
    测试2: 基本相机功能 - 显示图像和打印参数
    分辨率选项: 'HD2K', 'HD1080', 'HD720', 'SVGA', 'VGA'
    """
    # test_basic_camera('VGA')    # 672x376 @ 100fps
    # test_basic_camera('HD720')  # 1280x720 @ 60fps
    # test_basic_camera('HD1080') # 1920x1080 @ 30fps
    # test_basic_camera('HD2K')   # 2208x1242 @ 15fps

    """
    测试3: 点云渲染 (需要 Open3D)
    分辨率选项: 'HD2K', 'HD1080', 'HD720', 'SVGA', 'VGA'
    """
    # test_point_cloud('VGA')

    """
    测试4: 拍摄照片用于标定
    分辨率选项: 'HD2K', 'HD1080', 'HD720', 'SVGA', 'VGA'
    """
    # test_take_photo('HD720')

    """
    测试5: RGB-only 性能测试（无深度，最大FPS）
    分辨率选项: 'HD2K', 'HD1080', 'HD720', 'SVGA', 'VGA'
    """
    test_rgb_only('VGA')      # 672x376 @ 100fps
    # test_rgb_only('HD720')    # 1280x720 @ 60fps
    # test_rgb_only('HD1080')   # 1920x1080 @ 30fps
