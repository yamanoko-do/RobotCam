import cv2
import pyudev
import os
import glob
import cv2
import numpy as np
import math
from hardware.camera.basecamera import BaseCamera


class BinocularCam(BaseCamera):
    """
    仅在linux上可正常使用该相机
    """
    def __init__(self, device_id=None, vid_pid="1bcf:0b15"):
        """
        初始化双目摄像头
        Args:
            device_id (int): （可选）明确指定 /dev/videoX 的编号
            vid_pid (str): （可选）通过 USB VID:PID 自动查找设备，例如 "1bcf:0b15"
        """
        self.device_path = None
        width = 2560  # 双目合并后的宽度
        height = 720  # 高度
        fmt='MJPG' # MJPG 格式以提高帧率

        if vid_pid:
            self.device_path = self._find_video_device(vid_pid)
            if not self.device_path:
                raise RuntimeError(f"未找到 VID:PID = {vid_pid} 对应的摄像头")
            device_id = int(self.device_path.replace("/dev/video", ""))
            print(f"[INFO] 根据 VID:PID={vid_pid} 自动识别设备: {self.device_path}")
        elif device_id is not None:
            self.device_path = f"/dev/video{device_id}"
            print(f"[INFO] 使用手动指定的设备: {self.device_path}")
        else:
            raise ValueError(
                "必须指定参数：\n"
                "  device_id  → 可用命令查看：v4l2-ctl --list-devices\n"
                "  或 vid_pid → 可用命令查看：lsusb"
            )

        # 打开视频流
        print(device_id)
        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.device_path}")

        # 设置分辨率与格式
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # 验证设置是否生效
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] 摄像头初始化成功, 分辨率: {w}x{h},帧率: {self.cap.get(cv2.CAP_PROP_FPS)},格式: {fmt}")

    def get_frame(self):
        """获取一帧图像并返回左右目和完整帧
        Returns:
            dict: {'left': img_left, 'right': img_right, 'binocular': frame}
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise RuntimeError("读取相机帧失败")

        h, w = frame.shape[:2]
        mid = w // 2
        frame_left = frame[:, :mid]
        frame_right = frame[:, mid:]

        return {
            'left': frame_left,
            'right': frame_right,
            'binocular': frame
        }


    def adjust_focus_assistant(self, window_name="Focus Assistant", history_size=10):
        """
        调焦辅助工具：实时显示左右眼清晰度，帮助精确调焦
        Args:
            window_name (str): 窗口名称
            history_size (int): 历史数据记录长度，用于计算平均值
        """
        print("=" * 50)
        print("调焦辅助模式已启动")
        print("旋转焦距环时观察清晰度数值变化")
        print("清晰度数值越高表示图像越清晰")
        print("按 'q' 退出调焦模式")
        print("=" * 50)
        
        left_sharpness_history = []
        right_sharpness_history = []
        
        def calculate_sharpness(image):
            """
            计算图像清晰度（使用拉普拉斯方差法）
            Args:
                image: 输入图像
            Returns:
                float: 清晰度得分
            """
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        try:
            while True:
                frames = self.get_frame()
                left_frame = frames['left']
                right_frame = frames['right']
                
                # 计算当前帧的清晰度
                left_sharpness = calculate_sharpness(left_frame)
                right_sharpness = calculate_sharpness(right_frame)
                
                # 更新历史记录
                left_sharpness_history.append(left_sharpness)
                right_sharpness_history.append(right_sharpness)
                
                # 保持历史记录长度
                if len(left_sharpness_history) > history_size:
                    left_sharpness_history.pop(0)
                    right_sharpness_history.pop(0)
                
                # 计算平均清晰度（平滑显示）
                avg_left = np.mean(left_sharpness_history)
                avg_right = np.mean(right_sharpness_history)
                
                # 在图像上显示清晰度信息
                display_frame = frames['binocular'].copy()
                
                # 添加清晰度信息文本
                info_text = [
                    f"Left Sharpness: {avg_left:.1f}",
                    f"Right Sharpness: {avg_right:.1f}",
                    "Press 'q' to exit focus mode"
                ]
                
                # 在图像上绘制文本
                for i, text in enumerate(info_text):
                    y_position = 30 + i * 30
                    cv2.putText(display_frame, text, (10, y_position), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 添加分隔线
                h, w = display_frame.shape[:2]
                cv2.line(display_frame, (w//2, 0), (w//2, h), (0, 255, 0), 2)
                
                # 显示图像
                cv2.imshow(window_name, display_frame)
                
                # 同时在控制台输出清晰度信息（可选）
                print(f"\r左眼清晰度: {avg_left:8.1f} | 右眼清晰度: {avg_right:8.1f}", end="", flush=True)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n退出调焦模式")
                    break
                    
        except KeyboardInterrupt:
            print("\n用户中断调焦模式")
        finally:
            cv2.destroyWindow(window_name)

    def take_photo(self, save_dir="./data/chessboard_images/binocular"):
        """
        启动实时预览，按 's' 保存 left_N.jpg 和 right_N.jpg，按 'q' 或 ESC 退出。
        Args:
            save_dir (str): 保存图像的目录路径
        """
        os.makedirs(save_dir, exist_ok=True)

        # 确定起始编号（基于已存在的 left_*.jpg）
        photo_count = 1
        existing_files = glob.glob(os.path.join(save_dir, "left_*.jpg"))
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

        print("双目相机预览中... 按 's' 保存图像，按 'q' 或 ESC 退出。")
        try:
            while True:
                frames = self.get_frame()
                cv2.imshow('Binocular Camera - Press s to save, q/ESC to quit', frames['binocular'])
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    left_path = os.path.join(save_dir, f"left_{photo_count}.jpg")
                    right_path = os.path.join(save_dir, f"right_{photo_count}.jpg")
                    cv2.imwrite(left_path, frames['left'])
                    cv2.imwrite(right_path, frames['right'])
                    print(f"已保存: {left_path} | {right_path}")
                    photo_count += 1

                elif key in (ord('q'), 27):  # 'q' or ESC
                    break

        finally:
            cv2.destroyAllWindows()

    def stop(self):
        """释放摄像头资源"""
        if self.cap.isOpened():
            self.cap.release()
            print(f"[INFO] 摄像头 {self.device_path} 已释放")
    
    def pnp_check_chessboard(self, chessboard_size):
        """
        同时显示左右眼相机的PnP棋盘格检测结果
        Args:
            chessboard_size (list): 棋盘格参数 [width, height, square_size]
        """
        # 获取相机参数
        intrinsics = self.get_intrinsics()
        mtx_left = intrinsics['mtx_left']
        dist_left = intrinsics['dist_left']
        mtx_right = intrinsics['mtx_right']
        dist_right = intrinsics['dist_right']

        # 定义世界坐标系点
        p_world = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        p_world[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        p_world *= chessboard_size[2]

        # 角点细化条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # 创建显示窗口
        window_name = "Binocular PnP - Left & Right Cameras"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("=" * 60)
        print("双目PnP棋盘格检测模式已启动")
        print(f"棋盘格参数：内角点 {chessboard_size[0]}x{chessboard_size[1]}，方格边长 {chessboard_size[2]}mm")
        print("按 'q' 或 ESC 退出检测模式")
        print("=" * 60)

        try:
            while True:
                # 获取双目图像
                frames = self.get_frame()
                
                # 创建显示画布
                display_frame = np.zeros((720, 2560, 3), dtype=np.uint8)
                
                # 处理左眼图像
                left_processed, left_pnp = self._solve_and_visualize_pnp(
                    frames['left'], mtx_left, dist_left, chessboard_size, p_world, criteria, "LEFT"
                )
                
                # 处理右眼图像
                right_processed, right_pnp = self._solve_and_visualize_pnp(
                    frames['right'], mtx_right, dist_right, chessboard_size, p_world, criteria, "RIGHT"
                )
                
                # 拼接显示
                display_frame[:, :1280] = left_processed
                display_frame[:, 1280:] = right_processed
                
                # 添加分隔线和标题
                cv2.line(display_frame, (1280, 0), (1280, 720), (255, 255, 255), 2)
                cv2.putText(display_frame, "LEFT CAMERA", (500, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, "RIGHT CAMERA", (1780, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 显示图像
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('q'), 27):
                    print("\n退出双目PnP棋盘格检测模式")
                    break

        except KeyboardInterrupt:
            print("\n用户中断PnP检测模式")
        finally:
            cv2.destroyWindow(window_name)

    @staticmethod
    def get_intrinsics():
        """
        获取相机内参和畸变系数
        """
        # 左目相机内参矩阵
        mtx_left = np.array([[667.9168283725817, 0.0, 614.6288637957155], [0.0, 668.0460497679354, 346.0298183178211], [0.0, 0.0, 1.0]])
        
        # 左目相机畸变系数
        dist_left = np.array([-0.047259990825834194, -0.04440804570994635, -0.00030253164833697464, -0.000526020240064694, 0.042117973320535])
        dist_left = np.zeros((5, 1))
        # 右目相机内参矩阵
        mtx_right = np.array([[664.8425044613646, 0.0, 637.8140118078181], [0.0, 665.1471198265182, 352.7132595594718], [0.0, 0.0, 1.0]])
        
        # 右目相机畸变系数
        dist_right = np.array([-0.049725137737477484, -0.04598226254568722, 0.00027690607890856793, -1.8089702791640458e-05, 0.046316177986347636])
        dist_right = np.zeros((5, 1))
        return {
            'mtx_left': mtx_left,
            'dist_left': dist_left,
            'mtx_right': mtx_right,
            'dist_right': dist_right
        }

    @staticmethod
    def _find_video_device(vid_pid):
        """根据 VID:PID 查找对应的 /dev/videoX"""
        context = pyudev.Context()
        for device in context.list_devices(subsystem='video4linux'):
            parent = device.find_parent('usb', 'usb_device')
            if not parent:
                continue

            # 有的 pyudev 版本提示属性接口变化，这里兼容处理
            vid = parent.attributes.get('idVendor')
            pid = parent.attributes.get('idProduct')
            if not vid or not pid:
                continue

            if f"{vid.decode()}:{pid.decode()}".lower() == vid_pid.lower():
                return device.device_node
        return None


# 示例使用
if __name__ == "__main__":
    cam = BinocularCam()

    try:
        # 使用调焦辅助功能
        cam.adjust_focus_assistant()
        
    finally:
        cam.stop()
        cv2.destroyAllWindows()





