import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import time
from typing import Tuple

class CameraD435:
    def __init__(self):
        """
        初始化
        """
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.color_frame_enable=False
        self.depth_frame_enable=False
        self.pc = rs.pointcloud()# 创建相机点云对象，用于后续对齐深度和彩色
        self.align = rs.align(rs.stream.color)#创建对其对象，对齐到color坐标
        self.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
        self.start()

    def enable_stream(self,stream_type,width,height,format,framerate):
        if str(stream_type)=="stream.color":
            self.color_frame_enable=True
        elif str(stream_type)=="stream.depth":
            self.depth_frame_enable=True

        self.config.enable_stream(stream_type,width,height,format,framerate)


    def start(self):
        # 启动数据流
        self.pipeline.start(self.config)

    
    def stop(self):
        # 关闭数据流
        self.pipeline.stop()

    def get_frame(self,format2numpy=True)-> dict:
        frames = self.pipeline.wait_for_frames()
        #当color流和depth流都开启时，执行对齐
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
            # 获取深度帧和彩色帧
            depth_frame = frame_dict["depth"]
            color_frame = frame_dict["color"]

            self.pc.map_to(color_frame)#将点云的坐标映射到rgb坐标系下
            points = self.pc.calculate(depth_frame)#据深度帧生成点云和纹理映射
            # 获取点云数据
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)#(307200, 3)
            texcoords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)#(307200, 2)
            
            #verts[:, 0] *= -1  # 反转 X 轴方向
            verts[:, 1] *= -1  # 反转 Y 轴方向
            verts[:, 2] *= -1  # 反转 Z 轴方向

            verts = verts.astype(np.float64)#提前将数据转换为Open3D所需的float64类型，减少隐式转换开销
        
            # 获取彩色图像并转换为RGB
            color_image = np.asanyarray(color_frame.get_data())[..., ::-1]  # BGR转RGB,(720, 1280, 3)

            # 提取颜色信息
            u = (texcoords[:, 0] * (color_image.shape[1] - 1)).astype(int)
            v = (texcoords[:, 1] * (color_image.shape[0] - 1)).astype(int)
            u = np.clip(u, 0, color_image.shape[1] - 1)
            v = np.clip(v, 0, color_image.shape[0] - 1)
            #print(color_image)
            colors = color_image[v, u] / 255.0  # 归一化到[0,1]
            return verts,colors
    


    #获取支持的流参数
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
    
    #获取内参
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
        
        # 启用深度和彩色流
        config.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
        #config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # 启动设备
        pipeline.start(config)
        profile = pipeline.get_active_profile()

        # 获取彩色和深度相机的内参
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
        
        # print("=== Color Camera Intrinsics ===")
        # print(f"Width: {color_intrinsics.width}, Height: {color_intrinsics.height}")
        # print(f"fx: {color_intrinsics.fx}, fy: {color_intrinsics.fy}")
        # print(f"cx: {color_intrinsics.ppx}, cy: {color_intrinsics.ppy}")
        # print(f"Distortion Model: {color_intrinsics.model}")
        # print(f"Distortion Coeffs: {color_intrinsics.coeffs}")

        # print("\n=== Depth Camera Intrinsics ===")
        # print(f"Width: {depth_intrinsics.width}, Height: {depth_intrinsics.height}")
        # print(f"fx: {depth_intrinsics.fx}, fy: {depth_intrinsics.fy}")
        # print(f"cx: {depth_intrinsics.ppx}, cy: {depth_intrinsics.ppy}")
        # print(f"Distortion Model: {depth_intrinsics.model}")
        # print(f"Distortion Coeffs: {depth_intrinsics.coeffs}")

        pipeline.stop()
        return color_intrinsics_,np.array(color_intrinsics.coeffs),depth_intrinsics_,np.array(depth_intrinsics.coeffs)
    
    
    @staticmethod
    def get_extrinsics_depth2rgb():
        """
        获取外参
        """
        pipeline = rs.pipeline()
        config = rs.config()
        
        # 启用深度和彩色流
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # 启动设备
        pipeline.start(config)
        profile = pipeline.get_active_profile()

        # 获取外参（深度 -> 颜色）
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
    #CameraD435.get_support_config()

    """
    输出rgb流
    """
    # cam=CameraD435()
    # cam.enable_stream(rs.stream.color, 1920,1080, rs.format.bgr8, 30)
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
    # cam=CameraD435()
    # cam.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)
    # cam.start()
    # try:
    #     while True:
    #         frame_dict=cam.get_frame()
    #         depth_frame = frame_dict['depth']

    #         # 显示画面
    #         cv2.imshow('RealSense depth Stream', depth_frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    # finally:
    #         cam.stop()
    #         cv2.destroyAllWindows()

    """
    使用open3d渲染实时点云
    """
    cam=CameraD435()
    cam.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cam.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    cam.start()
    # 创建Open3D可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    first_iter = True
    try:
        while True:
            start_time = time.time()
            points,colors=cam.get_point_and_color()
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
    """
    打开实时rgb流,并检查鼠标所在像素点的距离rgb_frame
    """
    # def show_distance(event, x, y, args, params):
    #     global point
    #     point = (x, y)


    # cam = CameraD435()
    # cam.enable_stream(rs.stream.depth,  1280,  720, rs.format.z16, 6)
    # cam.enable_stream(rs.stream.color,  1280,  720, rs.format.bgr8, 30)
    # cam.start()
    # # Create mouse event
    # point = (400, 300)
    # cv2.namedWindow("Color frame")
    # cv2.setMouseCallback("Color frame", show_distance)

    # while True:
    #     frame_dict = cam.get_frame()
    #     color_frame = frame_dict['color']
    #     depth_frame = frame_dict['depth']

    #     # Show distance for a specific point
    #     cv2.circle(color_frame, point, 4, (0, 0, 255))
    #     distance = depth_frame[point[1], point[0]]

    #     cv2.putText(color_frame, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

    #     #cv2.imshow("depth frame", depth_frame)
    #     cv2.imshow("Color frame", color_frame)
    #     key = cv2.waitKey(1)# 按下 ESC 键
    #     if key == 27:
    #         cam.stop()
    #         break


