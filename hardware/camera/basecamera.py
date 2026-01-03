import abc
import cv2
import numpy as np
import math

class BaseCamera(abc.ABC):
    """
    相机抽象基类
    """

    @abc.abstractmethod
    def __init__(self):
        """初始化相机"""
        pass

    @abc.abstractmethod
    def get_frame(self):
        """获取一帧图像"""
        pass
    
    @abc.abstractmethod
    def stop(self):
        """释放相机资源"""
        pass
    
    def _solve_and_visualize_pnp(self, frame, mtx, dist, chessboard_size, p_world, criteria, camera_name):
        """
        PnP计算和可视化合并函数 - 对单帧图像进行PnP计算并直接绘制结果
        Args:
            frame: 输入图像
            mtx: 相机内参矩阵
            dist: 畸变系数
            chessboard_size: 棋盘格尺寸
            p_world: 世界坐标系点
            criteria: 角点细化条件
            camera_name: 相机名称（用于显示）
        Returns:
            processed_frame: 处理后的图像
            pnp_data: PnP计算结果字典
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = frame.copy()
        
        # 初始化返回数据
        pnp_data = {'success': False, 'detected': False}
        
        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(
            gray, chessboard_size[:2], None, cv2.CALIB_CB_FAST_CHECK
        )
        
        if not ret:
            cv2.putText(processed_frame, f"{camera_name}: No Chessboard", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return processed_frame, pnp_data
        
        pnp_data['detected'] = True
        
        # 角点细化
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # PnP求解
        success, rvec, tvec = cv2.solvePnP(p_world, corners_refined, mtx, dist)
        
        if not success:
            cv2.putText(processed_frame, f"{camera_name}: solvePnP Failed", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return processed_frame, pnp_data
        
        # 计算距离和坐标
        distance = math.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)
        x, y, z = tvec[0][0], tvec[1][0], tvec[2][0]
        
        # 绘制坐标轴
        axis_length = chessboard_size[2] * 3
        cv2.drawFrameAxes(processed_frame, mtx, dist, rvec, tvec, axis_length, 3)
        
        # 直接在图像上添加信息
        info_text = [
            f"{camera_name}",
            f"X: {x:.1f} mm",
            f"Y: {y:.1f} mm", 
            f"Z: {z:.1f} mm",
            f"Distance: {distance:.1f} mm"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(processed_frame, text, (20, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 更新返回数据
        pnp_data.update({
            'success': True,
            'rvec': rvec,
            'tvec': tvec,
            'distance': distance,
            'x': x,
            'y': y,
            'z': z
        })
        
        return processed_frame, pnp_data