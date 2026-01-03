'''
实时解算pnp,需要移动到根目录下运行
'''

import cv2
import numpy as np
import pyrealsense2 as rs
import math
import time
from hardware.camera.d435 import CameraD435

def pnp_Checkchessboard(chessboard_size,cam_intrinsic,camera):
    """
    实时计算T_world2cam(棋盘格为world)
    Arg:
        chessboard_size(list):棋盘格尺寸
        cam_intrinsic(numpy.ndarray):相机内参
    """
    # 棋盘格的尺寸（内角点数量，例如 9x6 棋盘格的内角点为 8x5）和边长（mm）
    # 相机内参（请根据实际标定数据调整）
    dist_coeffs = np.zeros((5, 1))  # 假设无畸变

    # 给定3d点
    p_world=np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    p_world[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * chessboard_size[2]



    # 定义角点细化的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    axis_length = chessboard_size[2] * 3

    cv2.namedWindow('RealSense - axis show', cv2.WINDOW_NORMAL)
    # 实时检测
    try:
        while True:
            # 获取 RealSense 帧数据
            frames = camera.get_frame()
            color_frame = frames["color"] 
            gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            start_time = time.time()
            #查找像素角点
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size[:2], None,cv2.CALIB_CB_FAST_CHECK)
            #ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size[:2], None)
            end_time = time.time()
            #print(f"{end_time-start_time}")
            if ret:
                # 角点细化
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                # 利用 solvePnP 计算旋转向量和平移向量
                success, rvec, tvec = cv2.solvePnP(p_world, corners, cam_intrinsic, dist_coeffs)
                if success:
                    # 绘制坐标轴（函数 cv2.drawFrameAxes 要求 OpenCV 编译时包含 calib3d 模块）
                    cv2.drawFrameAxes(color_frame, cam_intrinsic, dist_coeffs, rvec, tvec, axis_length, 3)
                    cv2.putText(color_frame, f"X:{tvec.T[0][0]:.1f},Y:{tvec.T[0][2]:.1f},Z:{tvec.T[0][2]:.1f}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.putText(color_frame, f"{math.sqrt(tvec[0]**2+tvec[1]**2+tvec[2]**2)}", (50, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.putText(color_frame, "solvePnP failed", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(color_frame, "no chess corners detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 显示实时图像
            cv2.imshow("RealSense - axis show", color_frame)
            key = cv2.waitKey(1)
            if key & 0xFF in (ord('q'), 27):
                break
    finally:
        # 释放 RealSense 流资源和关闭窗口
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    CHESSBOARD_SIZE = (11, 8, 15)
    cam = CameraD435()
    cam_intrinsic = np.array([[902.3352359413651, 0.0, 642.2675799117687], [0.0, 901.5064189761518, 371.8472754481835], [0.0, 0.0, 1.0]])
    pnp_Checkchessboard(chessboard_size=CHESSBOARD_SIZE,cam_intrinsic=cam_intrinsic, camera = cam)

    