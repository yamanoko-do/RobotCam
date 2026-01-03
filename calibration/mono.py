
import glob
import cv2
import numpy as np
from typing import Tuple
import re
import os

def calibrate_intrinsic(chessboard_glob_pattern: str, chessboard_size: Tuple[int, int, float], confirm: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    通过给定的棋盘格图片标定单目相机内参（可分别用于左/右目）
    
    Args:
        chessboard_glob_pattern (str): 图像文件的 glob 路径模式，例如 "data/left_*.jpg" 或 "data/right_*.jpg"
        chessboard_size (tuple): 包含三个元素的元组 (corners_x, corners_y, square_size_mm)
            - corners_x (int): 棋盘格内角点的列数（例如 8）
            - corners_y (int): 棋盘格内角点的行数（例如 5）
            - square_size_mm (float): 每个棋盘格方块的边长（单位：毫米）
        confirm (bool): 是否等待用户按键确认每张图的角点检测结果

    Returns:
        tuple: (camera_matrix, distortion_coeffs)
            - camera_matrix (np.ndarray): 3x3 相机内参矩阵
            - distortion_coeffs (np.ndarray): 5x1 畸变系数向量
    """
    # 获取所有匹配的图像路径
    images_path_list = glob.glob(chessboard_glob_pattern)
    if not images_path_list:
        raise ValueError(f"未找到匹配的图像文件: {chessboard_glob_pattern}")

    # 按文件名中的数字排序（支持 left_1.jpg, left_10.jpg 等）
    def sort_key(fname):
        match = re.search(r'(\d+)', fname)
        return int(match.group(1)) if match else 0

    images_path_list.sort(key=sort_key)

    # 读取第一张图以获取分辨率
    img0 = cv2.imread(images_path_list[0])
    if img0 is None:
        raise ValueError(f"无法读取图像: {images_path_list[0]}")
    resolution = img0.shape[:2]  # (height, width)

    # 构建世界坐标（所有图像共用同一棋盘格坐标系）
    nx, ny, square_size = chessboard_size
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * square_size

    objpoints = []  # 3D 点（世界坐标）
    imgpoints = []  # 2D 点（图像坐标）

    cv2.namedWindow('Chessboard Corners, press any key to next', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Chessboard Corners, press any key to next', 1280, 720)

    for fname in images_path_list:
        img = cv2.imread(fname)
        if img is None:
            print(f"警告：跳过无法读取的图像 {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            # 亚像素精化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners)

            # 可视化
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            for i, corner in enumerate(corners):
                pt = tuple(map(int, corner.ravel()))
                cv2.putText(img, str(i + 1), pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, os.path.basename(fname), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if confirm:
                cv2.imshow('Chessboard Corners, press any key to next', img)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    print("用户终止标定过程。")
                    cv2.destroyAllWindows()
                    exit()
        else:
            print(f"未检测到棋盘格角点，跳过: {fname}")

    cv2.destroyAllWindows()

    if not objpoints:
        raise RuntimeError("未找到任何有效的棋盘格图像用于标定！")

    print(f"成功检测 {len(objpoints)} 张图像的角点，开始标定...")
    
    # 执行标定（注意：resolution[::-1] 是 (width, height)）
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, resolution[::-1], None, None
    )

    print(f"重投影误差{ret},单位为像素,应小于1")#重投影误差：把三维点再投影回图像，看看计算出来的像素位置与真实检测位置差多少
    print("相机内参矩阵 (Intrinsic Camera Matrix):")
    print(mtx.tolist())
    print("\n畸变系数 (Distortion Coefficients):")
    print(dist.ravel().tolist())
    #dist=np.zeros((5, 1))

    # 可选：显示去畸变效果（使用第一张图）
    if confirm:
        img = cv2.imread(images_path_list[0])
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
        cv2.namedWindow('Undistorted Image (press any key to close)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Undistorted Image (press any key to close)', 1280, 720)
        cv2.imshow('Undistorted Image (press any key to close)', undistorted)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mtx, dist