import glob
import cv2
import numpy as np
from typing import Tuple, List
import ast
import math
def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    将欧拉角转换为旋转矩阵（ZYX顺序）
    
    参数:
    roll: 绕X轴旋转角度（弧度）
    pitch: 绕Y轴旋转角度（弧度）
    yaw: 绕Z轴旋转角度（弧度）
    
    返回:
    3x3 旋转矩阵
    """
    # 计算各个轴的旋转矩阵
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    
    # 组合旋转矩阵 (ZYX顺序)
    R = Rz @ Ry @ Rx
    return R

def calculate_transformation_error(transformations):
    """
    计算一系列变换矩阵之间的误差波动（使用李群平均法计算平均变换）
    
    参数:
    transformations: 变换矩阵列表（每个为4x4齐次变换矩阵）
    
    返回:
    旋转误差的平均值和标准差（度）
    平移误差的平均值和标准差
    """
    if len(transformations) < 2:
        return (0, 0), (0, 0)
    
    # 1. 提取所有旋转和平移部分
    rotations = [T[:3, :3] for T in transformations]
    translations = [T[:3, 3] for T in transformations]
    
    # 2. 计算平均旋转矩阵（使用SVD校正法，确保正交性）
    R_avg = np.mean(rotations, axis=0)
    
    # 对平均旋转矩阵进行SVD分解并校正
    U, _, Vt = np.linalg.svd(R_avg)
    R_corrected = U @ Vt
    
    # 确保行列式为+1（右手坐标系）
    if np.linalg.det(R_corrected) < 0:
        Vt[2, :] *= -1  # 调整最后一行
        R_corrected = U @ Vt
    
    # 3. 验证正交性
    # 检查是否正交: R^T * R = I
    orthogonality_error = np.linalg.norm(R_corrected.T @ R_corrected - np.eye(3))
    # 检查行列式是否为1
    det_error = abs(np.linalg.det(R_corrected) - 1.0)
    
    if orthogonality_error > 1e-5 or det_error > 1e-5:
        print(f"警告: 平均旋转矩阵正交性验证失败! "
              f"正交误差={orthogonality_error:.6f}, 行列式误差={det_error:.6f}")
    
    # 4. 计算平均平移（欧氏空间直接平均）
    t_avg = np.mean(translations, axis=0)
    print("棋盘到法兰盘的旋转矩阵：\n", R_corrected.tolist())
    print("棋盘到法兰盘的平移向量：\n", t_avg.reshape(-1).tolist())
    # 5. 构建校正后的平均变换矩阵
    T_avg = np.eye(4)
    T_avg[:3, :3] = R_corrected
    T_avg[:3, 3] = t_avg
    
    # 6. 计算每个变换与平均变换之间的差异
    rot_errors = []
    trans_errors = []
    
    for T in transformations:
        # 6.1 旋转部分误差 (使用角度差)
        # 计算相对旋转: R_diff = R_current * R_avg^{-1}
        R_diff = T[:3, :3] @ R_corrected.T
        # 将旋转矩阵转换为旋转向量（轴角表示）
        rvec, _ = cv2.Rodrigues(R_diff)
        # 旋转向量的范数即为旋转角度（弧度）
        angle_error = np.linalg.norm(rvec) * 180 / np.pi
        rot_errors.append(angle_error)
        
        # 6.2 平移部分误差 (欧氏距离)
        t_diff = T[:3, 3] - t_avg
        trans_error = np.linalg.norm(t_diff)
        trans_errors.append(trans_error)
    
    # 7. 计算统计量
    rot_mean = np.mean(rot_errors)
    rot_std = np.std(rot_errors)
    trans_mean = np.mean(trans_errors)
    trans_std = np.std(trans_errors)
    
    return (rot_mean, rot_std), (trans_mean, trans_std)

def calibrate_extrinsic_bysolvepnp(chessboard_picpath: str, chessboard_size: Tuple[int, int, float],cam_intrinsic: np.ndarray, confirm: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    通过PnP方法计算每张棋盘图片的外参world2cam(旋转矩阵和平移向量)
    Args:
        chessboard_picpath (str): 棋盘格照片路径
        chessboard_size (tuple): 包含三个元素的元组，表示棋盘格的尺寸，如 (8, 5, 26.36)
            - 内角点长 (int)
            - 内角点宽 (int)
            - 边长 (float, 毫米)
        cam_intrinsic (np.ndarray): 相机内参
        confirm (bool): 是否等待用户通过键盘确认 OpenCV 检测的角点
    
    Returns:
        tuple: 包含两个列表，分别是：
            - r_matrix_list (List[np.ndarray]): 每张图片的旋转矩阵
            - tvecs_list (List[np.ndarray]): 每张图片的平移向量
    """
    
    # 加载棋盘格照片
    images_path_list = glob.glob(chessboard_picpath + '/*.jpg')
    def sort_key(fname):
        # 提取文件名中的数字部分
        import re
        match = re.search(r'pose_(\d+)\.jpg', fname)
        if match:
            return int(match.group(1))
        return 0
    images_path_list.sort(key=sort_key)
    # 准备世界坐标点
    p_world = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    p_world[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * chessboard_size[2]
    #print(p_world)

    # 读取相机内参和畸变系数（假设已标定）
    # 这里需要你提供相机内参矩阵 mtx 和畸变系数 dist
    # 如果没有，可使用 cv2.calibrateCamera 事先标定
    mtx = cam_intrinsic
    dist = np.zeros((5, 1), dtype=np.float32)  # 假设无畸变
    
    r_matrix_list, tvecs_list = [], []
    cv2.namedWindow('Chessboard Corners - Press any key for next', cv2.WINDOW_NORMAL)
    for fname in images_path_list:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size[:2], None)
        #ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size[:2], cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if ret:
            # 定义亚像素迭代的终止条件（迭代次数或精度）
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            # 精化角点位置（需要灰度图、初始角点、搜索窗口大小、死区、终止条件）
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if ret:
            # 计算 PnP
            ret, rvec, tvec = cv2.solvePnP(p_world, corners, mtx, dist)
            r_matrix, _ = cv2.Rodrigues(rvec)  # 将旋转向量转换为旋转矩阵
            r_matrix_list.append(r_matrix)
            tvecs_list.append(tvec)
            
            # 绘制角点并显示
            cv2.drawChessboardCorners(img, chessboard_size[:2], corners, ret)
            #显示点的序号
            for i, corner in enumerate(corners):
                corner = tuple(map(int, corner.ravel()))  # 确保corner是一个包含两个整数值的元组
                cv2.putText(img, str(i+1), corner, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, fname, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            if confirm:
                cv2.imshow('Chessboard Corners - Press any key for next', img)
                key = cv2.waitKey(0)
                if key == ord('q'):  # 如果按下 'q' 键，则退出
                    print("用户终止标定过程。")
                    exit()
        else:
            print(f"{fname}检测棋盘格角点失败")
            
    cv2.destroyAllWindows()
    
    return r_matrix_list, tvecs_list

def eye2hand_calibration(cam_intrinsic,chessboard_picpath):
    """
    计算r_cam2base(列向量为基)和t_cam2base
    """
    pose_data_path =  chessboard_picpath+"/pose_data.txt"
    # 1. 由pnp计算外参
    r_world2cam,t_world2cam=calibrate_extrinsic_bysolvepnp(
        chessboard_picpath=chessboard_picpath,
        chessboard_size = (11, 8, 15),
        cam_intrinsic=cam_intrinsic,
        confirm=False
        )
    
    
    #print(t_world2cam)
    # 2. 处理T_end2base
    pose_data_path = "./data/eye2hand_images/pose_data.txt"
    R_base2end_list = []
    t_base2end_list= []

    with open(pose_data_path, "r") as file:
        for line in file:
            # 去除行首尾的空白字符
            line = line.strip()
            if line:  # 跳过空行
                # 解析字符串为字典
                data = ast.literal_eval(line)
                # 提取end_pose并添加到列表
                end_pose = data['end_pose']
                euler_end2base = [math.radians(end_pose[3]), math.radians(end_pose[4]), math.radians(end_pose[5])]#读取姿态角度值并转换为弧度


                #R_end2base, _ = cv2.Rodrigues(np.array(R_end2base, dtype=np.float32))#将旋转向量转换为矩阵，输入为弧度

                R_end2base = euler_to_rotation_matrix(euler_end2base[0], euler_end2base[1], euler_end2base[2])
                t_end2base = np.array([end_pose[0],end_pose[1],end_pose[2]], dtype=np.float32)

                t_base2end=-R_end2base.T@t_end2base
                R_base2end_list.append(R_end2base.T)
                t_base2end_list.append(t_base2end)


                # R_base2end_list.append(R_end2base)
                # t_base2end_list.append(t_end2base)


    # 3. 计算cam2base
    methods = [
        "cv2.CALIB_HAND_EYE_ANDREFF",
        "cv2.CALIB_HAND_EYE_TSAI",
        "cv2.CALIB_HAND_EYE_DANIILIDIS",
        "cv2.CALIB_HAND_EYE_PARK",
        "cv2.CALIB_HAND_EYE_HORAUD"
    ]
    # t_base2end_list = [array / 1000 for array in t_base2end_list]
    # t_world2cam = [array / 1000 for array in t_world2cam]
    # 4. 循环计算每种方法的结果
    for method in methods:
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_gripper2base=R_base2end_list,
            t_gripper2base=t_base2end_list,
            R_target2cam=r_world2cam,
            t_target2cam=t_world2cam,
            method=eval(method)
        )
        print("-" * 80)
        print(f"使用 {method[19:]} 计算的结果：")
        print("相机到基座的旋转矩阵：\n", R_cam2base.tolist())
        print("相机到基座的平移向量：\n", t_cam2base.reshape(-1).tolist())
        print()

        # 收集所有估计的世界到基座的变换矩阵
        T_w2b_list = []
        for R_b2e, t_b2e, R_w2c, t_w2c in zip(R_base2end_list, t_base2end_list, r_world2cam, t_world2cam):
            # 构造变换矩阵
            T_base2end = np.eye(4)
            T_base2end[:3, :3] = R_b2e
            T_base2end[:3, 3] = t_b2e

            T_cam2base = np.eye(4)
            T_cam2base[:3, :3] = R_cam2base
            T_cam2base[:3, 3] = t_cam2base.reshape(-1)

            T_board2cam = np.eye(4)
            T_board2cam[:3, :3] = R_w2c
            T_board2cam[:3, 3] = t_w2c.reshape(-1)

            # 计算世界到基座的变换
            T_w2b_estimated = T_base2end @ T_cam2base @ T_board2cam
            T_w2b_list.append(T_w2b_estimated)
            #np.set_printoptions(precision=2, suppress=True)
            #print("世界到基座的变换矩阵：\n", T_w2b_estimated)
        
        # 计算并打印误差波动
        rot_stats, trans_stats = calculate_transformation_error(T_w2b_list)
        print(f"旋转误差 - 平均值: {rot_stats[0]:.4f}°，标准差: {rot_stats[1]:.4f}°")
        print(f"平移误差 - 平均值: {trans_stats[0]:.4f}，标准差: {trans_stats[1]:.4f}")
        