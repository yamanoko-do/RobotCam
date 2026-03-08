import glob
import cv2
import numpy as np
import os

def stereo_calibrate(left_pattern, right_pattern, mtxL, distL, mtxR, distR,chessboard_size=(8, 5, 25), confirm=False):
    """
    双目标定，输出外参 R、T。
    参数：
        left_pattern  : 左相机图片路径模式，如 "data/chessboard_images/binocular/left_*.jpg"
        right_pattern : 右相机图片路径模式，如 "data/chessboard_images/binocular/right_*.jpg"
        chessboard_size: (cols, rows, square_size_mm)
        confirm       : 是否显示角点检测结果以人工确认
    返回：
        R_l2r: 旋转矩阵,右边是基 (3x3)
        t_l2r: 平移向量,右边是基 (3x1)
    """
    cols, rows, square_size = chessboard_size

    # === 棋盘格世界坐标 ===
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    obj_points = []         # 世界坐标
    img_points_left = []    # 左图角点
    img_points_right = []   # 右图角点

    images_left = sorted(glob.glob(left_pattern))
    images_right = sorted(glob.glob(right_pattern))

    if len(images_left) != len(images_right):
        raise ValueError(f"左右图数量不匹配：{len(images_left)} vs {len(images_right)}")

    print(f"\n共检测到 {len(images_left)} 对图像用于双目标定。")

    for i, (lf, rf) in enumerate(zip(images_left, images_right)):
        imgL = cv2.imread(lf)
        imgR = cv2.imread(rf)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # === 检测角点 ===
        retL, cornersL = cv2.findChessboardCorners(grayL, (cols, rows))
        retR, cornersR = cv2.findChessboardCorners(grayR, (cols, rows))

        if retL and retR:
            # 亚像素优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
            cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

            if confirm:
                visL = imgL.copy()
                visR = imgR.copy()
                cv2.drawChessboardCorners(visL, (cols, rows), cornersL, retL)
                cv2.drawChessboardCorners(visR, (cols, rows), cornersR, retR)
                cv2.imshow('Left', visL)
                cv2.imshow('Right', visR)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break
            obj_points.append(objp)
            img_points_left.append(cornersL)
            img_points_right.append(cornersR)
        else:
            print(f"[警告] 第{i+1}对图像角点检测失败：{os.path.basename(lf)}, {os.path.basename(rf)}")

    if len(obj_points) < 5:
        raise RuntimeError("有效的棋盘格配对太少，至少需要5组以上。")

    # === 双目标定（求外参） ===
    print("开始双目标定（外参计算）...")
    flags = cv2.CALIB_FIX_INTRINSIC  # 固定内参，只优化外参
    criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R_l2r, t_l2r, E, F = cv2.stereoCalibrate(
        obj_points, img_points_left, img_points_right,
        mtxL, distL, mtxR, distR,
        imageSize=grayL.shape[::-1],
        criteria=criteria_stereo,
        flags=flags
    )

    print(f"重投影误差{retval}")
    print("双目标定完成。")
    print("旋转矩阵 R_l2r =\n", R_l2r.tolist())
    print("平移向量 t_l2r =\n", t_l2r.tolist())
    print("E =", E.tolist())
    print("F =", F.tolist())
    return R_l2r, t_l2r, E, F


# def stereo_rectify(left_pattern, right_pattern, mtxL, distL, mtxR, distR, R, T, F, chessboard_size):
#     """
#     根据双目相机的内外参，计算左右相机的校正旋转矩阵、投影矩阵和重投影矩阵 Q。
#     并在校正前后分别绘制极线对比。
#     """
#     def draw_epilines(img1, img2, pts1, pts2, F, title="Epilines"):
#         """
#         在两幅图上绘制对应的极线，用于验证校正效果。
#         """
#         img1_color = img1.copy()
#         img2_color = img2.copy()

#         lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
#         lines1 = lines1.reshape(-1, 3)

#         for r, pt1, pt2 in zip(lines1, pts1, pts2):
#             color = tuple(int(c) for c in np.random.randint(0, 255, 3))
#             x0, y0 = map(int, [0, -r[2] / r[1]])
#             x1, y1 = map(int, [img1.shape[1], -(r[2] + r[0] * img1.shape[1]) / r[1]])
#             cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
#             cv2.circle(img1_color, tuple(np.int32(pt1)), 3, color, -1)
#             cv2.circle(img2_color, tuple(np.int32(pt2)), 3, color, -1)

#         vis = np.hstack((img1_color, img2_color))
#         cv2.imshow(title, vis)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     # --- 取任意一张图确定尺寸 ---
#     sample_img = cv2.imread(sorted(glob.glob(left_pattern))[0])
#     h, w = sample_img.shape[:2]

#     # --- 读取一对图像 ---
#     imgL = cv2.imread(sorted(glob.glob(left_pattern))[0])
#     imgR = cv2.imread(sorted(glob.glob(right_pattern))[0])

#     grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
#     grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

#     # --- 检测角点 ---
#     pattern_size = (chessboard_size[0], chessboard_size[1])
#     retL, cornersL = cv2.findChessboardCorners(grayL, pattern_size)
#     retR, cornersR = cv2.findChessboardCorners(grayR, pattern_size)


#     if not (retL and retR):
#         print("未能检测到棋盘角点，跳过极线绘制。")
#         return None

#     # 选取部分点用于可视化
#     idx = np.linspace(0, len(cornersL)-1, 10, dtype=int)
#     pts1 = cornersL[idx].reshape(-1, 2)
#     pts2 = cornersR[idx].reshape(-1, 2)

#     # --- 绘制校正前极线 ---
#     draw_epilines(imgL, imgR, pts1, pts2, F, title="Before Rectification")

#     # --- 计算立体校正参数 ---
#     R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
#         cameraMatrix1=mtxL,
#         distCoeffs1=distL,
#         cameraMatrix2=mtxR,
#         distCoeffs2=distR,
#         imageSize=(w, h),
#         R=R,
#         T=T,
#         flags=cv2.CALIB_ZERO_DISPARITY,
#     )
#     print(f"左相机的旋转矩阵 R1:\n", R1)
#     print(f"右相机的旋转矩阵 R2:\n", R2)
#     print(f"校正后左相机投影矩阵 P1:\n", P1)
#     print(f"校正后右相机投影矩阵 P1:\n", P1)
#     print(f"重投影矩阵 Q:\n", Q)
#     print(f"新的基线B(mm):{-1.0 / Q[3, 2]}")
#     print(f"左相机 ROI: {roi1}")
#     print(f"右相机 ROI: {roi2}")

#     # --- 生成映射表 ---
#     map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (w, h), cv2.CV_32FC1)
#     map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (w, h), cv2.CV_32FC1)

#     # --- 使用映射表进行校正 ---
#     rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
#     rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

#     gray_rectL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
#     gray_rectR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

#     # 再次检测角点（在校正图上）
#     retL2, cornersL2 = cv2.findChessboardCorners(gray_rectL, pattern_size)
#     retR2, cornersR2 = cv2.findChessboardCorners(gray_rectR, pattern_size)

#     if retL2 and retR2:
#         pts1_rect = cornersL2[idx].reshape(-1, 2)
#         pts2_rect = cornersR2[idx].reshape(-1, 2)
#         # 校正后极线（理论上应水平）
#         draw_epilines(rectL, rectR, pts1_rect, pts2_rect, F, title="After Rectification")
#     else:
#         print("校正后未检测到角点，跳过校正后极线绘制。")

#     return (map1x, map1y, map2x, map2y, Q)



import cv2
import numpy as np
import glob

def stereo_rectify(left_pattern, right_pattern, mtxL, distL, mtxR, distR, R, T, F, chessboard_size):
    """
    双目校正完整函数：包含报错修复、校正前斜极线绘制、校正后水平贯穿线绘制。
    新增：打印校正后的新内参和外参
    """
    print("\n开始极线校正")
    # --- 0. 内部辅助函数：绘制校正前的斜极线 ---
    def draw_epilines_before(imgL, imgR, ptsL, ptsR, F):
        h, w = imgL.shape[:2]
        # 拼接图像用于显示
        vis = np.hstack((imgL, imgR))
        
        # 计算极线: linesL 是右图点在左图上的投影线; linesR 是左图点在右图上的投影线
        linesR = cv2.computeCorrespondEpilines(ptsL.reshape(-1, 1, 2), 1, F).reshape(-1, 3)
        linesL = cv2.computeCorrespondEpilines(ptsR.reshape(-1, 1, 2), 2, F).reshape(-1, 3)

        for rL, rR, ptL, ptR in zip(linesL, linesR, ptsL, ptsR):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # 在左图区域画线 (rL: ax + by + c = 0)
            x0, y0 = map(int, [0, -rL[2]/rL[1]])
            x1, y1 = map(int, [w, -(rL[2] + rL[0]*w)/rL[1]])
            cv2.line(vis, (x0, y0), (x1, y1), color, 1)
            cv2.circle(vis, (int(ptL[0]), int(ptL[1])), 5, color, -1)

            # 在右图区域画线 (注意 x 坐标要加上 w)
            x0_r, y0_r = map(int, [w, -rR[2]/rR[1]])
            x1_r, y1_r = map(int, [2*w, -(rR[2] + rR[0]*w)/rR[1]])
            cv2.line(vis, (x0_r, y0_r), (x1_r, y1_r), color, 1)
            cv2.circle(vis, (int(ptR[0]) + w, int(ptR[1])), 5, color, -1)

        cv2.imshow("Before Rectification - Slanted Epilines", vis)
        cv2.waitKey(0)

    # --- 1. 内部辅助函数：绘制校正后的水平贯穿线 ---
    def draw_rectified_lines_after(imgL, imgR, ptsL, ptsR):
        h, w = imgL.shape[:2]
        vis = np.hstack((imgL, imgR))
        
        # 绘制贯穿全图的水平线
        for ptL, ptR in zip(ptsL, ptsR):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            y = int(ptL[1])
            # 贯穿线：从左图最左边到右图最右边
            cv2.line(vis, (0, y), (2 * w, y), color, 1)
            cv2.circle(vis, (int(ptL[0]), int(ptL[1])), 5, color, -1)
            cv2.circle(vis, (int(ptR[0]) + w, int(ptR[1])), 5, color, -1)
            
        # 辅助背景线
        for y in range(0, h, 40):
            cv2.line(vis, (0, y), (2 * w, y), (200, 200, 200), 1)

        cv2.imshow("After Rectification - Horizontal Lines", vis)
        cv2.waitKey(0)

    # --- 2. 准备数据 ---
    left_files = sorted(glob.glob(left_pattern))
    right_files = sorted(glob.glob(right_pattern))

    imgL = cv2.imread(left_files[0])
    imgR = cv2.imread(right_files[0])
    h, w = imgL.shape[:2]

    # 重要修复：确保 pattern_size 只有两个元素
    inner_size = (chessboard_size[0], chessboard_size[1])

    # --- 3. 校正前：检测角点并绘图 ---
    retL, cornersL = cv2.findChessboardCorners(imgL, inner_size)
    retR, cornersR = cv2.findChessboardCorners(imgR, inner_size)
    if retL and retR:
        idx = np.linspace(0, len(cornersL)-1, 10, dtype=int)
        draw_epilines_before(imgL, imgR, cornersL[idx].reshape(-1, 2), cornersR[idx].reshape(-1, 2), F)

    # --- 4. 计算立体校正参数 ---
    # R1,R2,为旋转补偿
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=mtxL, distCoeffs1=distL,
        cameraMatrix2=mtxR, distCoeffs2=distR,
        imageSize=(w, h), R=R, T=T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    
    print(f"R1: {R1.tolist()}")
    print(f"R2: {R2.tolist()}")
    print(f"P1: {P1.tolist()}")
    print(f"P2: {P2.tolist()}")
    print(f"Q: {Q.tolist()}")

    K_new = P1[:3, :3]
    fx_new = K_new[0, 0] # 提取新的焦距
    R_rect = R2[:3, :3] @ np.linalg.inv(R1[:3, :3])

    t_rect_phys = np.array([P2[0, 3], P2[1, 3], P2[2, 3]])/ fx_new
    
    print("\n" + "="*50)
    print("极线校正后的参数")
    print("="*50)

    print(f"\n1. 校正后统一的内参矩阵 K_new:\n{K_new.tolist()}")

    print(f"\n2. 校正后的外参 (Rectified Coordinate System):")
    print(f"   - 旋转矩阵 Rl2r: {R_rect.tolist()}")
    print(f"   - 平移向量 t_l2r: {t_rect_phys.tolist()}")


    # --- 5. 执行校正 ---
    map1x, map1y = cv2.initUndistortRectifyMap(mtxL, distL, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(mtxR, distR, R2, P2, (w, h), cv2.CV_32FC1)
    rectL = cv2.remap(imgL, map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, map2x, map2y, cv2.INTER_LINEAR)

    # --- 6. 校正后：重新检测角点并绘水平线 ---
    gray_rectL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    gray_rectR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    retL2, cornersL2 = cv2.findChessboardCorners(gray_rectL, inner_size)
    retR2, cornersR2 = cv2.findChessboardCorners(gray_rectR, inner_size)

    if retL2 and retR2:
        idx = np.linspace(0, len(cornersL2)-1, 10, dtype=int)
        draw_rectified_lines_after(rectL, rectR, cornersL2[idx].reshape(-1, 2), cornersR2[idx].reshape(-1, 2))
    else:
        print("校正后未检测到角点，绘制默认参考线。")
        draw_rectified_lines_after(rectL, rectR, np.array([]), np.array([]))

    cv2.destroyAllWindows()
    return (map1x, map1y, map2x, map2y)