import sys
sys.path.insert(0, './')
import os
import cv2
from hardware.camera.binocam import BinocularCam
from calibration.mono import calibrate_intrinsic
from calibration.bino import stereo_calibrate, stereo_rectify
import calibration.utils as utils
import time
import numpy as np

#chessboard_size=(8, 5, 25)
chessboard_size = (11, 8, 15)

'''
拍摄照片
'''
# cam = BinocularCam(vid_pid="1bcf:0b15")
# try:
#     cam.take_photo(save_dir="./data/chessboard_images/binocular")
# finally:
#     cam.stop()
'''
相机内参标定
'''

mtxL, distL = calibrate_intrinsic(
    chessboard_glob_pattern="data/chessboard_images/binocular/left_*.jpg",
    chessboard_size=chessboard_size,
    confirm=0
)
mtxR, distR = calibrate_intrinsic(
    chessboard_glob_pattern="data/chessboard_images/binocular/right_*.jpg",
    chessboard_size=chessboard_size,
    confirm=0
)
# # '''
# 实时解算pnp
# '''
# cam = BinocularCam()
# cam.pnp_check_chessboard(chessboard_size=chessboard_size)
"""
双目外参标定
"""
R_l2r, t_l2r,E, F= stereo_calibrate(
        left_pattern="data/chessboard_images/binocular/left_*.jpg",
        right_pattern="data/chessboard_images/binocular/right_*.jpg",
        mtxL=mtxL, distL=distL,
        mtxR=mtxR, distR=distR,
        chessboard_size=chessboard_size,
        confirm= False
    )
# """
# 立体校正,极线校正
# """
map1x, map1y, map2x, map2y = stereo_rectify(
    left_pattern="data/chessboard_images/binocular/left_*.jpg",
    right_pattern="data/chessboard_images/binocular/right_*.jpg",
    mtxL=mtxL, distL=distL,
    mtxR=mtxR, distR=distR,
    R=R_l2r, T=t_l2r, F=F,
    chessboard_size=chessboard_size
)
# map1x, map1y, map2x, map2y, Q = stereo_rectify(
#     left_pattern="data/chessboard_images/binocular/left_*.jpg",
#     right_pattern="data/chessboard_images/binocular/right_*.jpg",
#     mtxL=mtxL, distL=distL,
#     mtxR=mtxR, distR=distR,
#     R=R_l2r, T=t_l2r,
#     F=F,
#     chessboard_size=chessboard_size,
# )
np.save("data/output/map1x.npy", map1x)
np.save("data/output/map1y.npy", map1y)
np.save("data/output/map2x.npy", map2x)
np.save("data/output/map2y.npy", map2y)


# """
# 对图像应用极线校正，并测试不同插值方法的性能
# """
map1x = np.load("data/output/map1x.npy")
map1y = np.load("data/output/map1y.npy")
map2x = np.load("data/output/map2x.npy")
map2y = np.load("data/output/map2y.npy")
imgL = cv2.imread("data/chessboard_images/binocular/left_21.jpg")
imgR = cv2.imread("data/chessboard_images/binocular/right_21.jpg")

if imgL is None or imgR is None:
    print("错误：无法读取图像，请检查路径。")
else:
    # 定义待测试的插值方法
    methods = [
        ("NEAREST", cv2.INTER_NEAREST),
        ("LINEAR ", cv2.INTER_LINEAR),
        ("CUBIC  ", cv2.INTER_CUBIC),
        ("LANCZOS", cv2.INTER_LANCZOS4),
        ("AREA   ", cv2.INTER_AREA)
    ]

    # 设置测试参数
    iterations = 100  # 每种方法运行100次取平均值
    os.makedirs("data/output", exist_ok=True)

    print(f"{'Method':<10} | {'Total Time (s)':<15} | {'Avg Time per Frame (ms)':<25}")
    print("-" * 55)

    for name, method in methods:
        # 预热：运行一次以排除冷启动干扰
        _ = cv2.remap(imgL, map1x, map1y, interpolation=method)
        
        start_time = time.time()
        
        # 开始循环测试
        for _ in range(iterations):
            # 同时校正左右图，模拟真实双目场景
            rectL = cv2.remap(imgL, map1x, map1y, interpolation=method)
            rectR = cv2.remap(imgR, map2x, map2y, interpolation=method)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / iterations) * 1000 / 2 # 除以2是因为每次循环处理了2张图
        
        print(f"{name:<10} | {total_time:<15.4f} | {avg_time_ms:<25.4f}")

        # 将最后一次结果保存，以便肉眼观察画质区别
        cv2.imwrite(f"data/output/left_{name.strip()}_rect.jpg", rectL)

    print("-" * 55)
    print("测试完成！结果图像已保存至 data/output/。")

