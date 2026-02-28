import sys
sys.path.insert(0, '.')
import os
import cv2
from hardware.camera.binocam import BinocularCam
from calibration.mono import calibrate_intrinsic
from calibration.bino import stereo_calibrate, stereo_rectify
import calibration.utils as utils

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
"""
立体校正,极线校正
"""
map1x, map1y, map2x, map2y, Q = stereo_rectify(
    left_pattern="data/chessboard_images/binocular/left_*.jpg",
    right_pattern="data/chessboard_images/binocular/right_*.jpg",
    mtxL=mtxL, distL=distL,
    mtxR=mtxR, distR=distR,
    R=R_l2r, T=t_l2r,
    F=F,
    chessboard_size=chessboard_size,
)

print(f"map1x{map1x.shape}")
imgL = cv2.imread("data/chessboard_images/binocular/left_27.jpg")
imgR = cv2.imread("data/chessboard_images/binocular/right_27.jpg")
rectL = cv2.remap(imgL, map1x, map1y, interpolation=cv2.INTER_LINEAR)
rectR = cv2.remap(imgR, map2x, map2y, interpolation=cv2.INTER_LINEAR)
os.makedirs("data/output", exist_ok=True)
cv2.imwrite("data/output/left_1_rectified.jpg", rectL)
cv2.imwrite("data/output/right_1_rectified.jpg", rectR)

