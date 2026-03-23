'''
眼在手外标定
'''
import sys
sys.path.insert(0, './')
from hardware.camera.d435 import CameraD435
from hardware.robotarm.piper import PiperClass
from calibration.mono import calibrate_intrinsic
from calibration.eye2hand import eye2hand_calibration
import calibration.utils as utils

SAVE_DIR = "./data/eye2hand_images"
DATA_FILE = "./data/cali_pose.txt" # 包含您提供的那组位姿的文件
CHESSBOARD_SIZE = (11, 8, 15) #内角点长宽数和边长（毫米）
if __name__=="__main__":
    # 0. 初始化


    # 1. 收集数据
    # import pyrealsense2 as rs
    # cam = CameraD435()
    # cam.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
    # cam.start()
    # arm = piper = PiperClass(can_name = "can_piper")

    # utils.eye2hand_collect_manual(save_dir = SAVE_DIR, camera = cam, robotarm = arm)
    # utils.eye2hand_collect_auto(
    #     save_dir=SAVE_DIR,
    #     pose_file_path=DATA_FILE,
    #     camera=cam,
    #     robotarm=arm,
    #     wait_time=10.0
    # )

    # 2. 计算内参
    # cam_intrinsic,_,_,_= cam.get_intrinsics()
    # print(cam_intrinsic)
    intrinsic, dist = calibrate_intrinsic(chessboard_glob_pattern = SAVE_DIR + "/*.jpg", chessboard_size = CHESSBOARD_SIZE, confirm = 0)
    
    eye2hand_calibration(cam_intrinsic=intrinsic,chessboard_picpath=SAVE_DIR)
