import sys
sys.path.insert(0, './')
from hardware.camera.d435 import CameraD435
from calibration.mono import calibrate_intrinsic
import calibration.utils as utils


CHESSBOARD_SIZE = (11, 8, 15) #内角点长宽数和边长（毫米）
SAVE_DIR = "./data/chessboard_images/monocular"
if __name__=="__main__":
    '''
    拍摄标定板图像
    '''
    import pyrealsense2 as rs
    cam = CameraD435()
    cam.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
    cam.start()
    cam.take_photo(save_dir = SAVE_DIR)

    '''
    相机内参标定
    '''
    matrix, dist = calibrate_intrinsic(chessboard_glob_pattern = SAVE_DIR + "/*.jpg", chessboard_size = CHESSBOARD_SIZE, confirm = 1)