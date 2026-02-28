from hardware.camera.d435 import CameraD435
from calibration.mono import calibrate_intrinsic
import calibration.utils as utils

CHESSBOARD_SIZE = (11, 8, 15) #内角点长宽数和边长（毫米）
SAVE_DIR = "./data/eye2hand_images/pose_*.jpg"
if __name__=="__main__":
    '''
    拍摄标定板图像
    '''
    cam = CameraD435()

    # utils.take_photo(camera = cam, save_dir = SAVE_DIR)
    '''
    相机内参标定
    '''
    matrix,dist=calibrate_intrinsic(chessboard_glob_pattern = SAVE_DIR, chessboard_size = CHESSBOARD_SIZE, confirm = 1)
    