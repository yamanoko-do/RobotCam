import numpy as np
import cv2
import os
import glob
import time
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
def read_pose_list(pose_file_path):
    """
    读取 data/pose_data.txt
    每一行是一个 dict:
    {'end_pose': [...], 'joint_state': [...]}
    """
    pose_list = []
    with open(pose_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pose_dict = ast.literal_eval(line)
            pose_list.append(pose_dict)
    return pose_list

def eye2hand_collect_auto(
        save_dir,
        pose_file_path,
        camera,
        robotarm,
        wait_time=4.0
    ):
    """
    自动 eye-to-hand 数据采集
    - save_dir: 保存图片和pose的目录
    - pose_file_path: 预先录好的 joint_state 文件
    - wait_time: 每个姿态运动后等待时间（秒）
    """

    os.makedirs(save_dir, exist_ok=True)
    save_pose_path = os.path.join(save_dir, "pose_data.txt")

    # ====== 读取离线位姿 ======
    pose_list = read_pose_list(pose_file_path)
    print(f"共读取 {len(pose_list)} 个目标位姿")

    # ====== 图片编号 ======
    existing_files = glob.glob(os.path.join(save_dir, "pic_*.jpg"))
    photo_count = 1
    if existing_files:
        nums = []
        for f in existing_files:
            try:
                nums.append(int(os.path.splitext(os.path.basename(f))[0].split("_")[-1]))
            except:
                pass
        if nums:
            photo_count = max(nums) + 1

    # ====== 机械臂准备 ======
    robotarm.set_ctrl_mode2can()

    try:
        for idx, pose in enumerate(pose_list):
            joint_state = pose["joint_state"]

            print(f"\n[{idx+1}/{len(pose_list)}] 运动到关节角:")
            print(joint_state)

            # 1. 运动到目标位姿
            robotarm.control_joint(joint_state)

            # 2. 等待机械臂稳定
            time.sleep(wait_time)

            # 3. 获取图像
            frames = camera.get_frame()
            color_frame = frames["color"]

            # 4. 保存图片
            img_path = os.path.join(save_dir, f"pic_{photo_count}.jpg")
            cv2.imwrite(img_path, color_frame)

            # 5. 再读一次真实位姿（非常关键）
            real_pose = robotarm.getpose()

            with open(save_pose_path, "a") as f:
                f.write(str(real_pose) + "\n")

            print(f"✔ 已保存: {img_path}")
            photo_count += 1

    finally:
        camera.stop()
        cv2.destroyAllWindows()

def take_photo(save_dir, camera):
    """
    使用给定的相机对象cam拍摄照片并保存到指定目录save_dir。s保存照片，q或ESC退出。
    """
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 初始化照片计数器
    photo_count = 1
    
    # 检查现有文件并确定起始编号
    existing_files = glob.glob(os.path.join(save_dir, "pic_*.jpg"))
    if existing_files:
        max_num = 0
        for file_path in existing_files:
            try:
                # 从文件名中提取数字
                filename = os.path.splitext(os.path.basename(file_path))[0]
                number = int(filename.split('_')[-1])
                if number > max_num:
                    max_num = number
            except (ValueError, IndexError):
                continue
        photo_count = max_num + 1


    try:
        while True:
            frames = camera.get_frame()
            color_frame = frames["color"] 
            # 显示实时画面
            cv2.imshow('RealSense Camera', color_frame)
            key = cv2.waitKey(1)

            # 按下's'保存照片
            if key & 0xFF == ord('s'):
                save_path = os.path.join(save_dir, f"d435_{photo_count}.jpg")
                cv2.imwrite(save_path, color_frame)
                print(f"照片已保存至：{save_path}")
                photo_count += 1

            # 按下'q'或ESC退出
            if key & 0xFF in (ord('q'), 27):
                break

    finally:
        # 清理资源
        camera.stop()
        cv2.destroyAllWindows()


def eye2hand_collect_manual(save_dir, camera, robotarm):
    pose_file_path = save_dir+"/pose_data.txt"
    # 创建保存目录（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 初始化照片计数器
    photo_count = 1

    # 检查现有文件并确定起始编号
    existing_files = glob.glob(os.path.join(save_dir, "pic_*.jpg"))
    if existing_files:
        max_num = 0
        for file_path in existing_files:
            try:
                # 从文件名中提取数字
                filename = os.path.splitext(os.path.basename(file_path))[0]
                number = int(filename.split('_')[-1])
                if number > max_num:
                    max_num = number
            except (ValueError, IndexError):
                continue
        photo_count = max_num + 1


    #初始化一个窗口
    #cv2.namedWindow('RealSense Camera', cv2.WINDOW_NORMAL)
    try:
        while True:
            frames = camera.get_frame()
            color_frame = frames["color"] 
            # 显示实时画面
            cv2.imshow('RealSense Camera', color_frame)
            key = cv2.waitKey(1)

            # 按下's'保存照片
            if key & 0xFF == ord('s'):
                save_path = os.path.join(save_dir, f"pose_{photo_count}.jpg")
                cv2.imwrite(save_path, color_frame)
                print(f"照片已保存至：{save_path}")
                pose_list=robotarm.getpose()
                
                print(pose_list)
                with open(pose_file_path, "a") as f:  # 使用追加模式
                    f.write(str(pose_list) + "\n")  # 将 pose_list 转换为字符串并写入文件

                photo_count += 1

            # 按下'q'或ESC退出
            if key & 0xFF in (ord('q'), 27):
                break

    finally:
        # 清理资源
        camera.stop()

        # print(piper.piper.get_ctrl_mode())
        # if piper.get_ctrl_mode() == 0x02:#如果是示教模式
        #     piper.piper.MotionCtrl_1(0x02,0,0)
        #     piper.piper.MotionCtrl_1(0x02,0,0)

        # piper.control_gripper(isclose=False)
        # piper.disconnect()
        cv2.destroyAllWindows()



def show_image(image: np.ndarray, title: str = "Image", show_colorbar: bool = None, figsize=(6, 6)):
    """
    自动判断图像类型（RGB 彩色图 or 单通道深度图）并可视化

    参数:
        image (np.ndarray): 输入图像数组
            - 彩色图: (H, W, 3) 或 (H, W, 4)
            - 深度图: (H, W) 或 (H, W, 1)
        title (str): 图像标题
        show_colorbar (bool): 是否显示 colorbar（深度图默认 True，彩色图默认 False）
        figsize (tuple): 图像大小

    示例:
        visualize_image_auto(color_img)      # 自动识别为彩色图
        visualize_image_auto(depth_img)      # 自动识别为深度图
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("输入必须是 numpy.ndarray")
    print(f"图像形状: {image.shape}, 数据类型: {image.dtype}")
    # 处理 (H, W, 1) -> (H, W)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(-1)  # 变成 (H, W)

    # 判断图像类型
    is_color = (image.ndim == 3 and image.shape[2] in [3, 4])
    is_depth = (image.ndim == 2)

    if not (is_color or is_depth):
        raise ValueError(f"不支持的图像形状: {image.shape}。仅支持 (H, W, 3/4) 彩色图 或 (H, W) 深度图。")

    # 值域归一化处理（仅对彩色图需要，深度图 matplotlib 会自动映射）
    img_display = image.astype(np.float32)
    if is_color:
        if img_display.max() > 1.0:
            img_display = img_display / 255.0
        img_display = np.clip(img_display, 0.0, 1.0)

    # 设置默认 colorbar 行为
    if show_colorbar is None:
        show_colorbar = is_depth  # 深度图默认显示 colorbar

    # 绘图
    plt.figure(figsize=figsize)
    if is_color:
        plt.imshow(img_display)
    else:  # 深度图
        im = plt.imshow(img_display, cmap='jet')
        if show_colorbar:
            plt.colorbar(im, fraction=0.046, pad=0.04, label='Depth Value')

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()