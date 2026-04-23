"""
HTC Vive 控制器封装类
依赖: pip install openvr numpy scipy
"""
import openvr
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Dict, Tuple, Optional, List


class ViveClass:
    """
    HTC Vive 控制器封装类，支持获取位姿和按键状态
    """

    def __init__(self):
        """
        初始化 OpenVR 系统
        """
        self.vr_system = None
        self._init_openvr()

    def _init_openvr(self):
        """初始化 OpenVR"""
        try:
            openvr.init(openvr.VRApplication_Utility)
            self.vr_system = openvr.VRSystem()
            print("init: Vive 系统初始化成功")
        except Exception as e:
            print(f"init: Vive 初始化失败 - {e}")
            raise

    @staticmethod
    def _pose_to_matrix(pose) -> List[List[float]]:
        """
        将 OpenVR 的 pose 结构转换为 4x4 矩阵

        Args:
            pose: OpenVR pose 结构

        Returns:
            4x4 变换矩阵
        """
        matrix = [
            [pose[0][0], pose[0][1], pose[0][2], pose[0][3]],
            [pose[1][0], pose[1][1], pose[1][2], pose[1][3]],
            [pose[2][0], pose[2][1], pose[2][2], pose[2][3]],
            [0, 0, 0, 1]
        ]
        return matrix

    @staticmethod
    def _matrix_to_position(matrix: List[List[float]]) -> List[float]:
        """
        从变换矩阵中提取位置

        Args:
            matrix: 4x4 变换矩阵

        Returns:
            [x, y, z] 位置坐标
        """
        return [matrix[0][3], matrix[1][3], matrix[2][3]]

    @staticmethod
    def _matrix_to_euler(matrix: List[List[float]], degrees: bool = True) -> Tuple[float, float, float]:
        """
        将旋转矩阵转换为欧拉角 (roll, pitch, yaw)

        Args:
            matrix: 4x4 变换矩阵
            degrees: 是否返回角度制，默认 True

        Returns:
            (roll, pitch, yaw) 欧拉角
        """
        sy = math.sqrt(matrix[0][0] * matrix[0][0] + matrix[1][0] * matrix[1][0])

        singular = sy < 1e-6

        if not singular:
            roll = math.atan2(matrix[2][1], matrix[2][2])
            pitch = math.atan2(-matrix[2][0], sy)
            yaw = math.atan2(matrix[1][0], matrix[0][0])
        else:
            roll = math.atan2(-matrix[1][2], matrix[1][1])
            pitch = math.atan2(-matrix[2][0], sy)
            yaw = 0

        if degrees:
            return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
        else:
            return roll, pitch, yaw

    def _get_controller_buttons(self, device_index: int) -> Dict:
        """
        获取控制器按键状态

        Args:
            device_index: 控制器设备索引

        Returns:
            包含按键状态的字典
        """
        result, state = self.vr_system.getControllerState(device_index)

        if not result:
            return {
                'grip': False,
                'trigger': 0.0,
                'menu_pressed': False,
                'trackpad_touched': False,
                'trackpad_pressed': False,
                'trackpad_x': 0.0,
                'trackpad_y': 0.0,
                'trackpad_up': False,
                'trackpad_down': False,
                'trackpad_left': False,
                'trackpad_right': False,
                'button_pressed_bits': 0,
            }

        state = openvr.VRControllerState_t.from_buffer(bytearray(state))

        trigger_value = state.rAxis[1].x if len(state.rAxis) > 1 else 0.0
        trackpad_x = state.rAxis[0].x if len(state.rAxis) > 0 else 0.0
        trackpad_y = state.rAxis[0].y if len(state.rAxis) > 0 else 0.0

        trackpad_touched = bool(state.ulButtonTouched & (1 << 32))  # Trackpad 触摸
        trackpad_pressed = bool(state.ulButtonPressed & (1 << 32))  # Trackpad 按压

        # 检测 Grip 按键
        grip = bool(state.ulButtonPressed & (1 << 2))

        # Trackpad 方向检测 - 降低死区，独立检测触摸和按压状态
        deadzone = 0.3
        # 使用触摸状态来检测方向，更灵敏
        trackpad_active = trackpad_touched or trackpad_pressed
        trackpad_up = trackpad_active and (trackpad_y > deadzone)
        trackpad_down = trackpad_active and (trackpad_y < -deadzone)
        trackpad_left = trackpad_active and (trackpad_x < -deadzone)
        trackpad_right = trackpad_active and (trackpad_x > deadzone)

        return {
            'grip': grip,
            'trigger': trigger_value,
            'menu_pressed': bool(state.ulButtonPressed & (1 << 1)),
            'trackpad_touched': trackpad_touched,
            'trackpad_pressed': trackpad_pressed,
            'trackpad_x': trackpad_x,
            'trackpad_y': trackpad_y,
            'trackpad_up': trackpad_up,
            'trackpad_down': trackpad_down,
            'trackpad_left': trackpad_left,
            'trackpad_right': trackpad_right,
            'button_pressed_bits': state.ulButtonPressed,  # 原始按键位，用于调试
        }

    def get_state(self) -> Dict[str, Optional[Dict]]:
        """
        获取所有控制器的状态（位姿 + 按键）

        Returns:
            字典，键为 'left' 和 'right'，值为对应控制器的状态，
            每个状态包含 'pose' ([x,y,z,rx,ry,rz]) 和 'buttons'
        """
        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            0,
            openvr.k_unMaxTrackedDeviceCount
        )

        result = {
            'left': None,
            'right': None
        }

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if not poses[i].bPoseIsValid:
                continue

            device_class = self.vr_system.getTrackedDeviceClass(i)

            if device_class == openvr.TrackedDeviceClass_Controller:
                role = self.vr_system.getControllerRoleForTrackedDeviceIndex(i)

                role_str = None
                if role == openvr.TrackedControllerRole_LeftHand:
                    role_str = 'left'
                elif role == openvr.TrackedControllerRole_RightHand:
                    role_str = 'right'

                if role_str is None:
                    continue

                pose_matrix = self._pose_to_matrix(poses[i].mDeviceToAbsoluteTracking)
                position = self._matrix_to_position(pose_matrix)
                roll, pitch, yaw = self._matrix_to_euler(pose_matrix, degrees=True)

                buttons = self._get_controller_buttons(i)

                result[role_str] = {
                    'pose': [position[0], position[1], position[2], roll, pitch, yaw],
                    'buttons': buttons
                }

        return result

    def get_left_controller(self) -> Optional[Dict]:
        """
        获取左手控制器状态

        Returns:
            左手控制器状态字典，未连接返回 None
        """
        return self.get_state()['left']

    def get_right_controller(self) -> Optional[Dict]:
        """
        获取右手控制器状态

        Returns:
            右手控制器状态字典，未连接返回 None
        """
        return self.get_state()['right']

    def shutdown(self):
        """关闭 OpenVR 系统"""
        try:
            openvr.shutdown()
            print("shutdown: Vive 系统已关闭")
        except Exception as e:
            print(f"shutdown: 关闭失败 - {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


if __name__ == "__main__":
    """
    测试样例：循环打印左右手控制器的位姿和按键状态
    """
    print("=" * 80)
    print("ViveClass 测试样例")
    print("=" * 80)
    print("提示：按下不同按键查看响应，button_bits 显示原始按键位用于调试")
    print("=" * 80)

    try:
        vive = ViveClass()

        print("\n开始读取控制器数据 (按 Ctrl+C 退出)...\n")

        while True:
            state = vive.get_state()

            # 打印左手控制器
            if state['left'] is not None:
                left_pose = state['left']['pose']
                left_btns = state['left']['buttons']
                trackpad_dir = []
                if left_btns['trackpad_up']: trackpad_dir.append('↑')
                if left_btns['trackpad_down']: trackpad_dir.append('↓')
                if left_btns['trackpad_left']: trackpad_dir.append('←')
                if left_btns['trackpad_right']: trackpad_dir.append('→')
                dir_str = ''.join(trackpad_dir) if trackpad_dir else '-'

                print(f"[左手] 位置: ({left_pose[0]:.3f}, {left_pose[1]:.3f}, {left_pose[2]:.3f})  "
                      f"姿态: ({left_pose[3]:.1f}°, {left_pose[4]:.1f}°, {left_pose[5]:.1f}°)  "
                      f"扳机: {left_btns['trigger']:.2f}  "
                      f"Grip: {'✓' if left_btns['grip'] else '✗'}  "
                      f"Menu: {'✓' if left_btns['menu_pressed'] else '✗'}  "
                      f"Trackpad: {dir_str} ({left_btns['trackpad_x']:.2f}, {left_btns['trackpad_y']:.2f})  "
                      f"bits: {left_btns['button_pressed_bits']}")
            else:
                print("[左手] 未连接")

            # 打印右手控制器
            if state['right'] is not None:
                right_pose = state['right']['pose']
                right_btns = state['right']['buttons']
                trackpad_dir = []
                if right_btns['trackpad_up']: trackpad_dir.append('↑')
                if right_btns['trackpad_down']: trackpad_dir.append('↓')
                if right_btns['trackpad_left']: trackpad_dir.append('←')
                if right_btns['trackpad_right']: trackpad_dir.append('→')
                dir_str = ''.join(trackpad_dir) if trackpad_dir else '-'

                print(f"[右手] 位置: ({right_pose[0]:.3f}, {right_pose[1]:.3f}, {right_pose[2]:.3f})  "
                      f"姿态: ({right_pose[3]:.1f}°, {right_pose[4]:.1f}°, {right_pose[5]:.1f}°)  "
                      f"扳机: {right_btns['trigger']:.2f}  "
                      f"Grip: {'✓' if right_btns['grip'] else '✗'}  "
                      f"Menu: {'✓' if right_btns['menu_pressed'] else '✗'}  "
                      f"Trackpad: {dir_str} ({right_btns['trackpad_x']:.2f}, {right_btns['trackpad_y']:.2f})  "
                      f"bits: {right_btns['button_pressed_bits']}")
            else:
                print("[右手] 未连接")

            print("-" * 120)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        if 'vive' in locals():
            vive.shutdown()
        print("程序退出")
