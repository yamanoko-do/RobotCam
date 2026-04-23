#!/usr/bin/env python3
"""
Vive 右手遥控 Franka 机械臂 — 最新值模式（无指令积压）

改动核心：用 threading.Lock + 单个变量替代任何队列，
后台线程永远只追最新目标，松开 Grip 立即停止。
"""
import os
import sys
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hardware', 'robotarm'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hardware', 'VR'))
from franka import FrankaClass
from vive import ViveClass

FRANKA_IP = "172.16.0.2"
DYNAMICS_FACTOR = 0.1
POS_SCALE = 1.0
ORI_SCALE = 1.0
AXIS_MAP = np.array([[0, 0, 1],
                     [1, 0, 0],
                     [0, 1, 0]])
AXIS_SIGN = np.array([1, -1, -1])
GRIPPER_DEADZONE = 2.0


# ==================== 最新值追踪器（核心改动） ====================
class LatestTargetTracker:
    """
    只保留最新目标位姿，后台线程以固定频率持续发送给机械臂。
    没有队列，没有积压——旧目标直接被新目标覆盖。
    """

    def __init__(self, franka_cls, dynamics_factor=0.08, poll_hz=50):
        self._franka = franka_cls
        self._dynamics_factor = dynamics_factor
        self._poll_interval = 1.0 / poll_hz

        self._lock = threading.Lock()
        self._latest_target = None   # 唯一共享变量：最新目标 Affine，None 表示不追踪
        self._running = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_target(self, affine_target):
        """主循环调用：更新目标（旧目标直接丢弃）"""
        with self._lock:
            self._latest_target = affine_target

    def clear_target(self):
        """松开 Grip 时调用：立即停止追踪"""
        with self._lock:
            self._latest_target = None

    def _loop(self):
        """后台线程：每次只取最新目标，直接发送，绝不积压"""
        from franky import CartesianMotion, ReferenceType

        while self._running:
            with self._lock:
                target = self._latest_target

            if target is not None:
                try:
                    motion = CartesianMotion(
                        target,
                        reference_type=ReferenceType.Absolute,
                        relative_dynamics_factor=self._dynamics_factor,  # 直接传 float 即可
                    )
                    # stop_on_error=False：即使上一个 motion 被打断也继续
                    self._franka.robot.move(motion, asynchronous=True)
                except Exception as e:
                    print(f"\n[Tracker] motion 异常: {e}")
                    try:
                        self._franka.robot.recover_from_errors()
                    except:
                        pass

            time.sleep(self._poll_interval)

    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
# ==============================================================


class GripperController:
    """线程安全夹爪控制（与原版相同）"""

    def __init__(self, franka_cls):
        self.franka = franka_cls
        self._lock = threading.Lock()
        self._last_width = None
        self._running = True
        self._pending_width = None
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self):
        while self._running:
            time.sleep(0.05)
            with self._lock:
                cmd = self._pending_width
                self._pending_width = None
            if cmd is not None:
                width, force, is_grasp = cmd
                try:
                    if is_grasp:
                        self.franka._gripper.grasp(
                            width, 0.05, force,
                            epsilon_inner=0.5, epsilon_outer=0.5
                        )
                    else:
                        self.franka._gripper.move(width, 0.05)
                    self._last_width = width * 1000
                except Exception as e:
                    print(f"[Gripper] 命令失败: {e}")
                    try:
                        self.franka.robot.recover_from_errors()
                    except:
                        pass

    def set_width(self, width_mm: float, force: float = 20.0):
        width_m = width_mm / 1000.0
        is_grasp = width_mm < 5.0
        with self._lock:
            if (self._last_width is not None
                    and abs(width_mm - self._last_width) < GRIPPER_DEADZONE):
                return
            self._pending_width = (width_m, force, is_grasp)

    def shutdown(self):
        self._running = False


def vive_delta_to_franka_target(vive_pose, vive_ref, ref_pos_m, ref_quat):
    from franky import Affine

    dp = AXIS_MAP @ (np.array(vive_pose[:3]) - np.array(vive_ref[:3])) * POS_SCALE
    target_pos = np.array(ref_pos_m) + dp

    d_euler = np.array(vive_pose[3:]) - np.array(vive_ref[3:])
    d_euler = (d_euler + 180) % 360 - 180
    d_euler_franka = AXIS_SIGN * (AXIS_MAP @ d_euler * ORI_SCALE)
    target_quat = (
        R.from_euler('xyz', d_euler_franka, degrees=True)
        * R.from_quat(ref_quat)
    ).as_quat()

    return Affine(target_pos.tolist(), target_quat.tolist())


def main():
    print("=" * 60)
    print("Vive 遥控 Franka — 速度控制模式（高响应 + 立即停止）")
    print("=" * 60)
    print("  Grip 按下  → 实时跟随手柄（速度控制）")
    print("  Grip 松开  → 立即停止运动")
    print("  扳机       → 夹爪开合")
    print("  Ctrl+C     → 安全退出")
    print("  位置增益   → 2.0，速度上限 → 0.3 m/s")
    print("=" * 60)

    vive = ViveClass()
    franka_cls = FrankaClass(FRANKA_IP, dynamics_factor=DYNAMICS_FACTOR)
    franka_cls._init_gripper()

    # 启动速度控制遥操作（内部已包含后台线程）
    teleop = franka_cls.start_teleop(dynamics_factor=0.2)   # 可以适当提高动力学因子
    gripper_ctrl = GripperController(franka_cls)

    grip_pressed = False
    vive_ref = None
    ref_pos_m = None
    ref_quat = None

    try:
        while True:
            right = vive.get_right_controller()
            if right is None:
                time.sleep(0.02)
                continue

            pose = right['pose']
            trigger = right['buttons']['trigger']

            if right['buttons']['grip']:
                if not grip_pressed:
                    # 首次按下：记录参考位姿（当前末端位姿）
                    ee = franka_cls.robot.current_cartesian_state.pose.end_effector_pose
                    ref_pos_m = list(ee.translation)
                    ref_quat = list(ee.quaternion)
                    vive_ref = pose[:]
                    grip_pressed = True
                    print(f"\n[Grip ON] 参考位姿: pos={ref_pos_m}")

                # 计算目标位姿
                target = vive_delta_to_franka_target(pose, vive_ref, ref_pos_m, ref_quat)
                teleop.set_target(target)   # 更新目标，速度控制器会自动追踪

                tp = target.translation
                print(
                    f"\r[跟随] ({tp[0]*1000:.1f}, {tp[1]*1000:.1f}, {tp[2]*1000:.1f})mm  "
                    f"trigger:{trigger:.2f}",
                    end='', flush=True
                )
            else:
                if grip_pressed:
                    teleop.clear_target()   # 清除目标 → 立即发零速度 → 停止
                    print("\n[Grip OFF] 立即停止追踪")
                    grip_pressed = False

                print(f"\r[停止] trigger:{trigger:.2f}   ", end='', flush=True)

            # 夹爪控制不变
            target_width_mm = 80.0 * (1.0 - trigger)
            gripper_ctrl.set_width(target_width_mm, force=20.0)

            time.sleep(0.02)  # 50Hz 主循环

    except KeyboardInterrupt:
        print("\n\n用户中断...")
    finally:
        teleop.stop()           # 停止速度控制线程
        gripper_ctrl.shutdown()
        franka_cls.disable()
        vive.shutdown()
        print("程序已退出")


if __name__ == "__main__":
    main()