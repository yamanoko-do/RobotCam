"""
pip install franky-control=1.1.1
"""
import time
import threading
from franky import *
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
from typing import List, Tuple, Optional

class Teleop:
    """
    速度控制模式遥操作：
    每个控制周期计算末端到目标的位置误差，转换为速度指令发出。
    - 无位置队列积压
    - 无 stop() 打断
    - 无加速度不连续 reflex
    松开 Grip 时 clear_target()，速度自然归零停止。
    """

    def __init__(self, robot, dynamics_factor=0.5, hz=100):
        self.robot = robot
        self._dynamics_factor = dynamics_factor
        self._hz = hz
        self._dt_ms = int(1000 / hz)   # Duration 单位是毫秒
        self._dt_s = 1.0 / hz

        # P 增益：位置误差(m) -> 速度(m/s)，姿态误差(rad) -> 角速度(rad/s)
        self._kp_pos = 1.0
        self._kp_rot = 1.0
        # 放宽速度上限（根据实际需求调整）
        self._max_vel_pos = 0.5      # m/s，原1.0
        self._max_vel_rot = 0.5      # rad/s，原1.0

        self._target_affine = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._last_lin_vel = np.zeros(3)
        self._last_ang_vel = np.zeros(3)
        self._alpha = 0.3   # 平滑系数，越小越平滑

    def set_target(self, affine):
        with self._lock:
            self._target_affine = affine

    def clear_target(self):
        with self._lock:
            self._target_affine = None

    def _loop(self):
        from franky import CartesianVelocityMotion, RobotVelocity, Twist, Duration
        import time

        # 清错，设置动力学因子
        try:
            self.robot.recover_from_errors()
            time.sleep(0.1)
        except:
            pass
        self.robot.relative_dynamics_factor = self._dynamics_factor

        next_time = time.perf_counter()
        while self._running:
            with self._lock:
                target = self._target_affine

            try:
                if target is None:
                    # 零速度指令（异步）
                    zero_twist = Twist(np.zeros(3), np.zeros(3))
                    zero_vel = RobotVelocity(zero_twist)
                    motion = CartesianVelocityMotion(zero_vel, Duration(self._dt_ms),
                                                    relative_dynamics_factor=self._dynamics_factor)
                    self.robot.move(motion, asynchronous=True)   # ← 关键修改
                else:
                    # 读取当前位姿
                    ee = self.robot.current_cartesian_state.pose.end_effector_pose
                    cur_pos = np.array(ee.translation)
                    cur_quat = np.array(ee.quaternion)

                    tgt_pos = np.array(target.translation)
                    tgt_quat = np.array(target.quaternion)

                    # 位置误差 -> 线速度
                    pos_err = tgt_pos - cur_pos
                    lin_vel = np.clip(pos_err * self._kp_pos, -self._max_vel_pos, self._max_vel_pos)

                    # 姿态误差 -> 角速度
                    r_cur = R.from_quat(cur_quat)
                    r_tgt = R.from_quat(tgt_quat)
                    r_err = r_tgt * r_cur.inv()
                    rotvec_err = r_err.as_rotvec()
                    ang_vel = np.clip(rotvec_err * self._kp_rot, -self._max_vel_rot, self._max_vel_rot)

                    twist = Twist(lin_vel, ang_vel)
                    vel = RobotVelocity(twist)
                    motion = CartesianVelocityMotion(vel, Duration(self._dt_ms),
                                                    relative_dynamics_factor=self._dynamics_factor)
                    self.robot.move(motion, asynchronous=True)   # ← 关键修改

            except Exception as e:
                if self._running:
                    print(f"[Teleop] 运动异常: {e}")
                    try:
                        self.robot.recover_from_errors()
                        time.sleep(0.05)
                    except:
                        pass

            # 精确计时
            next_time += self._dt_s
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()

    def start(self):
        if self._running:
            return
        self._running = True
        ee_pose = self.robot.current_cartesian_state.pose.end_effector_pose
        self._target_affine = Affine(ee_pose.translation, ee_pose.quaternion)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Teleop] 速度控制遥操作已启动")

    def stop(self):
        self._running = False
        try:
            self.robot.stop()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[Teleop] 已停止")

class FrankaClass:
    """
    与PiperClass接口兼容的Franka机器人控制类
    注意单位转换：Franka内部使用米和弧度，Piper使用毫米和度
    """
    
    def __init__(self, ip_address: str = "172.16.0.2", dynamics_factor: float = 0.1):
        """
        初始化Franka机器人连接
        
        Args:
            ip_address: 机器人IP地址
            dynamics_factor: 相对动力学因子(0-1)，控制速度/加速度/加加速度限制
        """
        self.robot = Robot(ip_address)
        self.robot.relative_dynamics_factor = dynamics_factor
        self._gripper = None  # 夹爪对象，需要单独初始化
        print(f"init: 成功连接到Franka (IP: {ip_address})")
        
    def _init_gripper(self):
        """初始化夹爪连接（适配你当前版本的 franky）"""
        if self._gripper is None:
            try:
                from franky import Gripper
                # ✅ 你的版本必须这样创建：直接传入 IP
                self._gripper = Gripper(self.robot.fci_hostname)
                print("init: 夹爪已连接")
            except Exception as e:
                print(f"init: 夹爪连接失败 - {e}")
    def control_gripper(self, length: float, force: float = 20.0, is_grasp: bool = False):
        """
        控制夹爪（支持开合和夹持两种模式）
        
        Args:
            length: 目标开口距离，单位mm，0~80mm
            force: 夹持力，单位N，默认20N（建议10-60N）
            is_grasp: True=夹持模式（用力夹紧），False=位置模式（单纯开合）
        """
        if self._gripper is None:
            self._init_gripper()
            
        if self._gripper is None:
            print("control_gripper: 夹爪未连接")
            return
            
        # 限制范围
        length = max(0, min(length, 80))
        width_m = length / 1000.0
        
        try:
            if is_grasp:
                # ✅ 夹持模式：先移动到目标位置，然后用力夹紧
                # 参数：目标宽度(m), 速度(m/s), 力(N), 是否等待完成
                self._gripper.grasp(width_m, 0.05, force, epsilon_inner=0.5, epsilon_outer=0.5)
                print(f"control_gripper: 夹持模式，目标{width_m*1000:.1f}mm，力{force}N")
            else:
                # 单纯开合
                self._gripper.move(width_m, 0.05)
                print(f"control_gripper: 开合模式，开口 {length} mm")
                
        except Exception as e:
            print(f"control_gripper: 控制失败 - {e}")

    
    def control_joint(self, joint_angle: List[float], angle_type: str = "degree"):
            """
            控制关节角度（绝对位置控制）

            Args:
                joint_angle: 6个或7个关节角度列表
                angle_type: 角度单位，"degree"（默认，角度制）或 "radian"（弧度制）
            """
            # 验证angle_type参数
            if angle_type not in ["degree", "radian"]:
                raise ValueError(f'angle_type必须是"degree"或"radian"，当前为"{angle_type}"')

            # 根据输入单位转换为弧度
            if angle_type == "degree":
                q_input = list(np.radians(joint_angle))
            else:  # radian
                q_input = list(joint_angle)

            # 处理6个或7个关节的情况
            if len(joint_angle) == 6:
                # 如果提供6个关节，补充第7个关节为当前值或默认值
                try:
                    current_q = self.robot.current_joint_state.position
                    q7 = current_q[6] if len(current_q) >= 7 else 0.7
                except:
                    q7 = 0.7  # 默认安全值
                q_input.append(q7)
            elif len(joint_angle) != 7:
                raise ValueError(f"joint_angle必须是6或7个关节角度，当前{len(joint_angle)}个")

            # 执行关节运动
            motion = JointMotion(q_input)
            self.robot.move(motion)
    
    def getpose(self) -> dict:
        """
        获取机械臂末端位姿和关节角度
        
        Returns:
            dict: {
                "end_pose": [x, y, z, rx, ry, rz] (单位: mm和度),
                "joint_state": [j1, j2, j3, j4, j5, j6, j7] (单位: 度 Franka是7关节)
            }
        """
        # 获取末端位姿
        ee_affine = self.robot.current_cartesian_state.pose.end_effector_pose
        pos_m = ee_affine.translation  # [x, y, z] in meters
        quat = ee_affine.quaternion   # [x, y, z, w]
        
        # 转换为欧拉角（xyz顺序，单位度）
        rpy_deg = R.from_quat(quat).as_euler('xyz', degrees=True)
        
        # 转换为mm
        pose_list = [
            pos_m[0] * 1000,  # x mm
            pos_m[1] * 1000,  # y mm
            pos_m[2] * 1000,  # z mm
            rpy_deg[0],       # rx deg
            rpy_deg[1],       # ry deg
            rpy_deg[2]        # rz deg
        ]
        
        # 获取关节角度
        joint_state = self.robot.current_joint_state.position  # 弧度
        joint_state_list = list(np.degrees(joint_state))  # 转为度
        
        pose_dict = {
            "end_pose": pose_list,
            "joint_state": joint_state_list  # Franka有7个关节
        }
        return pose_dict
    

    def disable(self):
        """停止机器人（兼容 reflex 保护状态）"""
        try:
            # 优先尝试直接 stop（reflex 状态下也有效）
            self.robot.stop()
            print("disable: 机器人已停止 (stop)")
        except Exception:
            try:
                # stop 不可用时，尝试 recover 清除 reflex 再 stop
                self.robot.recover()
                time.sleep(0.5)
                self.robot.stop()
                print("disable: 机器人已恢复并停止 (recover + stop)")
            except Exception as e2:
                print(f"disable: 停止失败（可能需要手动解锁）- {e2}")

    

        
    @staticmethod
    def get_dh_params():
        """
        ✅ 严格匹配图片中的 Modified DH 参数表 (8行)
        Modified DH: (a, alpha, d, theta_offset)
        """
        dh_params = [
            (0.0,      0.0,      0.333,   0.0),          # Joint 1
            (0.0,     -np.pi/2,  0.0,     0.0),          # Joint 2
            (0.0,      np.pi/2,  0.316,   0.0),          # Joint 3
            (0.0825,   np.pi/2,  0.0,     0.0),          # Joint 4
            (-0.0825, -np.pi/2,  0.384,   0.0),          # Joint 5
            (0.0,      np.pi/2,  0.0,     0.0),          # Joint 6
            (0.088,    np.pi/2,  0.0,     -np.pi/4),     # Joint 7 (包含旋转修正)
            (0.0,      0.0,      0.107,   0.0),          # Flange (对应图片最后一行)
        ]
        return dh_params
    
    @staticmethod
    def dh_transform(a, alpha, d, theta):
        """
        根据DH参数计算齐次变换矩阵（Modified DH）
        """
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([
            [ct,    -st,    0,      a],
            [st*ca, ct*ca,  -sa,    -d*sa],
            [st*sa, ct*sa,  ca,     d*ca],
            [0 ,      0,      0,    1]
        ])
    
    @staticmethod
    def forward_kinematics(joint_angles, format: str = "euler"):
        """
        ✅ 修复版正运动学
        """
        dh_params = FrankaClass.get_dh_params()
        T = np.eye(4)

        # 1. 计算 1-7 关节的变换
        for i in range(7):
            a, alpha, d, theta_offset = dh_params[i]
            theta_rad = math.radians(joint_angles[i]) + theta_offset
            T = T @ FrankaClass.dh_transform(a, alpha, d, theta_rad)

        # 2. 计算法兰(Flange)变换 (即 DH 参数表的第 8 行)
        a_f, alpha_f, d_f, theta_f = dh_params[7]
        # 法兰是固定变换，theta 为 0
        T_flange = FrankaClass.dh_transform(a_f, alpha_f, d_f, theta_f)
        T = T @ T_flange

        # 3. 关键修复：加上夹爪 TCP 偏移 (从法兰面到指尖)
        # 如果你安装了 Franka Hand，这个值通常是 103.4mm
        T_gripper_tcp = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0.1034], # 夹爪长度 (单位: 米)
            [0, 0, 0, 1]
        ])
        T = T @ T_gripper_tcp

        if format == "matrix":
            return T
        return FrankaClass.matrix_to_pose(T, format2deg=True)
    
    @staticmethod
    def matrix_to_pose(T, format2deg: bool = False):
        """修复版：正确处理万向节锁和角度周期"""
        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        R_mat = T[:3, :3]
        
        # 使用 scipy 提取欧拉角（与 Franka 保持一致）
        from scipy.spatial.transform import Rotation as R_sci
        r = R_sci.from_matrix(R_mat)
        # 使用 'xyz' extrinsic（固定轴）或 'XYZ' intrinsic（动轴）
        # Franka 通常使用 intrinsic 'xyz'
        rx, ry, rz = r.as_euler('xyz', degrees=False)
        
        # 归一化到 [-pi, pi]
        rx = math.atan2(math.sin(rx), math.cos(rx))
        ry = math.atan2(math.sin(ry), math.cos(ry))
        rz = math.atan2(math.sin(rz), math.cos(rz))
        
        if format2deg:
            rx, ry, rz = np.degrees([rx, ry, rz])
            x, y, z = x * 1000, y * 1000, z * 1000
            
        return [x, y, z, rx, ry, rz]
    
    @staticmethod
    def inverse_kinematics(end_pose, initial_guess: Optional[List[float]] = None, 
                          max_iterations: int = 100, tolerance: float = 1e-4):
        """
        数值逆运动学（使用scipy.optimize.least_squares）
        
        Args:
            end_pose: [x, y, z, rx, ry, rz]，单位mm和度
            initial_guess: 初始关节角猜测（度），7个关节
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
        
        Returns:
            成功时返回7个关节角度列表（度），失败返回None
        """
        try:
            from scipy.optimize import least_squares

            # 输入检查
            if end_pose is None or len(end_pose) != 6:
                raise ValueError("end_pose必须是长度为6的[x, y, z, rx, ry, rz]")

            # 默认初始猜测（安全姿态）
            default_guess = [0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0]
            if initial_guess is None:
                initial_guess = default_guess

            # 确保是7个关节
            if len(initial_guess) == 6:
                initial_guess = list(initial_guess) + [45.0]
            elif len(initial_guess) != 7:
                raise ValueError("initial_guess必须是6或7个关节角度")

            # Franka关节限制（度）
            lower_bounds = np.array([-166.0, -101.0, -166.0, -176.0, -166.0, -1.0, -166.0], dtype=float)
            upper_bounds = np.array([166.0, 101.0, 166.0, -4.0, 166.0, 215.0, 166.0], dtype=float)

            # 目标位姿（位置:mm，姿态:旋转矩阵）
            target_pose = np.array(end_pose, dtype=float)
            target_pos_mm = target_pose[:3]
            target_rot = R.from_euler('xyz', target_pose[3:], degrees=True).as_matrix()

            def pose_error(joint_deg):
                """
                残差向量: [dx_mm, dy_mm, dz_mm, dRx_deg, dRy_deg, dRz_deg]
                其中旋转误差使用旋转向量，避免欧拉角跳变造成的假大误差。
                """
                T = FrankaClass.forward_kinematics(joint_deg, format="matrix")

                # 位置误差（mm）
                current_pos_mm = T[:3, 3] * 1000.0
                pos_err = current_pos_mm - target_pos_mm

                # 旋转误差（deg，旋转向量）
                current_rot = T[:3, :3]
                rot_err_mat = target_rot @ current_rot.T
                rot_err_deg = np.degrees(R.from_matrix(rot_err_mat).as_rotvec())

                return np.hstack([pos_err, rot_err_deg])

            def eval_solution(q):
                e = pose_error(q)
                pos_abs = np.abs(e[:3])
                rot_abs = np.abs(e[3:])
                # 综合评分：位置优先，其次姿态
                score = np.linalg.norm(e[:3]) + 0.5 * np.linalg.norm(e[3:])
                return {
                    "err": e,
                    "max_pos": float(np.max(pos_abs)),
                    "max_rot": float(np.max(rot_abs)),
                    "score": float(score),
                }

            # 多初值，降低局部最优概率
            seeds = []
            seed0 = np.clip(np.array(initial_guess, dtype=float), lower_bounds, upper_bounds)
            seeds.append(seed0)
            seeds.append(np.clip(np.array(default_guess, dtype=float), lower_bounds, upper_bounds))

            # 额外启发式初值（围绕常见安全姿态）
            heuristic_seeds = [
                [0.0, -30.0, 0.0, -120.0, 0.0, 90.0, 0.0],
                [0.0, -60.0, 0.0, -140.0, 0.0, 110.0, 60.0],
                [20.0, -50.0, 20.0, -130.0, -20.0, 100.0, 30.0],
            ]
            for s in heuristic_seeds:
                seeds.append(np.clip(np.array(s, dtype=float), lower_bounds, upper_bounds))

            best_q = None
            best_eval = None

            for seed in seeds:
                result = least_squares(
                    lambda q: pose_error(q),
                    seed,
                    bounds=(lower_bounds, upper_bounds),
                    method="trf",
                    loss="huber",
                    f_scale=1.0,
                    max_nfev=max_iterations * 7,
                    xtol=tolerance,
                    ftol=tolerance,
                    gtol=tolerance,
                )

                q_candidate = np.clip(result.x, lower_bounds, upper_bounds)
                cur_eval = eval_solution(q_candidate)

                if (best_eval is None) or (cur_eval["score"] < best_eval["score"]):
                    best_q = q_candidate
                    best_eval = cur_eval

            if best_q is None or best_eval is None:
                return None

            # 结果验收阈值：不满足则明确返回None，避免“错误逆解”被当成可用解
            # 默认 tolerance=1e-4 时，约为位置<=1mm、姿态<=5deg
            pos_threshold_mm = max(1.0, min(20.0, tolerance * 1e4))
            rot_threshold_deg = max(1.0, min(15.0, tolerance * 5e4))

            if best_eval["max_pos"] <= pos_threshold_mm and best_eval["max_rot"] <= rot_threshold_deg:
                return best_q.tolist()

            print(
                "inverse_kinematics: 未找到满足阈值的解 "
                f"(max_pos={best_eval['max_pos']:.3f}mm, max_rot={best_eval['max_rot']:.3f}deg, "
                f"阈值={pos_threshold_mm:.3f}mm/{rot_threshold_deg:.3f}deg)"
            )
            return None

        except ImportError:
            print("inverse_kinematics: 需要scipy库")
            return None
        except Exception as e:
            print(f"inverse_kinematics: 求解失败 - {e}")
            return None
    
    def start_teleop(self, dynamics_factor: float = 0.08):
        """
        启动遥操作模式：后台线程持续以低动力学因子执行 CartesianMotion。

        Args:
            dynamics_factor: 相对动力学因子 (0-1)，越低越慢越柔顺，默认0.08
        Returns:
            Teleop 实例，用于 set_target / stop
        """
        teleop = Teleop(self.robot, dynamics_factor)
        teleop.start()
        return teleop

    def move_cartesian(self, position: Optional[List[float]] = None,
                        orientation: Optional[List[float]] = None,
                        relative: bool = False, relative_dynamics: float = 0.1):
        """
        笛卡尔空间运动
        
        Args:
            position: [x, y, z] 目标位置
                - relative=True 时：单位 mm（偏移量），None=不移动
                - relative=False 时：单位 m（绝对位置），None=保持当前位置
            orientation: 四元数[x,y,z,w] 或 欧拉角[rx,ry,rz]（度），None=保持当前姿态
            relative: 是否相对运动
            relative_dynamics: 动力学缩放因子(0-1)
        """
        from franky import Affine, CartesianMotion
        
        # ---- 1. 获取当前状态 ----
        current_state = self.robot.current_cartesian_state
        ee_affine = current_state.pose.end_effector_pose
        current_pos = list(ee_affine.translation)       # [x, y, z] 米
        current_quat = list(ee_affine.quaternion)       # [x, y, z, w]
        
        # ---- 2. 计算目标位置（始终转为绝对坐标） ----
        if position is None:
            target_pos = current_pos                     # 保持当前位置
        elif relative:
            offset_m = [p / 1000.0 for p in position]   # mm → m
            target_pos = [current_pos[i] + offset_m[i] for i in range(3)]
            print(f"  相对偏移: {position} mm → 绝对目标(m): {[round(v, 4) for v in target_pos]}")
        else:
            target_pos = list(position)
        
        # ---- 3. 计算目标姿态 ----
        if orientation is None:
            target_quat = current_quat
        elif len(orientation) == 3:
            if relative:
                # 相对模式：将欧拉角增量叠加到当前姿态
                current_euler = R.from_quat(current_quat).as_euler('xyz', degrees=True)
                target_euler = [current_euler[i] + orientation[i] for i in range(3)]
                target_quat = list(R.from_euler('xyz', target_euler, degrees=True).as_quat())
                print(f"  相对旋转: {orientation} ° → 绝对姿态(°): {[round(v, 2) for v in target_euler]}")
            else:
                target_quat = list(R.from_euler('xyz', orientation, degrees=True).as_quat())
        else:
            if relative:
                # 四元数增量模式：将增量旋转叠加到当前姿态
                current_rot = R.from_quat(current_quat)
                delta_rot = R.from_quat(orientation)
                target_quat = list((delta_rot * current_rot).as_quat())
            else:
                target_quat = list(orientation)
        
        # ---- 4. 构建 Affine 并执行 ----
        target_affine = Affine(target_pos, target_quat)
        
        old_dynamics = self.robot.relative_dynamics_factor
        self.robot.relative_dynamics_factor = relative_dynamics
        
        try:
            motion = CartesianMotion(target_affine)
            self.robot.move(motion)
            print(f"  ✅ 运动完成")
        except Exception as e:
            print(f"  ❌ 笛卡尔运动失败: {e}")
        finally:
            self.robot.relative_dynamics_factor = old_dynamics



    def get_current_pose_euler(self) -> List[float]:
        """
        获取当前位姿（欧拉角格式，单位mm和度）
        
        Returns:
            [x, y, z, rx, ry, rz]
        """
        pose_dict = self.getpose()
        return pose_dict["end_pose"]


# ==================== 使用示例 ====================

if __name__ == "__main__":
    """
    获取机械臂位姿示例
    """
    # franka = FrankaClass(ip_address="172.16.0.2", dynamics_factor=0.1) 
    # try:
    #     while True:
    #         pose_dict = franka.getpose()
    #         print(f"末端位姿: {pose_dict['end_pose']}")
    #         print(f"关节角度: {pose_dict['joint_state']}")
    #         print("-" * 40)
    #         time.sleep(1)
    # except KeyboardInterrupt:
    #     print("\n程序已被用户中断...")
    # finally:
    #     franka.disable()
    #     print("程序已退出")
    """
    控制夹爪开合
    """
    # franka = FrankaClass("172.16.0.2", 0.1)

    # # 打开到 80mm
    # franka.control_gripper(80)
    # time.sleep(1)

    # # 闭合到 10mm
    # franka.control_gripper(10)
    # time.sleep(1)

    # # 完全闭合
    # franka.control_gripper(0)
    """
    运动学检查
    """
    # franka = FrankaClass("172.16.0.2", 0.1)
    # q_test = [-0.0, 0.0, 0.0, -2.0, 0.0, 1.8, 0.7]
    
    # # 运动到测试位姿
    # franka.control_joint(q_test,angle_type = "radian")
    # time.sleep(2)

    # # 1. 机器人真实读数
    # real = franka.getpose()["end_pose"]
    # print(len(franka.getpose()["joint_state"]))
    # print("【机器人真实位姿】\n", real)

    # # 2. DH正运动学计算
    # calc = FrankaClass.forward_kinematics(np.rad2deg(q_test), format="euler")
    # print("【DH计算位姿】\n", calc)

    # # 3. 误差
    # err = np.abs(np.array(real[:3]) - np.array(calc[:3]))
    # print(f"【位置误差】X: {err[0]:.2f}mm Y: {err[1]:.2f}mm Z: {err[2]:.2f}mm")
    # # 检查旋转矩阵是否相同（这才是金标准）
    # real_pose = franka.getpose()["end_pose"]
    # calc_T = FrankaClass.forward_kinematics(q_test, format="matrix")

    # # 从真实读数重建旋转矩阵
    # real_R = R.from_euler('xyz', real_pose[3:], degrees=True).as_matrix()

    # # 比较旋转矩阵
    # print("旋转矩阵差:\n", calc_T[:3,:3] - real_R)
    """
    新增：纯前向运动学（正解）测试代码
    输入7个关节角度 → 输出末端位姿（mm + deg）
    不依赖机器人连接，纯数学计算
    """
    # print("=" * 60)
    # print("【 正运动学 DH 计算测试 】")
    # print("=" * 60)

    # # ===================== 1. 输入：7个关节角度（单位：度） =====================
    # # 你可以随便改这组角度
    # joint_test = [0.0, -45.0, 0.0, -125.0, 0.0, 90.0, 40.0]
    # print(f"输入关节角度（7轴）：")
    # print([round(j, 2) for j in joint_test])

    # # ===================== 2. 正运动学计算 =====================
    # pose_calc = FrankaClass.forward_kinematics(joint_test)

    # # ===================== 3. 打印结果 =====================
    # print("\n" + "=" * 60)
    # print("【正运动学计算结果】")
    # print(f"X: {pose_calc[0]:.3f} mm")
    # print(f"Y: {pose_calc[1]:.3f} mm")
    # print(f"Z: {pose_calc[2]:.3f} mm")
    # print(f"RX: {pose_calc[3]:.3f} °")
    # print(f"RY: {pose_calc[4]:.3f} °")
    # print(f"RZ: {pose_calc[5]:.3f} °")
    # print("=" * 60)

    # # ===================== 4. 输出齐次变换矩阵（可选） =====================
    # T_matrix = FrankaClass.forward_kinematics(joint_test, format="matrix")
    # print("\n【齐次变换矩阵】")
    # print(np.round(T_matrix, 4))

    # print("\n✅ 正运动学测试完成")
    """
    新增：逆运动学 + 正运动学 闭环测试
    给定位姿a → 求关节角 → 正解算位姿b → 打印a、b、误差
    """
    # franka = FrankaClass(ip_address="172.16.0.2", dynamics_factor=0.1)

    # # ===================== 1. 自定义目标位姿 a =====================
    # # 单位：mm(位置) / deg(角度) [x,y,z,rx,ry,rz]
    # pose_a = [321.0, 0, 570.0, 180.0, -10.0, 5.0]
    # print("="*60)
    # print("【目标位姿 a】")
    # print(f"x: {pose_a[0]:.2f} mm  |  y: {pose_a[1]:.2f} mm  |  z: {pose_a[2]:.2f} mm")
    # print(f"rx:{pose_a[3]:.2f} deg | ry:{pose_a[4]:.2f} deg | rz:{pose_a[5]:.2f} deg")
    # print("="*60)

    # # ===================== 2. 逆运动学求关节角度 =====================
    # print("\n正在求解逆运动学...")
    # joint_sol = franka.inverse_kinematics(pose_a)

    # if joint_sol is None:
    #     print("❌ 逆解失败！")
    #     franka.disable()
    #     exit()

    # print(f"✅ 求得关节角度（7轴）：")
    # print([round(j,2) for j in joint_sol])

    # # ===================== 3. 正运动学计算位姿 b =====================
    # pose_b = franka.forward_kinematics(joint_sol)
    # print("\n" + "="*60)
    # print("【正解位姿 b】")
    # print(f"x: {pose_b[0]:.2f} mm  |  y: {pose_b[1]:.2f} mm  |  z: {pose_b[2]:.2f} mm")
    # print(f"rx:{pose_b[3]:.2f} deg | ry:{pose_b[4]:.2f} deg | rz:{pose_b[5]:.2f} deg")
    # print("="*60)

    # # ===================== 4. 计算并打印误差 =====================
    # pos_err = np.abs(np.array(pose_a[:3]) - np.array(pose_b[:3]))
    # ang_err = np.abs(np.array(pose_a[3:]) - np.array(pose_b[3:]))

    # print("\n" + "="*60)
    # print("【误差对比】")
    # print(f"位置误差：X={pos_err[0]:.4f} mm  Y={pos_err[1]:.4f} mm  Z={pos_err[2]:.4f} mm")
    # print(f"角度误差：rx={ang_err[0]:.4f}°  ry={ang_err[1]:.4f}°  rz={ang_err[2]:.4f}°")
    # print(f"总位置误差：{np.linalg.norm(pos_err):.4f} mm")
    # print("="*60)

    # # 可选：让机器人运动到该关节角（真实验证）
    # # print("\n机器人运动到该姿态...")
    # # franka.control_joint(joint_sol)
    # # time.sleep(2)

    # franka.disable()
    # print("\n✅ 测试完成，程序退出")
    """
    相对位置控制模式测试,沿着x,y,z方向分别前进后退20mm,rxryrz分别旋转15度并转回
    """
    # franka = FrankaClass(ip_address="172.16.0.2", dynamics_factor=0.1)

    # try:
    #     print("=" * 60)
    #     print("【 相对位置控制模式测试 】")
    #     print("=" * 60)

    #     # 获取初始位姿
    #     initial_pose = franka.get_current_pose_euler()
    #     print(f"\n初始位姿: {[round(v, 2) for v in initial_pose]}")

    #     # 测试参数
    #     translation_dist = 20.0  # 平移距离 mm
    #     rotation_angle = 10.0    # 旋转角度 deg
    #     wait_time = 2.0          # 等待时间 s

    #     # 轴名称
    #     axes = ['X', 'Y', 'Z']
    #     rot_axes = ['RX', 'RY', 'RZ']

    #     # ==================== 平移测试 ====================
    #     print("\n" + "-" * 60)
    #     print("【 平移测试 】")
    #     print("-" * 60)

    #     for i, axis in enumerate(axes):
    #         print(f"\n--- {axis} 轴测试 ---")

    #         # 正向移动
    #         delta = [0.0, 0.0, 0.0]
    #         delta[i] = translation_dist
    #         print(f"正向移动 {translation_dist}mm...")
    #         franka.move_cartesian(delta, relative=True)
    #         time.sleep(wait_time)

    #         # 获取当前位姿
    #         current_pose = franka.get_current_pose_euler()
    #         print(f"当前位姿: {[round(v, 2) for v in current_pose]}")

    #         # 反向移动（回到原位）
    #         delta[i] = -translation_dist
    #         print(f"反向移动 {translation_dist}mm...")
    #         franka.move_cartesian(delta, relative=True)
    #         time.sleep(wait_time)

    #         # 获取当前位姿
    #         current_pose = franka.get_current_pose_euler()
    #         print(f"当前位姿: {[round(v, 2) for v in current_pose]}")

    #     # ==================== 旋转测试 ====================
    #     print("\n" + "-" * 60)
    #     print("【 旋转测试 】")
    #     print("-" * 60)  # 正确

    #     for i, axis in enumerate(rot_axes):
    #         print(f"\n--- {axis} 轴测试 ---")

    #         # 获取当前姿态
    #         current_pose = franka.get_current_pose_euler()
    #         current_ori = current_pose[3:]  # [rx, ry, rz]

    #         # ✅ 正向旋转：position=None 保持当前位置
    #         target_ori = current_ori.copy()
    #         target_ori[i] += rotation_angle
    #         print(f"正向旋转 {rotation_angle}°...")
    #         franka.move_cartesian(position=None, orientation=target_ori, relative=False)
    #         time.sleep(wait_time)

    #         current_pose = franka.get_current_pose_euler()
    #         print(f"当前位姿: {[round(v, 2) for v in current_pose]}")

    #         # ✅ 反向旋转：回到旋转前的姿态
    #         target_ori_back = current_ori.copy()   # 旋转前的姿态
    #         print(f"反向旋转 {rotation_angle}°...")
    #         franka.move_cartesian(position=None, orientation=target_ori_back, relative=False)
    #         time.sleep(wait_time)

    #         current_pose = franka.get_current_pose_euler()
    #         print(f"当前位姿: {[round(v, 2) for v in current_pose]}")


    #     # ==================== 组合测试 ====================
    #     print("\n" + "-" * 60)
    #     print("【 组合运动测试 】")
    #     print("-" * 60)

    #     print("\n同时移动 X+10mm, Y+10mm, Z+10mm...")
    #     franka.move_cartesian([10, 10, 10], relative=True)
    #     time.sleep(wait_time)
    #     current_pose = franka.get_current_pose_euler()
    #     print(f"当前位姿: {[round(v, 2) for v in current_pose]}")

    #     print("\n同时移回 X-10mm, Y-10mm, Z-10mm...")
    #     franka.move_cartesian([-10, -10, -10], relative=True)
    #     time.sleep(wait_time)
    #     current_pose = franka.get_current_pose_euler()
    #     print(f"当前位姿: {[round(v, 2) for v in current_pose]}")

    #     print("\n" + "=" * 60)
    #     print("✅ 相对位置控制测试完成")
    #     print("=" * 60)

    # except KeyboardInterrupt:
    #     print("\n测试被用户中断")
    # except Exception as e:
    #     print(f"\n测试出错: {e}")
    # finally:
    #     franka.disable()
    #     print("\n程序已退出")
    """
    resetfranka
    """
    franka = FrankaClass(ip_address="172.16.0.2", dynamics_factor=0.1)

    # 安全姿态（弧度转角度）
    q_safe_rad = [-0.0, 0.0, 0.0, -2.0, 0.0, 1.8, 0.7]
    q_safe_deg = np.degrees(q_safe_rad).tolist()  # 安全姿态示例（避免奇异）
    franka.control_joint(q_safe_deg)

    def get_current_pose():
        pose_dict = franka.getpose()
        pos_mm = pose_dict["end_pose"][:3]  # [x,y,z] mm
        rpy_deg = pose_dict["end_pose"][3:]  # [rx,ry,rz] deg
        joint_deg = pose_dict["joint_state"]  # 7关节角度 deg

        # 转换为米（与原代码一致）
        pos_m = [p / 1000.0 for p in pos_mm]

        print("pos:", pos_m)
        # 从欧拉角转回四元数（与原代码输出格式一致）
        quat = R.from_euler('xyz', rpy_deg, degrees=True).as_quat()  # [x,y,z,w]
        print("cur_quat:", quat)
        print("rpy_deg:", rpy_deg)
        print('-'*40)

    get_current_pose()

    franka.disable()