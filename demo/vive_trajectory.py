"""
HTC Vive 右手控制器轨迹可视化
使用 Open3D 显示手柄姿态(T形)和历史轨迹

需要提前启动steamVR
"""
import open3d as o3d
import numpy as np
import time
import os
import sys

# 添加硬件模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'hardware', 'VR'))
from vive import ViveClass

os.environ["XDG_SESSION_TYPE"] = "x11"


def create_t_shape(scale=0.2):
    """
    创建T形几何体表示手柄姿态
    """
    # 手柄主体 - 沿X轴的横杆
    head = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=scale)
    head.rotate(o3d.geometry.get_rotation_matrix_from_xyz([np.pi/2, np.pi/2, 0]), center=(0, 0, 0))
    body = o3d.geometry.TriangleMesh.create_cylinder(radius=0.01, height=scale)
    body.rotate(o3d.geometry.get_rotation_matrix_from_xyz([0, 0, np.pi/2]), center=(0, 0, 0))


    # 【关键修改】：将横杆向上移动半个竖杆的高度，使其与竖杆顶部对齐形成T形
    body.translate([0, 0, scale * 0.4]) 

    # # 手柄头部 - 沿Y轴的竖杆 (默认中心在原点，高度0.8*scale，所以顶端在0.4*scale处)
    # head = o3d.geometry.TriangleMesh.create_cylinder(radius=0.015, height=scale*0.8)

    # 组合成T形 (此时原点刚好在竖杆的最底部)
    t_shape = head + body
    t_shape.paint_uniform_color([0.8, 0.2, 0.2])  # 红色



    down = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=0.012, cone_radius=0.02,
                                                    cylinder_height=0.12, cone_height=0.04)
    down.rotate(o3d.geometry.get_rotation_matrix_from_xyz([np.pi/2, 0, 0]), center=(0, 0, 0))
    down.paint_uniform_color([0, 0, 1])  # Z轴蓝色

    return t_shape + down



def euler_to_rotation_matrix(roll, pitch, yaw, degrees=True):
    """欧拉角转旋转矩阵"""
    if degrees:
        roll, pitch, yaw = np.radians(roll), np.radians(pitch), np.radians(yaw)

    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def main():
    print("=" * 80)
    print("Vive 右手控制器轨迹可视化")
    print("=" * 80)
    print("说明:")
    print("  - 红色T形: 手柄当前姿态")
    print("  - 彩色轨迹: 手柄历史位置")
    print("  - 按下右手扳机键清除轨迹")
    print("  - 按 ESC 或关闭窗口退出")
    print("=" * 80)

    try:
        vive = ViveClass()
    except Exception as e:
        print(f"错误: 无法初始化Vive - {e}")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Vive 轨迹可视化", width=1200, height=900)

    base_controller = create_t_shape()
    base_vertices = np.asarray(base_controller.vertices)
    base_triangles = np.asarray(base_controller.triangles)
    base_vertex_colors = np.asarray(base_controller.vertex_colors)

    controller_vis = o3d.geometry.TriangleMesh()
    controller_vis.vertices = o3d.utility.Vector3dVector(base_vertices.copy())
    controller_vis.triangles = o3d.utility.Vector3iVector(base_triangles.copy())
    controller_vis.vertex_colors = o3d.utility.Vector3dVector(base_vertex_colors.copy())
    vis.add_geometry(controller_vis)

    # ==================== 核心优化部分 ====================
    MAX_TRAIL_POINTS = 6000
    BG_COLOR = np.array([0.05, 0.05, 0.05])
    HIDDEN_POS = np.array([0.0, -100.0, 0.0])  # 隐藏到地下，避免深度遮挡
    FADE_TIME = 15.0  # 淡出时间(秒)

    # 预分配点云内存，只添加一次几何体
    trajectory = o3d.geometry.PointCloud()
    init_pts = np.tile(HIDDEN_POS, (MAX_TRAIL_POINTS, 1))
    init_cols = np.tile(BG_COLOR, (MAX_TRAIL_POINTS, 1))
    trajectory.points = o3d.utility.Vector3dVector(init_pts)
    trajectory.colors = o3d.utility.Vector3dVector(init_cols)
    vis.add_geometry(trajectory, reset_bounding_box=False)

    # 使用 Numpy 数组进行高速状态追踪
    trail_times = np.zeros(MAX_TRAIL_POINTS)
    trail_base_colors = np.tile(np.array([0.5, 0.5, 1.0]), (MAX_TRAIL_POINTS, 1))
    trail_index = 0
    # =======================================================

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
    vis.add_geometry(coord_frame)

    ground = o3d.geometry.TriangleMesh.create_box(width=2.0, height=0.02, depth=2.0)
    ground.translate([0, -0.01, 0])
    ground.paint_uniform_color([0.3, 0.3, 0.3])
    vis.add_geometry(ground)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.point_size = 4
    opt.mesh_show_back_face = True

    ctr = vis.get_view_control()
    ctr.set_lookat([0, 1, 0])
    ctr.set_front([0, 0.5, -1])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(1.0)

    last_trigger = 0.0
    print("\n开始可视化...\n")

    try:
        while True:
            state = vive.get_state()
            right = state['right']

            if right is not None:
                pose = right['pose']
                pos = np.array(pose[:3])
                roll, pitch, yaw = pose[3], pose[4], pose[5]
                current_time = time.time()

                # 1. 更新控制器姿态
                R = euler_to_rotation_matrix(roll, pitch, yaw)
                transform = np.eye(4)
                transform[:3, :3] = R
                transform[:3, 3] = pos
                vertices_homo = np.hstack([base_vertices, np.ones((base_vertices.shape[0], 1))])
                transformed_vertices = (transform @ vertices_homo.T).T[:, :3]
                controller_vis.vertices = o3d.utility.Vector3dVector(transformed_vertices)

                # 2. 添加新轨迹点 (环形缓冲区方式)
                pts = np.asarray(trajectory.points)
                cols = np.asarray(trajectory.colors)

                pts[trail_index] = pos
                trail_times[trail_index] = current_time
                
                # 动态渐变色 (基于时间平滑过渡)
                t = (current_time % 20.0) / 20.0 
                trail_base_colors[trail_index] = np.array([t, 0.5, 1.0 - t])
                
                trail_index = (trail_index + 1) % MAX_TRAIL_POINTS

                # 3. 检查扳机键清除轨迹 (通过将时间归零，让所有点瞬间过期)
                trigger = right['buttons']['trigger']
                if trigger > 0.5 and last_trigger <= 0.5:
                    print("清除轨迹")
                    trail_times[:] = 0
                last_trigger = trigger

                # 4. 批量计算所有点的淡出 (Numpy矩阵运算，无Python循环，极速)
                ages = current_time - trail_times
                alphas = np.clip(1.0 - (ages / FADE_TIME), 0.0, 1.0)

                # 计算混合后的颜色
                faded_colors = trail_base_colors * alphas[:, np.newaxis] + BG_COLOR * (1 - alphas[:, np.newaxis])
                
                # 找出已经完全消失的点 (alpha <= 0)
                hidden_mask = alphas <= 0.0

                # 更新颜色
                cols[~hidden_mask] = faded_colors[~hidden_mask]
                cols[hidden_mask] = BG_COLOR
                
                # 将消失的点踢到地下，避免产生黑色深度阴影遮挡新轨迹！
                pts[hidden_mask] = HIDDEN_POS

                # 强制更新数据到 GPU (必须重新赋值，否则 Open3D 不一定刷新显存)
                trajectory.points = o3d.utility.Vector3dVector(pts)
                trajectory.colors = o3d.utility.Vector3dVector(cols)

                # 统计可见点数
                visible_count = np.sum(~hidden_mask)
                print(f"\r位置: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  "
                      f"姿态: ({roll:.1f}°, {pitch:.1f}°, {yaw:.1f}°)  "
                      f"轨迹点: {visible_count}", end='')

            # 只 update_geometry，杜绝 remove/add
            vis.update_geometry(controller_vis)
            vis.update_geometry(trajectory)

            if not vis.poll_events():
                break
            vis.update_renderer()

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        vive.shutdown()
        vis.destroy_window()
        print("程序退出")


if __name__ == "__main__":
    main()
