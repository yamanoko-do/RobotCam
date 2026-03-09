"""
一个基于Open3D和Matplotlib的相机噪声分析和可视化工具。
功能包括：
1. 从BinocularCam采集多帧图像数据。
2. 计算每帧图像与平均图像的像素级差异，并统计不同颜色通道和亮度区间的噪声均值和方差。
3. 绘制噪声统计的柱状图和分布直方图。
4. 使用Open3D动态可视化每帧图像的RGB差异距离，颜色表示差异大小。
注意：此代码需要安装Open3D、Matplotlib、OpenCV和SciPy等库，并且需要正确配置BinocularCam的环境。 
"""
import sys
sys.path.insert(0, './')
import os
import time
import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from hardware.camera.binocam import BinocularCam

# 设置环境变量
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

def capture_and_visualize():
    # --- 1. 参数设置 ---
    num_frames = 1000
    scale = 1000.0
    playback_speed = 0.03
    downsample_rate = 4

    # --- 2. 数据采集 ---
    print(f"正在初始化相机并采集 {num_frames} 帧图像...")
    cam = BinocularCam(map_dir="./data/output")
    
    frame_list = []
    for i in range(num_frames):
        frame = cam.get_frame()
        img_bgr = frame["left"]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        frame_list.append(img_float)
        if i % 10 == 0:
            print(f"已采集 {i} 帧...")
    
    frames = np.array(frame_list)   # (N, H, W, 3)
    H, W, C = frames[0].shape
    print(f"单图分辨率: {W}x{H}")

    # --- 3. 计算平均图像 ---
    print("正在计算平均值...")
    mean_img = np.mean(frames, axis=0)      # (H, W, 3)

    # --- 4. 计算像素级差异（此处会占用大量内存，可考虑优化）---
    print("正在计算每帧与平均图像的差异...")
    diff_frames = frames - mean_img          # (N, H, W, 3)
    dist_frames = np.sqrt(np.sum(diff_frames**2, axis=-1))  # (N, H, W)
    mean_dist = np.mean(dist_frames, axis=0)   # (H, W)
    var_dist = np.var(dist_frames, axis=0)     # (H, W)
    # --- 5. 打印不同颜色通道的均值和方差 ---
    print("\n=== 各颜色通道噪声统计（基于RGB差异）===")
    channel_names = ['R', 'G', 'B']
    channel_means = []
    channel_vars = []
    for c in range(3):
        channel_data = diff_frames[:, :, :, c]   # (N, H, W)
        mean_c = np.mean(channel_data)
        var_c  = np.var(channel_data)
        channel_means.append(mean_c)
        channel_vars.append(var_c)
        print(f"{channel_names[c]} 通道: 均值 = {mean_c:.6f}, 方差 = {var_c:.6f}")

    # --- 6. 不同亮度下的噪声统计（基于dist_frames）---
    print("\n=== 不同亮度区间噪声统计 ===")
    # 将平均图像转为灰度
    gray_mean = 0.299 * mean_img[:,:,0] + 0.587 * mean_img[:,:,1] + 0.114 * mean_img[:,:,2]  # (H, W)
    
    # 定义亮度区间（0~1 分为20个bin）
    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # 初始化存储每个bin的噪声统计量
    bin_mean_dist = np.zeros(n_bins)
    bin_var_dist  = np.zeros(n_bins)
    bin_counts    = np.zeros(n_bins)
    
    # 注意：dist_frames 是 (N, H, W)，需要遍历每一帧，或使用高级索引
    # 由于内存可能较大，这里采用逐帧累加方式以避免存储所有dist_frames
    # 但我们已经有了dist_frames，可以直接利用
    for i in range(num_frames):
        dist_frame = dist_frames[i]      # (H, W)
        # 根据灰度值将像素分配到bin
        # 将灰度图展平，并获取每个像素对应的bin索引
        gray_flat = gray_mean.ravel()
        dist_flat = dist_frame.ravel()
        # 使用digitize获取bin索引（注意索引从1开始）
        bin_indices = np.digitize(gray_flat, bin_edges) - 1
        # 处理边界情况（灰度值等于1时，digitize会得到n_bins+1）
        bin_indices = np.clip(bin_indices, 0, n_bins-1)
        
        # 按bin累加和与平方和（用于计算均值和方差）
        # 更高效的方法是用np.bincount，但这里为了清晰，用循环累加
        for bin_idx in range(n_bins):
            mask = (bin_indices == bin_idx)
            count = np.sum(mask)
            if count > 0:
                bin_counts[bin_idx] += count
                bin_mean_dist[bin_idx] += np.sum(dist_flat[mask])
                bin_var_dist[bin_idx]  += np.sum(dist_flat[mask]**2)
    
    # 计算最终均值和方差
    for bin_idx in range(n_bins):
        if bin_counts[bin_idx] > 0:
            mean_val = bin_mean_dist[bin_idx] / bin_counts[bin_idx]
            var_val  = bin_var_dist[bin_idx] / bin_counts[bin_idx] - mean_val**2
            bin_mean_dist[bin_idx] = mean_val
            bin_var_dist[bin_idx]  = var_val
        else:
            bin_mean_dist[bin_idx] = np.nan
            bin_var_dist[bin_idx]  = np.nan
        print(f"亮度区间 [{bin_edges[bin_idx]:.2f}, {bin_edges[bin_idx+1]:.2f}]: "
              f"噪声均值 = {bin_mean_dist[bin_idx]:.6f}, 方差 = {bin_var_dist[bin_idx]:.6f}")

    # --- 7. 绘制柱状图和高斯曲线 ---
    plt.ion()   # 开启交互模式

    fig1, ax1 = plt.subplots()
    im1 = ax1.imshow(mean_dist, cmap='jet')
    ax1.set_title('Mean of Pixel-wise Noise (RGB difference magnitude)')
    plt.colorbar(im1, ax=ax1, label='Mean difference')
    fig1.canvas.manager.set_window_title('Noise Mean')

    fig2, ax2 = plt.subplots()
    im2 = ax2.imshow(var_dist, cmap='jet')
    ax2.set_title('Variance of Pixel-wise Noise (RGB difference magnitude)')
    plt.colorbar(im2, ax=ax2, label='Variance')
    fig2.canvas.manager.set_window_title('Noise Variance')
    fig_list = []

    # 7.1 各通道均值和方差的柱状图
    fig_channel, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(channel_names, channel_means, color=['red', 'green', 'blue'])
    ax1.set_title('Channel-wise Noise Mean')
    ax1.set_ylabel('Mean value')
    
    ax2.bar(channel_names, channel_vars, color=['red', 'green', 'blue'])
    ax2.set_title('Channel-wise Noise Variance')
    ax2.set_ylabel('Variance')
    fig_channel.canvas.manager.set_window_title('Channel Noise Stats')
    fig_list.append(fig_channel)

    # 7.2 亮度区间噪声均值和方差的柱状图
    fig_bright, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4))
    ax3.bar(bin_centers, bin_mean_dist, width=1/n_bins, edgecolor='k', alpha=0.7)
    ax3.set_xlabel('Brightness')
    ax3.set_ylabel('Noise Mean')
    ax3.set_title('Noise Mean vs Brightness')
    
    ax4.bar(bin_centers, bin_var_dist, width=1/n_bins, edgecolor='k', alpha=0.7)
    ax4.set_xlabel('Brightness')
    ax4.set_ylabel('Noise Variance')
    ax4.set_title('Noise Variance vs Brightness')
    fig_bright.canvas.manager.set_window_title('Brightness-based Noise')
    fig_list.append(fig_bright)

    # 7.3 各通道噪声分布直方图 + 高斯拟合
    fig_hist, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors_hist = ['red', 'green', 'blue']          # 正确的颜色名称
    for c in range(3):
        ax = axes[c]
        data = diff_frames[:, :, :, c].ravel()
        # 绘制直方图
        n, bins, patches = ax.hist(data, bins=100, density=True, alpha=0.6, color=colors_hist[c])
        
        # 拟合高斯曲线（需要 scipy.stats.norm）
        from scipy.stats import norm
        mu, std = norm.fit(data)
        x = np.linspace(data.min(), data.max(), 200)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2, label=f'Gaussian fit ($\mu={mu:.4f}$, $\sigma={std:.4f}$)')
        ax.set_title(f'{channel_names[c]} channel noise distribution')
        ax.set_xlabel('Noise value')
        ax.set_ylabel('Probability density')
        ax.legend()
        fig_hist.canvas.manager.set_window_title('Channel Noise Histograms')
        fig_list.append(fig_hist)

    # 让所有图窗显示
    plt.pause(0.1)

    # --- 8. 准备 Open3D 可视化索引 ---
    rows = np.arange(0, H, downsample_rate)
    cols = np.arange(0, W, downsample_rate)
    xv, yv = np.meshgrid(cols, rows)
    
    x_coords_flat = xv.flatten()
    y_coords_flat = yv.flatten()
    num_points = x_coords_flat.size

    pcd = o3d.geometry.PointCloud()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="RGB Z-Distance Visualization", width=1280, height=720)

    def update_pcd_by_frame(index):
        current_frame = frames[index]
        diff = current_frame - mean_img
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        
        sample_colors = current_frame[rows[:, None], cols, :] 
        sample_dists = dist[rows[:, None], cols]             
        
        xyz = np.zeros((num_points, 3))
        xyz[:, 0] = x_coords_flat
        xyz[:, 1] = -y_coords_flat
        xyz[:, 2] = sample_dists.flatten() * scale 
        
        return xyz, sample_colors.reshape(-1, 3)

    # 载入初始帧
    initial_xyz, initial_rgb = update_pcd_by_frame(0)
    pcd.points = o3d.utility.Vector3dVector(initial_xyz)
    pcd.colors = o3d.utility.Vector3dVector(initial_rgb)
    
    vis.add_geometry(pcd)
    
    print("开始自动播放...")
    # --- 7. 循环动态更新 ---
    frame_idx = 0
    try:
        while True:
            xyz, rgb = update_pcd_by_frame(frame_idx)
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            
            vis.update_geometry(pcd)
            if not vis.poll_events():
                break
            vis.update_renderer()
            
            frame_idx = (frame_idx + 1) % num_frames
            time.sleep(playback_speed)

            # 允许 matplotlib 窗口处理事件，保持响应
            plt.pause(0.01)
            
    except KeyboardInterrupt:
        print("用户终止")

    vis.destroy_window()
    plt.close('all')  # 关闭所有matplotlib窗口

if __name__ == "__main__":
    capture_and_visualize()