"""
计算多帧图像每个像素的样本方差，并分析边缘区域与非边缘区域的噪声差异。

步骤：
1. 采集 N 帧图像（左目）。
2. 分批计算每个像素的均值。
3. 分批计算每个像素的方差（样本方差，除以 N-1）。
4. 在平均图像上检测边缘，生成边缘掩码。
5. 统计边缘区域和非边缘区域的方差均值、方差等。
6. 可视化平均图像、边缘掩码、方差热力图。
7. 进行 t 检验，判断差异显著性。
"""
import sys
sys.path.insert(0, './')
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from hardware.camera.binocam import BinocularCam
from hardware.camera.d435 import CameraD435
def compute_pixel_variance_gpu():
    # 参数设置
    num_frames = 1000          # 总帧数
    batch_size = 200           # 每批处理的帧数（根据 GPU 显存调整）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ------------------- 采集帧 -------------------
    print("初始化相机...")
    cam = BinocularCam(map_dir="./data/output")
    # import pyrealsense2 as rs
    # cam = CameraD435()
    # cam.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
    # cam.start()

    frames_cpu = []             # 在 CPU 内存中保留所有帧 (归一化 float32)
    print("采集图像...")
    for i in range(num_frames):
        frame = cam.get_frame()
        # 转换为 RGB、归一化到 [0,1]
        img = cv2.cvtColor(frame["left"], cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        frames_cpu.append(img)
        if i % 10 == 0:
            print(f"frame {i}")

    frames_cpu = np.stack(frames_cpu, axis=0)   # (N, H, W, 3)
    N, H, W, _ = frames_cpu.shape
    print(f"采集完成，总帧数: {N}, 分辨率: {H}x{W}")

    # ------------------- 第1遍：计算均值图像 -------------------
    print("计算均值图像...")
    mean_img = torch.zeros((H, W, 3), dtype=torch.float32, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_np = frames_cpu[start:end]                     # (b, H, W, 3)
        batch_t = torch.from_numpy(batch_np).to(device)      # 移至 GPU
        mean_img += batch_t.sum(dim=0)                        # 按 batch 求和

    mean_img /= N                                             # 全局平均
    print("均值图像计算完成")

    # ------------------- 第2遍：计算方差图像 -------------------
    print("计算方差图像（每个像素每个通道的样本方差）...")
    # 使用二阶矩方法：Var = E[x^2] - (E[x])^2，最后乘以 N/(N-1) 得样本方差
    mean_sq = torch.zeros((H, W, 3), dtype=torch.float32, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_np = frames_cpu[start:end]
        batch_t = torch.from_numpy(batch_np).to(device)      # (b, H, W, 3)
        mean_sq += (batch_t ** 2).sum(dim=0)                  # 累加平方和

    mean_sq /= N                                               # E[x^2]
    var_img = mean_sq - mean_img ** 2                          # 方差（除以 N）
    # 转换为样本方差（除以 N-1）
    var_img = var_img * N / (N - 1)

    print("方差图像计算完成")

    # ------------------- 边缘检测（在均值图像上） -------------------
    print("检测边缘...")
    # 转换为灰度图
    gray = mean_img.mean(dim=-1) * 255.0                       # (H, W), 值范围 [0,255]
    gray = gray.to(torch.float32).unsqueeze(0).unsqueeze(0)    # (1,1,H,W)

    # 定义 Sobel 核
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=device)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=device)
    sobel_x = sobel_x.view(1,1,3,3)
    sobel_y = sobel_y.view(1,1,3,3)

    grad_x = F.conv2d(gray, sobel_x, padding=1)                # (1,1,H,W)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze() # (H,W)

    # 归一化梯度幅值到 [0,1]，并设定阈值
    grad_norm = grad_mag / grad_mag.max()
    edge_threshold = 0.2
    edge_mask = grad_norm > edge_threshold                     # (H,W) 布尔张量
    non_edge_mask = ~edge_mask

    edge_pixels = edge_mask.sum().item()
    non_edge_pixels = non_edge_mask.sum().item()
    print(f"边缘像素: {edge_pixels}, 非边缘像素: {non_edge_pixels}")

    # ------------------- 统计边缘与非边缘的方差 -------------------
    # 将方差图像转换为单通道（取三个通道的平均值，作为总体噪声度量）
    var_gray = var_img.mean(dim=-1)                              # (H,W)

    edge_var = var_gray[edge_mask]                               # 一维张量
    non_edge_var = var_gray[non_edge_mask]

    edge_mean = edge_var.mean().item()
    edge_var_value = edge_var.var().item()
    non_edge_mean = non_edge_var.mean().item()
    non_edge_var_value = non_edge_var.var().item()

    print("\n===== 结果 =====")
    print(f"边缘区域方差均值   : {edge_mean:.6f}")
    print(f"边缘区域方差方差   : {edge_var_value:.6f}")
    print(f"非边缘区域方差均值 : {non_edge_mean:.6f}")
    print(f"非边缘区域方差方差 : {non_edge_var_value:.6f}")

    # ------------------- 显著性检验（传回 CPU） -------------------
    t, p = ttest_ind(edge_var.cpu().numpy(), non_edge_var.cpu().numpy(), equal_var=False)
    print(f"\nT-test: t = {t:.4f}, p = {p:.4e}")
    if p < 0.05:
        print("结论：边缘区域噪声方差显著不同")
    else:
        print("结论：统计上没有显著差异")

    # ------------------- 可视化（传回 CPU） -------------------
    mean_img_np = mean_img.cpu().numpy()
    edge_mask_np = edge_mask.cpu().numpy()
    var_gray_np = var_gray.cpu().numpy()                        # 方差热力图

    plt.figure(figsize=(15,5))
    plt.subplot(131)
    plt.title("Mean Image")
    plt.imshow(mean_img_np)
    plt.axis('off')

    plt.subplot(132)
    plt.title("Edge Mask")
    plt.imshow(edge_mask_np, cmap='gray')
    plt.axis('off')

    plt.subplot(133)
    plt.title("Variance Map (average over channels)")
    im = plt.imshow(var_gray_np, cmap='jet')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 可选：绘制边缘与非边缘的方差分布直方图
    plt.figure(figsize=(8,5))
    plt.hist(edge_var.cpu().numpy(), bins=100, alpha=0.6, label='Edge')
    plt.hist(non_edge_var.cpu().numpy(), bins=100, alpha=0.6, label='Non-edge')
    plt.xlabel('Variance')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of pixel variance')
    plt.show()


if __name__ == "__main__":
    compute_pixel_variance_gpu()