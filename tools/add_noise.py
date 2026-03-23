import sys
sys.path.insert(0, './')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def add_region_based_noise(image, edge_mean=0, edge_var=0.002,
                           nonedge_mean=0, nonedge_var=0.0005,
                           edge_threshold=0.2):
    """
    对图像添加区域相关的高斯噪声。
    边缘区域噪声 ~ N(edge_mean, edge_var)
    非边缘区域噪声 ~ N(nonedge_mean, nonedge_var)

    参数:
        image: uint8 格式 [0,255] 的彩色图像 (H,W,3)
        edge_mean, edge_var: 边缘区域的噪声均值和方差
        nonedge_mean, nonedge_var: 非边缘区域的噪声均值和方差
        edge_threshold: 边缘检测阈值 (归一化梯度幅值)

    返回:
        noisy_image: uint8 格式的加噪图像
    """
    # 转换为 float 范围 [0,1] 便于处理
    img_float = image.astype(np.float32) / 255.0
    h, w, c = img_float.shape

    # ---------- 边缘检测 ----------
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # Sobel 梯度
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    # 归一化梯度幅值到 [0,1]
    grad_norm = grad_mag / (grad_mag.max() + 1e-8)
    # 生成边缘掩膜
    edge_mask = grad_norm > edge_threshold   # 布尔型 (H,W)

    # ---------- 生成噪声 ----------
    noise = np.zeros_like(img_float)         # 三通道噪声
    # 边缘区域噪声
    edge_noise = np.random.normal(loc=edge_mean, scale=np.sqrt(edge_var),
                                  size=(h, w, c))
    # 非边缘区域噪声
    nonedge_noise = np.random.normal(loc=nonedge_mean, scale=np.sqrt(nonedge_var),
                                     size=(h, w, c))

    # 根据掩膜填充噪声 (将边缘区域的噪声赋值给对应像素)
    # 注意：掩膜是二维的，需要扩展到三个通道
    edge_mask_3c = np.stack([edge_mask]*c, axis=-1)
    noise = np.where(edge_mask_3c, edge_noise, nonedge_noise)

    # ---------- 叠加噪声并裁剪 ----------
    noisy_float = img_float + noise
    noisy_float = np.clip(noisy_float, 0.0, 1.0)
    noisy_image = (noisy_float * 255).astype(np.uint8)

    return noisy_image

def main():
    # 图像路径
    img_path = os.path.expanduser("~/docker_share/dataset/test_image/left_1_rectified.jpg")
    if not os.path.exists(img_path):
        print(f"图像文件不存在: {img_path}")
        return

    # 读取图像（OpenCV 默认 BGR，转换为 RGB）
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print("图像读取失败")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 加噪声
    noisy_image = add_region_based_noise(image_rgb)

    # 可视化对比
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(image_rgb)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Noisy Image (region-based noise)")
    plt.imshow(noisy_image)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()