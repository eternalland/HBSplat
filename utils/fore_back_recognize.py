import numpy as np
import torch
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_utils

def get_foreground_mask_improved(depth_maps):
    b, _, h, w = depth_maps.shape
    foreground_masks = torch.zeros_like(depth_maps)

    for i in range(b):
        depth_map = depth_maps[i, 0].cpu().numpy()

        # 使用高斯模糊平滑深度图
        blurred = cv2.GaussianBlur(depth_map, (5, 5), 0)

        # 计算直方图
        hist, bins = np.histogram(blurred.flatten(), bins=256)

        # 寻找两个主要峰值
        peaks = np.argsort(hist)[-2:]  # 两个最高峰
        valley = np.argmin(hist[min(peaks):max(peaks)]) + min(peaks)
        threshold = bins[valley]

        # 生成掩码并应用形态学操作
        foreground_mask = depth_map > threshold
        foreground_mask = cv2.morphologyEx(foreground_mask.astype(np.uint8),
                                           cv2.MORPH_CLOSE,
                                           np.ones((5, 5), np.uint8))

        foreground_masks[i, 0] = torch.from_numpy(foreground_mask.astype(np.float32))

    return foreground_masks


import cv2


def get_foreground_mask_otsu(depth_maps):
    b, _, h, w = depth_maps.shape
    foreground_masks = torch.zeros_like(depth_maps)

    for i in range(b):
        depth_map = depth_maps[i, 0].cpu().numpy()

        # 归一化到0-255
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = depth_normalized.astype(np.uint8)

        # 使用Otsu阈值法
        _, thresholded = cv2.threshold(depth_normalized, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        foreground_mask = thresholded > 0
        foreground_masks[i, 0] = torch.from_numpy(foreground_mask.astype(np.float32))

    return foreground_masks


# image = cv2.imread("/home/mayu/thesis/SCGaussian/output/fortress000_1/mono_depth_map/IMG_1801.jpg")
# depth_maps = plot_utils.image_numpy_torch(image).unsqueeze(0)
# # 假设 depth_maps 是你的输入张量
# foreground_mask = get_foreground_mask_otsu(depth_maps)
#
# # 可视化结果
# plt.figure(figsize=(12,4))
# plt.subplot(131)
# plt.imshow(depth_maps[0,0].cpu().numpy(), cmap='gray')
# plt.title('Depth Map')
# plt.subplot(132)
# plt.hist(depth_maps[0,0].cpu().numpy().flatten(), bins=256)
# plt.title('Depth Histogram')
# plt.subplot(133)
# plt.imshow(foreground_mask[0,0].cpu().numpy(), cmap='gray')
# plt.title('Foreground Mask')
# plt.show()

def get_foreground_mask_otsu2(depth_map):
    depth_map = depth_map[0].cpu().numpy()
    # 归一化到0-255
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype(np.uint8)

    # 使用Otsu阈值法
    _, thresholded = cv2.threshold(depth_normalized, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    foreground_mask = thresholded > 0
    foreground_mask = torch.from_numpy(foreground_mask.astype(np.float32))

    return foreground_mask


def get_foreground_mask(depth_map):
    """
    使用Otsu阈值法获取前景掩码
    参数:
        depth_map: 输入深度图 tensor(1, h, w), 值范围[0., 1.]
    返回:
        foreground_mask: 前景掩码 tensor(1, h, w), 前景为1，背景为0
    """
    # 转换为numpy数组并缩放到0-255
    depth_np = (depth_map[0].cpu().numpy() * 255).astype(np.uint8)

    # 使用Otsu阈值法
    _, mask = cv2.threshold(depth_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 转换为tensor并归一化到0-1
    foreground_mask = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)

    return foreground_mask


def get_foreground_mask_hist(depth_map):
    """
    基于直方图双峰性的前景分割
    """
    depth_np = depth_map[0].cpu().numpy()
    hist, bins = np.histogram(depth_np.flatten(), bins=256, range=(0, 1))

    # 寻找直方图谷底作为阈值
    # 简单实现：取直方图最小值的位置
    threshold = bins[np.argmin(hist)]

    foreground_mask = (depth_map > threshold).float()
    return foreground_mask

import numpy as np

def get_sky_mask_simple(depth_map, threshold=0.95):
    """
    通过固定阈值提取天空掩码
    参数:
        depth_map: 归一化深度图 (h, w), 值范围[0, 1]
        threshold: 天空深度阈值（默认0.95，需根据数据调整）
    返回:
        sky_mask: 天空区域为True, 其他为False
    """
    return depth_map > threshold

def get_sky_mask_with_gradient2(depth_map, depth_thresh=0.9, gradient_thresh=0.01):
    """
    结合深度值和梯度信息（天空区域通常梯度平缓）
    参数:
        depth_map: 归一化深度图
        depth_thresh: 深度阈值
        gradient_thresh: 梯度阈值（天空区域梯度应小于此值）
    返回:
        sky_mask: 综合掩码
    """
    # 计算梯度
    sobel_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # 联合条件
    return (depth_map > depth_thresh) & (gradient < gradient_thresh)

# # image = cv2.imread("/home/mayu/thesis/SCGaussian/output/fortress000_1/mono_depth_map/IMG_1801.jpg")
# image = cv2.imread("/home/mayu/thesis/SCGaussian/output/leaves000_3/mono_depth_map/IMG_2998_color.jpg")
# depth_map = plot_utils.image_numpy_torch(image)[0].unsqueeze(0)
# # 假设 depth_maps 是你的输入张量
# foreground_mask = get_foreground_mask(depth_map)
#
# plt.figure(figsize=(12,4))
# plt.subplot(131)
# plt.imshow(depth_map[0].cpu().numpy(), cmap='gray')
# plt.title('Depth Map')
# plt.subplot(132)
# plt.hist(depth_map[0].cpu().numpy().flatten(), bins=50)
# plt.title('Depth Histogram')
# plt.subplot(133)
# plt.imshow(foreground_mask[0].cpu().numpy(), cmap='gray')
# plt.title('Foreground Mask')
# plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def read_depth_map(path, normalize=True):
    """
    读取深度图并归一化（支持.png/.npy/.exr格式）
    参数:
        path: 文件路径
        normalize: 是否归一化到[0,1]
    返回:
        depth_map: numpy数组 (h,w)
    """
    path = Path(path)
    if path.suffix == '.npy':
        depth_map = np.load(path)
    elif path.suffix == '.png' or path.suffix == '.jpg':
        depth_map = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth_map.ndim == 3:
            depth_map = depth_map[:, :, 0]  # 取单通道
    elif path.suffix == '.exr':
        depth_map = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    else:
        raise ValueError("Unsupported file format")

    if normalize:
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
    return depth_map


def get_sky_mask_with_gradient(depth_map, depth_thresh=0.9, gradient_thresh=0.01):
    """ 梯度法天空分割（完整实现） """
    # 计算梯度
    sobel_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # 联合条件
    sky_mask = (depth_map > depth_thresh)

    # 形态学后处理
    # kernel = np.ones((5, 5), np.uint8)
    # sky_mask = cv2.morphologyEx(sky_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return sky_mask.astype(bool)


def visualize_results(depth_map, sky_mask):
    """ 可视化深度图、直方图和掩码 """
    plt.figure(figsize=(15, 5))

    # 原始深度图
    plt.subplot(131)
    plt.imshow(depth_map, cmap='viridis')
    plt.colorbar(label='Depth Value')
    plt.title('Original Depth Map')

    # 深度直方图
    plt.subplot(132)
    plt.hist(depth_map.flatten(), bins=50, color='teal')
    plt.axvline(x=np.percentile(depth_map, 95), color='red', linestyle='--',
                label='95th percentile')
    plt.xlabel('Depth Value')
    plt.ylabel('Pixel Count')
    plt.title('Depth Distribution')
    plt.legend()

    # 天空掩码
    plt.subplot(133)
    plt.imshow(sky_mask, cmap='gray')
    plt.title(f'Sky Mask (Area={np.mean(sky_mask) * 100:.1f}%)')

    plt.tight_layout()
    plt.show()

def get_sky_mask_adaptive(depth_map, top_percent=5):
    """
    基于深度值分布的百分比阈值
    参数:
        depth_map: 归一化深度图 (h, w)
        top_percent: 视为天空的深度值最高百分比（默认5%）
    返回:
        sky_mask: 天空掩码
    """
    threshold = np.percentile(depth_map, 100 - top_percent)
    return depth_map > threshold



import numpy as np
from sklearn.mixture import GaussianMixture

def visualize_sky_detection(depth_map, sky_mask):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(depth_map, cmap='viridis')
    plt.title('Original Depth')

    plt.subplot(132)
    plt.hist(depth_map.flatten(), bins=50, color='blue')
    plt.axvline(x=np.percentile(depth_map, 99), color='red', linestyle='--')
    plt.title('Depth Histogram')

    plt.subplot(133)
    plt.imshow(sky_mask, cmap='gray')
    plt.title(f'Sky Mask ({np.mean(sky_mask) * 100:.2f}%)')

    plt.tight_layout()
    plt.show()



def refine_mask_with_morphology(raw_mask, min_ratio):
    """ 形态学优化掩码 """
    # 去除小连通区域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask.astype(np.uint8))

    if num_labels <= 1:
        return raw_mask

    # 保留面积前N%的区域
    area_threshold = np.prod(raw_mask.shape) * min_ratio
    kept_labels = [i for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] >= area_threshold]

    refined_mask = np.isin(labels, kept_labels)

    # 闭运算填充空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(refined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)




def generate_foreground_mask2(args, image_name, mono_depth_map):
    # background_mask = get_sky_mask_adaptive(mono_depth_map, top_percent=args.top_percent)
    threshold = np.percentile(mono_depth_map, args.top_percent)
    sky_mask = np.zeros_like(mono_depth_map, dtype=np.uint8).astype(bool)
    if threshold < 0.4:
        sky_mask = (mono_depth_map >= threshold)
        sky_mask = morphology.remove_small_holes(sky_mask, area_threshold=64)
        sky_mask = morphology.remove_small_objects(sky_mask, min_size=64)
    foreground_mask = (sky_mask == False)
    image_path = os.path.join(args.mono_depth_map_dir, f'{image_name}_foreground_mask.png')
    cv2.imwrite(image_path, foreground_mask.astype(np.uint8) * 255)
    return foreground_mask

def normalize_depth(depth_map):
    """ 线性归一化到0-1范围 """
    depth_min = np.min(depth_map)
    depth_max = np.max(depth_map)
    return (depth_map - depth_min) / (depth_max - depth_min + 1e-8)  # 避免除零

def fore_back(args, viewpoint_stack):
    for i, cam in enumerate(viewpoint_stack):
        foreground_mask = get_foreground_mask(cam.mono_depth_map.unsqueeze(0))
        cam.foreground_mask = (foreground_mask.int() ^ 1).float().cuda()
        foreground_mask = foreground_mask.repeat(3, 1, 1)
        image = plot_utils.image_torch_numpy(foreground_mask)
        image_path = os.path.join(args.mono_depth_map_dir, f'{i}_foreground_mask.png')
        cv2.imwrite(image_path, image)




def compute_sky_mask_cdf(depth_map, pixel_ratio=0.1, depth_ratio=0.5):
    """
    基于CDF判断天空：前 pixel_ratio 的像素是否占据 depth_ratio 的深度范围
    :param depth_map: 归一化深度图 (h, w), 范围 [0, 1]
    :param pixel_ratio: 前多少比例的像素（默认前10%）
    :param depth_ratio: 这些像素占据的深度范围（默认50%）
    :return: sky_mask (h, w), 天空=1, 非天空=0
    """
    # 1. 深度值排序并计算CDF
    sorted_depths = np.sort(depth_map.flatten())
    cdf = np.linspace(0, 1, len(sorted_depths))  # 累积分布

    # 2. 计算前 pixel_ratio 像素占据的深度范围
    n_pixels = len(sorted_depths)
    idx = int(pixel_ratio * n_pixels)
    depth_span = sorted_depths[idx] - sorted_depths[0]  # 前 pixel_ratio 像素的深度跨度

    # 3. 判断是否满足条件
    if depth_span >= depth_ratio:
        # 提取深度值较大的前 pixel_ratio 像素作为天空
        p = sorted_depths[-idx]  # 后 pixel_ratio 像素的起始深度
        sky_mask = (depth_map >= p).astype(np.uint8)
    else:
        sky_mask = np.zeros_like(depth_map, dtype=np.uint8)

    return sky_mask, depth_span

def visualize_sky_detection2(depth_map, sky_mask):
    # 测试
    depth_map = np.random.rand(256, 256)  # 随机深度图
    depth_map[50:150, 50:150] = 0.95  # 模拟天空（高深度）

    sky_mask, depth_span = compute_sky_mask_cdf(depth_map, pixel_ratio=0.1, depth_ratio=0.5)
    print(f"前10%像素占据的深度范围: {depth_span:.3f}")

    # 可视化
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.title("Depth Map"), plt.imshow(depth_map, cmap='gray')
    plt.subplot(132), plt.title("Sky Mask"), plt.imshow(sky_mask, cmap='gray')

    # 绘制CDF
    plt.subplot(133), plt.title("CDF")
    sorted_depths = np.sort(depth_map.flatten())
    cdf = np.linspace(0, 1, len(sorted_depths))
    plt.plot(sorted_depths, cdf, label="CDF")
    plt.axvline(sorted_depths[int(0.1 * len(sorted_depths))], color='r', linestyle='--', label="Top 10%")
    plt.legend()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def generate_sky_mask(depth_map, percentile=95, depth_threshold=0.5):
    """
    根据深度图的分布特征生成天空掩码

    参数:
        depth_map: 归一化的深度图(h,w)，值范围[0,1]
        percentile: 用于确定深度阈值的百分位(默认95，即取前5%的深度值)
        depth_threshold: 辅助深度阈值，用于验证候选天空区域

    返回:
        sky_mask: 二值掩码，天空区域为True/1
    """
    # 将深度图展平为一维数组
    depths = depth_map.flatten()

    # 计算指定百分位的深度阈值
    percentile_threshold = np.percentile(depths, percentile)

    # 初始掩码：深度大于百分位阈值的区域
    candidate_mask = depth_map > percentile_threshold

    # 结合绝对深度阈值验证天空区域
    # 天空应该同时满足高百分位和绝对深度较大两个条件
    sky_mask = (depth_map > percentile_threshold) & (depth_map > depth_threshold)

    # 可选：形态学操作去除小噪点
    from skimage import morphology
    sky_mask = morphology.remove_small_holes(sky_mask, area_threshold=64)
    sky_mask = morphology.remove_small_objects(sky_mask, min_size=64)

    return sky_mask


# 示例使用
# 假设depth_map是您的归一化深度图
# sky_mask = generate_sky_mask(depth_map)

# 可视化
def visualize_results2(depth_map, sky_mask):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(depth_map, cmap='gray')
    plt.title('Depth Map')

    plt.subplot(1, 3, 2)
    plt.hist(depth_map.flatten(), bins=50)
    plt.title('Depth Histogram')

    plt.subplot(1, 3, 3)
    plt.imshow(sky_mask, cmap='gray')
    plt.title('Sky Mask')

    plt.tight_layout()
    plt.show()



from skimage.filters import threshold_otsu
from skimage import morphology
# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 使用示例
    # depth_map = np.load('depth.npy')  # 加载归一化深度图

    # 1. 读取深度图（替换为你的文件路径）image = cv2.imread("/home/mayu/thesis/SCGaussian/output/fortress000_1/mono_depth_map/IMG_1801.jpg")
    # depth_path = "/home/mayu/thesis/SCGaussian/output/fern000_1/mono_depth_map/IMG_4027.jpg"  # 支持.png/.npy/.exr
    # depth_path = "/home/mayu/thesis/SCGaussian/output/fortress000_1/mono_depth_map/IMG_1801.jpg"  # 支持.png/.npy/.exr
    depth_path = "/home/mayu/thesis/SCGaussian/output/leaves000_3/mono_depth_map/IMG_2998.jpg"  # 支持.png/.npy/.exr
    depth_map = read_depth_map(depth_path)

    # 2. 获取天空掩码
    # sky_mask = get_sky_mask_with_gradient(
    #     depth_map,
    #     depth_thresh=0.5,  # 深度阈值（可调）
    #     gradient_thresh=0.2  # 梯度阈值（可调）
    # )
    # sky_mask = get_sky_mask_simple(depth_map, 0.2)
    # sky_mask = get_sky_mask_adaptive(depth_map, 2)

    threshold = np.percentile(depth_map, 99.5)
    if threshold < 0.4:
        sky_mask = (depth_map >= threshold)
        sky_mask = morphology.remove_small_holes(sky_mask, area_threshold=64)
        sky_mask = morphology.remove_small_objects(sky_mask, min_size=64)
    else:
        sky_mask = np.zeros_like(depth_map, dtype=np.uint8).astype(bool)

    # # 生成掩码
    # test_mask = generate_sky_mask(depth_map)

    # 可视化
    visualize_results2(depth_map, sky_mask)

    # 3. 可视化
    # visualize_results(depth_map, sky_mask)