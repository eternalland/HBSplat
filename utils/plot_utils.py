import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import torch
from torchvision import transforms


# percentile_range=(1, 98) # 可选：使用 1% 分位点作为下限，确保负值也能显示
percentile_range=(0, 100) # 可选：使用 1% 分位点作为下限，确保负值也能显示

def draw_histogram(output_dir, data, title, num_bins=7, remove_outliers=False):
    """绘制极线距离的折线直方图"""
    plt.figure(figsize=(8, 6))

    range_bound = np.percentile(data, percentile_range)

    if remove_outliers:
        counts, bins, _ = plt.hist(data, bins=num_bins, color='skyblue', edgecolor='black', alpha=0.7, range=range_bound)
    else:
        counts, bins, _ = plt.hist(data, bins=7, color='skyblue', edgecolor='black', alpha=0.7)

    # 在柱子上标注点个数
    for i, count in enumerate(counts):
        if count > 0:
            plt.text(bins[i] + (bins[i + 1] - bins[i]) / 2, count, f'{int(count)}',
                     ha='center', va='bottom', fontsize=10)

    # 绘制折线连接柱子顶部
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, counts, 'r-', marker='o', linewidth=2, markersize=8)

    plt.xlabel(title)
    plt.ylabel('Number of Points')
    plt.title(f'Histogram of {title}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{title}_直方图.png')
    # plt.show()
    plt.close()


def draw_points(output_dir, img_path, points_3d, title, remove_outliers=False):
    """
    绘制匹配点，点的颜色深浅代表第三列值的大小
    points_3d: (N, 3) 数组，前两列是归一化xy坐标，第三列是值（深度/距离/置信度/差值）
    title: 用于生成图像标题
    remove_outliers: 是否去除极端值
    percentile_range: 分位点范围，用于去除极端值
    """
    if len(points_3d) == 0:
        points_3d = np.array([[0, 0, 0]])

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # 转换归一化坐标到像素坐标
    pixel_points = (points_3d[:, :2] * np.array([w, h])).astype(int)

    # 获取第三列的值
    values = points_3d[:, 2]

    # 去除极端值（基于分位点）
    if remove_outliers:
        lower_bound, upper_bound = np.percentile(values, percentile_range)
        mask = (values >= lower_bound) & (values <= upper_bound)
        pixel_points = pixel_points[mask]
        values = values[mask]

    # 归一化值到 [0, 1] 用于颜色映射
    normalized_values = (values - values.min()) / (values.max() - values.min() + 1e-8)

    # 使用 viridis colormap
    cmap = plt.get_cmap('viridis')
    colors = cmap(normalized_values)

    # 创建可视化
    plt.figure(figsize=(15, 8))
    plt.imshow(img)
    scatter = plt.scatter(pixel_points[:, 0], pixel_points[:, 1], c=colors, s=5, alpha=0.7)

    # 添加颜色条
    cbar = plt.colorbar(scatter)
    cbar.set_label('Normalized Depth Difference (epi_depth - ray_depth)' if 'depth_diff' in title else 'Normalized Value')

    plt.title(title)
    plt.xlabel(f'点数: {len(pixel_points)}')
    plt.savefig(f'{output_dir}/{title}_{len(pixel_points)}_稀疏深度点.png', dpi=600, bbox_inches='tight')
    plt.close()


def draw_depth_diff_histogram(output_dir, depth_range, diff_data, title, num_bins=7):
    """绘制基于深度范围的差值均值折线直方图
    Args:
        output_dir: 输出目录
        depth_range: 深度范围元组 (min_depth, max_depth)
        diff_data: 差值数据数组
        title: 图表标题
        num_bins: 分箱数量 (默认为7)
    """
    plt.figure(figsize=(8, 6))

    # 直接从输入的depth_range获取最小最大值
    range_min, range_max = depth_range

    # 创建等宽的 bins，宽度为 (range_max - range_min) / num_bins
    bins = np.linspace(range_min, range_max, num_bins + 1)
    bin_width = (range_max - range_min) / num_bins

    # 计算每个 bin 内差值的均值
    bin_means = []
    for i in range(len(bins) - 1):
        mask = (diff_data >= bins[i]) & (diff_data < bins[i + 1])  # 注意这里改为对diff_data进行分箱
        if np.sum(mask) > 0:  # 仅计算有数据的 bin
            mean_diff = np.mean(diff_data[mask])
        else:
            mean_diff = 0  # 如果 bin 内无数据，均值为 0
        bin_means.append(mean_diff)

    # 绘制直方图，纵坐标为差值均值
    counts = bin_means
    bins = bins[:-1]  # 柱子以 bin 左边界为起点
    plt.bar(bins, counts, width=bin_width, color='skyblue', edgecolor='black', alpha=0.7)

    # 在柱子上标注均值
    for i, count in enumerate(counts):
        if count != 0:  # 仅标注非零均值
            plt.text(bins[i] + bin_width / 2, count, f'{count:.2f}',
                     ha='center', va='bottom', fontsize=10)

    # 绘制折线连接柱子顶部
    bin_centers = bins + bin_width / 2
    plt.plot(bin_centers, counts, 'r-', marker='o', linewidth=2, markersize=8)

    plt.xlabel('深度值范围')
    plt.ylabel('差值均值')
    plt.title(f'直方图: {title}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{title}_直方图.png')
    plt.close()


def draw_depth_diff_histogram5(output_dir, depth_data, diff_data, title, num_bins=7):
    """绘制基于深度范围的差值均值折线直方图"""
    plt.figure(figsize=(8, 6))

    # 计算深度范围的 1% 和 99% 分位点
    range_min, range_max = np.percentile(depth_data, percentile_range)

    # 创建等宽的 bins，宽度为 (range_max - range_min) / num_bins
    bins = np.linspace(range_min, range_max, num_bins + 1)
    bin_width = (range_max - range_min) / num_bins

    # 计算每个 bin 内差值的均值
    bin_means = []
    for i in range(len(bins) - 1):
        mask = (depth_data >= bins[i]) & (depth_data < bins[i + 1])
        if np.sum(mask) > 0:  # 仅计算有数据的 bin
            mean_diff = np.mean(diff_data[mask])
        else:
            mean_diff = 0  # 如果 bin 内无数据，均值为 0
        bin_means.append(mean_diff)

    # 绘制直方图，纵坐标为差值均值
    counts = bin_means
    bins = bins[:-1]  # 柱子以 bin 左边界为起点
    plt.bar(bins, counts, width=bin_width, color='skyblue', edgecolor='black', alpha=0.7)

    # 在柱子上标注均值
    for i, count in enumerate(counts):
        if count != 0:  # 仅标注非零均值
            plt.text(bins[i] + bin_width / 2, count, f'{count:.2f}',
                     ha='center', va='bottom', fontsize=10)

    # 绘制折线连接柱子顶部
    bin_centers = bins + bin_width / 2
    plt.plot(bin_centers, counts, 'r-', marker='o', linewidth=2, markersize=8)

    plt.xlabel('深度值范围 (ray_depth)')
    plt.ylabel('差值均值 (epi_depth - ray_depth)')
    plt.title(f'直方图: {title}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{title}_直方图.png')
    plt.close()



# if 'mconf' in title:
#     normalized_values = (values - values.min()) / (values.max() - values.min() + 1e-8)
# else:
#     normalized_values = 1 - (values - values.min()) / (values.max() - values.min() + 1e-8)


# def draw_points(output_dir, img_path, points, title):
#     img_name0 = os.path.basename(img_path).split(".")[0]
#     # Load image
#     img0 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#     h0, w0 = img0.shape[:2]
#
#     # 将归一化坐标转换为像素坐标
#     pixel_points0 = (points[:, :2] * np.array([w0, h0])).astype(int)  # 假设points前两列是xy坐标
#
#     # 获取深度值（假设points第三列是深度值）
#     depth_values = points[:, 2] if points.shape[1] > 2 else np.zeros(len(points))
#
#     # 归一化深度值到[0,1]范围
#     normalized_depths = (depth_values - depth_values.min()) / (depth_values.max() - depth_values.min() + 1e-8)
#
#     # 使用matplotlib的colormap根据深度值分配颜色
#     cmap = plt.get_cmap('viridis')  # 可以使用其他colormap如'plasma', 'magma', 'inferno'等
#     colors = cmap(normalized_depths)  # 这将生成RGBA颜色
#
#     # 创建可视化图形
#     plt.figure(figsize=(15, 8))
#
#     # 子图 1：参考点
#     plt.imshow(img0)
#     scatter = plt.scatter(pixel_points0[:, 0], pixel_points0[:, 1], c=colors, s=1)
#
#     # 添加颜色条
#     cbar = plt.colorbar(scatter)
#     cbar.set_label('Normalized Depth')
#
#     plt.title(title)
#     plt.xlabel(f'点数: {len(points)}')
#     plt.savefig(f'{output_dir}/{title}_稀疏深度点.png', dpi=600, bbox_inches='tight')
#     plt.close()

# def draw_points2(output_dir, img_path0, img_path1, points_3d0, points_3d1, title):
#     """
#     points_3d: (N, 3) 数组，前两列是归一化xy坐标，第三列是值（深度/距离/置信度）
#     title_suffix: 用于生成图像标题的后缀
#     """
#     img_name0 = os.path.basename(img_path0).split(".")[0]
#     img_name1 = os.path.basename(img_path1).split(".")[0]
#     img0 = cv2.cvtColor(cv2.imread(img_path0), cv2.COLOR_BGR2RGB)
#     img1 = cv2.cvtColor(cv2.imread(img_path1), cv2.COLOR_BGR2RGB)
#     h0, w0 = img0.shape[:2]
#     h1, w1 = img1.shape[:2]
#
#     # 转换归一化坐标到像素坐标
#     pixel_points0 = (points_3d0[:, :2] * np.array([w0, h0])).astype(int)
#     pixel_points1 = (points_3d1[:, :2] * np.array([w1, h1])).astype(int)
#
#     # 获取第三列的值并归一化到[0,1]用于颜色映射
#     values0 = points_3d0[:, 2]
#     values1 = points_3d1[:, 2]
#     normalized_values0 = (values0 - values0.min()) / (values0.max() - values0.min() + 1e-8)
#     normalized_values1 = (values1 - values1.min()) / (values1.max() - values1.min() + 1e-8)
#
#     # 使用viridis colormap
#     cmap = plt.get_cmap('viridis')
#     colors0 = cmap(normalized_values0)
#     colors1 = cmap(normalized_values1)
#
#     # 创建可视化
#     plt.figure(figsize=(15, 8))
#
#     plt.subplot(1, 2, 1)
#     plt.imshow(img0)
#     scatter = plt.scatter(pixel_points0[:, 0], pixel_points0[:, 1], c=colors0, s=5, alpha=0.7)
#     plt.title(img_name0)
#     plt.xlabel(f'点数: {len(points_3d0)}')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(img1)
#     scatter = plt.scatter(pixel_points1[:, 0], pixel_points1[:, 1], c=colors1, s=5, alpha=0.7)
#     plt.title(img_name1)
#     plt.xlabel(f'点数: {len(points_3d1)}')
#
#     # 添加颜色条
#     cbar = plt.colorbar(scatter)
#     cbar.set_label('Normalized Value')
#     plt.savefig(f'{output_dir}/{title}_{len(points_3d0)}_稀疏深度点.png', dpi=600, bbox_inches='tight')
#     plt.close()



def plot_binned_depth_histogram(output_dir, depths, n=100, depth_range=(0, 1000), title="Depth Distribution"):
    """
    绘制分箱深度直方图（每n条数据的均值为一个柱子）

    参数:
        depths: (N,) 深度值数组
        n: 每个柱子包含的数据条数
        outlier_threshold: Z-score异常值阈值
        title: 图表标题
    """
    # 移除异常值
    # z_scores = np.abs(stats.zscore(depths))
    # mask0 = depths > depth_range[0]
    # mask1 = depths < depth_range[1]
    # valid_depths = depths[mask0 & mask1]
    valid_depths = depths
    print(f"原始数据: {len(depths)}条, 有效数据: {len(valid_depths)}条")

    # 数据分箱
    num_bins = len(valid_depths) // n
    binned_depths = valid_depths[:num_bins * n].reshape(-1, n)
    bin_means = np.mean(binned_depths, axis=1)

    # 创建图表
    plt.figure(figsize=(12, 6))

    # 绘制柱状图（折线式）
    x = np.arange(len(bin_means)) * n  # 横坐标：数据个数
    plt.bar(x, bin_means, width=n * 0.8,
            edgecolor='steelblue', linewidth=1,
            alpha=0.7, label=f'每{n}条均值')

    # 添加趋势线
    plt.plot(x, bin_means, 'r-', linewidth=2, marker='o', markersize=5, label='趋势线')

    # 标注统计信息
    plt.axhline(np.mean(valid_depths), color='green', linestyle='--',
                label=f'全局均值: {np.mean(valid_depths):.2f}')

    # 美化图表
    plt.title(f"{title}\n(排除{len(depths) - len(valid_depths)}个异常值)", fontsize=14)
    plt.xlabel(f'数据序号 (每{n}条为一个区间)', fontsize=12)
    plt.ylabel('深度均值', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 显示/保存
    plt.savefig('depth_binned_hist.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/{title}_直方图.png')
    plt.show()

def image_torch_pil(image, mode="RGB"):
    return transforms.ToPILImage()(image).convert(mode)

def image_pil_torch(image):
    # Define the transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # Apply the transform
    image = transform(image)
    return image


def cal_scale_matrix(inpainting_depth_map: torch.Tensor):
    """
    归一化深度图并生成缩放矩阵
    Args:
        inpainting_depth_map: 输入深度图 (1, H, W)
    Returns:
        depth_map_normalized: 归一化的三通道深度图 (3, H, W)
        scale_matrix: 缩放矩阵 [scale, offset]
    """
    assert inpainting_depth_map.dim() == 3 and inpainting_depth_map.shape[0] == 1

    depth_map_min = inpainting_depth_map.min()
    depth_map_max = inpainting_depth_map.max()

    # 归一化处理
    if torch.isclose(depth_map_max, depth_map_min):
        depth_map_normalized = torch.zeros_like(inpainting_depth_map)
    else:
        depth_map_normalized = (inpainting_depth_map - depth_map_min) / (depth_map_max - depth_map_min)

    # 扩展为三通道
    depth_map_normalized = depth_map_normalized.repeat(3, 1, 1)  # (3, H, W)

    # 缩放矩阵 [scale_factor, min_value]
    scale_matrix = torch.tensor([(depth_map_max - depth_map_min).item(), depth_map_min.item()])

    return depth_map_normalized, scale_matrix


def rec_scale(inpainted_depth: torch.Tensor, scale_matrix: torch.Tensor):
    """
    将归一化的深度图恢复原始范围
    Args:
        inpainted_depth: 归一化后的深度图 (3, H, W)
        scale_matrix: 缩放矩阵 [scale, offset]
    Returns:
        re_inpainted_depth: 恢复原始范围的深度图 (3, H, W)
    """
    scale = scale_matrix[0]
    offset = scale_matrix[1]

    if inpainted_depth.shape[0] == 3:
        inpainted_depth = inpainted_depth[0:1, :, :]  # 转为 (1, H, W)

    # 恢复原始范围
    re_inpainted_depth = inpainted_depth * scale + offset

    return re_inpainted_depth


def image_torch_numpy(image: torch.tensor):
    assert image.dim() == 3
    image = (image * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return image


def image_numpy_torch(image: np.array):
    assert image.ndim == 3
    image = torch.from_numpy(image / 255.).permute(2, 0, 1).float().cuda()
    return image

def image_2_mask_torch(image: np.array):
    assert image.ndim == 3
    mask = torch.from_numpy(image / 255.)[:, :, 0].float().cuda()
    return mask

def image_pil_numpy(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image

def save_image2(image: torch.tensor, save_path: str):
    assert image.dim() == 3
    image_np = (image * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    cv2.imwrite(save_path, image_np)

def save_image(image: torch.tensor, save_path: str):
    assert image.dim() == 3
    image_np = (image * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if image_np.shape[-1] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image_np)

def save_mask(mask: torch.tensor, save_path: str):
    assert mask.dim() == 1
    mask = mask.repeat(3, 1, 1)
    mask = image_torch_numpy(mask)
    cv2.imwrite(save_path, mask)

def filter_outliers(depth_map, percentile = (3, 97), is_normal = False):
    depth_map_min = depth_map.min()
    depth_map_max = depth_map.max()
    middle = (depth_map_max - depth_map_min) / 2.
    depth_map = np.nan_to_num(depth_map, nan=depth_map_min)
    min_quantile, max_quantile = np.percentile(depth_map, percentile)
    if max_quantile < middle: # 97%的点 挤在左半区
        depth_map = np.clip(depth_map, None, max_quantile)
    if min_quantile > middle: # 97%的点 挤在右半区
        depth_map = np.clip(depth_map, None, min_quantile)

    if is_normal:
        depth_map_min = depth_map.min()
        depth_map_max = depth_map.max()
        if depth_map_max > depth_map_min:
            depth_map_normalized = (depth_map - depth_map_min) / (depth_map_max - depth_map_min)
        else:
            depth_map_normalized = np.zeros_like(depth_map)
        return depth_map_normalized
    return depth_map

def save_depth_map(depth_map: torch.tensor, save_path: str, pseudo_color = True):
    """
    保存深度图为伪彩色图像。

    Args:
        depth_map (torch.Tensor): 深度图张量，形状 (H, W) 或 (1, H, W)。
        save_path (str): 保存路径。
        pseudo_color (bool): 伪彩色图像。
    """
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze(0)

    depth_map = depth_map.detach().cpu().numpy()  # 确保转换为numpy数组

    depth_map_normalized = filter_outliers(depth_map, is_normal = True)

    depth_map_normalized = 1 - depth_map_normalized

    depth_map = (depth_map_normalized * 255).astype(np.uint8)
    cv2.imwrite(save_path, depth_map)
    if pseudo_color:
        parent_dir, filename = os.path.split(save_path)
        filename_without_ext, file_extension = os.path.splitext(filename)
        save_path = os.path.join(parent_dir, filename_without_ext + '_color' + file_extension)
        depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        cv2.imwrite(save_path, depth_map_colored)
    return depth_map_normalized

def save_depth_map_mask(depth_map: np.array, mask : np.array, save_path: str, pseudo_color = True):
    """
    保存深度图为伪彩色图像。

    Args:
        depth_map (np.array): 深度图张量，形状 (H, W) 或 (1, H, W)。
        mask (np.array): 深度图张量，形状 (H, W) 或 (1, H, W)。
        save_path (str): 保存路径。
        pseudo_color (bool): 伪彩色图像。
    """
    if len(depth_map.shape) == 3:
        depth_map = depth_map.squeeze(0)

    depth_map = (depth_map * 255).astype(np.uint8)
    cv2.imwrite(save_path, depth_map * mask)
    if pseudo_color:
        parent_dir, filename = os.path.split(save_path)
        filename_without_ext, file_extension = os.path.splitext(filename)
        save_path = os.path.join(parent_dir, filename_without_ext + '_color' + file_extension)
        depth_map_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
        cv2.imwrite(save_path, depth_map_colored * mask)
    return depth_map

from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

def plot_depth_histogram(depth_map, n_bins=20, save_path=None):
    """
    绘制深度图直方图并可选择保存到文件

    参数:
        depth_map: 深度图张量，形状为(1, h, w)
        n_bins: 直方图的柱子数量
        save_path: 图片保存路径(包含文件名)，如为None则不保存
    """
    # 确保输入是PyTorch张量
    if not isinstance(depth_map, torch.Tensor):
        depth_map = torch.tensor(depth_map)

    # 将深度图转换为numpy数组并展平
    depth_values = depth_map.squeeze().cpu().numpy().flatten()

    # 计算直方图
    counts, bin_edges = np.histogram(depth_values, bins=n_bins)

    # 计算每个柱子的中心位置作为横坐标
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], align='center')

    # 设置图表标题和标签
    plt.title('深度图直方图')
    plt.xlabel('深度范围')
    plt.ylabel('点数')
    plt.grid(True, alpha=0.3)

    # 如果提供了保存路径，则保存图片
    if save_path is not None:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"直方图已保存到: {save_path}")

    plt.show()



from mpl_toolkits.mplot3d import Axes3D
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
def visual_3d_points(points):

    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               c='blue', marker='o', s=20, alpha=0.6)

    # 设置坐标轴标签
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('Z轴')

    # 设置标题
    ax.set_title('三维点云可视化')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 生成随机三维点数据
    np.random.seed(42)
    points = np.random.rand(100, 3)  # 100个点，每个点有x,y,z坐标

    visual_3d_points(points)


import numpy as np
# Function to create point cloud file
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')     # 必须先写入，然后利用write()在头部插入ply header
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)


if __name__ == '__main__':
    # Define name for output file
    output_file = 'Andre_Agassi_0015.ply'
    a = np.load("Andre_Agassi_0015.npy")
    b = np.float32(a)
#   43867是我的点云的数量，用的时候记得改成自己的
    one = np.ones((43867,3))
    one = np.float32(one)*255
#    points_3D = np.array([[1,2,3],[3,4,5]]) # 得到的3D点（x，y，z），即2个空间点
#    colors = np.array([[0, 255, 255], [0, 255, 255]])   #给每个点添加rgb
    # Generate point cloud
    print("\n Creating the output file... \n")
#    create_output(points_3D, colors, output_file)
    create_output(b, one, output_file)



# def save_ply():
    # output = str(os.path.join(args.virtual_camera_dir, "output.ply"))
    # points = np.array(all_repaired_points[0])
    #
    # plot_utils.save_ply(os.path.join(args.virtual_camera_dir, "output.ply"), points.reshape(-1, 6), )
    # # plot_utils.visual_3d_points(all_repaired_points[0])
    # pcd = o3d.geometry.PointCloud()
    # a = all_repaired_points[0][:, :3]
    # pcd.points = o3d.utility.Vector3dVector(a)
    # o3d.io.write_point_cloud(Path(output), pcd)

from plyfile import PlyData, PlyElement

def store_ply(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)



import matplotlib.pyplot as plt
# import seaborn as sns


def plot_enhanced_depth_hist(viewpoint_stack, save_path):
    for cam in viewpoint_stack:
        depth_values = cam.mono_depth_map.cpu().numpy().flatten()

        plt.figure(figsize=(12, 6))
        n, bins, patches = plt.hist(depth_values, bins=50, color='teal', alpha=0.7, edgecolor='black')

        # 添加统计信息
        mean_val = depth_values.mean()
        median_val = np.median(depth_values)
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.3f}')
        plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.3f}')

        plt.title('Enhanced Depth Distribution with Statistics', fontsize=14)
        plt.xlabel('Depth Value', fontsize=12)
        plt.ylabel('Pixel Count (log scale)', fontsize=12)
        plt.yscale('log')  # 对数坐标可以更好显示长尾分布
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.savefig(f"{save_path}/{cam.image_name}_mono_depth_distribution.png")
        plt.show()

        # 打印关键统计量
        print(f"Depth Range: [{depth_values.min():.3f}, {depth_values.max():.3f}]")
        print(f"Mean: {mean_val:.3f}, Median: {median_val:.3f}")
        print(f"Standard Deviation: {depth_values.std():.3f}")

def plot_depth_histogram(viewpoint_stack, save_path):
    for cam in viewpoint_stack:
        """
        绘制深度图直方图
        参数:
            depth_tensor: 输入深度图 tensor(h, w)
        """
        # 转换为numpy数组并展平
        depth_values = cam.mono_depth_map.cpu().numpy().flatten()

        plt.figure(figsize=(10, 5))
        plt.hist(depth_values, bins=50, color='blue', alpha=0.7)
        plt.title('Depth Distribution Histogram')
        plt.xlabel('Depth Value')
        plt.ylabel('Pixel Count')
        plt.grid(True)
        plt.savefig(f"{save_path}/{cam.image_name}_mono_depth_distribution.png")
        plt.show()


# def plot_loss_distribution(repro_loss_state, save_path):
#     for key0, value in repro_loss_state.items():
#         for key1 in value.keys():
#             loss = repro_loss_state[key0][key1].clone().detach().cpu()
#             plt.figure(figsize=(10, 4))
#             # 直方图 + 密度估计
#             sns.histplot(loss, bins=50, kde=True)
#             plt.axvline(torch.median(loss), color='r', label='Median')
#             plt.xlabel('Reprojection Error (pixels)')
#             plt.ylabel('Frequency')
#             plt.legend()
#             plt.savefig(f"{save_path}/{key0}_{key1}_sample_plot.png")
#             plt.show()

# def plot_loss_with_threshold(repro_loss_state, threshold, save_path):
#     for key0, value in repro_loss_state.items():
#         for key1 in value.keys():
#             loss = repro_loss_state[key0][key1].clone().detach().cpu()
#             plt.figure(figsize=(12, 5))
#             ax = sns.histplot(loss, bins=100, kde=True)
#
#             # 标记关键统计量
#             stats = {
#                 'Median': torch.median(loss),
#                 'Mean': torch.mean(loss),
#                 'Threshold': threshold
#             }
#             for name, val in stats.items():
#                 ax.axvline(val, color='r' if name == 'Threshold' else 'k', linestyle='--', label=f'{name}: {val:.1f}')
#
#             plt.xlabel('Reprojection Error (pixels)')
#             plt.legend()
#             plt.savefig(f"{save_path}/{key0}_{key1}_localOrContinue_tail.png")
#             plt.show()


from scipy import stats


def test_distribution(repro_loss_state):
    for key0, value in repro_loss_state.items():
        for key1 in value.keys():
            loss = repro_loss_state[key0][key1].clone().detach().cpu()
            # Shapiro-Wilk检验正态性
            _, p_normal = stats.shapiro(loss.numpy())
            # Kolmogorov-Smirnov检验拉普拉斯分布
            loc = torch.median(loss).item()
            scale = torch.mean(torch.abs(loss - loc)).item()
            _, p_laplace = stats.kstest(loss.numpy(), 'laplace', args=(loc, scale))

            print(f"Normal分布检验p值: {p_normal:.3f} (p>0.05时接受正态假设)")
            print(f"Laplace分布检验p值: {p_laplace:.3f} (p>0.05时接受拉普拉斯假设)")