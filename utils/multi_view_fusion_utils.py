import torch
import torch.nn.functional as F
import cv2
import numpy as np
from utils import plot_utils
import torchvision.transforms as transforms
import matplotlib.cm as cm
import os

def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)
def vis_depth(depth):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    percentile = 95
    eps = 1e-10

    lo_auto, hi_auto = weighted_percentile(
        depth, np.ones_like(depth), [50 - percentile / 2, 50 + percentile / 2])
    lo = None or (lo_auto - eps)
    hi = None or (hi_auto + eps)

    curve_fn = lambda x: 1/x + eps

    depth, lo, hi = [curve_fn(x) for x in [depth, lo, hi]]

    depth = np.nan_to_num(
            np.clip((depth - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))

    colorized = cm.get_cmap('turbo')(depth)[:, :, :3]


    #### for blender visualize
    # depth = depth/np.max(depth)
    # colorized = cm.get_cmap('turbo')(depth)[:, :, :3]


    return np.uint8(colorized[..., ::-1] * 255)

def visualization(depth, save_path):
    import matplotlib as mpl
    import matplotlib.cm as cm
    from PIL import Image

    vmax = np.percentile(depth, 98)
    vmin = depth.min()
    # print(save_path, vmax, vmin)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='turbo')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    im = Image.fromarray(colormapped_im)
    im.save(save_path)

def erode_mask(mask, erode_pixels):
    """
    向内腐蚀mask

    参数:
        mask (torch.Tensor): 输入掩码，形状(1, h, w)
        erode_pixels (int): 腐蚀的像素数

    返回:
        torch.Tensor: 腐蚀后的掩码
    """
    if erode_pixels <= 0:
        return mask

    # 确保mask是二值化的
    binary_mask = (mask > 0.5).float()

    # 创建腐蚀核
    kernel_size = 2 * erode_pixels + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)

    # 计算需要多少卷积才能完全腐蚀
    # 使用最大值卷积，然后检查是否等于核面积（表示所有像素都是1）
    convolved = F.conv2d(binary_mask.unsqueeze(0), kernel, padding=erode_pixels)
    eroded = (convolved == (kernel_size * kernel_size)).float()

    return eroded.squeeze(0)

def expand_visible_region_fusion(warped_images, warped_depths, valid_masks, k, erode_pixels=1):
    """
    扩展可视区域的融合函数

    参数:
        warped_images (torch.Tensor): RGB图像，形状(b, 3, h, w)，归一化值[0,1]
        warped_depths (torch.Tensor): 深度图，形状(b, 1, h, w)
        valid_masks (torch.Tensor): 有效掩码，形状(b, 1, h, w)，1表示有效，0表示无效
        k (int): 每组图像数量

    返回:
        tuple: (fused_images, fused_depths, fused_masks)
    """
    b, c, h, w = warped_images.shape
    n = b // k

    # 初始化输出
    fused_images = torch.zeros((n, 3, h, w), device=warped_images.device)
    fused_depths = torch.zeros((n, 1, h, w), device=warped_images.device)
    fused_masks = torch.zeros((n, 1, h, w), device=warped_images.device, dtype=torch.float32)

    for i in range(n):
        # 获取当前组
        group_start = i * k
        group_end = (i + 1) * k
        group_images = warped_images[group_start:group_end]
        group_depths = warped_depths[group_start:group_end]
        group_masks = valid_masks[group_start:group_end]

        # 以第一张图像为基础
        base_image = group_images[0].clone()
        base_depth = group_depths[0].clone()
        base_mask = group_masks[0].clone()

        # 向内腐蚀base_mask
        if erode_pixels > 0:
            base_mask = erode_mask(base_mask, erode_pixels)

        # 找到基础图像中需要填充的区域（mask为0的区域）
        holes_mask = (base_mask < 0.5).float()  # 需要填充的区域为1

        if holes_mask.sum() == 0:
            # 如果没有需要填充的区域，直接使用基础图像
            fused_images[i] = base_image
            fused_depths[i] = base_depth
            fused_masks[i] = base_mask
            continue

        # 对每个其他图像进行处理
        for j in range(1, k):
            current_image = group_images[j]
            current_depth = group_depths[j]
            current_mask = group_masks[j]

            # 只考虑当前图像有效且基础图像无效的区域
            fill_candidate = (current_mask > 0.5) & (holes_mask > 0.5)

            if fill_candidate.sum() > 0:
                # 使用当前图像填充空洞
                base_image = torch.where(fill_candidate.repeat(1, 3, 1, 1),
                                         current_image, base_image)
                base_depth = torch.where(fill_candidate, current_depth, base_depth)
                base_mask = torch.where(fill_candidate, current_mask, base_mask)

                # 更新空洞掩码（减去已填充的区域）
                holes_mask = (base_mask < 0.5).float()

                if holes_mask.sum() == 0:
                    break  # 所有空洞都已填充

        # 应用边缘平滑
        base_image = smooth_transitions(base_image, base_mask)

        fused_images[i] = base_image
        fused_depths[i] = base_depth
        fused_masks[i] = base_mask

    return fused_images, fused_depths, fused_masks


def smooth_transitions(image, mask, kernel_size=5):
    """
    对融合边界进行平滑处理
    """
    # 创建边界掩码（从无效到有效的过渡区域）
    boundary_mask = create_boundary_mask(mask, kernel_size)

    if boundary_mask.sum() == 0:
        return image

    # 对边界区域进行高斯模糊
    blurred_image = transforms.functional.gaussian_blur(image, kernel_size=kernel_size, sigma=0.1)

    # 混合原始图像和模糊图像
    alpha = boundary_mask.repeat(1, 3, 1, 1)
    smoothed_image = image * (1 - alpha) + blurred_image * alpha

    return smoothed_image


def create_boundary_mask(mask, kernel_size=5):
    """
    创建边界区域的掩码
    """
    # 膨胀有效区域
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    dilated_mask = F.conv2d(mask, kernel, padding=kernel_size // 2)
    dilated_mask = (dilated_mask > 0).float()

    # 腐蚀有效区域
    eroded_mask = F.conv2d(mask, kernel, padding=kernel_size // 2)
    eroded_mask = (eroded_mask == kernel_size * kernel_size).float()

    # 边界区域 = 膨胀区域 - 腐蚀区域
    boundary_mask = dilated_mask - eroded_mask

    return boundary_mask


def advanced_expand_visible_region_fusion(warped_images, warped_depths, valid_masks, k):
    """
    高级版本：考虑深度一致性的融合
    """
    b, c, h, w = warped_images.shape
    n = b // k

    fused_images = torch.zeros((n, 3, h, w), device=warped_images.device)
    fused_depths = torch.zeros((n, 1, h, w), device=warped_images.device)
    fused_masks = torch.zeros((n, 1, h, w), device=warped_images.device)

    for i in range(n):
        group_start = i * k
        group_images = warped_images[group_start:group_start + k]
        group_depths = warped_depths[group_start:group_start + k]
        group_masks = valid_masks[group_start:group_start + k]

        base_image = group_images[0].clone()
        base_depth = group_depths[0].clone()
        base_mask = group_masks[0].clone()

        holes_mask = (base_mask < 0.5).float()

        if holes_mask.sum() == 0:
            fused_images[i] = base_image
            fused_depths[i] = base_depth
            fused_masks[i] = base_mask
            continue

        # 按深度排序（从近到远），优先使用近距离信息填充
        depth_values = []
        for j in range(k):
            if group_masks[j].sum() > 0:
                median_depth = torch.median(group_depths[j][group_masks[j] > 0.5])
                depth_values.append((j, median_depth.item()))

        # 按深度排序（近的优先）
        depth_values.sort(key=lambda x: x[1])
        sorted_indices = [idx for idx, _ in depth_values if idx != 0]  # 排除基础图像

        for j in sorted_indices:
            current_image = group_images[j]
            current_depth = group_depths[j]
            current_mask = group_masks[j]

            # 深度一致性检查：只填充深度相近的区域
            depth_diff = torch.abs(base_depth - current_depth)
            depth_consistent = (depth_diff < 0.1) | (base_mask < 0.5)  # 基础无效或深度相近

            fill_candidate = (current_mask > 0.5) & (holes_mask > 0.5) & depth_consistent

            if fill_candidate.sum() > 0:
                base_image = torch.where(fill_candidate.repeat(1, 3, 1, 1), current_image, base_image)
                base_depth = torch.where(fill_candidate, current_depth, base_depth)
                base_mask = torch.where(fill_candidate, torch.ones_like(base_mask), base_mask)
                holes_mask = (base_mask < 0.5).float()

                if holes_mask.sum() == 0:
                    break

        # 最终平滑处理
        base_image = smooth_transitions(base_image, base_mask)

        fused_images[i] = base_image
        fused_depths[i] = base_depth
        fused_masks[i] = base_mask

    return fused_images, fused_depths, fused_masks




# 使用示例
def multi_view_fusion(sparse_args, warped_images, warped_depths, valid_masks, k):

    fused_images, fused_depths, fused_masks = expand_visible_region_fusion(
        warped_images, warped_depths, valid_masks, k
    )

    depth0, depth1 = warped_depths[0], warped_depths[1]
    mask0, mask1 = valid_masks[0], valid_masks[1]

    fused_image = fused_images[0]
    fused_depth = fused_depths[0]
    fused_mask = fused_masks[0]

    os.makedirs(virtual_visual_dir := os.path.join(sparse_args.virtual_camera_dir, 'visual'), exist_ok=True)
    plot_utils.save_image2(fused_image, f'{virtual_visual_dir}/fused_image01.png')
    plot_utils.save_image2(fused_mask, f'{virtual_visual_dir}/fused_mask01.png')

    depth0 = (depth0 - depth0.min()) / (depth0.max() - depth0.min()) + 1 * (1 - mask0)
    depth1 = (depth1 - depth1.min()) / (depth1.max() - depth1.min()) + 1 * (1 - mask1)
    fused_depth = (fused_depth - fused_depth.min()) / (fused_depth.max() - fused_depth.min()) + 1 * (1 - fused_mask)

    depth_map0 = vis_depth(depth0[0].detach().cpu().numpy())
    depth_map1 = vis_depth(depth1[0].detach().cpu().numpy())
    depth_map01 = vis_depth(fused_depth[0].detach().cpu().numpy())

    depth_map0 = depth_map0 * (mask0.permute(1, 2, 0).cpu().numpy())
    depth_map1 = depth_map1 * (mask1.permute(1, 2, 0).cpu().numpy())
    depth_map01 = depth_map01 * (fused_mask.permute(1, 2, 0).cpu().numpy())

    cv2.imwrite(f'{virtual_visual_dir}/depth_color0.png', depth_map0)
    cv2.imwrite(f'{virtual_visual_dir}/depth_color1.png', depth_map1)
    cv2.imwrite(f'{virtual_visual_dir}/fused_depth_color01.png', depth_map01)

    # plot_utils.save_depth_map(fused_depth, '/home/mayu/thesis/scg_0905_ll/o_ll/fern110_3/virtual_camera/save_image2.jpg')

    return fused_images, fused_depths, fused_masks