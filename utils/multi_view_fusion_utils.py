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
    Erode mask inward

    Args:
        mask (torch.Tensor): Input mask, shape (1, h, w)
        erode_pixels (int): Number of pixels to erode

    Returns:
        torch.Tensor: Eroded mask
    """
    if erode_pixels <= 0:
        return mask

    # Ensure mask is binarized
    binary_mask = (mask > 0.5).float()

    # Create erosion kernel
    kernel_size = 2 * erode_pixels + 1
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)

    # Calculate how many convolutions needed for complete erosion
    # Use max pooling, then check if equal to kernel area (indicating all pixels are 1)
    convolved = F.conv2d(binary_mask.unsqueeze(0), kernel, padding=erode_pixels)
    eroded = (convolved == (kernel_size * kernel_size)).float()

    return eroded.squeeze(0)

def expand_visible_region_fusion(warped_images, warped_depths, valid_masks, k, erode_pixels=1):
    """
    Fusion function for expanding visible regions

    Args:
        warped_images (torch.Tensor): RGB images, shape (b, 3, h, w), normalized values [0,1]
        warped_depths (torch.Tensor): Depth maps, shape (b, 1, h, w)
        valid_masks (torch.Tensor): Valid masks, shape (b, 1, h, w), 1=valid, 0=invalid
        k (int): Number of images per group

    Returns:
        tuple: (fused_images, fused_depths, fused_masks)
    """
    b, c, h, w = warped_images.shape
    n = b // k

    # Initialize output
    fused_images = torch.zeros((n, 3, h, w), device=warped_images.device)
    fused_depths = torch.zeros((n, 1, h, w), device=warped_images.device)
    fused_masks = torch.zeros((n, 1, h, w), device=warped_images.device, dtype=torch.float32)

    for i in range(n):
        # Get current group
        group_start = i * k
        group_end = (i + 1) * k
        group_images = warped_images[group_start:group_end]
        group_depths = warped_depths[group_start:group_end]
        group_masks = valid_masks[group_start:group_end]

        # Use first image as base
        base_image = group_images[0].clone()
        base_depth = group_depths[0].clone()
        base_mask = group_masks[0].clone()

        # Erode base_mask inward
        if erode_pixels > 0:
            base_mask = erode_mask(base_mask, erode_pixels)

        # Find regions in base image that need filling (regions where mask=0)
        holes_mask = (base_mask < 0.5).float()  # Regions to fill are 1

        if holes_mask.sum() == 0:
            # If no regions to fill, directly use base image
            fused_images[i] = base_image
            fused_depths[i] = base_depth
            fused_masks[i] = base_mask
            continue

        # Process each other image
        for j in range(1, k):
            current_image = group_images[j]
            current_depth = group_depths[j]
            current_mask = group_masks[j]

            # Only consider regions that are valid in current image and invalid in base image
            fill_candidate = (current_mask > 0.5) & (holes_mask > 0.5)

            if fill_candidate.sum() > 0:
                # Use current image to fill holes
                base_image = torch.where(fill_candidate.repeat(1, 3, 1, 1),
                                         current_image, base_image)
                base_depth = torch.where(fill_candidate, current_depth, base_depth)
                base_mask = torch.where(fill_candidate, current_mask, base_mask)

                # Update holes mask (subtract filled regions)
                holes_mask = (base_mask < 0.5).float()

                if holes_mask.sum() == 0:
                    break  # All holes filled

        # Apply edge smoothing
        base_image = smooth_transitions(base_image, base_mask)

        fused_images[i] = base_image
        fused_depths[i] = base_depth
        fused_masks[i] = base_mask

    return fused_images, fused_depths, fused_masks


def smooth_transitions(image, mask, kernel_size=5):
    """
    Smooth fusion boundaries
    """
    # Create boundary mask (transition region from invalid to valid)
    boundary_mask = create_boundary_mask(mask, kernel_size)

    if boundary_mask.sum() == 0:
        return image

    # Apply Gaussian blur to boundary regions
    blurred_image = transforms.functional.gaussian_blur(image, kernel_size=kernel_size, sigma=0.1)

    # Blend original and blurred images
    alpha = boundary_mask.repeat(1, 3, 1, 1)
    smoothed_image = image * (1 - alpha) + blurred_image * alpha

    return smoothed_image


def create_boundary_mask(mask, kernel_size=5):
    """
    Create mask for boundary regions
    """
    # Dilate valid regions
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)
    dilated_mask = F.conv2d(mask, kernel, padding=kernel_size // 2)
    dilated_mask = (dilated_mask > 0).float()

    # Erode valid regions
    eroded_mask = F.conv2d(mask, kernel, padding=kernel_size // 2)
    eroded_mask = (eroded_mask == kernel_size * kernel_size).float()

    # Boundary region = dilated region - eroded region
    boundary_mask = dilated_mask - eroded_mask

    return boundary_mask


def advanced_expand_visible_region_fusion(warped_images, warped_depths, valid_masks, k):
    """
    Advanced version: fusion considering depth consistency
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

        # Sort by depth (near to far), prioritize using nearby information for filling
        depth_values = []
        for j in range(k):
            if group_masks[j].sum() > 0:
                median_depth = torch.median(group_depths[j][group_masks[j] > 0.5])
                depth_values.append((j, median_depth.item()))

        # Sort by depth (near first)
        depth_values.sort(key=lambda x: x[1])
        sorted_indices = [idx for idx, _ in depth_values if idx != 0]  # Exclude base image

        for j in sorted_indices:
            current_image = group_images[j]
            current_depth = group_depths[j]
            current_mask = group_masks[j]

            # Depth consistency check: only fill regions with similar depth
            depth_diff = torch.abs(base_depth - current_depth)
            depth_consistent = (depth_diff < 0.1) | (base_mask < 0.5)  # Base invalid or depth similar

            fill_candidate = (current_mask > 0.5) & (holes_mask > 0.5) & depth_consistent

            if fill_candidate.sum() > 0:
                base_image = torch.where(fill_candidate.repeat(1, 3, 1, 1), current_image, base_image)
                base_depth = torch.where(fill_candidate, current_depth, base_depth)
                base_mask = torch.where(fill_candidate, torch.ones_like(base_mask), base_mask)
                holes_mask = (base_mask < 0.5).float()

                if holes_mask.sum() == 0:
                    break

        # Final smoothing
        base_image = smooth_transitions(base_image, base_mask)

        fused_images[i] = base_image
        fused_depths[i] = base_depth
        fused_masks[i] = base_mask

    return fused_images, fused_depths, fused_masks




# Usage example
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