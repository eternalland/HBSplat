import numpy as np
import torch
import matplotlib.cm
import cv2
from typing import Tuple, Optional
import os
from tqdm import tqdm
from skimage import morphology


def optimize_depth_alignment(
        source_depth: np.ndarray,
        target_depth: np.ndarray,
        valid_mask: np.ndarray,
        depth_weights: np.ndarray,
        prune_ratio: float = 0.001,
        max_iterations: int = 10000,
        convergence_threshold: float = 1e-6
):
    # Convert to PyTorch tensor
    device = torch.device('cuda')
    source = torch.from_numpy(source_depth).float().to(device)
    target = torch.from_numpy(target_depth).float().to(device)
    mask = torch.from_numpy(valid_mask).bool().to(device)
    weights = torch.from_numpy(depth_weights).float().to(device)

    # Trim outliers
    with torch.no_grad():
        valid_depths = target[target > 1e-7]
        sorted_depths = torch.sort(valid_depths).values
        n = sorted_depths.numel()

        min_thresh = sorted_depths[int(n * prune_ratio)]
        max_thresh = sorted_depths[int(n * (1 - prune_ratio))]

        mask = mask & (target > min_thresh) & (target < max_thresh)

    # Prepare optimization data
    source_masked = source[mask]
    target_masked = target[mask]
    weights_masked = weights[mask]

    # Initialize optimizable parameters
    scale = torch.ones(1, device=device, requires_grad=True)
    shift = torch.zeros(1, device=device, requires_grad=True)

    # Set up optimizer
    optimizer = torch.optim.Adam([scale, shift], lr=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8 ** (1 / 100))

    # Optimization loop
    prev_loss = float('inf')
    ema_loss = 0.0
    best_params = None
    progress_bar = tqdm(range(1, max_iterations + 1), desc="aligned_iteration")
    for iteration in progress_bar:
        aligned = scale * source_masked + shift

        # Compute loss
        mse_loss = torch.mean(((target_masked - aligned) ** 2) * weights_masked)
        hinge_loss = torch.mean(torch.relu(-aligned) ** 2) * 2.0
        total_loss = mse_loss + hinge_loss

        # Optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Check convergence
        ema_loss = total_loss.item() * 0.2 + ema_loss * 0.8
        if abs(ema_loss - prev_loss) < convergence_threshold:
            best_params = (scale.item(), shift.item())
            break

        if iteration % 1000 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss:.7f}",
                                      "Scale": f"{scale.item():.4f}",
                                      "Shift": f"{shift.item():.4f}",
                                      "LR": f"{optimizer.param_groups[0]['lr']:.4f}"})
            prev_loss = total_loss.item()

    # Apply best parameters
    with torch.no_grad():
        final_scale, final_shift = best_params or (scale.item(), shift.item())
        aligned_depth = (final_scale * source + final_shift).cpu().numpy()

    torch.cuda.empty_cache()
    return aligned_depth, ema_loss, final_scale, final_shift


def create_depth_map_from_sparse(
        sparse_depth_map: np.ndarray,  # (n, 3) format: [u_norm, v_norm, depth]
        height: int,
        width: int,
        depth_weights: Optional[np.ndarray] = None  # (n,) optional weights
) -> Tuple[np.ndarray, np.ndarray]:
    # Initialize output matrix
    depth_map = np.zeros((height, width))
    weight_map = np.zeros((height, width))

    # Convert normalized coordinates to pixel coordinates
    u_pixel = np.round(sparse_depth_map[:, 0] * (width - 1)).astype(int)
    v_pixel = np.round(sparse_depth_map[:, 1] * (height - 1)).astype(int)

    # Ensure coordinates are within valid range
    valid_mask = (u_pixel >= 0) & (u_pixel < width) & \
                 (v_pixel >= 0) & (v_pixel < height)

    u_valid = u_pixel[valid_mask]
    v_valid = v_pixel[valid_mask]
    depths = sparse_depth_map[valid_mask, 2]

    # Fill depth map
    depth_map[v_valid, u_valid] = depths

    # Handle weights
    if depth_weights is not None:
        weights = depth_weights[valid_mask, 2]
        weight_map[v_valid, u_valid] = weights / weights.max()
    else:
        weight_map[v_valid, u_valid] = 1.0

    return depth_map, weight_map


def gather_in_one(cam_dir, min_loss_state):
    li = list(cam_dir.keys())
    for key in li:
        pixels = []
        z_vals = []
        z_cams = []
        reproj_losses = []
        small_transform_z_vals = []
        small_transform_z_cams = []
        for key1 in li:
            if str(key) == str(key1):
                continue
            page = cam_dir[key]['match_infos'][key1]
            pixels.append(page['match_pixel'])

            z_val = page['z_val'].clone().detach().cuda()
            z_vals.append(z_val)
            z_cam = (z_val.squeeze(-1) * page["cam_rays_d"][:, 2]).unsqueeze(-1)
            z_cams.append(z_cam)

            reproj_loss = min_loss_state[key][key1].clone().detach().cuda().unsqueeze(-1)
            reproj_losses.append(reproj_loss)

            small_transform_z_val = page['small_transform_z_val'].clone().detach().cuda()
            small_transform_z_vals.append(z_val)
            small_transform_z_cam = (small_transform_z_val.squeeze(-1) * page["cam_rays_d"][:, 2]).unsqueeze(-1)
            small_transform_z_cams.append(small_transform_z_cam)

        cam_dir[key]['pixel_gather'] = torch.concatenate(pixels, dim=0)
        cam_dir[key]['z_gather'] = torch.concatenate(z_vals, dim=0)
        cam_dir[key]['z_cam_gather'] = torch.concatenate(z_cams, dim=0)
        cam_dir[key]['small_transform_z_vals'] = torch.concatenate(small_transform_z_vals, dim=0)
        cam_dir[key]['small_transform_z_cam_gather'] = torch.concatenate(small_transform_z_cams, dim=0)
        cam_dir[key]['reproj_loss'] = torch.concatenate(reproj_losses, dim=0)



def aligned_depth_scale(args, cam_infos, gaussians):
    cam_dir = gaussians.view_gs
    os.makedirs(output_dir := os.path.join(args.hybrid_depth_dir, "aligned_mono_depth"), exist_ok=True)
    for cam_info in tqdm(cam_infos, desc="aligned_depth_scale"):
        image_name = cam_info.image_name
        z_points = torch.hstack((cam_dir[image_name]['pixel_gather'], cam_dir[image_name]['z_cam_gather'])).cpu().numpy()
        mono_depth_map = np.array(cam_info.mono_depth_map.cpu().numpy())
        foreground_mask = generate_foreground_mask(args, image_name, mono_depth_map)

        height, width = mono_depth_map.shape[:2]
        sparse_depth_map, weight_map = create_depth_map_from_sparse(z_points, height = height, width = width)
        sparse_depth_map = foreground_mask * sparse_depth_map
        aligned_mono_depth, ema_loss, final_scale, final_shift = optimize_depth_alignment(source_depth = mono_depth_map, target_depth = sparse_depth_map, valid_mask = sparse_depth_map>0, depth_weights = weight_map)
        cam_info.aligned_mono_depth_map = torch.from_numpy(aligned_mono_depth).cuda()
        if args.switch_intermediate_result:
            aligned_mono_depth_path = str(os.path.join(output_dir, f'{image_name}_aligned_mono_depth.jpg'))
            save_depth_map(aligned_mono_depth, aligned_mono_depth_path)

def aligned_depth_scale_small(args, cam_infos, gaussians):
    cam_dir = gaussians.view_gs
    os.makedirs(output_dir := os.path.join(args.hybrid_depth_dir, "aligned_mono_depth"), exist_ok=True)
    for cam_info in tqdm(cam_infos, desc="aligned_depth_scale_small"):
        image_name = cam_info.image_name
        z_points = torch.hstack((cam_dir[image_name]['pixel_gather'], cam_dir[image_name]['small_transform_z_cam_gather'])).cpu().numpy()
        mono_depth_map = np.array(cam_info.mono_depth_map.cpu().numpy())
        height, width = mono_depth_map.shape[:2]
        sparse_depth_map, weight_map = create_depth_map_from_sparse(z_points, height = height, width = width)
        aligned_mono_depth, ema_loss, final_scale, final_shift = optimize_depth_alignment(source_depth = mono_depth_map, target_depth = sparse_depth_map, valid_mask = sparse_depth_map>0, depth_weights = weight_map)
        cam_info.aligned_mono_depth_map_small = torch.from_numpy(aligned_mono_depth).cuda()
        if args.switch_intermediate_result:
            aligned_mono_depth_path = str(os.path.join(output_dir, f'{image_name}_aligned_mono_depth_small.jpg'))
            save_depth_map(aligned_mono_depth, aligned_mono_depth_path)



def save_depth_map(depth_map: np.ndarray, save_path: str) -> None:
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze(0)

    depth_map = np.nan_to_num(depth_map, nan=depth_map.min())
    depth_map_min = depth_map.min()
    depth_map_max = depth_map.max()
    if depth_map_max > depth_map_min:
        depth_map_normalized = (depth_map - depth_map_min) / (depth_map_max - depth_map_min)
    else:
        depth_map_normalized = np.zeros_like(depth_map)
    depth_map_uint8 = (depth_map_normalized * 255).astype(np.uint8)
    cv2.imwrite(save_path, depth_map_uint8)
    depth_map_colored = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_JET)
    cv2.imwrite(save_path, depth_map_colored)


def generate_foreground_mask(args, image_name, mono_depth_map):
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