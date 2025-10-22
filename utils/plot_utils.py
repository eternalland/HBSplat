import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import torch
from torchvision import transforms


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
    if max_quantile < middle: # 97% of points squeezed in left half
        depth_map = np.clip(depth_map, None, max_quantile)
    if min_quantile > middle: # 97% of points squeezed in right half
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
    Save depth map as pseudo-color image.

    Args:
        depth_map (torch.Tensor): Depth map tensor, shape (H, W) or (1, H, W).
        save_path (str): Save path.
        pseudo_color (bool): Pseudo-color image.
    """
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze(0)

    depth_map = depth_map.detach().cpu().numpy()  # Ensure conversion to numpy array

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
    Save depth map as pseudo-color image.

    Args:
        depth_map (np.array): Depth map tensor, shape (H, W) or (1, H, W).
        mask (np.array): Depth map tensor, shape (H, W) or (1, H, W).
        save_path (str): Save path.
        pseudo_color (bool): Pseudo-color image.
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





