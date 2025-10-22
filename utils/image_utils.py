#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene.cameras import TempCamera


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


from utils.graphics_utils import getWorld2View, focal2fov


import os
import torch
import cv2
import numpy as np
from utils import graphics_utils
import torch.nn.functional as F
from PIL import Image


# Set Chinese display font
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]

def read_image(image_path: str, grayscale: bool = False) -> np.ndarray:
    """Read image, supports color or grayscale format"""
    if grayscale:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image from {image_path}")
    return image



def resize_intrinsic(K_ori: np.ndarray, scale: float) -> np.ndarray:
    K = K_ori.copy()
    K[0, 0] *= scale  # fx
    K[1, 1] *= scale  # fy
    K[0, 2] *= scale  # cx
    K[1, 2] *= scale  # cy
    return K.astype(np.float32)





def resize_image_with_scale(image: np.ndarray, scale: float = None, dfactor: int = 8):
    """Resize image, returns resized image and new dimensions"""
    size = image.shape[:2][::-1]  # (width, height)

    # First resize: scale based on resize_max
    if scale is not None and scale < 1.0:
        print('scale: ', scale)
        new_size = tuple(int(round(x * scale)) for x in size)
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Second resize: ensure dimensions divisible by dfactor
    new_size = tuple(int(x // dfactor * dfactor) for x in image.shape[:2][::-1])
    if new_size != image.shape[:2][::-1]:
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    print(f'ori_size -> scaled_size: {size} -> {new_size}')

    return image, new_size


def save_image(image: np.ndarray, save_path: str):
    """Save image to specified path"""
    save_image_data = image.clip(0, 255).astype(np.uint8)
    cv2.imwrite(save_path, save_image_data)


def resize_to_multiple(image_data: torch.Tensor, multiple: int = 8):
    """
    Scale image and mask to size that is multiple of specified integer
    Args:
        image_data: Input image (3, H, W) RGB format
        multiple: Integer multiple to scale to (default 32, common CNN downsampling multiple)
    Returns:
        resized_image: Scaled image (3, H', W')
    """
    assert image_data.dim() == 3
    h, w = image_data.shape[1:]

    # Calculate target size (round down to nearest multiple)
    new_h = (h // multiple) * multiple
    new_w = (w // multiple) * multiple

    # Return directly if already a multiple
    if new_h == h and new_w == w:
        return image_data

    # Scale image with bilinear interpolation
    resized_image = F.interpolate(
        image_data.unsqueeze(0),  # Add batch dimension (1, 3, H, W)
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # Remove batch dimension

    return resized_image


def resize_pil_to_multiple(
        image_data: Image.Image,
        multiple: int = 32
) -> Image.Image:
    """
    Scale PIL image to size that is multiple of specified integer (maintain aspect ratio)

    Args:
        image_data: Input image in PIL format (RGB mode)
        multiple: Integer multiple to scale to (default 32)

    Returns:
        resized_image: Scaled PIL image
    """
    assert isinstance(image_data, Image.Image), "Input must be PIL.Image"

    w, h = image_data.size

    # Calculate target size (round down to nearest multiple)
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple

    # Return directly if already a multiple
    if new_w == w and new_h == h:
        return image_data

    # Use LANCZOS high-quality scaling filter
    return image_data.resize((new_w, new_h), Image.Resampling.LANCZOS)





def to_torch_cuda(image: np.ndarray, grayscale: bool = False, device: str = '') -> torch.Tensor:
    """Convert NumPy image to PyTorch Tensor and put on CUDA"""
    # Channel adjustment and normalization
    if grayscale:
        assert image.ndim == 2, f"Expected grayscale image with 2 dims, got {image.shape}"
        image = image[None]  # (1, H, W)
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = image / 255.0  # Normalize to [0, 1]

    # Convert to PyTorch Tensor
    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.to(device)[None]

    return image_tensor


def resize_images(args, cam_infos):
    resized_image_dir = args.resized_image_dir
    scale = 1.0 / args.inv_scale
    dfactor = args.dfactor

    temp_cameras = []
    for cam_info in cam_infos:
        image_data = np.array(cam_info.image)
        resized_image_data, new_size = resize_image_with_scale(image_data, scale, dfactor)
        resized_image_data = cv2.cvtColor(resized_image_data, cv2.COLOR_RGB2BGR).astype(np.uint8)

        img_name = cam_info.image_name + '.jpg'
        save_image_path = str(os.path.join(resized_image_dir, img_name))
        save_image(resized_image_data, save_image_path)

        K = resize_intrinsic(cam_info.K, scale)
        height, width = resized_image_data.shape[:2]
        FovX, FovY = focal2fov(K[0, 0], width), focal2fov(K[1, 1], height)
        temp_camera = TempCamera(R=cam_info.R, T=cam_info.T, K=K, w2c=cam_info.w2c, width=width, height=height, FovX=FovX, FovY=FovY,
                                 image_name = cam_info.image_name, image_data= resized_image_data, image_path=save_image_path)

        temp_camera.point3D_ids = cam_info.point3D_ids
        temp_camera.bounds = cam_info.bounds

        temp_cameras.append(temp_camera)
    return temp_cameras