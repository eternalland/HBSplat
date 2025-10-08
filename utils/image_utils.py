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

from skimage.measure import shannon_entropy
from utils import plot_utils

# 设置中文显示字体
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
import matplotlib.pyplot as plt

def read_image(image_path: str, grayscale: bool = False) -> np.ndarray:
    """读取图像，支持彩色或灰度格式"""
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

def get_intrinsic(focal_length_x, focal_length_y, height, width) -> np.ndarray:
    K = np.array([
        [focal_length_x, 0., intr.params[2]],
        [0., focal_length_y, intr.params[3]],
        [0., 0., 1.]
    ], dtype=np.float32)


def resize_image_with_max(image: np.ndarray, resize_max: int = None, dfactor: int = 8):
    """调整图像尺寸，返回调整后的图像和新尺寸"""
    size = image.shape[:2][::-1]  # (width, height)
    scale = np.array([1.0, 1.0])

    # 第一次 resize：根据 resize_max 缩放
    if resize_max and max(size) > resize_max:
        scale = resize_max / max(size)
        size_new = tuple(int(round(x * scale)) for x in size)
        image = cv2.resize(image, size_new, interpolation=cv2.INTER_AREA)
        scale = np.array(size) / np.array(size_new)

    # 第二次 resize：确保尺寸能被 dfactor 整除
    size_new = tuple(int(x // dfactor * dfactor) for x in image.shape[:2][::-1])
    if size_new != image.shape[:2][::-1]:
        image = cv2.resize(image, size_new, interpolation=cv2.INTER_LINEAR)
        scale = np.array(size) / np.array(size_new)

    return image, size_new


def resize_image_with_scale(image: np.ndarray, scale: float = None, dfactor: int = 8):
    """调整图像尺寸，返回调整后的图像和新尺寸"""
    size = image.shape[:2][::-1]  # (width, height)

    # 第一次 resize：根据 resize_max 缩放
    if scale is not None and scale < 1.0:
        print('scale: ', scale)
        new_size = tuple(int(round(x * scale)) for x in size)
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # 第二次 resize：确保尺寸能被 dfactor 整除
    new_size = tuple(int(x // dfactor * dfactor) for x in image.shape[:2][::-1])
    if new_size != image.shape[:2][::-1]:
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    print(f'ori_size -> scaled_size: {size} -> {new_size}')

    return image, new_size


def save_image(image: np.ndarray, save_path: str):
    """保存图像到指定路径"""
    save_image_data = image.clip(0, 255).astype(np.uint8)
    cv2.imwrite(save_path, save_image_data)


def resize_to_multiple(image_data: torch.Tensor, multiple: int = 8):
    """
    将图像和掩膜缩放到指定整数倍的尺寸
    Args:
        image_data: 输入图像 (3, H, W) RGB格式
        multiple: 需要缩放的整数倍 (默认32，常用CNN下采样倍数)
    Returns:
        resized_image: 缩放后的图像 (3, H', W')
    """
    assert image_data.dim() == 3
    h, w = image_data.shape[1:]

    # 计算目标尺寸（向下取整到最近的multiple倍数）
    new_h = (h // multiple) * multiple
    new_w = (w // multiple) * multiple

    # 如果已经是整数倍则直接返回
    if new_h == h and new_w == w:
        return image_data

    # 双线性插值缩放图像
    resized_image = F.interpolate(
        image_data.unsqueeze(0),  # 添加batch维度 (1, 3, H, W)
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # 移除batch维度

    return resized_image


def resize_pil_to_multiple(
        image_data: Image.Image,
        multiple: int = 32
) -> Image.Image:
    """
    将PIL图像缩放到指定整数倍的尺寸（保持宽高比）

    Args:
        image_data: PIL格式的输入图像 (RGB模式)
        multiple: 需要缩放的整数倍 (默认32)

    Returns:
        resized_image: 缩放后的PIL图像
    """
    assert isinstance(image_data, Image.Image), "输入必须是PIL.Image"

    w, h = image_data.size

    # 计算目标尺寸（向下取整到最近的multiple倍数）
    new_w = (w // multiple) * multiple
    new_h = (h // multiple) * multiple

    # 如果已经是整数倍则直接返回
    if new_w == w and new_h == h:
        return image_data

    # 使用LANCZOS高质量缩滤波
    return image_data.resize((new_w, new_h), Image.Resampling.LANCZOS)

from skimage.filters.rank import entropy
from skimage.morphology import disk
from multiprocessing import Pool, cpu_count

import time

def compute_entropy_patch(args):
    """计算单个图像区块的熵"""
    patch, patch_size = args
    ent = entropy(patch, disk(patch_size))
    return np.mean(ent)


def is_cluttered_entropy_parallel(image, threshold=5.5, patch_size=15, n_workers=None):
    """
    并行计算局部熵判断杂乱场景
    Args:
        image: 灰度图像 (H x W)
        threshold: 熵阈值 (典型5.0-6.5)
        patch_size: 计算局部熵的窗口大小
        n_workers: 进程数（默认使用所有CPU核心）
    Returns:
        bool: 是否杂乱场景
    """
    if n_workers is None:
        n_workers = cpu_count()  # 默认使用所有CPU核心

    # 分割图像为多个区块（这里按行分割）
    height = image.shape[0]
    patch_height = height // n_workers
    patches = [
        (image[i * patch_height: (i + 1) * patch_height, :], patch_size)
        for i in range(n_workers)
    ]

    # 使用多进程并行计算
    with Pool(n_workers) as pool:
        entropy_means = pool.map(compute_entropy_patch, patches)

    avg_entropy = np.mean(entropy_means)
    print('平均熵:', avg_entropy)
    return avg_entropy > threshold

def calculate_scene_entropy2(temp_cameras):
    # 计算耗时
    start = time.perf_counter()
    gray_images = cv2.cvtColor(temp_cameras[0].image_data, cv2.COLOR_BGR2GRAY)
    is_cluttered = is_cluttered_entropy_parallel(gray_images, threshold=5.5, patch_size=15)
    end = time.perf_counter()
    print(f"是否杂乱: {is_cluttered}")
    print(f"耗时: {end - start:.4f} 秒")
    return is_cluttered



def calculate_scene_entropy(train_cam_infos):
    # 方法1：基于图像纹理熵
    gray_images = [cv2.cvtColor(plot_utils.image_pil_numpy(train_cam_info.image), cv2.COLOR_BGR2GRAY) for train_cam_info in train_cam_infos]
    entropies = [shannon_entropy(img) for img in gray_images]
    entropy = np.mean(entropies) / 10.0  # 假设归一化到[0,1]
    print('entropy: ', entropy)
    return entropy




def to_torch_cuda(image: np.ndarray, grayscale: bool = False, device: str = '') -> torch.Tensor:
    """将NumPy图像转换为PyTorch Tensor并放到CUDA上"""
    # 通道调整并归一化
    if grayscale:
        assert image.ndim == 2, f"Expected grayscale image with 2 dims, got {image.shape}"
        image = image[None]  # (1, H, W)
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = image / 255.0  # 归一化到 [0, 1]

    # 转换为 PyTorch Tensor
    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.to(device)[None]

    return image_tensor



from scene.cameras import TempCamera

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