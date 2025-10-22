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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import random
import torch.nn as nn
import numpy as np
from torchmetrics.functional.regression import pearson_corrcoef
import torch.nn as nn
import kornia
import math


def get_smooth_loss(depth, guide=None):
    grad_disp_x = torch.abs(depth[:, :-1] - depth[:, 1:])
    grad_disp_y = torch.abs(depth[:-1, :] - depth[1:, :])

    if guide is None:
        guide = torch.ones_like(depth).detach()

    if len(guide.shape)==3:
        grad_img_x = torch.abs(guide[:, :, :-1] - guide[:, :, 1:]).mean(dim=0)
        grad_img_y = torch.abs(guide[:, :-1, :] - guide[:, 1:, :]).mean(dim=0)
    else:
        grad_img_x = torch.abs(guide[:, :-1] - guide[:, 1:])
        grad_img_y = torch.abs(guide[:-1, :] - guide[1:, :])

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    smooth_loss = grad_disp_x.mean() + grad_disp_y.mean()

    return smooth_loss

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    if mask is not None:
        if len(mask.shape)==2:
            mask = mask.unsqueeze(0)
        mask = F.conv2d(mask, window[:1], padding=window_size // 2, groups=1)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask is not None:
        ssim_map = ssim_map * mask

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def margin_l2_loss(network_output, gt, mask_patches, margin, return_mask=False):
    network_output = network_output[mask_patches]
    gt = gt[mask_patches]
    mask = (network_output - gt).abs() > margin
    if not return_mask:
        return ((network_output - gt)[mask] ** 2).mean()
    else:
        return ((network_output - gt)[mask] ** 2).mean(), mask

def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2*torch.std(input.reshape(-1)))

def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0,2,1).view(-1, 1*patch_size*patch_size)
    return patches

def patch_norm_mse_loss(input, target, fore_mask, patch_size, margin=0.2, return_mask=False):
    input_patches = normalize(patchify(input, patch_size))
    target_patches = normalize(patchify(target, patch_size))
    mask_patches = patchify(fore_mask, patch_size).sum(dim=1) < (patch_size*patch_size / 3)
    return margin_l2_loss(input_patches, target_patches, mask_patches, margin, return_mask)




class SSIM(nn.Module):
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(5, 1)
        self.mu_y_pool = nn.AvgPool2d(5, 1)
        self.sig_x_pool = nn.AvgPool2d(5, 1)
        self.sig_y_pool = nn.AvgPool2d(5, 1)
        self.sig_xy_pool = nn.AvgPool2d(5, 1)
        self.mask_pool = nn.AvgPool2d(5, 1)
        self.refl = nn.ReflectionPad2d(2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask=None):
        x = self.refl(x)
        y = self.refl(y)
        if mask is not None:
            mask = self.refl(mask)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        if mask is not None:
            SSIM_mask = self.mask_pool(mask)
            output = SSIM_mask * torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        else:
            output = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return output

def get_pixel_loss(image, gt_image):

    l1 = (image - gt_image).abs().mean(dim=0)

    ssim_func = SSIM()

    ssim_l = ssim_func(image.unsqueeze(0), gt_image.unsqueeze(0)).squeeze(0)

    pl = l1 * 0.5 + ssim_l.mean(dim=0) * 0.5

    return pl



def tv(inp):

    dx  = inp[:, :, :-1] - inp[:, :, 1:]
    dy  = inp[:, :-1, :] - inp[:, 1:, :]
    loss_smooth_ren = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))

    return loss_smooth_ren

def second_order_tv(ren_disparity, gamma):
    temp = ren_disparity.contiguous()
    loss_smooth_ren = (gamma) * tv(temp) + (1 - gamma) * tv(ren_disparity)

    return loss_smooth_ren


def composite_loss(image, gt_image, mask, alpha=0.8):
    # Add batch dimension if needed (from [C,H,W] to [1,C,H,W])
    if image.dim() == 3:
        image = image.unsqueeze(0)
        gt_image = gt_image.unsqueeze(0)
        mask = mask.unsqueeze(0)

    # Structure-preserving L1 loss (computed in gradient domain)
    grad_x = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1])
    grad_y = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :])
    gt_grad_x = torch.abs(gt_image[:, :, :, 1:] - gt_image[:, :, :, :-1])
    gt_grad_y = torch.abs(gt_image[:, :, 1:, :] - gt_image[:, :, :-1, :])

    # Make sure mask matches the gradient dimensions
    mask_x = mask[:, :, :, 1:]
    mask_y = mask[:, :, 1:, :]

    gradient_loss = (l1_loss(grad_x * mask_x, gt_grad_x * mask_x) +
                     l1_loss(grad_y * mask_y, gt_grad_y * mask_y)) / 2

    # Improved SSIM calculation (excluding invalid regions)
    ssim_loss = 1 - ssim(image * mask, gt_image * mask, size_average=False)

    return alpha * gradient_loss + (1 - alpha) * ssim_loss



# Replace original Pearson loss
def adaptive_depth_loss(rendered_depth, virtual_depth):
    # Compute depth gradient as confidence
    depth_grad = torch.abs(sobel_filter(virtual_depth))
    conf = torch.exp(-depth_grad * 10).squeeze(0)  # High confidence in smooth regions

    # Weighted Pearson loss
    valid = (virtual_depth > 0)
    if valid.sum() > 10:  # Ensure enough valid points
        corr = pearson_corrcoef(rendered_depth[valid], -virtual_depth[valid])
        # return corr
        return (1 - corr.item()) * conf[valid].mean().item()
    return 0.0



# Create Sobel filter
def sobel_filter(image: torch.Tensor) -> torch.Tensor:
    """
    Args:
        image: (B, C, H, W) input image
    Returns:
        (B, C, H, W) gradient magnitude
    """
    # Compute x/y direction gradients
    image = image.unsqueeze(0)
    grad_x = kornia.filters.sobel(image)

    # Merge gradients
    return grad_x




def linear_ramp(n, m, p, q, current_iter):
    if current_iter < n:
        return p
    elif current_iter <= m:
        return p + (q - p) * (current_iter - n) / (m - n)
    else:
        return q

def exponential_ramp(n, m, p, q, current_iter, k=5):
    if current_iter < n:
        return p
    elif current_iter <= m:
        t = (current_iter - n) / (m - n)  # Normalize to [0, 1]
        return p + (q - p) * (1 - math.exp(-k * t))
    else:
        return q
