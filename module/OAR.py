import os.path

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from diffusers import StableDiffusionInpaintPipeline
from simple_lama_inpainting import SimpleLama
import torch
from PIL import Image
from skimage.segmentation import active_contour
from skimage.feature import local_binary_pattern
import sys
from utils import plot_utils



def inpainting2(select_pipe, rgb_image, depth_map, mask, simple_lama1):
    simple_lama = None
    pipe = None
    method = 'NS'
    radius = 3

    if select_pipe == 'simple_lama':
        print('simple_lama used for inpainting')
        simple_lama = simple_lama1
    elif select_pipe == 'diffusion_inpaint':
        print('StableDiffusionInpaintPipeline used for inpainting')
        # pipe = StableDiffusionInpaintPipeline.from_pretrained(
        #     # "stabilityai/stable-diffusion-2-inpainting",
        #     "/home/mayu/thesis/inpainting",
        #     torch_dtype=torch.float16,
        # )
        # pipe.to("cuda")
    else:
        print('cv2 used for inpainting: NS')
        method = 'NS'
        radius = 3

    if select_pipe == 'simple_lama' or select_pipe == 'diffusion_inpaint':
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        depth_map = (depth_map * 255.).astype(np.uint8)

        rgb_image = Image.fromarray(rgb_image)
        depth_map = Image.fromarray(depth_map)
        depth_map_rgb = Image.merge("RGB", (depth_map, depth_map, depth_map))

        inpainting_mask = Image.fromarray(mask)

        if select_pipe == 'simple_lama':
            inpainted_image = simple_lama(image=rgb_image, mask=inpainting_mask)
            inpainted_depth = simple_lama(image=depth_map_rgb, mask=inpainting_mask)
        else:
            inpainted_image = pipe(
                image=rgb_image,
                mask_image=inpainting_mask,
                prompt="Complete the missing part of the mask based on the surrounding information of the mask",
                height=752,
                width=1008
            ).images[0]
        inpainted_image = np.array(inpainted_image)
        inpainted_depth = np.array(inpainted_depth)
        inpainted_depth = inpainted_depth[:, :, 0]
        inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_RGB2BGR)
    else:
        if method.upper() == 'TELEA':
            flags = cv2.INPAINT_TELEA
        else:  # method.upper() == 'NS':
            flags = cv2.INPAINT_NS
        inpainted_image = cv2.inpaint(rgb_image, mask, radius, flags)

    return inpainted_image, inpainted_depth



def get_edge_boundaries(depth_edges, mode='min_rect', padding=5, min_area=100):

    contours, _ = cv2.findContours(depth_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundaries = []

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        if mode == 'min_rect':  # Minimum bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            boundaries.append((
                [x - padding, y - padding, x + w + padding, y + h + padding],
                'rect'
            ))

        elif mode == 'rotated_rect':  # Rotated rectangle
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            boundaries.append((
                np.int0(box),
                'rotated_rect'
            ))

        elif mode == 'convex':  # Convex hull
            hull = cv2.convexHull(cnt)
            boundaries.append((
                hull,
                'convex'
            ))

        elif mode == 'multi_rect':  # Multiple independent rectangles (split large contours)
            if cv2.contourArea(cnt) > min_area * 10:  # Split large contours
                mask = np.zeros_like(depth_edges)
                cv2.drawContours(mask, [cnt], -1, 1, -1)
                kernel = np.ones((15, 15), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                sub_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for sub_cnt in sub_contours:
                    x, y, w, h = cv2.boundingRect(sub_cnt)
                    boundaries.append((
                        [x - padding, y - padding, x + w + padding, y + h + padding],
                        'rect'
                    ))
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                boundaries.append((
                    [x - padding, y - padding, x + w + padding, y + h + padding],
                    'rect'
                ))

    return boundaries

def generate_local_foreground_mask(foreground_mask, depth_edges, padding=10, min_area=50):
    # 1. Get multiple rectangle bounding boxes
    boundaries = get_edge_boundaries(depth_edges, mode='min_rect', padding=padding, min_area=min_area)
    rect_boundaries = [bbox for bbox, btype in boundaries if btype == 'rect']

    # 2. Initialize local mask
    local_foreground_mask = np.zeros_like(foreground_mask)

    # 3. Extract foreground regions for each rectangle box
    for (x1, y1, x2, y2) in rect_boundaries:
        # Crop foreground mask within rectangle region
        rect_mask = foreground_mask[y1:y2, x1:x2]
        local_foreground_mask[y1:y2, x1:x2] = np.logical_or(
            local_foreground_mask[y1:y2, x1:x2],
            rect_mask
        )

    return local_foreground_mask, rect_boundaries

def get_foreground_mask_from_edges(planes, depth_edges):

    edge_pixels = np.argwhere(depth_edges == 1)  # Edge pixel coordinates
    plane_votes = np.zeros(len(planes), dtype=int)

    for y, x in edge_pixels:
        for plane_idx, plane in enumerate(planes):
            if plane[y, x] == 1:
                plane_votes[plane_idx] += 1
                break

    foreground_plane_idx = np.argmax(plane_votes)
    foreground_mask = planes[foreground_plane_idx].copy()

    return foreground_mask, foreground_plane_idx

def detect_depth_edges(depth_map, gradient_threshold=0.1, blur_kernel=5):

    depth_blur = cv2.GaussianBlur(depth_map, (blur_kernel, blur_kernel), 0)

    # Compute gradient magnitude (Sobel operator)
    grad_x = cv2.Sobel(depth_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_blur, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Normalize gradient and apply threshold
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min())
    edges = (grad_mag > gradient_threshold).astype(np.uint8) * 255

    return edges

def generate_depth_planes(depth_map, n_planes=5, mode='uniform'):

    planes = []
    if mode == 'uniform':
        thresholds = np.linspace(0, 1, n_planes + 1)  # Include 0 and 1
    elif mode == 'percentile':
        thresholds = np.percentile(depth_map, np.linspace(0, 100, n_planes + 1))

    # kernel = np.ones((5, 5), np.uint8)
    for i in range(n_planes):
        lower, upper = thresholds[i], thresholds[i + 1]
        mask = (depth_map >= lower) & (depth_map < upper)

        # mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
        planes.append(mask.astype(np.uint8))

    return planes, thresholds







def fill_holes_by_closing(mask, kernel_size=3):
    """
    Fill small holes using morphological closing operation
    Args:
        mask: Binary mask (0 or 1)
        kernel_size: Kernel size (odd number)
    Returns:
        filled_mask: Repaired mask
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return filled_mask








def occlusion(args, cam_infos):
    simple_lama = SimpleLama()
    for cam_info in cam_infos:
        image_name = cam_info.image_name
        rgb_image = cam_info.original_image
        depth_map = cam_info.mono_depth_map

        rgb_image = (rgb_image.cpu().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        depth_map = depth_map.cpu().numpy()

        n_planes = 6
        planes, thresholds = generate_depth_planes(depth_map, n_planes, mode='uniform')

        depth_edges = detect_depth_edges(depth_map, gradient_threshold=0.1)

        foreground_mask, foreground_plane_idx = get_foreground_mask_from_edges(planes, depth_edges)

        threshold = thresholds[foreground_plane_idx + 1]

        local_foreground_mask, rects = generate_local_foreground_mask(foreground_mask, depth_edges, padding=5, min_area=1000)

        local_foreground_mask = (local_foreground_mask * 255).astype(np.uint8) if local_foreground_mask.max() <= 1 else local_foreground_mask
        local_foreground_mask = fill_holes_by_closing(local_foreground_mask, kernel_size=5)

        kernel = np.ones((int(args.test_dilate), int(args.test_dilate)), np.uint8)
        local_foreground_mask = cv2.dilate(local_foreground_mask, kernel, iterations=1)

        cv2.imwrite(f'{args.occlusion_image_dir}/{image_name}_local_foreground_mask.png', local_foreground_mask)

        inpainted_image, inpainted_depth = inpainting2(args.occ_select_pipe, rgb_image, depth_map, local_foreground_mask, simple_lama)

        inpainted_image = cv2.resize(inpainted_image, rgb_image.shape[:2][::-1])
        inpainted_depth = cv2.resize(inpainted_depth, depth_map.shape[:2][::-1])

        if args.switch_intermediate_result:
            output_path = os.path.join(args.occlusion_image_dir, "{}.png".format(image_name))
            output_path2 = os.path.join(args.occlusion_image_dir, "{}_depth.png".format(image_name))
            cv2.imwrite(output_path, inpainted_image)
            cv2.imwrite(output_path2, inpainted_depth)

        inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)
        inpainted_image = torch.from_numpy(inpainted_image.transpose(2, 0, 1) / 255.).float().cuda()
        inpainted_depth = torch.from_numpy(inpainted_depth / 255.).float().cuda()
        depth_min = torch.min(inpainted_depth)
        depth_max = torch.max(inpainted_depth)
        if depth_max != depth_min:
            inpainted_depth = (inpainted_depth - depth_min) / (depth_max - depth_min)
        local_foreground_mask = torch.from_numpy(local_foreground_mask / 255.).int().cuda()

        cam_info.occlusion_image = inpainted_image
        cam_info.occlusion_depth = inpainted_depth
        cam_info.occlusion_mask = local_foreground_mask
        cam_info.occlusion_depth_threshold = threshold


