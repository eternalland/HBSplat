import cv2
import numpy as np
import torch

from utils import (warp_utils, recover_depth_scale_utils)

def virtual_view_synthesis(gaussians, scene, sparse_args, min_loss_state, viewpoint_stack):

    cam_dir = gaussians.view_gs

    recover_depth_scale_utils.gather_in_one(cam_dir, min_loss_state)

    recover_depth_scale_utils.aligned_depth_scale(sparse_args, viewpoint_stack, gaussians)

    if sparse_args.switch_small_transform:
        recover_depth_scale_utils.aligned_depth_scale_small(sparse_args, viewpoint_stack, gaussians)

    virtual_cams = warp_utils.generate_virtual_poses(sparse_args, viewpoint_stack)

    nearest_indices, scores = warp_utils.compute_important_scores(sparse_args, viewpoint_stack, virtual_cams)

    small_transform_mask = warp_utils.compute_small_transform_mask(viewpoint_stack, virtual_cams)

    if sparse_args.switch_nearest_warping:
        print("Top scores count:", sparse_args.nearest_warping_num)
        selected_indices = np.argsort(-scores)[:sparse_args.nearest_warping_num]
        virtual_cams = [virtual_cams[i] for i in selected_indices]
        nearest_indices = np.array([nearest_indices[i] for i in selected_indices])
        small_transform_mask = torch.tensor([small_transform_mask[i] for i in selected_indices], dtype=torch.bool)
    else:
        print("nearest_warping_num disabled:", len(virtual_cams))

    warp_utils.generate_virtual_cams_blend(sparse_args, viewpoint_stack, virtual_cams,
                                              nearest_indices, small_transform_mask)

    virtual_cameras = warp_utils.set_camera(virtual_cams)

    scene.setVirtualCameras(virtual_cameras)