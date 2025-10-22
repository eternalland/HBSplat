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

import os

import torch
from random import randint

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, SparseParams
from module import (OAR, HLDE, VVS)
from utils import loss_utils
import cv2
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, sparse_args, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, sparse_args)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()


    min_loss_state = HLDE.initial_depth_estimation(gaussians, sparse_args, iter_start, iter_end, viewpoint_stack)

    if 0b010 & sparse_args.run_module:
        VVS.virtual_view_synthesis(gaussians, scene, sparse_args, min_loss_state, viewpoint_stack)

    if 0b001 & sparse_args.run_module:
        OAR.occlusion(sparse_args, viewpoint_stack)

    print("\n[Init Stage] Saving Gaussians")
    scene.save_init(2000)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    gaussians.create_from_pcd()
    gaussians.training_setup(opt)

    if 0b001 & sparse_args.run_module:
        whole_indices = gaussians.get_occ_mask(viewpoint_stack)
        gaussians.mask_occlusion2(sparse_args, whole_indices, opt)

    viewpoint_stack = None
    unseen_viewpoint_stack = None
    ema_loss_for_log = 0.0
    render_match_loss = 0.0
    render_propagate_loss = 0.0
    loss_smooth_ren = 0.0
    image_loss = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if 0b010 & sparse_args.run_module:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            if not unseen_viewpoint_stack:
                unseen_viewpoint_stack = scene.getVirtualCameras().copy()

            if 0b001 & sparse_args.run_module:
                if iteration == sparse_args.occlusion_interval + 1:
                    print(f"Enable both 010 and 001\nocclusion_interval: {sparse_args.occlusion_interval}\nreg_interval: {sparse_args.reg_interval}")
                if iteration > sparse_args.occlusion_interval and iteration % sparse_args.reg_interval == 0:
                    viewpoint_cam = unseen_viewpoint_stack.pop(randint(0, len(unseen_viewpoint_stack) - 1))
                else:
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            else:
                if iteration == sparse_args.reg_interval:
                    print(f"Enable 010, without 001\nstart_virtual_render: {sparse_args.start_virtual_render}\nreg_interval: {sparse_args.reg_interval}")
                if iteration > sparse_args.start_virtual_render and iteration % sparse_args.reg_interval == 0:
                    viewpoint_cam = unseen_viewpoint_stack.pop(randint(0, len(unseen_viewpoint_stack) - 1))
                else:
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        else:
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if 0b001 & sparse_args.run_module and iteration == sparse_args.occlusion_interval:
            print('restore_all-----', sparse_args.run_module)
            gaussians.restore_all(opt)

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        if 0b001 & sparse_args.run_module:
            if iteration >= sparse_args.occlusion_interval:
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image = viewpoint_cam.occlusion_image.cuda()
        else:
            gt_image = viewpoint_cam.original_image.cuda()

        bg_mask = None
        if "dtu" in args.source_path:
            if 'scan110' not in args.source_path:
                bg_mask = (gt_image.max(0, keepdim=True).values < 30 / 255)
            else:
                bg_mask = (gt_image.max(0, keepdim=True).values < 15 / 255)
            bg_mask_clone = bg_mask.clone()
            for i in range(1, 50):
                bg_mask[:, i:] *= bg_mask_clone[:, :-i]
            gt_image[bg_mask.repeat(3, 1, 1)] = 0.

        loss = 0.

        if not viewpoint_cam.is_virtual:
            Ll1 = l1_loss(image, gt_image)
            image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss += image_loss

            if 'blender' not in sparse_args.dataset:
                render_match_loss = gaussians.get_matchloss_from_renderdepth(viewpoint_cam, render_pkg["rendered_depth"])
                loss += render_match_loss * 0.3

        else:  # unseen views
            mask = viewpoint_cam.mask.squeeze().cuda().bool()

            if sparse_args.dataset == 'DTU':
                image[:, ~mask] = 1.0
                gt_image[:, ~mask] = 1.0

            # # Improvement: soften constraints on mask edge regions
            def get_soft_mask(mask, kernel_size=(5, 5), sigma=1.0):
                blurred_mask = cv2.GaussianBlur(mask, kernel_size, sigma)
                return torch.from_numpy(blurred_mask).unsqueeze(0).cuda()  # (1,H,W)

            soft_mask = get_soft_mask(mask.float().cpu().numpy())  # Values in [0,1] range

            image_masked = image * soft_mask
            gt_image_masked = gt_image * soft_mask

            unseen_v_loss = loss_utils.composite_loss(image_masked, gt_image_masked, soft_mask)
            render_weight = loss_utils.linear_ramp(1000, 2000, 0.8, 0.3, iteration)
            loss += render_weight * unseen_v_loss
            depth_map = viewpoint_cam.depth_map * soft_mask
            rendered_depth = render_pkg["rendered_depth"] * soft_mask
            # Depth loss (improved version)
            depth_loss = loss_utils.adaptive_depth_loss(rendered_depth, depth_map)
            loss += 0.1 * depth_loss

        if 0b100 & sparse_args.run_module and (viewpoint_cam.mono_depth_map is not None) and 'blender' not in sparse_args.dataset:
            if iteration == sparse_args.start_render_propagate_iteration:
                print("\nEnable propagate_from_renderdepth:", sparse_args.render_propagate_weight)
            if iteration >= sparse_args.start_render_propagate_iteration:
                render_propagate_loss = gaussians.get_propagate_from_renderdepth(viewpoint_cam, render_pkg["rendered_depth"])
                loss += sparse_args.render_propagate_weight * render_propagate_loss

                # TV regularization
                ren_disp = 1 / (render_pkg["rendered_depth"] + 1)
                loss_smooth_ren = loss_utils.second_order_tv(ren_disp, iteration / opt.iterations)

                loss += 0.2 * loss_smooth_ren

        if "dtu" in args.source_path:
            loss += render_pkg["rendered_alpha"][bg_mask].mean()

        loss.backward()

        iter_end.record()

        with (torch.no_grad()):
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if 0b100 & sparse_args.run_module and iteration % 10 == 0:
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                progress_bar.set_postfix({
                    "image_l": f"{image_loss:.4f}",
                    "mat_l": f"{render_match_loss:.4f}",
                    "mat_a": f"{0.3 * render_match_loss:.4f}",
                    "pro_l": f"\n{render_propagate_loss:.4f}",
                    "pro_a": f"{sparse_args.render_propagate_weight * render_propagate_loss:.4f}",
                    "sm_l": f"\n{loss_smooth_ren:.4f}",
                    "sm_a": f"{0.1 * loss_smooth_ren:.4f}",

                    "Total": f"{loss:.4f}",
                })

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                gaussians.optimizer_bg.step()
                gaussians.optimizer_bg.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    if viewpoint.dtumask is not None:
                        mask = viewpoint.dtumask > 0
                        l1_test += l1_loss(image[:, mask], gt_image[:, mask]).mean().double()
                        psnr_test += psnr(image[:, mask], gt_image[:, mask]).mean().double()
                    else:
                        l1_test += l1_loss(image, gt_image).mean().double()
                        psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


import time


def format_time(seconds):
    """Convert seconds to minutes and seconds format"""
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}m {remaining_seconds:.4f}s"


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    sp = SparseParams(parser, is_training=True)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 1_000, 1_500, 2_000, 2_500, 3_000, 4_000, 5_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[500, 1_000, 1_500, 2_000, 2_500, 3_000, 4_000, 5_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    sp.update_from_args(args)

    sp.device = device

    print("------------Optimizing ", args.model_path)
    sp.create_dir(args.model_path)
    sp.print_log()

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), sp
             , args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    total_time = time.time() - start_time
    print(f'[Time] training time took: {total_time:.4f} seconds | {format_time(total_time)}')
    # All done
    print("\nTraining complete.")