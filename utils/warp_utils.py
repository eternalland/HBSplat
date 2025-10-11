import torch
import os
import numpy as np
from typing import NamedTuple, List
from utils import pose_utils, plot_utils, middle_pose_utils, graphics_utils, multi_view_fusion_utils
from utils.WarperPytorch import Warper
from scene.cameras import Camera, TempCamera, save_pose


def generate_virtual_poses(args, viewpoint_stack) -> List['TempCamera']:
    # input_cams = list(cam_dir.values())
    input_cams = viewpoint_stack
    print(f'输入视图数量: {len(input_cams)}')
    v_num = args.virtual_cam_num
    print(f'生成虚拟位姿数量: {v_num}')

    # 1. 生成目标虚拟位姿（所有输入视图共享同一组目标位姿）
    # bounds = np.stack([cam.bounds for cam in input_cams])
    bounds = np.stack([cam.bounds for cam in input_cams])
    # 从所有输入位姿生成目标位姿
    all_input_extrs = np.stack([cam.w2c.cpu().numpy() for cam in input_cams])

    print('scene_type : ', args.scene_type)
    target_w2cs = np.empty((0, 4, 4))
    if v_num > 0:
        target_c2ws = np.empty((0, 4, 4))
        if args.scene_type == 'fore':
            target_c2ws = pose_utils.generate_pseudo_poses_llff(all_input_extrs, bounds, n_poses=v_num)
        if args.scene_type == '360':
            target_c2ws = pose_utils.generate_ellipse_path_from_camera_infos(all_input_extrs, n_frames=v_num)
        # 转换为外参矩阵
        target_w2cs = np.stack([np.linalg.inv(pose) for pose in target_c2ws])

    if args.switch_middle_pose:
        interpolated_w2cs = middle_pose_utils.generate_virtual_poses(all_input_extrs, args.interpolate_middle_pose_num).astype(np.float32)
        target_w2cs = np.concatenate([target_w2cs, interpolated_w2cs], axis=0)

    # 保存结果
    virtual_cams = []
    camera = input_cams[0]
    for index, target_w2c in enumerate(target_w2cs):
        R = target_w2c[:3, :3].T
        T = target_w2c[:3, 3]
        K = camera.intr.cuda()
        w2c = torch.from_numpy(graphics_utils.getWorld2View(R, T)).cuda()
        height, width = camera.image_height, camera.image_width
        FovX, FovY = camera.FoVx, camera.FoVy

        # 创建虚拟相机
        cam_info = TempCamera(R=R, T=T, K=K, w2c=w2c, width=width, height=height, FovX=FovX, FovY=FovY,
                              image_data=None, image_name=None, image_path=None)
        cam_info.id = str(index)
        cam_info.is_virtual = True
        virtual_cams.append(cam_info)
    save_pose(args.virtual_camera_dir, virtual_cams)

    return virtual_cams


def compute_important_scores(sparse_args, input_cams, virtual_cams):
    input_w2cs = np.stack([graphics_utils.getWorld2View(input_cam.R, input_cam.T) for input_cam in input_cams])
    target_w2cs = np.stack([graphics_utils.getWorld2View(virtual_cam.R, virtual_cam.T) for virtual_cam in virtual_cams])

    # 确保输入是NumPy数组
    input_extrs = np.asarray(input_w2cs)
    target_poses = np.asarray(target_w2cs)

    # 提取位置和旋转
    input_pos = input_extrs[:, :3, 3]  # (N, 3)
    target_pos = target_poses[:, :3, 3]  # (M, 3)
    input_rot = input_extrs[:, :3, :3]  # (N, 3, 3)
    target_rot = target_poses[:, :3, :3]  # (M, 3, 3)

    # 计算距离矩阵 (M, N)
    dist_matrix = np.sqrt(np.sum((target_pos[:, np.newaxis] - input_pos) ** 2, axis=2))

    # 计算旋转差异矩阵 (M, N)
    rot_diff = np.einsum('mik,njk->mnij', target_rot, input_rot)  # (M,N,3,3)
    traces = np.diagonal(rot_diff, axis1=2, axis2=3).sum(axis=2)  # (M,N)
    rot_diff_matrix = np.arccos(np.clip((traces - 1) / 2, -1, 1))

    # 综合评分 (平移权重0.8，角度权重0.2)
    dist_scores = 0.7 * np.exp(-dist_matrix / np.mean(dist_matrix))
    angle_scores = 0.3 * np.exp(-rot_diff_matrix / np.mean(rot_diff_matrix))
    combined_scores = dist_scores + angle_scores

    # 旋转、平移小于阈值，指示


    # 为每个目标位姿选择top-K输入视图
    nearest_indices = np.argsort(-combined_scores, axis=1)[:, :min(sparse_args.virtual_source_num, input_extrs.shape[0])]
    print('virtual_source_num: ', sparse_args.virtual_source_num)
    # 距离分数，与所有训练视图的距离分数
    scores = np.sum(combined_scores, axis=1)


    return nearest_indices, scores



def set_camera(virtual_cams):
    virtual_cameras = []
    for virtual_cam in virtual_cams:
        virtual_camera = Camera(colmap_id=None, R=virtual_cam.R, T=virtual_cam.T,
                               FoVx=virtual_cam.FovX, FoVy=virtual_cam.FovY, near_far=None, height_in=virtual_cam.height, width_in=virtual_cam.width,
                               image=virtual_cam.image_data, gt_alpha_mask=None, dtumask=None, blendermask=None,
                               bounds=None, image_path=virtual_cam.image_path, mono_depth_map=None, mono_depth_map_ori_size=None,
                               image_name=virtual_cam.image_name, uid=id, data_device="cuda")
        virtual_camera.is_virtual = True
        virtual_camera.depth_map = virtual_cam.warped_depth
        virtual_camera.mask = virtual_cam.mask
        virtual_cameras.append(virtual_camera)
    return virtual_cameras


import sys
def generate_virtual_cams_blend(args, source_cams, target_cams, nearest_indices, small_transform_mask):
    n, k = nearest_indices.shape[:2]

    torch_source_image_data = torch.stack([source_cams[index].original_image for group in nearest_indices for index in group])  # (b, 3, h, w)
    torch_source_depth_map = torch.stack([source_cams[index].aligned_mono_depth_map.unsqueeze(0) for group in nearest_indices for index in group])  # (b, 1, h, w)
    torch_source_w2cs = torch.stack([source_cams[index].w2c for group in nearest_indices for index in group])   # (b, 4, 4)
    torch_target_w2cs = torch.stack([target_cam.w2c for target_cam in target_cams for _ in range(len(nearest_indices[0]))]) # (b, 4, 4)
    torch_source_K = torch.stack([source_cams[index].intr for group in nearest_indices for index in group]) # (b, 3, 3)
    torch_target_K = torch.stack([target_cam.K for target_cam in target_cams for _ in range(len(nearest_indices[0]))]) # (b, 3, 3)

    if args.switch_small_transform:
        torch_source_depth_map_small = torch.stack([source_cams[index].aligned_mono_depth_map_small.unsqueeze(0) for group in nearest_indices for index in group])  # (b, 1, h, w)
        torch_small_transform_mask = torch.stack([mask for mask in small_transform_mask for _ in range(len(nearest_indices[0]))]) # (b, 3, 3)
        torch_source_depth_map[torch_small_transform_mask] = torch_source_depth_map_small[torch_small_transform_mask]
        print('small_transform_mask, ', small_transform_mask.sum().item())


    # 3. 初始化Warper和结果存储
    warper = Warper()

    # 多视图前向扭曲（逐个处理）
    '''
    warped_images=tensor(b, 3, h, w), valid_masks=tensor(b, 1, h, w), warped_depths=tensor(b, 1, h, w)
    '''

    if args.warping_mode == 'forward_warping':
        print("warper.forward_warp")
        warped_images, valid_masks, warped_depths, flow12s = warper.forward_warp(
            frame1=torch_source_image_data,  # (b, 3, h, w)
            mask1=None,                 # (b, 1, h, w)
            depth1=torch_source_depth_map,  # (b, 1, h, w)
            transformation1=torch_source_w2cs,  # (b, 4, 4)
            transformation2=torch_target_w2cs,  # (b, 4, 4)
            intrinsic1=torch_source_K,  # (b, 3, 3)
            intrinsic2=torch_target_K  # (b, 3, 3)
        )
    else:
        print("warper.bidirectional_warp")
        warped_images, warped_depths, valid_masks = warper.bidirectional_warp(
            img1=torch_source_image_data,  # (b, 3, h, w)
            depth_map1=torch_source_depth_map,  # (b, 1, h, w)
            w2c1=torch_source_w2cs,  # (b, 4, 4)
            w2c2=torch_target_w2cs,  # (b, 4, 4)
            K1=torch_source_K,  # (b, 3, 3)
            K2=torch_target_K  # (b, 3, 3)
        )

    assert len(warped_images) == len(target_cams) * k
    os.makedirs(warped_dir := os.path.join(args.virtual_camera_dir, "warped"), exist_ok=True)
    os.makedirs(depths_dir := os.path.join(args.virtual_camera_dir, "depths"), exist_ok=True)
    os.makedirs(masks_dir := os.path.join(args.virtual_camera_dir, "masks"), exist_ok=True)
    os.makedirs(fused_warped_dir := os.path.join(args.virtual_camera_dir, "fused_warped"), exist_ok=True)
    os.makedirs(fused_depths_dir := os.path.join(args.virtual_camera_dir, "fused_depths"), exist_ok=True)
    os.makedirs(fused_masks_dir := os.path.join(args.virtual_camera_dir, "fused_masks"), exist_ok=True)

    fused_warped_images, fused_depth_maps, fused_valid_masks = multi_view_fusion_utils.multi_view_fusion(args, warped_images, warped_depths, valid_masks, k)

    for index in range(len(target_cams)):
        warped_image_name = str(index) + "_image.jpg"
        save_path = str(os.path.join(fused_warped_dir, warped_image_name))

        if args.switch_intermediate_result:
            plot_utils.save_image(fused_warped_images[index], save_path)
            plot_utils.save_depth_map(fused_depth_maps[index], os.path.join(fused_depths_dir, f"{index}_depth.jpg"))
            plot_utils.save_image(fused_valid_masks[index], os.path.join(fused_masks_dir, f"{index}_mask.jpg"))

        target_cams[index].image_data=fused_warped_images[index]
        target_cams[index].image_name=warped_image_name
        target_cams[index].image_path=save_path
        target_cams[index].warped_depth = fused_depth_maps[index]
        target_cams[index].mask = fused_valid_masks[index]

    if args.switch_intermediate_result:
        for index in range(len(warped_images)):
            # 保存结果
            plot_utils.save_image(warped_images[index], os.path.join(warped_dir, f"{index}_image.jpg"))
            plot_utils.save_depth_map(warped_depths[index], os.path.join(depths_dir, f"{index}_depth.jpg"))
            # depth_map = vis_depth(warped_depths[index].detach().squeeze(0).cpu().numpy())
            # cv2.imwrite(os.path.join(depths_dir, f"{index}_vis.jpg"), depth_map)
            plot_utils.save_image(valid_masks[index], os.path.join(masks_dir, f"{index}_mask.jpg"))
            np.save(os.path.join(depths_dir, f"{index}_np.npy"), np.array(warped_depths[index].cpu().numpy()))


def compute_small_transform_mask(input_cams, virtual_cams,
                                 rotation_factor=1, translation_factor=1):

    # 提取输入和虚拟相机的R和T（确保是PyTorch张量）
    input_w2cs = np.stack([graphics_utils.getWorld2View(input_cam.R, input_cam.T) for input_cam in input_cams])
    target_w2cs = np.stack([graphics_utils.getWorld2View(virtual_cam.R, virtual_cam.T) for virtual_cam in virtual_cams])

    # 提取位置和旋转
    input_rot = input_w2cs[:, :3, :3]  # (N, 3, 3)
    input_pos = input_w2cs[:, :3, 3]  # (N, 3)
    target_rot = target_w2cs[:, :3, :3]  # (M, 3, 3)
    target_pos = target_w2cs[:, :3, 3]  # (M, 3)

    input_R = torch.from_numpy(input_rot).cuda()   # (B, 3, 3)
    input_T = torch.from_numpy(input_pos).cuda() # (B, 3)
    virtual_R = torch.from_numpy(target_rot).cuda()  # (M, 3, 3)
    virtual_T = torch.from_numpy(target_pos).cuda() # (M, 3)

    # 计算输入相机之间的平均旋转和平移差异（用于动态阈值）
    with torch.no_grad():
        # 1. 计算输入相机之间的旋转差异（基于R的迹）
        input_rot_diff = torch.einsum('bik,bjk->bij', input_R, input_R.transpose(1, 2))  # (B,3,3)
        traces = input_rot_diff.diagonal(dim1=1, dim2=2).sum(dim=1)  # (B,)
        mean_rot_diff = torch.acos(torch.clamp((traces - 1) / 2, min=-1, max=1)).mean().item()

        # 2. 计算输入相机之间的平均平移距离
        input_pos = input_T  # (B, 3)
        dist_matrix = torch.cdist(input_pos, input_pos, p=2)  # (B, B)
        mean_translation = dist_matrix.mean().item()

    # 动态阈值：基于输入数据的统计
    max_rotation_rad = rotation_factor * mean_rot_diff
    max_translation = translation_factor * mean_translation

    # 计算每个虚拟视点与最近参考视图的差异
    mask = []
    for i in range(len(virtual_cams)):
        # 计算平移距离（取最近的参考视图）
        dist = torch.norm(virtual_T[i] - input_T.squeeze(-1), dim=1)  # (B,)
        min_dist = dist.min().item()

        # 计算旋转差异（与最近参考视图的旋转矩阵迹）
        nearest_input_idx = dist.argmin()
        rot_diff = input_R[nearest_input_idx].T @ virtual_R[i]  # (3,3)
        trace = torch.trace(rot_diff)
        angle_diff = torch.acos(torch.clamp((trace - 1) / 2, min=-1, max=1)).item()

        # 判断是否满足阈值条件
        if angle_diff < max_rotation_rad and min_dist < max_translation:
            mask.append(True)
        else:
            mask.append(False)

    return torch.tensor(mask, dtype=torch.bool)



def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)

import matplotlib.cm as cm

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

    return np.uint8(colorized[..., ::-1] * 255)