import torch
import numpy as np
from tqdm import tqdm
from utils import graphics_utils
import torch.nn.functional as F
from itertools import combinations



def compute_dynamic_tau_depth(z_i, z_j, base_thresh=0.05, range_sensitivity=0.3):
    # 计算联合深度统计量
    z_combined = torch.cat([z_i, z_j])
    z_min = torch.quantile(z_combined, 0.02)  # 避免极端小值影响
    z_max = torch.quantile(z_combined, 0.98)  # 避免极端大值影响
    z_range = z_max - z_min

    # 动态调整策略 (近处严格，远处宽松)
    z_avg = (z_i + z_j) / 2
    normalized_depth = (z_avg - z_min) / (z_range + 1e-6)


    # 非线性阈值曲线 (sigmoid形式)
    # sigmoid_term = torch.sigmoid(sigmoid_scale * normalized_depth - sigmoid_scale / 2)
    tau_depth = base_thresh + range_sensitivity * torch.sigmoid(2 * normalized_depth - 1)

    return tau_depth


def filter_outliers_dynamic(e_ij, e_ji, z_i, z_j, switch_dynamic_filter, tau_reproj=0.1, tau_depth=3, base_thresh=0.2, range_sensitivity=0.2):
    """
    输入:
        z_i, z_j: 深度值 (n,)
        e_ij, e_ji: 重投影误差 (n,)
        tau_reproj: 重投影误差阈值（单位：像素）
        tau_depth: 深度相对差异阈值
    输出:
        mask: 内点掩码 (True=内点)
    """
    # 重投影误差检验
    reproj_mask = (e_ij <= tau_reproj) & (e_ji <= tau_reproj)

    # 深度一致性检验（相对差异）
    depth_mask = filter_outliers_dy_depth(z_i, z_j, switch_dynamic_filter, tau_depth, base_thresh, range_sensitivity)

    # 联合条件
    inlier_mask = reproj_mask & depth_mask
    return inlier_mask


def filter_outliers_dy_depth(z_i, z_j, switch_dynamic_filter=True, tau_depth=0.1, base_thresh=0.2, range_sensitivity=0.2):
    """
    改进版深度一致性过滤器
    输入:
        z_i, z_j: 深度值 (n,)
        dynamic_thresh: 是否启用动态阈值
        kwargs: 可传递base_thresh/range_sensitivity等参数
    输出:
        mask: 内点掩码 (True=内点)
    """
    if switch_dynamic_filter:
        tau_depth = compute_dynamic_tau_depth(z_i, z_j, base_thresh, range_sensitivity)
        print("tau_depth: ", tau_depth[:2].tolist())
    else:
        print("tau_depth: ", tau_depth)
    min_depth = torch.minimum(z_i, z_j)
    depth_diff = torch.abs(z_i - z_j) / (min_depth + 1e-6)
    depth_mask = (depth_diff <= tau_depth)

    return depth_mask







def blender_filter(train_cam_infos, match_data, sparse_args):
    for i in range(len(train_cam_infos) - 1):
        cam0 = train_cam_infos[i]
        name0 = cam0.image_name
        bm0 = cam0.blendermask  # 获取cam0的掩码
        image0 = cam0.image  # 获取cam0的掩码
        height0, width0 = bm0.shape  # 掩码的尺寸

        for j in range(i + 1, len(train_cam_infos)):
            cam1 = train_cam_infos[j]
            name1 = cam1.image_name
            bm1 = cam1.blendermask  # 获取cam1的掩码
            image1 = cam1.image  # 获取cam0的掩码
            height1, width1 = bm1.shape  # 掩码的尺寸

            # 获取原始匹配数据
            matches_0_to_1 = match_data[name0][name1]['points']  # (n, 2)
            matches_1_to_0 = match_data[name1][name0]['points']  # (n, 2)

            # 将归一化坐标转换为像素坐标
            uv0_x = (matches_0_to_1[:, 0] * width0).astype(int)
            uv0_y = (matches_0_to_1[:, 1] * height0).astype(int)
            uv1_x = (matches_1_to_0[:, 0] * width1).astype(int)
            uv1_y = (matches_1_to_0[:, 1] * height1).astype(int)

            # 确保坐标在图像范围内
            uv0_x = np.clip(uv0_x, 0, width0 - 1)
            uv0_y = np.clip(uv0_y, 0, height0 - 1)
            uv1_x = np.clip(uv1_x, 0, width1 - 1)
            uv1_y = np.clip(uv1_y, 0, height1 - 1)

            # 根据掩码过滤匹配点
            # 对于cam0：检查匹配点在cam0的掩码中是否为True
            valid_mask0 = bm0[uv0_y, uv0_x]

            # 对于cam1：检查匹配点在cam1的掩码中是否为True
            valid_mask1 = bm1[uv1_y, uv1_x]

            # 求交集：两个掩码都为True的点
            valid_mask = valid_mask0 & valid_mask1

            # 获取有效的匹配点索引
            valid_indices = np.where(valid_mask)[0]

            if len(valid_indices) > 0:
                # 更新match_data，只保留有效的匹配点
                match_data[name0][name1]['points'] = matches_0_to_1[valid_indices]
                match_data[name1][name0]['points'] = matches_1_to_0[valid_indices]

                print(f"Filtered {name0}-{name1}: {len(matches_0_to_1)} -> {len(valid_indices)} matches")
                # image_data0 = plot_utils.image_pil_numpy(image0)
                # image_data1 = plot_utils.image_pil_numpy(image1)
                # draw_matching2(sparse_args.matched_image_dir, image_data0, image_data1,
                #                               name0, name1, match_data[name0][name1]['points'], match_data[name1][name0]['points'],
                #                'maskFilter')
            else:
                # 如果没有有效匹配点，可以删除该匹配对或设置为空数组
                match_data[name0][name1]['points'] = np.empty((0, 2), dtype=np.float32)
                match_data[name1][name0]['points'] = np.empty((0, 2), dtype=np.float32)
                print(f"No valid matches after filtering for {name0}-{name1}")


def generate_ray_point_cloud_filtering(train_cam_infos, matched_data, sparse_args):
    """Generate point cloud from matched features across images."""

    for i, train_cam_info0 in tqdm(enumerate(train_cam_infos[:-1]), desc="generate_ray_point_cloud"):
        for train_cam_info1 in train_cam_infos[i + 1:]:

            short_image_name0 = train_cam_info0.image_name
            short_image_name1 = train_cam_info1.image_name
            points_norm0 = matched_data[short_image_name0][short_image_name1]['points']
            points_norm1 = matched_data[short_image_name1][short_image_name0]['points']


            R = torch.stack([torch.from_numpy(train_cam_info0.R), torch.from_numpy(train_cam_info1.R)])
            T = torch.stack([torch.from_numpy(train_cam_info0.T), torch.from_numpy(train_cam_info1.T)])
            K = torch.tensor(train_cam_info0.K, dtype=torch.float32)

            point_3d0, point_3d1, distance = generate_matched_points(points_norm0, points_norm1, R, T, K, train_cam_info0.width, train_cam_info0.height)

            mask = distance < 0.01
            matched_data[short_image_name0][short_image_name1]['ray_depth'] = point_3d0[:, 2]
            matched_data[short_image_name0][short_image_name1]['ray_distance'] = distance
            matched_data[short_image_name0][short_image_name1]['ray_point_cloud'] = point_3d0

            matched_data[short_image_name1][short_image_name0]['ray_depth'] = point_3d1[:, 2]
            matched_data[short_image_name1][short_image_name0]['ray_distance'] = distance
            matched_data[short_image_name1][short_image_name0]['ray_point_cloud'] = point_3d1

            a = mask.sum().item()
            print('ray filter: ', short_image_name0, short_image_name1, a)
            point_2d0 = points_norm0[mask]
            point_2d1 = points_norm1[mask]
            matched_data[short_image_name0][short_image_name1]['points'] = point_2d0
            matched_data[short_image_name1][short_image_name0]['points'] = point_2d1


def generate_pixel_rays(W, H, K):
    """Generate normalized ray directions for all pixels in an image.

    Args:
        W: Image width
        H: Image height
        K: Camera intrinsic matrix [3, 3]

    Returns:
        directions: Ray directions for each pixel [H, W, 3]
    """
    x, y = torch.meshgrid(torch.arange(W, device=K.device),
                          torch.arange(H, device=K.device),
                          indexing="xy")
    directions = torch.stack([
        (x - K[0, 2] + 0.5) / K[0, 0],
        (y - K[1, 2] + 0.5) / K[1, 1],
        torch.ones_like(x)
    ], dim=-1)
    return directions

def nearest_points_between_lines(P1, u1, P2, u2):
    """
    Compute nearest points between two lines and their distance.

    Args:
        P1 (torch.Tensor): Start point of line 1 (N, 3)
        u1 (torch.Tensor): Direction vector of line 1 (N, 3)
        P2 (torch.Tensor): Start point of line 2 (N, 3)
        u2 (torch.Tensor): Direction vector of line 2 (N, 3)

    Returns:
        tuple: (Q1, Q2, distance) where Q1 and Q2 are nearest points on each line
    """
    # Precompute dot products
    u1u1 = torch.sum(u1 * u1, dim=-1)
    u2u2 = torch.sum(u2 * u2, dim=-1)
    u1u2 = torch.sum(u1 * u2, dim=-1)
    u1P1 = torch.sum(u1 * P1, dim=-1)
    u1P2 = torch.sum(u1 * P2, dim=-1)
    u2P1 = torch.sum(u2 * P1, dim=-1)
    u2P2 = torch.sum(u2 * P2, dim=-1)

    denominator = u2u2 * u1u1 - u1u2 * u1u2 + 1e-8

    # Compute parameters
    n = (u1u2 * (u1P2 - u1P1) - u1u1 * (u2P2 - u2P1)) / denominator
    m = (u2u2 * (u1P2 - u1P1) - u1u2 * (u2P2 - u2P1)) / denominator

    # Compute nearest points
    Q1 = P1 + m.unsqueeze(-1) * u1
    Q2 = P2 + n.unsqueeze(-1) * u2

    distance = torch.norm(Q1 - Q2, dim=-1)

    return Q1, Q2, distance

def generate_matched_points(pix0, pix1, R, T, K, W, H):
    """Generate 3D points from matched pixels in two views."""
    # Convert numpy arrays to PyTorch tensors if needed
    if isinstance(pix0, np.ndarray):
        pix0 = torch.from_numpy(pix0).float()  # 确保是float类型，避免整数运算问题
    if isinstance(pix1, np.ndarray):
        pix1 = torch.from_numpy(pix1).float()

    # Generate all ray directions
    directions = generate_pixel_rays(W, H, K)

    # Get camera-to-world transforms
    RT0 = graphics_utils.getWorld2View2(R[0].numpy(), T[0].numpy())
    RT1 = graphics_utils.getWorld2View2(R[1].numpy(), T[1].numpy())
    C2W0, C2W1 = np.linalg.inv(RT0), np.linalg.inv(RT1)

    R0, R1 = torch.tensor(C2W0[:3, :3]), torch.tensor(C2W1[:3, :3])
    T0, T1 = torch.tensor(C2W0[:3, -1]), torch.tensor(C2W1[:3, -1])

    # Get pixel coordinates (now pix0/pix1 are PyTorch tensors)
    pix0_coords = (pix0 * torch.tensor([W, H], device=pix0.device)).long()
    pix1_coords = (pix1 * torch.tensor([W, H], device=pix1.device)).long()

    # Get and normalize ray directions
    dirs0 = (R0 @ directions[pix0_coords[..., 1], pix0_coords[..., 0]].unsqueeze(-1)).squeeze(-1)
    dirs1 = (R1 @ directions[pix1_coords[..., 1], pix1_coords[..., 0]].unsqueeze(-1)).squeeze(-1)
    dirs0 = F.normalize(dirs0, dim=-1)
    dirs1 = F.normalize(dirs1, dim=-1)

    # Find nearest points between rays
    Q0, Q1, dis = nearest_points_between_lines(T0, dirs0, T1, dirs1)

    return Q0.cpu().numpy(), Q1.cpu().numpy(), dis.cpu().numpy()



def filter_propagate_data_common(gaussian, min_loss_state, sparse_args):
    common_matched_data = gaussian.common_matched_data
    # vmask_dict = gaussian.vmask_dict
    cam_dict = gaussian.view_gs

    common_keys = list(common_matched_data.keys())
    for key1 in common_keys:
        # for key1 in keys:
        other_keys = list(common_matched_data[key1].keys())
        # other_keys = [key for key in keys if key != key1]
        for i, key0 in enumerate(other_keys[:-1]):
            for key2 in other_keys[i + 1:]:
                # if key1 not in list(vmask_dict[key0].keys()) or key1 not in list(vmask_dict[key2].keys()):
                #     continue

                prop_mask102 = common_matched_data[key1][key0][key2]
                prop_mask120 = common_matched_data[key1][key2][key0]

                ori_num = prop_mask102.sum().item()

                # ------------------

                z_val01 = cam_dict[key0]['match_infos'][key1]["z_val"]
                z_val21 = cam_dict[key2]['match_infos'][key1]["z_val"]

                true_list102 = torch.nonzero(prop_mask102).flatten()
                true_list120 = torch.nonzero(prop_mask120).flatten()

                if true_list102.numel() == 0 or true_list120.numel() == 0:
                    del common_matched_data[key1][key0][key2]
                    del common_matched_data[key1][key2][key0]
                    continue

                t2o_dict102 = {pos: idx.item() for pos, idx in enumerate(true_list102)}
                t2o_dict120 = {pos: idx.item() for pos, idx in enumerate(true_list120)}

                depth_mask02 = filter_outliers_dy_depth(z_i=z_val01[true_list102], z_j=z_val21[true_list120],
                                                                       switch_dynamic_filter=sparse_args.switch_dynamic_filter,
                                                                       tau_depth=sparse_args.tau_depth * 1.2,
                                                                       base_thresh=sparse_args.base_thresh * 1.5, range_sensitivity=sparse_args.range_sensitivity * 1.5)

                true_list02 = torch.nonzero(depth_mask02).flatten()
                if true_list02.numel() == 0:
                    del common_matched_data[key1][key0][key2]
                    del common_matched_data[key1][key2][key0]
                    continue

                ori_id01 = torch.tensor([t2o_dict102[id.item()] for id in true_list02],
                                        device=sparse_args.device)
                ori_id21 = torch.tensor([t2o_dict120[id.item()] for id in true_list02],
                                        device=sparse_args.device)

                new_mask01 = torch.zeros(len(prop_mask102), dtype=torch.bool).cuda()
                new_mask21 = torch.zeros(len(prop_mask120), dtype=torch.bool).cuda()

                # if true_id01.numel() != 0 or true_id21.numel() != 0:
                new_mask01[ori_id01] = True
                new_mask21[ori_id21] = True

                prop_mask102 = new_mask01
                prop_mask120 = new_mask21

                assert prop_mask102.sum().item() == prop_mask120.sum().item()

                # ------------------

                common_matched_data[key1][key0][key2] = prop_mask102
                common_matched_data[key1][key2][key0] = prop_mask120

                print("common ", key1, key0, key2, ori_num, "->", prop_mask102.sum().item())

    gaussian.common_matched_data = common_matched_data


def filter_outside(gaussian, cam_infos, min_loss_state, sparse_args):
    cam_dict = gaussian.view_gs
    all_keys = [cam_info.image_name for cam_info in cam_infos]
    keys = list(cam_dict.keys())
    outside_mask_dict = {key: {} for key in keys}

    for i, cam_info0 in enumerate(cam_infos[:-1]):
        key0 = cam_info0.image_name
        if key0 not in cam_dict:
            continue
        for cam_info1 in cam_infos[i + 1:]:
            key1 = cam_info1.image_name
            if key1 not in cam_dict or key1 not in cam_dict[key0]["match_infos"]:
                continue

            page01 = cam_dict[key0]['match_infos'][key1]
            page10 = cam_dict[key1]['match_infos'][key0]

            # 获取初始匹配点数量
            num_matches = len(page01["rays_o"])
            if num_matches == 0:
                continue

            rayso0 = page01["rays_o"]
            raysd0 = page01["rays_d"]
            zvald0 = page01["z_val"]
            world_pts0 = (rayso0 + raysd0 * zvald0).permute(1, 0)

            rayso1 = page10["rays_o"]
            raysd1 = page10["rays_d"]
            zvald1 = page10["z_val"]
            world_pts1 = (rayso1 + raysd1 * zvald1).permute(1, 0)

            keys2 = [current_key for current_key in all_keys if current_key != key0 and current_key != key1]

            # 初始化middle_mask为全True（所有点都有效）
            middle_mask = torch.ones(num_matches, dtype=torch.bool, device='cuda')

            for key2 in keys2:
                if key2 not in cam_dict:
                    continue

                intr2, w2c2 = cam_dict[key2]["intr"], cam_dict[key2]["w2c"]
                blender_mask2 = torch.tensor(cam_dict[key2]["blender_mask"]).bool().cuda()

                cam_pts_0to2 = torch.matmul(w2c2, torch.cat([world_pts0, torch.ones_like(world_pts0[:1])]))[:3]
                depth0 = cam_pts_0to2[2, :]  # 获取深度
                xyz_0to2 = torch.matmul(intr2, cam_pts_0to2)
                valid_z0 = xyz_0to2[2:] > 1e-8  # 深度有效性检查
                xy_0to2 = xyz_0to2[:2] / (xyz_0to2[2:] + 1e-8)

                cam_pts_1to2 = torch.matmul(w2c2, torch.cat([world_pts1, torch.ones_like(world_pts1[:1])]))[:3]
                depth1 = cam_pts_1to2[2, :]  # 获取深度
                xyz_1to2 = torch.matmul(intr2, cam_pts_1to2)
                valid_z1 = xyz_1to2[2:] > 1e-8  # 深度有效性检查
                xy_1to2 = xyz_1to2[:2] / (xyz_1to2[2:] + 1e-8)

                # 重新排列维度并提取坐标
                uv0_x = xy_0to2[0, :]  # x坐标
                uv0_y = xy_0to2[1, :]  # y坐标
                uv1_x = xy_1to2[0, :]  # x坐标
                uv1_y = xy_1to2[1, :]  # y坐标

                width2 = cam_dict[key2]['width']
                height2 = cam_dict[key2]['height']

                # 转换为整数索引 (torch版本)
                uv0_x_int = torch.clamp(uv0_x, 0, width2 - 1).long()
                uv0_y_int = torch.clamp(uv0_y, 0, height2 - 1).long()
                uv1_x_int = torch.clamp(uv1_x, 0, width2 - 1).long()
                uv1_y_int = torch.clamp(uv1_y, 0, height2 - 1).long()

                # 确保索引在有效范围内
                # valid_mask0 = bm0[uv0_y, uv0_x]
                valid_depth = (depth0 > 0) & (depth1 > 0)
                valid_coords0 = (uv0_x_int >= 0) & (uv0_x_int < width2) & (uv0_y_int >= 0) & (uv0_y_int < height2)
                valid_coords1 = (uv1_x_int >= 0) & (uv1_x_int < width2) & (uv1_y_int >= 0) & (uv1_y_int < height2)
                valid_coords = valid_coords0 & valid_coords1 & valid_depth & valid_z0.squeeze(0) & valid_z1.squeeze(0)

                if not torch.any(valid_coords):
                    # 如果没有有效坐标，所有点都无效
                    middle_mask[:] = False
                    continue

                # 创建当前key2的有效掩码
                current_valid_mask = torch.zeros(num_matches, dtype=torch.bool, device='cuda')

                # 只对有效坐标进行索引
                valid_coords_indices = torch.where(valid_coords)[0]
                uv0_y_valid = uv0_y_int[valid_coords_indices]
                uv0_x_valid = uv0_x_int[valid_coords_indices]
                uv1_y_valid = uv1_y_int[valid_coords_indices]
                uv1_x_valid = uv1_x_int[valid_coords_indices]

                mask0_valid = blender_mask2[uv0_y_valid, uv0_x_valid]
                mask1_valid = blender_mask2[uv1_y_valid, uv1_x_valid]

                current_valid_mask[valid_coords_indices] = mask0_valid & mask1_valid

                # 累积到middle_mask中（只有所有key2都有效的点才保留）
                middle_mask = middle_mask & current_valid_mask


                # uv0_x_norm = uv0_x_int[current_valid_mask].float() / (width2 - 1)
                # uv0_y_norm = uv0_y_int[current_valid_mask].float() / (height2 - 1)
                # uv0_coords_norm = torch.stack([uv0_x_norm, uv0_y_norm], dim=1).cpu().numpy()
                # image_data = (cam_dict[key2]['image_color'].cpu().numpy() * 255).astype(np.uint8)
                # # pixel_2d = (uv0_x_valid, uv0_y_valid) / torch.tensor([width2, height2])
                # plot_utils.draw_2d_points(f'{sparse_args.matched_image_dir}/{key2}+{key0}+{key1}.jpg', image_data, uv0_coords_norm)

            # 获取最终的有效索引
            valid_indices = torch.where(middle_mask)[0]

            if len(valid_indices) > 0:
                print(f'filter outside: {key0}-{key1}-{len(valid_indices)}')
                # 更新match_data，只保留有效的匹配点
                cam_dict[key0]['match_infos'][key1] = {
                    "z_val": page01["z_val"][valid_indices],
                    "rays_o": page01["rays_o"][valid_indices],
                    "rays_d": page01["rays_d"][valid_indices],
                    "cam_rays_d": page01["cam_rays_d"][valid_indices],
                    "color": page01["color"][valid_indices],
                    "uv": page01["uv"][valid_indices],
                    "match_pixel": page01["match_pixel"][valid_indices],
                    "blender_mask": page01["blender_mask"][valid_indices],
                    "small_transform_z_val": page01["small_transform_z_val"][valid_indices]
                }

                cam_dict[key1]['match_infos'][key0] = {
                    "z_val": page10["z_val"][valid_indices],
                    "rays_o": page10["rays_o"][valid_indices],
                    "rays_d": page10["rays_d"][valid_indices],
                    "cam_rays_d": page10["cam_rays_d"][valid_indices],
                    "color": page10["color"][valid_indices],
                    "uv": page10["uv"][valid_indices],
                    "match_pixel": page10["match_pixel"][valid_indices],
                    "blender_mask": page10["blender_mask"][valid_indices],
                    "small_transform_z_val": page10["small_transform_z_val"][valid_indices]
                }

                min_loss_state[key0][key1] = min_loss_state[key0][key1][valid_indices]
                min_loss_state[key1][key0] = min_loss_state[key1][key0][valid_indices]

                # 将mask移到CPU用于存储（如果需要）
                outside_mask_dict[key0][key1] = middle_mask
                outside_mask_dict[key1][key0] = middle_mask
            else:
                print(f'{key0}-{key1}无有效点')
                # 如果没有有效点，清空数据
                del cam_dict[key0]['match_infos'][key1]
                del cam_dict[key1]['match_infos'][key0]

    gaussian.outside_mask_dict = outside_mask_dict



def filter_outliers(gaussian, sparse_args, min_loss_state):

    common_matched_data = gaussian.common_matched_data
    cam_dict = gaussian.view_gs
    keys = list(cam_dict.keys())
    vmask_dict = {key: {} for key in keys}

    for i, key0 in enumerate(keys[:-1]):
        for key1 in keys[i + 1:]:
            page01 = cam_dict[key0]['match_infos'][key1]
            page10 = cam_dict[key1]['match_infos'][key0]

            vmask01 = vmask10 = filter_outliers_dynamic(e_ij=min_loss_state[key0][key1], e_ji=min_loss_state[key1][key0],
                                                                       z_i=page01['z_val'].squeeze(), z_j=page10['z_val'].squeeze(),
                                                                       switch_dynamic_filter = sparse_args.switch_dynamic_filter,
                                                                       tau_reproj=sparse_args.tau_reproj, tau_depth=sparse_args.tau_depth,
                                                                       base_thresh=sparse_args.base_thresh, range_sensitivity=sparse_args.range_sensitivity)
            print("switch_dynamic_filter: ", sparse_args.switch_dynamic_filter, len(min_loss_state[key0][key1]), "->", vmask01.sum().item())

            vmask_dict[key0][key1] = vmask01
            vmask_dict[key1][key0] = vmask10

            # Create new Parameter objects for the filtered tensors
            def create_filtered_param(original_param, mask):
                """Helper to create new Parameter while preserving gradients"""
                filtered_data = original_param.data[mask]  # Filter the data
                new_param = torch.nn.Parameter(filtered_data.clone())  # Create new Parameter
                if original_param.requires_grad:
                    new_param.requires_grad_(True)  # Preserve gradient requirement
                return new_param

            gaussian.view_gs[key0]['match_infos'][key1] = {
                "z_val": create_filtered_param(page01["z_val"], vmask01),
                "rays_o": page01["rays_o"][vmask01],
                "rays_d": page01["rays_d"][vmask01],
                "cam_rays_d": page01["cam_rays_d"][vmask01],
                "color": page01["color"][vmask01],
                "uv": page01["uv"][vmask01],
                "match_pixel": page01["match_pixel"][vmask01],
                "blender_mask": page01["blender_mask"][vmask01],
                "small_transform_z_val": page01["small_transform_z_val"][vmask01]
            }
            gaussian.view_gs[key1]['match_infos'][key0] = {
                "z_val": create_filtered_param(page10["z_val"], vmask10),
                "rays_o": page10["rays_o"][vmask10],
                "rays_d": page10["rays_d"][vmask10],
                "cam_rays_d": page10["cam_rays_d"][vmask10],
                "color": page10["color"][vmask10],
                "uv": page10["uv"][vmask10],
                "match_pixel": page10["match_pixel"][vmask10],
                "blender_mask": page10["blender_mask"][vmask10],
                "small_transform_z_val": page10["small_transform_z_val"][vmask10]
            }

    for key1 in keys:
        other_keys = [key for key in keys if key != key1]
        for i, key0 in enumerate(other_keys[:-1]):
            for key2 in other_keys[i + 1:]:
                reproj_error_mask01 = vmask_dict[key0][key1]
                reproj_error_mask21 = vmask_dict[key2][key1]

                prop_mask102 = common_matched_data[key1][key0][key2]
                prop_mask120 = common_matched_data[key1][key2][key0]

                ori_num = prop_mask102.sum().item()
                assert prop_mask102.sum().item() == prop_mask120.sum().item()


                true_list102 = torch.nonzero(prop_mask102).flatten()
                true_list120 = torch.nonzero(prop_mask120).flatten()

                o2t_dict102 = {idx.item(): pos for pos, idx in enumerate(true_list102)}
                o2t_dict120 = {idx.item(): pos for pos, idx in enumerate(true_list120)}
                t2o_dict102 = {pos: idx.item() for pos, idx in enumerate(true_list102)}
                t2o_dict120 = {pos: idx.item() for pos, idx in enumerate(true_list120)}

                pr_mask102 = prop_mask102 * reproj_error_mask01
                pr_mask120 = prop_mask120 * reproj_error_mask21

                # id_pr_mask102 不一定== id_pr_mask120
                id_pr_mask102 = torch.nonzero(pr_mask102).flatten()
                id_pr_mask120 = torch.nonzero(pr_mask120).flatten()

                true_id102 = torch.tensor([o2t_dict102[id.item()] for id in id_pr_mask102],
                                          device=id_pr_mask102.device)
                true_id120 = torch.tensor([o2t_dict120[id.item()] for id in id_pr_mask120],
                                          device=id_pr_mask120.device)

                intersection_true_id02 = true_id102[torch.isin(true_id102, true_id120)]

                ori_id102 = torch.tensor([t2o_dict102[id.item()] for id in intersection_true_id02],
                                          device=id_pr_mask102.device)
                ori_id120 = torch.tensor([t2o_dict120[id.item()] for id in intersection_true_id02],
                                          device=id_pr_mask120.device)


                # ------------------

                id_pr_mask01 = torch.nonzero(reproj_error_mask01).flatten()
                id_pr_mask21 = torch.nonzero(reproj_error_mask21).flatten()

                o2t_dict01 = {idx.item(): pos for pos, idx in enumerate(id_pr_mask01)}
                o2t_dict21 = {idx.item(): pos for pos, idx in enumerate(id_pr_mask21)}

                intersection_ori_id01 = ori_id102[torch.isin(ori_id102, id_pr_mask01)]
                intersection_ori_id21 = ori_id120[torch.isin(ori_id120, id_pr_mask21)]

                true_id01 = torch.tensor([o2t_dict01[id.item()] for id in intersection_ori_id01],
                                          device=id_pr_mask102.device)
                true_id21 = torch.tensor([o2t_dict21[id.item()] for id in intersection_ori_id21],
                                          device=id_pr_mask120.device)

                new_mask01 = torch.zeros(len(id_pr_mask01), dtype=torch.bool).cuda()
                new_mask21 = torch.zeros(len(id_pr_mask21), dtype=torch.bool).cuda()

                if true_id01.numel() != 0 or true_id21.numel() != 0:
                    new_mask01[true_id01] = True
                    new_mask21[true_id21] = True

                prop_mask102 = new_mask01
                prop_mask120 = new_mask21

                assert prop_mask102.sum().item() == prop_mask120.sum().item()

                # ------------------

                z_val01 = cam_dict[key0]['match_infos'][key1]["z_val"]
                z_val21 = cam_dict[key2]['match_infos'][key1]["z_val"]

                true_list102 = torch.nonzero(prop_mask102).flatten()
                true_list120 = torch.nonzero(prop_mask120).flatten()

                if true_list102.numel() == 0 or true_list120.numel() == 0:
                    del common_matched_data[key1][key0][key2]
                    del common_matched_data[key1][key2][key0]
                    continue

                t2o_dict102 = {pos: idx.item() for pos, idx in enumerate(true_list102)}
                t2o_dict120 = {pos: idx.item() for pos, idx in enumerate(true_list120)}

                depth_mask02 = filter_outliers_dy_depth(z_i=z_val01[true_list102], z_j=z_val21[true_list120],
                                                                           switch_dynamic_filter=sparse_args.switch_dynamic_filter,
                                                                           tau_depth=sparse_args.tau_depth * 1.2,
                                                                            base_thresh=sparse_args.base_thresh * 1.5, range_sensitivity=sparse_args.range_sensitivity * 1.5)

                true_list02 = torch.nonzero(depth_mask02).flatten()
                if true_list02.numel() == 0:
                    del common_matched_data[key1][key0][key2]
                    del common_matched_data[key1][key2][key0]
                    continue

                ori_id01 = torch.tensor([t2o_dict102[id.item()] for id in true_list02],
                                         device=id_pr_mask102.device)
                ori_id21 = torch.tensor([t2o_dict120[id.item()] for id in true_list02],
                                         device=id_pr_mask120.device)

                new_mask01 = torch.zeros(len(prop_mask102), dtype=torch.bool).cuda()
                new_mask21 = torch.zeros(len(prop_mask120), dtype=torch.bool).cuda()

                if true_id01.numel() != 0 or true_id21.numel() != 0:
                    new_mask01[ori_id01] = True
                    new_mask21[ori_id21] = True

                prop_mask102 = new_mask01
                prop_mask120 = new_mask21

                assert prop_mask102.sum().item() == prop_mask120.sum().item()

                # ------------------

                common_matched_data[key1][key0][key2] = prop_mask102
                common_matched_data[key1][key2][key0] = prop_mask120

                print("common ", key1, key0, key2, ori_num, "->", prop_mask102.sum().item())
                print("common ", key1, key2, key0, ori_num, "->", prop_mask120.sum().item())
    gaussian.common_matched_data = common_matched_data



def second_filter(gaussian, sparse_args):
    def count_leaf_values(d):
        count = 0
        for value in d.values():
            if isinstance(value, dict):
                count += count_leaf_values(value)
            else:
                count += 1
        return count
    cam_dict = gaussian.view_gs
    keys = list(cam_dict.keys())
    common_matched_data = gaussian.common_matched_data
    print('count_leaf_before: ', count_leaf_values(common_matched_data))

    for key1 in keys:
        if key1 not in common_matched_data:
            continue
        key1_data = common_matched_data[key1]

        # 遍历所有 (key0, key2) 组合（key0 != key2 != key1）
        for key0, key2 in combinations((k for k in keys if k != key1), 2):
            if key0 not in key1_data or key2 not in key1_data.get(key0, {}):
                continue
            prop_mask102 = key1_data[key0][key2]
            prop_mask120 = key1_data[key2][key0]

            assert prop_mask102.sum().item() == prop_mask120.sum().item()

            if prop_mask102.sum().item() < sparse_args.secondary_filtering_number:
                del key1_data[key0][key2]
                del key1_data[key2][key0]

        if not key1_data:  # 检查是否为空
            del common_matched_data[key1]

    print('count_leaf: ', count_leaf_values(common_matched_data))
    gaussian.common_matched_data = common_matched_data

