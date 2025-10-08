
import torch





def compute_dynamic_tau_depth(z_i, z_j, base_thresh=0.05, range_sensitivity=0.3):
    sigmoid_scale = 4
    """
    动态深度阈值生成器
    输入:
        z_i, z_j: 深度值 (n,)
        base_thresh: 基础相对差异阈值 (0.05=5%)
        range_sensitivity: 深度范围敏感系数 (0-1)
    输出:
        tau_depth: 动态阈值 (n,)
    """
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

