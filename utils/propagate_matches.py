import torch
import numpy as np
from tqdm import tqdm


from scipy.spatial import cKDTree

def mutual_nearest_neighbor(points0, points1, threshold=0.001):
    """
    使用 Mutual Nearest Neighbor (MNN) 算法对齐两组点集。

    Args:
        points0 (np.ndarray): 第一组点集，形状为 (n, 2)，归一化坐标。
        points1 (np.ndarray): 第二组点集，形状为 (m, 2)，归一化坐标。
        threshold (float): 距离阈值（归一化坐标空间），默认 1e-3。

    Returns:
        tuple: (matched_points0, matched_points1)
            - matched_points0: points0 中匹配的点，形状为 (k, 2)。
            - matched_points1: points1 中匹配的点，形状为 (k, 2)。
    """
    # 转换为 numpy 数组
    # {xy∈img1,img1|img1-img0,img1-img0} 缩写 {xy∈img1,img1|img1-img0}
    # {xy∈img1,img1|img1-img2,img1-img2}
    points0 = np.array(points0)
    points1 = np.array(points1)

    # 构建 KD 树
    tree0 = cKDTree(points0)
    tree1 = cKDTree(points1)

    # 从 points1 到 points0 的最近邻
    # 对应0索引
    # {d,img1|,img1-img2|img1-img0<-img1-img2}
    # {idx∈img1,img1|img1-img0,img1-img2|img1-img0<-img1-img2}
    distances_1to0, indices_1to0 = tree0.query(points1, k=1)
    mask_1to0 = distances_1to0 < threshold

    # 从 points0 到 points1 的最近邻
    # {d,img1|,img1-img0|img1-img0->img1-img2}
    # {idx∈img1,img1|img1-img2,img1-img0|img1-img0->img1-img2}
    distances_0to1, indices_0to1 = tree1.query(points0, k=1)
    mask_0to1 = distances_0to1 < threshold

    # 互近邻检查
    # {f,img1|,img1-img2}
    mutual_mask = np.zeros(len(points1), dtype=bool)
    for i in range(len(points1)):
        # mask_0to1[i] = true/false
        if mask_1to0[i]:  # points0[i] 到 points1 的最近邻满足阈值
            j = indices_1to0[i]  # points0[i] 在 points1 中的最近邻
            if mask_0to1[j] and indices_0to1[j] == i:  # points1[j] 的最近邻是 points0[i]
                # {tf,img1|,img1-img2|img1-img0<->img1-img2}
                mutual_mask[i] = True

    # # 提取互近邻匹配点
    # # 过滤
    # matched_points0 = points0[mutual_mask]
    # matched_indices = indices_0to1[mutual_mask]
    # # 过滤
    # matched_points1 = points1[matched_indices]

    # 提取互近邻匹配点及其索引
    # 保留索引
    # mutual_mask：img1的每个像素对应的img0的最近邻的坐标
    # {idx∈img1,|img1-img2,|img1-img0<->img1-img2}
    indices1 = np.where(mutual_mask)[0]
    # 保留索引
    # indices_1to0={idx∈img1,img1|img1-img0,img1-img2|img1-img0<-img1-img2}
    # mutual_mask={tf,img1|,img1-img2|img1-img0<->img1-img2}
    # {idx∈img1,|img1-img0,|img1-img0<->img1-img2}
    indices0 = indices_1to0[mutual_mask]

    # points0={xy∈img1,img1|img1-img0}
    # indices0={idx∈img1,|img1-img0,|img1-img0<->img1-img2}
    # {xy∈img1,|img1-img0,|img1-img0<->img1-img2}
    matched_points0 = points0[indices0]
    # {xy∈img1,|img1-img2,|img1-img0<->img1-img2}
    matched_points1 = points1[indices1]

    return matched_points0, matched_points1, indices0, indices1


def common_matched_data_mnn(cam_infos, matched_data, dic=None, threshold=1e-3):

    common_matched_data = {}

    image_keys = [cam_info.image_name for cam_info in cam_infos]
    width, height = cam_infos[0].width, cam_infos[0].height
    for current_key in image_keys:

        common_matched_data[current_key] = {}
        if dic is None:
            other_keys = [key for key in image_keys if key != current_key]
        else:
            other_keys = dic[current_key]
        for other_key in other_keys:
            common_matched_data[current_key][other_key] = {}
        for i, prev_key in tqdm(enumerate(other_keys[:-1]), desc="common_matched_data"):
            for next_key in other_keys[i + 1:]:

                # 获取匹配点（归一化坐标）
                points_prev_to_current = np.array(matched_data[current_key][prev_key]['points'])  # T10
                points_current_to_next = np.array(matched_data[current_key][next_key]['points'])  # T12

                pixel_points10 = (points_prev_to_current * np.array([width, height])).astype(int)
                pixel_points12 = (points_current_to_next * np.array([width, height])).astype(int)

                # 使用 MNN 对齐 T01 和 T12
                aligned_points_t10, aligned_points_t12, indices_t10, indices_t12 = mutual_nearest_neighbor(
                    pixel_points10, pixel_points12, threshold=threshold
                )
                mask10 = torch.zeros(len(pixel_points10), dtype=torch.bool).cuda()
                mask12 = torch.zeros(len(pixel_points12), dtype=torch.bool).cuda()
                mask10[indices_t10] = True
                mask12[indices_t12] = True

                assert mask10.sum().item() == len(indices_t10) and mask12.sum().item() == len(indices_t12)
                a = len(indices_t12)
                print(current_key, prev_key, next_key, a)

                common_matched_data[current_key][prev_key][next_key] = mask10
                common_matched_data[current_key][next_key][prev_key] = mask12

    return common_matched_data



def common_matched_data_mnn2(cam_infos, matched_data, dic=None, threshold=1e-3):

    common_matched_data = {}

    image_keys = [cam_info.image_name for cam_info in cam_infos]
    for current_key in image_keys:
        common_matched_data[current_key] = {}
        if dic is None:
            other_keys = [key for key in image_keys if key != current_key]
        else:
            other_keys = dic[current_key]
        for other_key in other_keys:
            common_matched_data[current_key][other_key] = {}
        for i, prev_key in enumerate(other_keys[:-1]):
            for next_key in other_keys[i + 1:]:

                # 获取匹配点（归一化坐标）
                points_prev_to_current = np.array(matched_data[current_key][prev_key]['points'])  # T10
                points_current_to_next = np.array(matched_data[current_key][next_key]['points'])  # T12

                # 使用 MNN 对齐 T01 和 T12
                aligned_points_t10, aligned_points_t12, indices_t10, indices_t12 = mutual_nearest_neighbor(
                    points_prev_to_current, points_current_to_next, threshold=0.003
                )
                mask10 = torch.zeros(len(points_prev_to_current), dtype=torch.bool).cuda()
                mask12 = torch.zeros(len(points_current_to_next), dtype=torch.bool).cuda()
                mask10[indices_t10] = True
                mask12[indices_t12] = True

                assert mask10.sum().item() == len(indices_t10) and mask12.sum().item() == len(indices_t12)
                a = len(indices_t12)
                print('common points: ', current_key, prev_key, next_key, a)

                common_matched_data[current_key][prev_key][next_key] = mask10
                common_matched_data[current_key][next_key][prev_key] = mask12

    return common_matched_data


