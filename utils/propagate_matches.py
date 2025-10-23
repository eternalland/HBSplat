import torch
import numpy as np
from tqdm import tqdm


from scipy.spatial import cKDTree

def mutual_nearest_neighbor(points0, points1, threshold=0.001):
    """
    Align two point sets using Mutual Nearest Neighbor (MNN) algorithm.

    Args:
        points0 (np.ndarray): First point set, shape (n, 2), normalized coordinates.
        points1 (np.ndarray): Second point set, shape (m, 2), normalized coordinates.
        threshold (float): Distance threshold (normalized coordinate space), default 1e-3.

    Returns:
        tuple: (matched_points0, matched_points1)
            - matched_points0: Matched points from points0, shape (k, 2).
            - matched_points1: Matched points from points1, shape (k, 2).
    """
    # Convert to numpy arrays
    # {xy∈img1,img1|img1-img0,img1-img0} abbreviated as {xy∈img1,img1|img1-img0}
    # {xy∈img1,img1|img1-img2,img1-img2}
    points0 = np.array(points0)
    points1 = np.array(points1)

    # Build KD trees
    tree0 = cKDTree(points0)
    tree1 = cKDTree(points1)

    # Nearest neighbors from points1 to points0
    # Corresponding to 0 index
    # {d,img1|,img1-img2|img1-img0<-img1-img2}
    # {idx∈img1,img1|img1-img0,img1-img2|img1-img0<-img1-img2}
    distances_1to0, indices_1to0 = tree0.query(points1, k=1)
    mask_1to0 = distances_1to0 < threshold

    # Nearest neighbors from points0 to points1
    # {d,img1|,img1-img0|img1-img0->img1-img2}
    # {idx∈img1,img1|img1-img2,img1-img0|img1-img0->img1-img2}
    distances_0to1, indices_0to1 = tree1.query(points0, k=1)
    mask_0to1 = distances_0to1 < threshold

    # Mutual nearest neighbor check
    # {f,img1|,img1-img2}
    mutual_mask = np.zeros(len(points1), dtype=bool)
    for i in range(len(points1)):
        # mask_0to1[i] = true/false
        if mask_1to0[i]:  # Nearest neighbor from points0[i] to points1 satisfies threshold
            j = indices_1to0[i]  # Nearest neighbor of points0[i] in points1
            if mask_0to1[j] and indices_0to1[j] == i:  # Nearest neighbor of points1[j] is points0[i]
                # {tf,img1|,img1-img2|img1-img0<->img1-img2}
                mutual_mask[i] = True

    # # Extract mutual nearest neighbor matching points
    # # Filter
    # matched_points0 = points0[mutual_mask]
    # matched_indices = indices_0to1[mutual_mask]
    # # Filter
    # matched_points1 = points1[matched_indices]

    # Extract mutual nearest neighbor matching points and their indices
    # Keep indices
    # mutual_mask: coordinates of nearest neighbor in img0 for each pixel in img1
    # {idx∈img1,|img1-img2,|img1-img0<->img1-img2}
    indices1 = np.where(mutual_mask)[0]
    # Keep indices
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

                # Get matched points (normalized coordinates)
                points_prev_to_current = np.array(matched_data[current_key][prev_key]['points'])  # T10
                points_current_to_next = np.array(matched_data[current_key][next_key]['points'])  # T12

                pixel_points10 = (points_prev_to_current * np.array([width, height])).astype(int)
                pixel_points12 = (points_current_to_next * np.array([width, height])).astype(int)

                # Use MNN to align T01 and T12
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

                # Get matched points (normalized coordinates)
                points_prev_to_current = np.array(matched_data[current_key][prev_key]['points'])  # T10
                points_current_to_next = np.array(matched_data[current_key][next_key]['points'])  # T12

                # Use MNN to align T01 and T12
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


