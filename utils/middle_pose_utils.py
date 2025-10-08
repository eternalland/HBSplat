from scipy.spatial.transform import Slerp
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from typing import Tuple



def generate_virtual_poses(poses: np.ndarray,
                          x: int = 5,
                          min_distance_ratio: float = 2.0):

    positions = poses[:, :3, 3]  # (N, 3)

    # 2. 计算最短距离并确定邻居
    if len(poses) >= 2:
        pairwise_dist = cdist(positions, positions)
        np.fill_diagonal(pairwise_dist, np.inf)
        d_min = np.min(pairwise_dist)
        t = min_distance_ratio * d_min
        neighbors = [(i, j) for i in range(len(poses)) for j in range(i+1, len(poses))
                     if pairwise_dist[i, j] < t]
    else:
        neighbors = []

    # 3. 初始化输出
    virtual_poses = []

    # 5. 对邻居位姿进行插值
    for i, j in neighbors:
        pose_i, pose_j = poses[i], poses[j]

        # 位置插值 (LERP)
        pos_i, pos_j = pose_i[:3, 3], pose_j[:3, 3]
        positions_interp = [ (1-alpha)*pos_i + alpha*pos_j
                           for alpha in np.linspace(0, 1, x+2)[1:-1] ]


        # 旋转插值（兼容旧版SciPy）
        key_rots = R.from_matrix(np.stack([pose_i[:3, :3], pose_j[:3, :3]]))
        slerp = Slerp([0, 1], key_rots)  # 时间点0和1对应两个关键帧
        rotations_interp = slerp(np.linspace(0, 1, x+2)[1:-1]).as_matrix()

        # 构造虚拟位姿
        for rot, pos in zip(rotations_interp, positions_interp):
            new_pose = np.eye(4)
            new_pose[:3, :3] = rot
            new_pose[:3, 3] = pos
            virtual_poses.append(new_pose)

    return np.array(virtual_poses)

