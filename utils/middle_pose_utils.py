from scipy.spatial.transform import Slerp
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from typing import Tuple



def generate_virtual_poses(poses: np.ndarray,
                          x: int = 5,
                          min_distance_ratio: float = 2.0):

    positions = poses[:, :3, 3]  # (N, 3)

    # 2. Compute minimum distance and determine neighbors
    if len(poses) >= 2:
        pairwise_dist = cdist(positions, positions)
        np.fill_diagonal(pairwise_dist, np.inf)
        d_min = np.min(pairwise_dist)
        t = min_distance_ratio * d_min
        neighbors = [(i, j) for i in range(len(poses)) for j in range(i+1, len(poses))
                     if pairwise_dist[i, j] < t]
    else:
        neighbors = []

    # 3. Initialize output
    virtual_poses = []

    # 5. Interpolate neighbor poses
    for i, j in neighbors:
        pose_i, pose_j = poses[i], poses[j]

        # Position interpolation (LERP)
        pos_i, pos_j = pose_i[:3, 3], pose_j[:3, 3]
        positions_interp = [ (1-alpha)*pos_i + alpha*pos_j
                           for alpha in np.linspace(0, 1, x+2)[1:-1] ]


        # Rotation interpolation (compatible with older SciPy versions)
        key_rots = R.from_matrix(np.stack([pose_i[:3, :3], pose_j[:3, :3]]))
        slerp = Slerp([0, 1], key_rots)  # Time points 0 and 1 correspond to two keyframes
        rotations_interp = slerp(np.linspace(0, 1, x+2)[1:-1]).as_matrix()

        # Construct virtual poses
        for rot, pos in zip(rotations_interp, positions_interp):
            new_pose = np.eye(4)
            new_pose[:3, :3] = rot
            new_pose[:3, 3] = pos
            virtual_poses.append(new_pose)

    return np.array(virtual_poses)

