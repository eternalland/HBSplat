import numpy as np
from typing import Tuple

def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def focus_pt_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def poses_avg(poses):
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world

def viewmatrix(lookdir, up, position, subtract_position=False):
    """Construct lookat view matrix."""
    vec2 = normalize((lookdir - position) if subtract_position else lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def generate_random_poses_dtu(extrinsics, n_poses=120, r_scale=4.0):
    poses = np.stack([np.linalg.inv(extrinsics[i]) for i in range(extrinsics.shape[0])])
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 100, 0) * r_scale
    radii = np.concatenate([radii, [1.]])
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    z_axis = focus_pt_fn(poses)
    random_poses = []
    for _ in range(n_poses):
        random_pose = np.eye(4, dtype=np.float32)
        t = radii * np.concatenate([
            2 * 1.0 * (np.random.rand(3) - 0.5), [1,]])
        position = cam2world @ t
        z_axis_i = z_axis + np.random.randn(*z_axis.shape) * 0.125
        random_pose[:3, :4] = viewmatrix(z_axis_i, up, position, True)
        random_poses.append(random_pose)
    random_poses = np.stack(random_poses, axis=0)
    return random_poses

def generate_pseudo_poses_llff(extrinsics, bounds, n_poses, r_scale=2.0):
    poses = np.stack([np.linalg.inv(extrinsics[i]) for i in range(extrinsics.shape[0])])

    # 计算合理的焦点深度，基于近深度和远深度的加权平均（视差空间）
    close_depth, inf_depth = bounds.min() * 0.9, bounds.max() * 5.0
    dt = 0.75
    focal = 1 / (((1 - dt) / close_depth + dt / inf_depth))

    # 使用相机位置的 100 百分位数计算螺旋路径的半径，并应用缩放因子
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 100, 0) * r_scale
    radii = np.concatenate([radii, [1.0]])  # 添加齐次坐标维度

    # 生成随机姿态
    random_poses = []
    cam2world = poses_avg(poses)  # 计算平均相机到世界变换
    up = poses[:, :3, 1].mean(0)  # 计算平均上向量
    for _ in range(n_poses):
        random_pose = np.eye(4, dtype=np.float32)  # 初始化 4x4 单位矩阵
        # 生成随机平移向量，范围 [-1, 1] 并按半径缩放
        t = radii * np.concatenate([2 * np.random.rand(3) - 1.0, [1.0]])
        position = cam2world @ t  # 将平移向量变换到世界坐标系
        lookat = cam2world @ [0, 0, -focal, 1.0]  # 计算相机注视点
        z_axis = position - lookat  # 计算相机 z 轴（从位置指向注视点）
        # 使用视图矩阵生成相机姿态
        random_pose[:3, :4] = viewmatrix(z_axis, up, position)
        random_poses.append(random_pose)

    # 堆叠生成的姿态
    return np.stack(random_poses, axis=0)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :4, :4]

def pad_poses(p):
    """
    为姿态矩阵添加齐次底行 [0,0,0,1]。

    将形状为 [..., 3, 4] 的姿态矩阵扩展为 [..., 4, 4] 的齐次矩阵。

    参数
    ----------
    p : np.ndarray
        输入姿态矩阵，形状 [..., 3, 4]。

    返回
    -------
    np.ndarray
        扩展后的齐次姿态矩阵，形状 [..., 4, 4]。
    """
    # 创建形状与 p[..., :1, :4] 相同的底行 [0, 0, 0, 1]
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    # 将底行拼接到姿态矩阵，扩展为齐次矩阵
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return poses_recentered, transform

def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def generate_random_poses_360(extrinsics, n_frames=10000, z_variation=0.1, z_phase=0):
    # poses = []
    # for view in views:
    #     tmp_view = np.eye(4)
    #     tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
    #     tmp_view = np.linalg.inv(tmp_view)
    #     tmp_view[:, 1:3] *= -1
    #     poses.append(tmp_view)
    # poses = np.stack(poses, 0)

    poses = np.stack([np.linalg.inv(extrinsics[i]) for i in range(extrinsics.shape[0])])
    poses, transform = transform_poses_pca(poses)


    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0] , center[1],  0 ])
    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)

    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)


    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            (low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5)),
            (low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5)),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.random.rand(n_frames) * 2. * np.pi
    positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    # up = normalize(poses[:, :3, 1].sum(0))

    render_poses = []
    for p in positions:
        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(p - center, up, p)
        render_pose = np.linalg.inv(transform) @ render_pose
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    return render_poses














































from typing import NamedTuple, Optional, List, Tuple
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

def transform_poses_pca2(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Transforms poses so principal components lie on XYZ axes.

    Args:
        poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
        A tuple (poses, transform), with the transformed poses and the applied
        camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor

    return poses_recentered, transform, scale_factor

def generate_ellipse_path_from_poses(poses: np.ndarray,
                          n_frames: int = 120,
                          const_speed: bool = True,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 100, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 0, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 100, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
            low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                        (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)
    print('theta[0]', theta[0])

    if const_speed:
        # Resample theta angles so that the velocity is closer to constant.
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        theta = sample(None, theta, np.log(lengths), n_frames + 1)
        positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])

def invert_transform_poses_pca(poses_recentered, transform, scale_factor):
    poses_recentered[:, :3, 3] /= scale_factor
    transform_inv = np.linalg.inv(transform)
    poses_original = unpad_poses(transform_inv @ pad_poses(poses_recentered))
    return poses_original


def generate_ellipse_path_from_camera_infos(
        extrinsics,
        n_frames: int = 120,
        const_speed: bool = False,
        z_variation: float = 0.,
        z_phase: float = 0.
    ):
    # print(f'Generating ellipse path from {len(cam_infos)} camera infos ...')
    # poses = np.array([np.linalg.inv(getWorld2View2(cam_info.R, cam_info.T))[:3, :4] for cam_info in cam_infos])
    poses = np.stack([np.linalg.inv(extrinsics[i]) for i in range(extrinsics.shape[0])])
    poses[:, :, 1:3] *= -1
    poses, transform, scale_factor = transform_poses_pca2(poses)
    render_poses = generate_ellipse_path_from_poses(poses, n_frames, const_speed, z_variation, z_phase)
    render_poses = invert_transform_poses_pca(render_poses, transform, scale_factor)
    render_poses[:, :, 1:3] *= -1

    return render_poses


