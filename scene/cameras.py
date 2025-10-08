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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal
import torch.nn.functional as F
from utils import loss_utils, get_mono_depth


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, dtumask, near_far, blendermask, height_in, width_in,
                 bounds, image_path, mono_depth_map, mono_depth_map_ori_size,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.bounds = bounds
        self.mono_depth_map = mono_depth_map
        self.mono_depth_map_ori_size = mono_depth_map_ori_size
        self.foreground_mask = None
        self.depth_map = None
        self.is_virtual = False
        self.mask = None
        self.occlusion_image = None
        self.occlusion_depth = None
        self.occlusion_mask = None
        self.occlusion_depth_threshold = None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.near_far = near_far
        self.dtumask = torch.tensor(dtumask).float().cuda() if dtumask is not None else None
        self.blendermask = torch.tensor(blendermask).float().cuda() if blendermask is not None else None
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device) if image is not None else None
        self.image_width = self.original_image.shape[2] if image is not None else width_in
        self.image_height = self.original_image.shape[1] if image is not None else height_in

        
        if image is not None:
            if gt_alpha_mask is not None:
                self.original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        
        fx, fy = fov2focal(FoVx, self.image_width), fov2focal(FoVy, self.image_height)
        cx, cy = self.image_width / 2.0, self.image_height / 2.0
        self.intr = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).float().cuda()
        
        self.w2c = torch.zeros((4, 4)).float().cuda()
        self.w2c[:3, :3] = torch.from_numpy(R).float().cuda().transpose(0, 1)
        self.w2c[:3, 3] = torch.from_numpy(T).float().cuda()
        self.w2c[3, 3] = 1.0

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]



from dataclasses import dataclass

@dataclass
class TempCamera:
    def __init__(self, R, T, K, w2c, width, height, FovX, FovY, image_name, image_data, image_path):
        self.id = int
        # R:c2w
        self.R = R
        self.T = T
        self.K = K
        self.w2c = w2c
        self.FovX = FovX
        self.FovY = FovY
        self.image_data = image_data
        self.image_path = image_path
        self.image_name = image_name
        self.width = width
        self.height = height
        self.points = np.array([])
        self.uv = np.array([])
        self.rays_o = np.array([])
        self.rays_d = np.array([])
        self.mconf = np.array([])
        self.epi_depth = np.array([])
        self.ray_depth = np.array([])
        self.epi_distance = np.array([])
        self.ray_distance = np.array([])
        self.epi_ray_diff = np.array([])
        self.epi_point_cloud = np.array([])
        self.ray_point_cloud = np.array([])
        self.backproj_point_cloud = np.array([])
        self.backproj_depth = np.array([])
        self.colmap_depth_range = ()
        self.fused_depth_point = None
        self.mono_depth_map = None
        self.aligned_mono_depth_map = None
        self.aligned_mono_depth_map_small = None
        self.warped_depth = None
        self.point3D_ids = None
        self.color = None
        self.mask = None
        self.mask_dilation = None
        self.mask_no_buffer = None
        self.mask_epi_depth = None
        self.mask_ray_depth = None
        self.mask_reproj_depth = None
        self.mask_eip_ray_diff = None
        self.mask_reproj_ray_diff = None
        self.is_virtual = False


import os
import json

def save_pose(output_dir, cameras):
    json_cams = []
    camlist = []
    camlist.extend(cameras)
    for id, cam in enumerate(camlist):
        json_cams.append(camera_to_JSON(id, cam))
    with open(os.path.join(output_dir, "virtual_cameras.json"), 'w') as file:
        json.dump(json_cams, file)
#
def read_pose(input_dir):
    """
    从 JSON 文件读取相机参数并返回 Camera 对象列表。

    Args:
        input_dir (str): 包含 virtual_cameras.json 的目录路径。
        CameraClass (class): Camera 类的定义（需支持字段：R, T, image_name, width, height, FovX, FovY）。

    Returns:
        list: 包含所有 Camera 对象的列表。
    """
    file_path = os.path.join(input_dir, "virtual_cameras.json")
    with open(file_path, 'r') as file:
        json_cams = json.load(file)

    cameras = []
    for cam_data in json_cams:
        # 从 JSON 恢复旋转矩阵 R 和平移向量 T
        rotation = np.array(cam_data['rotation'])
        position = np.array(cam_data['position'])

        # 计算 c2w 矩阵并转换为 w2c（Rt）
        c2w = np.eye(4)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = position
        w2c = np.linalg.inv(c2w)
        K = np.array([[cam_data['fx'], 0., cam_data['width'] / 2.],
                      [0., cam_data['fy'], cam_data['height'] / 2.],
                      [0., 0., 1.]
                      ], dtype=np.float32)

        # 提取 R 和 T
        R = w2c[:3, :3].transpose()  # 转置回原始 R
        T = w2c[:3, 3]

        # 创建 Camera 对象
        camera = TempCamera(
            R=R,
            T=T,
            K=K,
            w2c=w2c,
            width=cam_data['width'],
            height=cam_data['height'],
            FovY=focal2fov(cam_data['fy'], cam_data['height']),  # 需实现 focal2fov
            FovX=focal2fov(cam_data['fx'], cam_data['width']),
            image_data=None,
            image_path=None,
            image_name=None
        )
        camera.id = cam_data['id']
        camera.is_virtual = True
        cameras.append(camera)

    return cameras


def camera_to_JSON(id, camera : Camera):

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    c2w = np.linalg.inv(Rt)
    pos = c2w[:3, 3]
    rot = c2w[:3, :3]
    # assert
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        # 'intrinsic': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry