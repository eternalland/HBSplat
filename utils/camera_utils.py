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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, PILtoTorch2
from utils.graphics_utils import fov2focal
import cv2

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    resized_mono_depth_map = None
    mono_depth_map_ori_size = None
    if  cam_info.image is not None:
        orig_w, orig_h = cam_info.image.size

        if args.resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        else:  # should be a type that converts to float
            if args.resolution == -1:
                if orig_w > 1600:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / args.resolution

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))

        resized_image_rgb = PILtoTorch(cam_info.image, resolution)

        if cam_info.mono_depth_map is not None:
            resized_mono_depth_map = PILtoTorch2(cam_info.mono_depth_map, resolution).squeeze(dim=0)
            mono_depth_map_ori_size = PILtoTorch2(cam_info.mono_depth_map, (orig_w, orig_h)).squeeze(dim=0)


        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = None
        if resized_image_rgb.shape[1] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
        blendermask = cam_info.blendermask
        if blendermask is not None:
            blendermask = blendermask.astype(np.uint8) * 255  # Convert True to 255, False to 0
            blendermask = cv2.resize(blendermask, resolution)
            blendermask = blendermask > 128  # Convert back to boolean if needed
        dtumask = cam_info.dtumask
        if dtumask is not None:
            dtumask = cv2.resize(dtumask, resolution)
            
    else:
        gt_image = None
        loaded_mask = None
        dtumask = None
        blendermask = None
    
    height_in = cam_info.height
    width_in = cam_info.width
    if args.resolution in [1, 2, 4, 8]:
        height_in /= args.resolution
        width_in /= args.resolution

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, near_far=cam_info.near_far, height_in=height_in, width_in=width_in,
                  image=gt_image, gt_alpha_mask=loaded_mask, dtumask=dtumask, blendermask=blendermask,
                  bounds = cam_info.bounds, image_path=cam_info.image_path, mono_depth_map = resized_mono_depth_map, mono_depth_map_ori_size= mono_depth_map_ori_size,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
