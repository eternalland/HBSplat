import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import sys

sys.path.insert(0, '/home/mayu/thesis/ml-depth-pro/src')
from depth_pro import depth_pro, utils
from utils import image_utils, plot_utils


def load_model(device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> tuple:
    """
    加载深度估计模型和预处理变换，移动到指定设备。

    Args:
        device (str): 运行设备 ('cuda' 或 'cpu')。

    Returns:
        tuple: (model, transform)，加载的模型和变换函数。
    """

    config = depth_pro.DepthProConfig(
        patch_encoder_preset="dinov2l16_384",
        image_encoder_preset="dinov2l16_384",
        checkpoint_uri="/home/mayu/thesis/ml-depth-pro/checkpoints/depth_pro.pt",
        decoder_features=256,
        use_fov_head=True,
        fov_encoder_preset="dinov2l16_384",
    )
    model, transform = depth_pro.create_model_and_transforms(config =config)
    model.eval()
    model = model.to(device)
    return model, transform


def load_and_preprocess_image(image_path: str, transform, device: str) -> tuple:
    """
    加载并预处理图像。

    Args:
        image_path (str): 图像文件路径。
        transform: 预处理变换函数（来自 depth_pro）。
        device (str): 运行设备。

    Returns:
        tuple: (image, f_px)，预处理后的张量和焦距（像素单位）。
    """
    try:
        image, _, f_px = utils.load_rgb(image_path)
        image = transform(image).to(device)
        return image, f_px
    except Exception as e:
        print(f"加载图像 {image_path} 失败: {e}")
        return None, None


def infer_depth(model, image: torch.Tensor, f_px: float) -> torch.Tensor:
    """
    运行深度估计推理。

    Args:
        model: 深度估计模型。
        image (torch.Tensor): 预处理后的图像张量。
        f_px (float): 焦距（像素单位）。

    Returns:
        torch.Tensor: 深度图张量，形状 (H, W) 或 (1, H, W)。
    """
    with torch.no_grad():
        prediction = model.infer(image, f_px=f_px)
    return prediction["depth"]


def generate_mono_depths(cam_infos, args):
    print(f"run mono_depth")

    output_dir = args.mono_depth_map_dir
    device = args.device

    model, transform = load_model(device)

    for idx, cam_info in tqdm(enumerate(cam_infos), desc="generate mono depth map"):
        image, f_px = load_and_preprocess_image(cam_info.image_path, transform, device)
        if image is None:
            continue
        depth = infer_depth(model, image, f_px)
        image_name = cam_info.image_name
        print(image_name)
        save_path = os.path.join(output_dir, f'{image_name}.jpg')
        depth_normalized = plot_utils.save_depth_map(depth, save_path)
        mono_depth_map = Image.fromarray(depth_normalized)
        cam_infos[idx] = cam_info._replace(mono_depth_map=mono_depth_map)
        print(f"深度图已保存到: {save_path}")



def get_mono_depth(imgs):
    # get Mono depth (from https://github.com/Wanggcong/SparseNeRF/blob/main/get_depth_map_for_llff_dtu.py)
    # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    H, W = imgs[0].shape[0:2]
    imgs = torch.concat([transform(img).to(device) for img in imgs])

    with torch.no_grad():
        prediction = midas(imgs)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(H, W),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()


def load_depth_maps(cam_infos, depth_image_dir: str):
    image_name_dic = {cam_info.image_name: cam_info for cam_info in cam_infos}
    image_names = list(image_name_dic.keys())

    image_extensions = ('.jpg', '.jpeg', '.png')

    image_list = []
    for depth_image_name in os.listdir(depth_image_dir):
        if not depth_image_name.lower().endswith(image_extensions):
            continue
        short_depth_image_name = os.path.basename(depth_image_name).split(".")[0]
        if short_depth_image_name not in image_names:
            continue

        image_list.append(depth_image_name)

    for idx, cam_info in tqdm(enumerate(cam_infos), desc="Loading depth maps"):
        for image in image_list:
            if cam_info.image_name in image:
                image_depth_path = os.path.join(depth_image_dir, image)
                try:
                    # 读取深度图（假设为灰度或伪彩色）
                    depth_img = cv2.imread(image_depth_path, cv2.IMREAD_GRAYSCALE)
                    if depth_img is None:
                        print(f"无法读取深度图: {image_depth_path}")
                        continue
                    # 如果是伪彩色 (H, W, 3)，转换为灰度 (H, W) 以近似深度值
                    if len(depth_img.shape) == 3:
                        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
                    # 存储深度图，键为文件名（不含扩展名

                    # 归一化深度图到 [0, 1]
                    depth_min = np.min(depth_img)
                    depth_max = np.max(depth_img)
                    if depth_max != depth_min:
                        depth_img = (depth_img - depth_min) / (depth_max - depth_min)
                    else:
                        depth_img = np.zeros_like(depth_img)  # 或 depth_img.fill(0.5)

                    depth_img = depth_img.astype(np.float32)
                    image = Image.fromarray(depth_img)
                    cam_infos[idx] = cam_info._replace(mono_depth_map=image)
                except Exception as e:
                    print(f"加载深度图 {image_depth_path} 失败: {e}")
                    continue
