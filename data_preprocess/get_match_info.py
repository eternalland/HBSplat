# -*- coding: utf-8 -*-
# @Author  : mayu
import os.path

from tqdm import tqdm

from utils import image_utils

from data_preprocess.read_models import *

import matplotlib.pyplot as plt



def draw_matching2(save_dir, img_data0, img_data1, img_name0, img_name1, points0, points1, type):
    # Load image
    img0 = cv2.cvtColor(img_data0, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img_data1, cv2.COLOR_BGR2RGB)
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    # Convert normalized coordinates to pixel coordinates
    pixel_points0 = (points0 * np.array([w0, h0])).astype(int)
    pixel_points1 = (points1 * np.array([w1, h1])).astype(int)

    # Generate random RGB color for each pair of points
    num_points0 = len(points0)
    num_points1 = len(points1)
    colors = np.random.rand(num_points0, 3)  # Generate random RGB colors with shape (N, 3), range [0, 1]

    # Create visualization figure
    plt.figure(figsize=(15, 8))

    # Subplot 1: reference points
    plt.subplot(1, 2, 1)
    plt.imshow(img0)
    plt.scatter(pixel_points0[:, 0], pixel_points0[:, 1], c=colors, s=1)
    plt.title(f'{img_name0}')
    plt.xlabel(f'matching_point_number: {num_points0}')

    # Subplot 2: aligned points
    plt.subplot(1, 2, 2)
    plt.imshow(img1)
    plt.scatter(pixel_points1[:, 0], pixel_points1[:, 1], c=colors, s=1)
    plt.title(f'{img_name1}')
    plt.xlabel(f'matching_point_number: {num_points1}')

    plt.savefig(f'{save_dir}/{img_name0}_{img_name1}_{len(points0)}_{type}.png', dpi=600, bbox_inches='tight')

    plt.close()

def draw_matching(args, img_path0, img_path1, points0, points1, matching_number):
    print("points1 type:", type(points1))
    print("points1 content:", points1)
    img_name0 = os.path.basename(img_path0).split(".")[0]
    img_name1 = os.path.basename(img_path1).split(".")[0]
    # Load image
    img0 = cv2.cvtColor(cv2.imread(img_path0), cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(cv2.imread(img_path1), cv2.COLOR_BGR2RGB)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    # Convert normalized coordinates to pixel coordinates
    pixel_points0 = (points0 * np.array([w0, h0])).astype(int)
    pixel_points1 = (points1 * np.array([w1, h1])).astype(int)

    # Generate random RGB color for each pair of points
    num_points0 = len(points0)
    num_points1 = len(points1)
    colors = np.random.rand(num_points0, 3)  # Generate random RGB colors with shape (N, 3), range [0, 1]

    # Create visualization figure
    plt.figure(figsize=(15, 8))

    # Subplot 1: reference points
    plt.subplot(1, 2, 1)
    plt.imshow(img0)
    plt.scatter(pixel_points0[:, 0], pixel_points0[:, 1], c=colors, s=1)
    plt.title(f'{img_name0} ({args.matching_method})')
    plt.xlabel(f'Before ransac points: {matching_number}, aligned points: {num_points0}')

    # Subplot 2: aligned points
    plt.subplot(1, 2, 2)
    plt.imshow(img1)
    plt.scatter(pixel_points1[:, 0], pixel_points1[:, 1], c=colors, s=1)
    plt.title(f'{img_name1} ({args.matching_method})')
    plt.xlabel(f'Before ransac points: {matching_number}, aligned points: {num_points1}')

    plt.savefig(f'{args.matched_image_dir}/{img_name0}_{img_name1}_{args.sign}_{len(points0)}.png', dpi=600, bbox_inches='tight')

    plt.close()



def match_pair(args, image_name0, image_name1, image_data0, image_data1, matched_datas):

    image_tensor0 = image_utils.to_torch_cuda(image_data0, device = args.device)
    image_tensor1 = image_utils.to_torch_cuda(image_data1, device = args.device)

    if args.matching_method == 'loftr' or args.matching_method == 'aspan':
        data = dict(color0=image_tensor0, color1=image_tensor1, image0=image_tensor0, image1=image_tensor1)
    else:
        data = dict(color0=image_data0, color1=image_data1, image0=image_tensor0, image1=image_tensor1)

    print('top_k', args.top_k)
    # ==============================================================================

    data = match_op(args, data, image_tensor0, image_tensor1)

    # ==============================================================================

    norm_kpts0 = data['norm_kpts0']
    norm_kpts1 = data['norm_kpts1']

    matched_datas[image_name0][image_name1] = {'points': norm_kpts0, 'mconf': data['mconf']}
    matched_datas[image_name1][image_name0] = {'points': norm_kpts1, 'mconf': data['mconf']}

    matching_number = len(data['norm_kpts0'])
    print(f"finish {image_name0}_{image_name1}: {matching_number}")



def cv_ransac(norm_kpts0, norm_kpts1, size0, size1):

    kpts0 = (norm_kpts0 * np.array(size0)).astype(int)
    kpts1 = (norm_kpts1 * np.array(size1)).astype(int)

    # robust fitting
    F, mask = cv2.findFundamentalMat(kpts0, kpts1,
                                     cv2.USAC_MAGSAC, ransacReprojThreshold=1.0,
                                     confidence=0.999999, maxIters=10000)

    mask = mask.ravel() > 0

    return F, mask



def generate_matched_datas(args, cam_infos):
    print('------------method: ', args.matching_method)
    if args.matching_method == 'loftr' or args.matching_method == 'aspan':
        args.sign = str(f"{args.matching_method}_{args.input_views}")
    else:
        args.sign = str(f"{args.matching_method}_{args.top_k}_{args.input_views}")

    # ==============================================================================

    read_model(args)

    # ==============================================================================

    matched_datas = {cam_info.image_name: {} for cam_info in cam_infos}

    for i, cam_info0 in tqdm(enumerate(cam_infos[:-1]), desc="generate_matched_datas"):
        image_name0 = cam_info0.image_name
        image_data0 = cam_info0.image_data

        for cam_info1 in cam_infos[i + 1:]:
            image_name1 = cam_info1.image_name
            image_data1 = cam_info1.image_data

            match_pair(args, image_name0, image_name1, image_data0, image_data1, matched_datas)

    np.save(f"{args.matched_image_dir}/matched_data.npy", np.array(matched_datas))

    return matched_datas

def draw_matched_point(args, cam_infos, matched_datas, type):
    for i, cam_info0 in tqdm(enumerate(cam_infos[:-1]), desc="draw_matched_point"):
        image_name0 = cam_info0.image_name
        image_data0 = cam_info0.image_data

        for cam_info1 in cam_infos[i + 1:]:
            image_name1 = cam_info1.image_name
            image_data1 = cam_info1.image_data
            print(image_name0, image_name1)

            matched_point0 = matched_datas[image_name0][image_name1]['points']
            matched_point1 = matched_datas[image_name1][image_name0]['points']

            draw_matching2(args.matched_image_dir,
                           image_data0, image_data1,
                           image_name0, image_name1,
                           matched_point0, matched_point1, type)


def ransac(args, cam_infos, matched_datas):

    for i, cam_info0 in tqdm(enumerate(cam_infos[:-1]), desc="ransac"):
        for cam_info1 in cam_infos[i + 1:]:
            image_name0 = cam_info0.image_name
            image_name1 = cam_info1.image_name
            size0 = (cam_info0.width, cam_info0.height)
            size1 = (cam_info1.width, cam_info1.height)
            if image_name0 not in matched_datas or image_name1 not in matched_datas[image_name0]:
                continue
            data0 = matched_datas[image_name0][image_name1]
            data1 = matched_datas[image_name1][image_name0]
            ori_matching_number = len(data0['points'])

            F, mask = cv_ransac(data0['points'], data1['points'], size0, size1)

            matched_datas[image_name0][image_name1] = {'points': data0['points'][mask], 'mconf': data0['mconf'][mask], 'F': F}
            matched_datas[image_name1][image_name0] = {'points': data1['points'][mask], 'mconf': data1['mconf'][mask], 'F': F}

            ran_matching_number = len(data0['points'][mask])

            np.save(f"{args.matched_image_dir}/matched_data_ransac.npy", np.array(matched_datas))

            print(f"finish {image_name0}_{image_name1}: {ori_matching_number} -> {ran_matching_number}")

