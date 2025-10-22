import os
import torch
import cv2
import warnings
import numpy as np
import sys

import data_preprocess.config as config

sys.path.insert(0, config.get_gim_path())
from networks.lightglue.superpoint import SuperPoint
from networks.lightglue.models.matchers.lightglue import LightGlue
from networks.loftr.loftr import LoFTR
from networks.loftr.misc import lower_config
from networks.loftr.config import get_cfg_defaults
from networks.dkm.models.model_zoo.DKMv3 import DKMv3
from networks.roma.roma import RoMa

# sys.path.insert(1, config.get_aspan_path())
# from src.ASpanFormer.aspanformer import ASpanFormer
# from src.config.default import get_cfg_defaults as get_cfg_defaults_asp
# from src.utils.misc import lower_config as lower_config_asp


global detector
global model

def lightglue_model(args):
    global detector
    global model
    # Load model
    detector = SuperPoint({
        'max_num_keypoints': args.top_k,
        'force_num_keypoints': True,
        'detection_threshold': 0.0,
        'nms_radius': 3,
        'trainable': False,
    }).eval().to(args.device)

    model = LightGlue({
        'filter_threshold': 0.01,
        'flash': False,
        'checkpointed': True,
    }).eval().to(args.device)


    checkpoints_path = config.get_lightglue_weights_path()
    state_dict = torch.load(checkpoints_path, map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('model.'): state_dict.pop(k)
        if k.startswith('superpoint.'):
            state_dict[k.replace('superpoint.', '', 1)] = state_dict.pop(k)
    detector.load_state_dict(state_dict)

    state_dict = torch.load(checkpoints_path, map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('superpoint.'): state_dict.pop(k)
        if k.startswith('model.'):
            state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
    model.load_state_dict(state_dict)


def loftr_model(args):
    global model
    # load model
    model = LoFTR(lower_config(get_cfg_defaults())['loftr'])
    # weights path
    checkpoints_path = config.get_loftr_weights_path()
    # load state dict
    state_dict = torch.load(checkpoints_path, map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    # load state dict
    model.load_state_dict(state_dict)
    # eval mode
    model = model.eval().to(args.device)


# def aspan_model(args):
#     global model
#
#     config = get_cfg_defaults_asp()
#     config.merge_from_file('/home/mayu/thesis/ml-aspanformer/configs/aspan/indoor/aspan_test.py')
#     _config = lower_config_asp(config)
#     model = ASpanFormer(config=_config['aspan'])
#
#     checkpoints_path = config.get_aspan_weights_path()
#     state_dict = torch.load(checkpoints_path, map_location='cpu')['state_dict']
#     model.load_state_dict(state_dict, strict=False)
#     model.eval().to(args.device)


def dkm_model(args):
    global model

    # load model
    model = DKMv3(weights=None, h=args.height, w=args.width)
    # weights path
    checkpoints_path = config.get_dkm_weights_path()
    # load state dict
    state_dict = torch.load(checkpoints_path, map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('model.'):
            state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)
        if 'encoder.net.fc' in k:
            state_dict.pop(k)
    # load state dict
    model.load_state_dict(state_dict)
    # eval mode
    model = model.eval().to(args.device)

def roma_model(args):
    global model

    # load model
    model = RoMa(img_size=[args.width])
    
    # weights path
    checkpoints_path = config.get_roma_weights_path()
    # load state dict
    state_dict = torch.load(checkpoints_path, map_location='cpu')
    if 'state_dict' in state_dict.keys(): state_dict = state_dict['state_dict']
    for k in list(state_dict.keys()):
        if k.startswith('model.'):
            state_dict[k.replace('model.', '', 1)] = state_dict.pop(k)

    # load state dict
    model.load_state_dict(state_dict)

    # eval mode
    model = model.eval().to(args.device)


# ==============================================================================# ==============================================================================


def lightglue_matching(data, image0, image1):

    pred = {}
    with torch.no_grad():
        pred.update({k + '0': v for k, v in detector({
            "image": data["gray0"],
        }).items()})
        pred.update({k + '1': v for k, v in detector({
            "image": data["gray1"],
        }).items()})
        pred.update(model({**pred, **data,
                           **{'image_size0': data['size0'],
                              'image_size1': data['size1']}}))
    torch.cuda.empty_cache()
    kpts0 = pred['keypoints0'].squeeze()
    kpts1 = pred['keypoints1'].squeeze()
    m_bids = torch.nonzero(pred['keypoints0'].sum(dim=2) > -1)[:, 0]
    matches = pred['matches']
    bs = data['image0'].size(0)
    kpts0 = torch.cat([kpts0[m_bids == b_id][matches[b_id][..., 0]] for b_id in range(bs)])
    kpts1 = torch.cat([kpts1[m_bids == b_id][matches[b_id][..., 1]] for b_id in range(bs)])
    b_ids = torch.cat([m_bids[m_bids == b_id][matches[b_id][..., 0]] for b_id in range(bs)])
    mconf = torch.cat(pred['scores'])

    kpts0 = kpts0.cpu().numpy()
    kpts1 = kpts1.cpu().numpy()
    b_ids = b_ids.cpu().numpy()
    mconf = mconf.cpu().numpy()

    # Get original image size for normalization
    orig_h0, orig_w0 = image0.shape[-2:]
    orig_h1, orig_w1 = image1.shape[-2:]

    # Normalize keypoints to [0,1] range (based on original image size)
    norm_kpts0 = np.copy(kpts0)
    norm_kpts0[:, 0] /= orig_w0  # x coordinate divided by image width
    norm_kpts0[:, 1] /= orig_h0  # y coordinate divided by image height

    norm_kpts1 = np.copy(kpts1)
    norm_kpts1[:, 0] /= orig_w1  # x coordinate divided by image width
    norm_kpts1[:, 1] /= orig_h1  # y coordinate divided by image height


    data.update({'norm_kpts0': norm_kpts0, 'norm_kpts1': norm_kpts1, 'b_ids': b_ids, 'mconf': mconf})

    return data


def loftr_matching(data, image0, image1):

    with torch.no_grad():
        model(data)

    torch.cuda.empty_cache()
    kpts0 = data['mkpts0_f'].cpu().numpy()
    kpts1 = data['mkpts1_f'].cpu().numpy()
    b_ids = data['m_bids'].cpu().numpy()
    mconf = data['mconf'].cpu().numpy()

    # Get original image size for normalization
    orig_h0, orig_w0 = image0.shape[-2:]
    orig_h1, orig_w1 = image1.shape[-2:]
    # Normalize keypoints to [0,1] range (based on original image size)
    norm_kpts0 = np.copy(kpts0)
    norm_kpts0[:, 0] /= orig_w0  # x coordinate divided by image width
    norm_kpts0[:, 1] /= orig_h0  # y coordinate divided by image height
    norm_kpts1 = np.copy(kpts1)
    norm_kpts1[:, 0] /= orig_w1  # x coordinate divided by image width
    norm_kpts1[:, 1] /= orig_h1  # y coordinate divided by image height


    data.update({'norm_kpts0': norm_kpts0, 'norm_kpts1': norm_kpts1, 'b_ids': b_ids, 'mconf': mconf})

    return data


def aspan_matching(data, image0, image1):
    with (torch.no_grad()):   
      model(data,online_resize=True)

    torch.cuda.empty_cache()
    kpts0 = data['mkpts0_f'].cpu().numpy()
    kpts1 = data['mkpts1_f'].cpu().numpy()
    b_ids = data['m_bids'].cpu().numpy()
    mconf = data['mconf'].cpu().numpy()

    # Get original image size for normalization
    orig_h0, orig_w0 = image0.shape[-2:]
    orig_h1, orig_w1 = image1.shape[-2:]
    # Normalize keypoints to [0,1] range (based on original image size)
    norm_kpts0 = np.copy(kpts0)
    norm_kpts0[:, 0] /= orig_w0  # x coordinate divided by image width
    norm_kpts0[:, 1] /= orig_h0  # y coordinate divided by image height
    norm_kpts1 = np.copy(kpts1)
    norm_kpts1[:, 0] /= orig_w1  # x coordinate divided by image width
    norm_kpts1[:, 1] /= orig_h1  # y coordinate divided by image height


    data.update({'norm_kpts0': norm_kpts0, 'norm_kpts1': norm_kpts1, 'b_ids': b_ids, 'mconf': mconf})

    return data
    

def dkm_matching(data, image0, image1, top_k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dense_matches, dense_certainty = model.match(image0, image1)
        sparse_matches, mconf = model.sample(dense_matches, dense_certainty, top_k)

    torch.cuda.empty_cache()
    norm_kpts0 = ((sparse_matches[:, :2] + 1) / 2).cpu().numpy()
    norm_kpts1 = ((sparse_matches[:, 2:] + 1) / 2).cpu().numpy()
    b_ids = (torch.where(mconf[None])[0]).cpu().numpy()
    mconf = (mconf).cpu().numpy()

    data.update({'norm_kpts0': norm_kpts0, 'norm_kpts1': norm_kpts1, 'b_ids': b_ids, 'mconf': mconf})

    return data


def roma_matching(data, image0, image1, top_k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dense_matches, dense_certainty = model.match(image0, image1)
        sparse_matches, mconf = model.sample(dense_matches, dense_certainty, top_k)
    torch.cuda.empty_cache()
    norm_kpts0 = ((sparse_matches[:, :2] + 1) / 2).cpu().numpy()
    norm_kpts1 = ((sparse_matches[:, 2:] + 1) / 2).cpu().numpy()
    b_ids = (torch.where(mconf[None])[0]).cpu().numpy()
    mconf = (mconf).cpu().numpy()

    data.update({'norm_kpts0': norm_kpts0, 'norm_kpts1': norm_kpts1, 'b_ids': b_ids, 'mconf': mconf})

    return data


def read_model(args):
    if args.matching_method == 'lightglue':
        lightglue_model(args)
    elif args.matching_method == 'loftr':
        loftr_model(args)
    # elif args.matching_method == 'aspan':
    #     aspan_model(args)
    elif args.matching_method == 'dkm':
        dkm_model(args)
    elif args.matching_method == 'roma':
        roma_model(args)
    else:
        raise ValueError('Unknown matching method')

def match_op(args, data, image0, image1):
    if args.matching_method == 'lightglue':
        return lightglue_matching(data, image0, image1)
    elif args.matching_method == 'loftr':
        return loftr_matching(data, image0, image1)
    # elif args.matching_method == 'aspan':
    #     return aspan_matching(data, image0, image1)
    elif args.matching_method == 'dkm':
        return dkm_matching(data, image0, image1, args.top_k)
    elif args.matching_method == 'roma':
        return roma_matching(data, image0, image1, args.top_k)
    return None


