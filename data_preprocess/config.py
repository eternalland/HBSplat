_mono_depth = '/home/mayu/thesis/ml-depth-pro/src'
_mono_ckpt = '/home/mayu/thesis/ml-depth-pro/checkpoints/depth_pro.pt'

_gim = '/home/mayu/thesis/gim'
_lightglue_weights_path = '/home/mayu/thesis/gim/weights/gim_lightglue_100h.ckpt'
_loftr_weights_path = '/home/mayu/thesis/gim/weights/gim_loftr_50h.ckpt'
_dkm_weights_path = '/home/mayu/thesis/gim/weights/gim_dkm_100h.ckpt'
_roma_weights_path = '/home/mayu/thesis/gim/weights/gim_roma_100h.ckpt'
# _aspan = '/home/mayu/thesis/ml-aspanformer'
# _aspan_weights_path = '/home/mayu/thesis/ml-aspanformer/weights/indoor.ckpt'

def get_mono_ckpt():
    return _mono_ckpt

def get_gim_path():
    return _gim

# def get_aspan_path():
#     return _aspan

def get_mono_depth():
    return _mono_depth

def get_lightglue_weights_path():
    return _lightglue_weights_path

def get_loftr_weights_path():
    return _loftr_weights_path

# def get_aspan_weights_path():
#     return _aspan_weights_path

def get_dkm_weights_path():
    return _dkm_weights_path

def get_roma_weights_path():
    return _roma_weights_path