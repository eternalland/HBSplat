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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 2_000
        # self.iterations = 2_500
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 2_000
        self.feature_lr = 0.002
        self.opacity_lr = 0.055
        self.scaling_lr = 0.0055
        self.rotation_lr = 0.0015
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 200
        self.opacity_reset_until_iter = 2_000
        self.densify_from_iter = 500
        self.densify_until_iter = 2_000
        self.densify_grad_threshold = 0.0004
        self.random_background = False
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


from datetime import datetime

# 获取当前日期
current_date = datetime.now()
# 格式化为 yymmdd（250530）
date_str = current_date.strftime('%y%m%d')
time_str = current_date.strftime('%H%M%S')


class SparseParams(ParamGroup):
    def __init__(self, parser, sentinel=False, is_training=False):
        # self.is_train = False

        self.input_root = '/home/mayu/thesis/SCGaussian/data'
        self.input_views = 3
        self.dataset = 'nerf_llff_data'
        self.scene_type = ''
        self.category = ''
        self.output_root = '/home/mayu/thesis/SCGaussian/output'
        self.is_draw = True
        self.top_k = 10000
        self.inv_scale = 4
        self.dfactor = 8
        self.depth_range_coeff = (0.8, 1.2)

        self.switch_virtual_batch_size = True
        self.switch_intermediate_result = 0
        # self.switch_virtual_batch_size = False
        self.virtual_batch_size = 100
        self.matching_method = 'dkm'
        self.matched_image_basedir = 'matched_image'
        self.resized_image_basedir = 'resized_image'
        self.hybrid_depth_basedir = 'hybrid_depth'
        self.virtual_camera_basedir = 'virtual_camera'
        self.mono_depth_map_basedir = 'mono_depth_map'
        self.middle_camera_basedir = 'middle_camera'
        self.occlusion_image_basedir = 'occlusion_image'

        self.switch_generate_matching_mono = 0


        self.run_module = 0b110
        self.log_file_path = '/home/mayu/thesis/SCGaussian/outputtest/flower110/123.txt'

        self.num_iterations = 2000
        self.filter_iteration = 1_000
        self.neighbor_dis = 3.
        self.propagate_weight = 0.5
        self.start_propagate_iteration = 1_000

        self.render_neighbor_dis = 3
        self.render_propagate_weight = 0.05
        self.start_render_propagate_iteration = 1

        self.switch_setup_init = True
        # self.switch_setup_init = False
        self.switch_dynamic_filter = True
        # self.switch_dynamic_filter = False
        self.tau_depth = 0.3
        self.tau_reproj = 0.1
        self.base_thresh = 0.2
        self.range_sensitivity = 0.2
        self.switch_second_filter = True
        # self.switch_second_filter = False
        self.smooth_iteration = 750
        self.mconf_threshold = 0
        self.use_mask = False
        self.distance_percentile = (0, 98)
        self.buffer_size = 30
        self.lr_decay_steps = [500, 1000, 1500]
        self.ratio = 0.5

        self.virtual_cam_num = 50
        self.warping_mode = 'backward_warping'
        # self.warping_mode = 'forward_warping'
        self.switch_small_transform = 1
        # self.switch_small_transform = 0
        self.small_transform_iter = 50
        # self.switch_nearest_warping = 1
        self.switch_nearest_warping = 0
        self.nearest_warping_num = 20
        self.switch_middle_pose = 1
        # self.switch_middle_pose = 0
        self.interpolate_middle_pose_num = 1
        self.virtual_source_num = 2
        self.start_virtual_render = 1000
        self.occ_select_pipe = 'simple_lama'
        self.top_percent = 99.5
        # self.mvs_depth_loss_w = 0.1
        # self.mono_depth_loss_w = 0.005
        self.unseen_lambda_dssim = 0.5
        self.reg_interval = 6

        self.test_dilate = 8
        self.test_max_distance = 4
        # self.occlusion_from_iter = 1_500
        # self.occlusion_until_iter = 2_200
        self.occlusion_interval = 1500
        # self.occlusion_iter = 50

        self.date = date_str
        self.time_str = time_str
        self.width = 1008
        self.height = 752
        self.output_dir = str(os.path.join(self.output_root, self.date))
        self.resized_image_dir = None
        self.mono_depth_map_dir = None
        self.matched_image_dir = None
        self.hybrid_depth_dir = None
        self.virtual_camera_dir = None
        self.middle_camera_dir = None
        self.occlusion_image_dir = None
        self.is_training = is_training

        super().__init__(parser, "Loading Parameters", sentinel)

    def update_from_args(self, args):
        """从解析的参数更新实例属性"""
        for arg_name in vars(self):
            if hasattr(args, arg_name):
                setattr(self, arg_name, getattr(args, arg_name))

    def create_dir(self, model_path):
        self.resized_image_dir = str(os.path.join(model_path, self.resized_image_basedir))
        os.makedirs(self.resized_image_dir, exist_ok=True)
        self.mono_depth_map_dir = str(os.path.join(model_path, self.mono_depth_map_basedir))
        os.makedirs(self.mono_depth_map_dir, exist_ok=True)
        self.matched_image_dir = os.path.join(model_path, self.matched_image_basedir)
        os.makedirs(self.matched_image_dir, exist_ok=True)
        self.hybrid_depth_dir = str(os.path.join(model_path, self.hybrid_depth_basedir))
        os.makedirs(self.hybrid_depth_dir, exist_ok=True)
        self.virtual_camera_dir = os.path.join(model_path, self.virtual_camera_basedir)
        os.makedirs(self.virtual_camera_dir, exist_ok=True)
        self.middle_camera_dir = os.path.join(model_path, self.middle_camera_basedir)
        os.makedirs(self.middle_camera_dir, exist_ok=True)
        self.occlusion_image_dir = os.path.join(model_path, self.occlusion_image_basedir)
        os.makedirs(self.occlusion_image_dir, exist_ok=True)


    def print_log(self):
        print("------------input_views: ", self.input_views)
        print("------------run_module: ", self.run_module)
        print("------------neighbor_dis: ", self.neighbor_dis)
        print("------------warping_mode: ", self.warping_mode)
        print("------------tau_reproj: ", self.tau_reproj)
        print("------------base_thresh: ", self.base_thresh)
        print("------------range_sensitivity: ", self.range_sensitivity)
        print("------------inv_scale: ", self.inv_scale)
        print("------------switch_generate_matching_mono: ", self.switch_generate_matching_mono)


