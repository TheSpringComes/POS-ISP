from util import Dict
from isp.filters import *

cfg = Dict()
cfg.depth_test_dir = 'Dataset/kitti/KITTI_depth/kitti_depth_test'
cfg.depth_train_dir = 'Dataset/kitti/KITTI_depth/KITTI_sc'
cfg.max_iter_step = 20001
cfg.val_freq = 500
cfg.save_model_freq = 500
cfg.print_freq = 100
cfg.summary_freq = 10
cfg.show_img_num = 2

cfg.parameter_lr_mul = 1
cfg.value_lr_mul = 1
cfg.critic_lr_mul = 1

cfg.filters = [
    ExposureFilter(),
    GammaFilter(),
    CCMFilter(),
    SharpenFilter(),
    DenoiseFilter(),
    ToneFilter(),
    ContrastFilter(),
    SaturationPlusFilter(),
    WNBFilter(),
    ImprovedWhiteBalanceFilter(),
]

cfg.filter_runtime_penalty = False
cfg.filters_runtime = [1.7, 2.0, 1.9, 6.3, 10, 2.7, 2.1, 2.0, 1.9, 1.7]
cfg.filter_runtime_penalty_lambda = 0.01

cfg.curve_steps = 8
cfg.gamma_range = 3
cfg.exposure_range = 3.5
cfg.wb_range = 1.1
cfg.color_curve_range = (0.90, 1.10)
cfg.lab_curve_range = (0.90, 1.10)
cfg.tone_curve_range = (0.5, 2)
cfg.usm_sharpen_range = (0.00001, 2.0)
cfg.sharpen_range = (0.00001, 10.0)
cfg.ccm_range = (-2.0, 2.0)
cfg.denoise_range = (0.00001, 1.0)
cfg.masking = False
cfg.minimum_strength = 0.3
cfg.maximum_sharpness = 1
cfg.clamp = False

cfg.critic_logit_multiplier = 100
cfg.discount_factor = 1.0
cfg.filter_usage_penalty = 1.0
cfg.use_TD = True
cfg.replay_memory_size = 128
cfg.maximum_trajectory_length = 7
cfg.over_length_keep_prob = 0.5
cfg.all_reward = 1.0
cfg.img_include_states = True
cfg.exploration = 0.05
cfg.exploration_penalty = 0.05
cfg.early_stop_penalty = 1.0
cfg.detect_loss_weight = 1.0
cfg.seg_loss_weight = 0.01

cfg.base_channels = 32
cfg.dropout_keep_prob = 0.5
cfg.shared_feature_extractor = True
cfg.fc1_size = 128
cfg.bnw = False
cfg.feature_extractor_dims = 4096
cfg.use_penalty = True
cfg.z_type = "uniform"
cfg.z_dim_per_filter = 16

cfg.num_state_dim = 3 + len(cfg.filters)
cfg.z_dim = 3 + len(cfg.filters) * cfg.z_dim_per_filter
cfg.test_steps = 5
