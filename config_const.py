"""
Script: config_const.py
Description:
    Central configuration hub for RLBench-based imitation learning and diffusion
    policy experiments. This file defines lightweight helpers, observation/camera
    settings, method-specific policy & training configs, and deployment/distillation
    presets used across training, evaluation, and analysis scripts.

Contents:
    1) Helper Data Class
       - MyNameTuple: Dict wrapper enabling both attribute- and key-based access.

    2) Conversion Utilities
       - convert_nametuple2dict: Recursively convert MyNameTuple → dict.
       - convert_dict_valuetype: Recursively map values of a given dtype via a function.

    3) Observation Configuration
       - Cam_cfg: Shared RLBench CameraConfig (RGB only, 128×128, OpenGL3).
       - Obs_config: Full ObservationConfig (multi-cam RGB + proprio signals).

    4) Global Policy / Training Defaults
       - Seeds, device selection, data splits, action dimensions, chunk length, etc.

    5) Demonstration Configuration
       - Demo_conifg: Task list, number of demos per task, show/hide demo, action type.

    6) Method Registry (Config_list)
       - Per-method `policy_config` and `train_config` blocks for:
           • ACT
           • DiffusionPolicy (and variants: visonly, switch, wrec)
           • D3P (and D3P_LSTM)
           • Default (minimal placeholder)
         Each block captures vision encoders, diffusion/dynamics heads, optimizers,
         EMA/early-stopping, dataset types, batching, schedulers, and misc knobs.

    7) Deployment / Sweep Utilities
       - Run dataclass and RunBuilder: Cartesian product of param grids into runs.

    8) Evaluation & Model Paths (check_cfg)
       - Episode limits per task, default query frequency, random init flags,
         test set size, and filesystem paths to trained checkpoints for each task/method.

    9) Distillation Config (distill_cfg)
       - Paths to distilled student checkpoints (per task) for D3P.

Usage:
    - Import specific blocks from this module in training/eval scripts:
        from config_const import Obs_config, Config_list, Demo_conifg, check_cfg, distill_cfg
    - Extend/modify method entries in `Config_list` to add new models or ablations.
    - Update `check_cfg.model_dir` and `distill_cfg.model_dir` when checkpoints move.

Notes:
    - Camera/observation configs assume RGB inputs from {front, wrist, left/right shoulder, overhead}.
    - Action space defaults to joint-space with gripper open/close appended (dim=8).
    - Many configs use MyNameTuple for ergonomic access; use the provided converters
      if you need plain dicts (e.g., for serialization).

"""

# ===============================
#        Helper Data Class
# ===============================
class MyNameTuple:
    """
    A simple wrapper class providing both attribute-style and dictionary-style access
    to key-value pairs stored in an internal dictionary.
    """

    def __init__(self, my_dict):
        self.data = my_dict

    def __getattr__(self, name):
        """
        Allow access via attribute syntax (e.g., obj.key instead of obj['key']).
        Raises AttributeError if the key does not exist.
        """
        if name in self.data:
            return self.data[name]
        raise AttributeError(f"'MyNameTuple' object has no attribute '{name}'")

    def __str__(self):
        """Return a string representation of the stored dictionary."""
        return str(self.data)

    def keys(self):
        """Return all keys in the stored dictionary."""
        return self.data.keys()

    def __getitem__(self, key):
        """Dictionary-style access: obj[key]."""
        return self.data[key]

    def __setitem__(self, key, value):
        """Allow assignment using obj[key] = value."""
        self.data[key] = value

    def __len__(self):
        """Return number of key-value pairs."""
        return len(self.data)

# ===============================
#        Conversion Utilities
# ===============================
def convert_nametuple2dict(input: dict):
    """
    Recursively convert nested MyNameTuple objects into plain Python dictionaries.

    Args:
        input (dict or MyNameTuple): Data structure to be converted.

    Returns:
        dict: A dictionary representation of the original data.
    """
    if isinstance(input, dict):
        return {key: convert_nametuple2dict(value)
                for key, value in input.items()}
    elif isinstance(input, MyNameTuple):
        return {key: convert_nametuple2dict(value)
                for key, value in input.data.items()}
    else:
        return input

def convert_dict_valuetype(input: dict, src_dtype, convert_func):
    """
    Recursively convert values of a given type in a dictionary using 'convert_func'.

    Args:
        input (dict): Input dictionary (possibly nested).
        src_dtype (type): The source type to look for (e.g., np.ndarray).
        convert_func (callable): A function that converts values of src_dtype.

    Returns:
        dict: New dictionary with converted values.
    """
    if isinstance(input, dict):
        return {key: convert_dict_valuetype(value, src_dtype, convert_func)
                for key, value in input.items()}
    elif isinstance(input, src_dtype):
        return convert_func(input)
    else:
        return input

# ===============================
#   Observation Configuration
# ===============================
import os
import torch
from rlbench.observation_config import ObservationConfig, CameraConfig
from rlbench.noise_model import NoiseModel, Identity
from pyrep.const import RenderMode

# Define per-camera settings for all views
Cam_cfg = CameraConfig(
    rgb=True,
    rgb_noise=Identity(),
    depth=False,
    depth_noise=Identity(),
    point_cloud=False,
    mask=False,
    image_size=(128, 128),
    render_mode=RenderMode.OPENGL3,
    masks_as_one_channel=False,
    depth_in_meters=False
)

# Define which sensory and proprioceptive signals to record
Obs_config = ObservationConfig(
    left_shoulder_camera=Cam_cfg,
    right_shoulder_camera=Cam_cfg,
    overhead_camera=Cam_cfg,
    wrist_camera=Cam_cfg,
    front_camera=Cam_cfg,
    joint_velocities=True,
    joint_velocities_noise=Identity(),
    joint_positions=True,
    joint_positions_noise=Identity(),
    joint_forces=False,
    joint_forces_noise=Identity(),
    gripper_open=True,
    gripper_pose=True,
    gripper_matrix=False,
    gripper_joint_positions=False,
    gripper_touch_forces=False,
    wrist_camera_matrix=False,
    record_gripper_closing=False,
    task_low_dim_state=False,
)

# ===============================
#      Policy / Training Config
# ===============================
seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_ratio = 0.8          # ratio of data for training vs testing
test_num = 20              # number of test samples
act_joint = True           # use joint-space actions
num_worker = 4             # dataloader workers
pin_memory = True          # enable pinned memory for GPU transfer
action_dim = 8             # dimension of robot action vector
chunking_size = 16         # temporal chunk size for sequence modeling
action_type = 'joint'      # action mode type: 'joint' or 'eepose'

# Export device variable to environment (useful for other scripts)
os.environ['DEVICE'] = device


# ===============================
#       Demonstration Config
# ===============================
from rlbench.tasks import (
    OpenDrawer, PushButtons, StackWine, SlideBlockToTarget,
    SweepToDustpan, PlaceCups, TurnTap 
)

# Bundle demonstration-related configurations into a MyNameTuple for convenience
Demo_conifg = MyNameTuple({
    'seed': seed,
    'task_list': [
        'OpenDrawer',
        'PushButtons',
        'StackWine',
        'SlideBlockToTarget',
        'SweepToDustpan',
        'TurnTap'
    ],
    'num_episode': 100,     # number of different demonstrations per task
    'show_demo': False,     # whether to visualize the demo generation
    'action_type': action_type,
})


# ===============================
#       Training Config
# ===============================
n_propri = 1 # the more past actions the lower robustness
act_dim  = 8
Config_list = {
    'ACT':{
        'policy_config' : {
            'lr': 1e-5,
            'num_queries': chunking_size, # number of the output actions, chunking size
            'hidden_dim': 512,
            'action_type': action_type, # ee_pose # absolute value
            'action_dim': 8,   # dimension of each output action, jpos + open or ee_pose + open
            'propri_dim': 8,   # dimension of the proprioception feature, joint position + gripper states
            'env_dim': 0,      # dimension of env states
            'kl_weight': 10,   # for ACT the loss is defined as: loss = l1_loss + kl_weight*kl_loss
            'dim_feedforward': 3200,
            'rl_backbone': 1e-5,
            'backbone': 'resnet18',
            'enc_layers': 4,
            'dec_layers': 7,
            'n_header': 8,
            'camera_names': ['front_rgb', 'wrist_rgb'], #, 'left_shoulder_rgb', 'right_shoulder_rgb'], 
        },
        'train_config' : MyNameTuple({
            'seed': seed,
            'device': device,
            # dataset
            'train_ratio': train_ratio,
            'test_num'   : test_num,
            'act_joint'  : act_joint,
            'num_worker' : num_worker,
            'pin_memory' : pin_memory,
            'prefetch_factor': 1,
            'train_batchsize': 8,
            'valid_batchsize': 8,
            # training 
            'use_indices': True, # if False, then num_epochs is learning steps
            'num_epochs': 200, 
            'log_every_epoch': 20,
            'log_every_step' : 100,
            'lr_schduler_cfg': MyNameTuple({
                'name': 'cosine',
                'lr_warmup_steps': 500,
                'gradient_accumulate_every': 1,
            }),
            'use_ema': False,
            'early_stop_cfg': MyNameTuple({
                'activate': False, 
                'patience': 200,
                'verbose': True, 
            }),
        }),
    },
    'DiffusionPolicy': {
        'policy_config': MyNameTuple({
            'action_type': action_type,
            'num_queries': chunking_size,
            'action_dim' : action_dim,
            'vision_config': MyNameTuple({
                'shape_meta': {
                    'obs': {
                            'front_rgb': {
                                'shape': [3, 128, 128],
                                'type': 'rgb'
                            },
                            'wrist_rgb': {
                                'shape': [3, 128, 128],
                                'type': 'rgb'
                            },
                        }
                },
                'resize_shape': None,
                'random_crop': False,
                'crop_shape': None, # [76, 76], # shrink a little if random_crop is True
                'use_group_norm': True,
                'share_rgb_model': False,
                'imagenet_norm': True,
                'rgb_model_cfg': {
                    'input_shape': [3, 128, 128],
                    'output_size': 64,
                }
            }),
            'diffuser_config': MyNameTuple({
                'num_train_timesteps': 30, # infer step, 50 for instance, the larger the more stable
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'beta_schedule': 'squaredcos_cap_v2',
                'variance_type': 'fixed_small',
                'clip_sample': True,
                'prediction_type': 'epsilon'
            }),
            'diffmodel_config': MyNameTuple({
                'input_dim': 8,  # equals to action dimension
                'local_cond_dim': None,
                'global_cond_dim': 64*2 + 8, # output size of the rgb_model x num_cams + propri_dim, 
                                             # observation as global dimension (plus proprioception feature)
                'diffusion_step_embed_dim': 128,
                'down_dims': [512, 1024, 2048],
                'kernel_size': 5,
                'n_groups': 8,
                'cond_predict_scale': True
            }),
            'optimizer_config': MyNameTuple({
                'lr': 1.0e-4,
                'betas': [0.95, 0.999],
                'eps': 1.0e-8,
                'weight_decay': 1.0e-6,
            }),
        }),
        'train_config': MyNameTuple({
            # dataset settings
            'seed': seed,
            'device': device,
            'train_ratio': train_ratio,
            'train_num': 30,
            'num_worker' : num_worker,
            'pin_memory' : pin_memory,
            'prefetch_factor': 1,
            'train_batchsize': 8,
            'valid_batchsize': 8,
            'test_num'   : test_num,
            'use_indices': True,
            # training settings
            'num_epochs': 200, # 8000
            'log_every_epoch': 20,
            'log_every_step' : 100,
            'lr_schduler_cfg': MyNameTuple({
                'name': 'cosine',
                'lr_warmup_steps': 500,
                'gradient_accumulate_every': 1,
            }),
            'use_ema': True,
            'ema_config': MyNameTuple({
                'update_after_step': 0,
                'inv_gamma': 1.0,
                'power': 0.75,
                'min_value': 0.0,
                'max_value': 0.9999
            }),
            'early_stop_cfg': MyNameTuple({
                'activate': False, 
                'patience': 80,
                'verbose': True, 
            })
        })
    },
    'DiffusionPolicy_visonly': {
        'policy_config': MyNameTuple({
            'action_type': action_type,
            'num_queries': chunking_size,
            'action_dim' : action_dim,
            'vision_config': MyNameTuple({
                'shape_meta': {
                    'obs': {
                            'front_rgb': {
                                'shape': [3, 128, 128],
                                'type': 'rgb'
                            },
                            'wrist_rgb': {
                                'shape': [3, 128, 128],
                                'type': 'rgb'
                            },
                        }
                },
                'resize_shape': None,
                'random_crop': False,
                'crop_shape': None,
                'use_group_norm': True,
                'share_rgb_model': False,
                'imagenet_norm': True,
                'rgb_model_cfg': {
                    'input_shape': [3, 128, 128],
                    'output_size': 64,
                }
            }),
            'diffuser_config': MyNameTuple({
                'num_train_timesteps': 30, # infer step, 50 for instance, the larger the more stable
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'beta_schedule': 'squaredcos_cap_v2',
                'variance_type': 'fixed_small',
                'clip_sample': True,
                'prediction_type': 'epsilon'
            }),
            'diffmodel_config': MyNameTuple({
                'input_dim': 8,  # equals to action dimension
                'local_cond_dim': None,
                'global_cond_dim': 64*2, # output size of the rgb_model x num_cams + propri_dim, 
                                         # observation as global dimension (plus proprioception feature)
                'diffusion_step_embed_dim': 128,
                'down_dims': [512, 1024, 2048],
                'kernel_size': 5,
                'n_groups': 8,
                'cond_predict_scale': True
            }),
            'optimizer_config': MyNameTuple({
                'lr': 1.0e-4,
                'betas': [0.95, 0.999],
                'eps': 1.0e-8,
                'weight_decay': 1.0e-6,
            }),
        }),
        'train_config': MyNameTuple({
            # dataset settings
            'seed': seed,
            'device': device,
            'train_ratio': train_ratio,
            'num_worker' : num_worker,
            'pin_memory' : pin_memory,
            'prefetch_factor': 1,
            'train_batchsize': 8,
            'valid_batchsize': 8,
            'test_num'   : test_num,
            'use_indices': True,
            # training settings
            'num_epochs': 200,
            'log_every_epoch': 20,
            'log_every_step' : 100,
            'lr_schduler_cfg': MyNameTuple({
                'name': 'cosine',
                'lr_warmup_steps': 500,
                'gradient_accumulate_every': 1,
            }),
            'use_ema': True,
            'ema_config': MyNameTuple({
                'update_after_step': 0,
                'inv_gamma': 1.0,
                'power': 0.75,
                'min_value': 0.0,
                'max_value': 0.9999
            }),
            'early_stop_cfg': MyNameTuple({
                'activate': False, 
                'patience': 80,
                'verbose': True, 
            })
        })
    },
    'D3P': { 
        'policy_config': MyNameTuple({
            'action_type': action_type,
            'action_dim' : action_dim,
            'n_propri': n_propri,
            'query_every': 4, # tried 8 not good
            'chunking_size': chunking_size,
            'switch_prob': 0.4, # probability of selecting fuse feature as input for diffusion model
            'fuse_config': MyNameTuple({
                'fea_act_dim':n_propri*act_dim,  # n_propri x act_dim
                'fea_vis_dim':2*64, # n_cam x output_size, should fuse with latent action to avoid too frequent modification of the visual features
                'out_dim':64,
            }),
            'vision_config': MyNameTuple({
                'shape_meta': {
                    'obs': {
                            'front_rgb': {
                                'shape': [3, 128, 128],
                                'type': 'rgb'
                            },
                            'wrist_rgb': {
                                'shape': [3, 128, 128],
                                'type': 'rgb'
                            },
                        }
                },
                'resize_shape': None,
                'random_crop': False,
                'crop_shape': None, 
                'use_group_norm': True,
                'share_rgb_model': False, # each input camera view has it own ResNet18 model
                'imagenet_norm': True,
                'rgb_model_cfg': {
                    'input_shape': [3, 128, 128],
                    'output_size': 64,
                }
            }),
            'dko_config': MyNameTuple({ # deep koopman operator configuration
                'act_dim': action_dim,
                'obs_dim': 2*64, # related to the output size of 'rgb_model_cfg'/'output_size'
                'chunking_size': chunking_size+n_propri-1, # chunking_size + n_propri - 1
                'latent_act_dim': 64,
                'lp_cfg':{ # MLP: latent policy
                    'input_dim': 2*64, # related to 'obs_dim'
                    'hidden_dim': 128,
                    'num_hidden_layers': 4, # x mod 2 == 0
                    'output_dim': 64,
                    'dropout': 0.0,
                    'activation': 'ReLU',
                    'use_spectral_norm': False, 
                    'use_norm': False, 
                    'norm_style': 'BatchNorm'
                },
                'act_header_cfg':{
                    'input_dim': 64, # related to the output of lp_cfg
                    'hidden_dim': 256,
                    'num_hidden_layers': 4, # x mod 2 == 0
                    'output_dim': (chunking_size+n_propri-1)*action_dim, # chunking_size + n_propri - 1
                    'dropout': 0.0,
                    'activation': 'ReLU',
                    'use_spectral_norm': False,
                    'use_norm': False, 
                    'norm_style': 'BatchNorm'
                },
                'reg_header_cfg':{ 
                    'input_dim': 2*64, # related to the output of visual encoder, equals to obs_dim
                    'hidden_dim': 256,
                    'num_hidden_layers': 4, # x mod 2 == 0
                    'output_dim': (chunking_size+n_propri-1)*action_dim, # chunking_size + n_propri - 1
                    'dropout': 0.0,
                    'activation': 'ReLU',
                    'use_spectral_norm': False,
                    'use_norm': False, 
                    'norm_style': 'BatchNorm'
                },
            }),
            'diffuser_config': MyNameTuple({
                'num_train_timesteps': 30, # infer step, 50 for instance, the larger the more stable
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'beta_schedule': 'squaredcos_cap_v2',
                'variance_type': 'fixed_small',
                'clip_sample': True,
                'prediction_type': 'epsilon'
            }),
            'diffmodel_config': MyNameTuple({
                'input_dim': act_dim,    # equals to action dimension
                'global_cond_dim': 64, # fea_act_dim / fea_fuse_dim, switch
                'local_cond_dim': None,
                'diffusion_step_embed_dim': 128,
                'down_dims': [512, 1024, 2048],
                'kernel_size': 5,
                'n_groups': 8,
                'cond_predict_scale': True
            }),
            'optimizer_config': MyNameTuple({
                'lr': 1.0e-4,
                'betas': [0.95, 0.999],
                'eps': 1.0e-8,
                'weight_decay': 1.0e-6,
            }),
        }),
        'train_config': MyNameTuple({
            # dataset settings
            'seed': seed,
            'device': device,
            'train_ratio': train_ratio,
            'train_num': 80,
            'num_worker' : num_worker,
            'pin_memory' : pin_memory,
            'prefetch_factor': 1,
            'train_batchsize': 8,
            'valid_batchsize': 8,
            'test_num'   : test_num,
            'use_indices': True,
            'dataset_type': 'EpisodePairDataset_pro',
            # training settings
            'num_epochs': 200, 
            'log_every_epoch': 10,
            'log_every_step' : 50,
            'lr_schduler_cfg': MyNameTuple({
                'name': 'cosine',
                'lr_warmup_steps': 500,
                'gradient_accumulate_every': 1,
            }),
            'use_ema': True,
            'ema_config': MyNameTuple({
                'update_after_step': 0,
                'inv_gamma': 1.0,
                'power': 0.75,
                'min_value': 0.0,
                'max_value': 0.9999
            }),
            'early_stop_cfg': MyNameTuple({
                'activate': False, 
                'patience': 80,
                'verbose': True, 
            })
        })
    },
    'DiffusionPolicy_switch': {
        'policy_config': MyNameTuple({
            'action_type': action_type,
            'num_queries': chunking_size,
            'action_dim' : action_dim,
            'vision_config': MyNameTuple({
                'shape_meta': {
                    'obs': {
                            'front_rgb': {
                                'shape': [3, 128, 128],
                                'type': 'rgb'
                            },
                            'wrist_rgb': {
                                'shape': [3, 128, 128],
                                'type': 'rgb'
                            },
                        }
                },
                'resize_shape': None,
                'random_crop': False,
                'crop_shape': None,
                'use_group_norm': True,
                'share_rgb_model': False,
                'imagenet_norm': True,
                'rgb_model_cfg': {
                    'input_shape': [3, 128, 128],
                    'output_size': 64,
                }
            }),
            'diffuser_config': MyNameTuple({
                'num_train_timesteps': 30, # infer step, 50 for instance, the larger the more stable
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'beta_schedule': 'squaredcos_cap_v2',
                'variance_type': 'fixed_small',
                'clip_sample': True,
                'prediction_type': 'epsilon'
            }),
            'diffmodel_config': MyNameTuple({
                'input_dim': 8,  # equals to action dimension
                'local_cond_dim': None,
                'global_cond_dim': 64*2, # output size of the rgb_model x num_cams + propri_dim, 
                                         # observation as global dimension (plus proprioception feature)
                'diffusion_step_embed_dim': 128,
                'down_dims': [512, 1024, 2048],
                'kernel_size': 5,
                'n_groups': 8,
                'cond_predict_scale': True
            }),
            'optimizer_config': MyNameTuple({
                'lr': 1.0e-4,
                'betas': [0.95, 0.999],
                'eps': 1.0e-8,
                'weight_decay': 1.0e-6,
            }),
        }),
        'train_config': MyNameTuple({
            # dataset settings
            'seed': seed,
            'device': device,
            'train_ratio': train_ratio,
            'train_num': -1, 
            'num_worker' : num_worker,
            'pin_memory' : pin_memory,
            'prefetch_factor': 1,
            'train_batchsize': 8,
            'valid_batchsize': 8,
            'test_num'   : test_num,
            'use_indices': True,
            # training settings
            'num_epochs': 200, 
            'log_every_epoch': 10,
            'log_every_step' : 50,
            'lr_schduler_cfg': MyNameTuple({
                'name': 'cosine',
                'lr_warmup_steps': 500,
                'gradient_accumulate_every': 1,
            }),
            'use_ema': True,
            'ema_config': MyNameTuple({
                'update_after_step': 0,
                'inv_gamma': 1.0,
                'power': 0.75,
                'min_value': 0.0,
                'max_value': 0.9999
            }),
            'early_stop_cfg': MyNameTuple({
                'activate': False, 
                'patience': 80,
                'verbose': True, 
            })
        })
    },
    'DiffusionPolicy_wrec': {
        'policy_config': MyNameTuple({
            'action_type': action_type,
            'num_queries': chunking_size,
            'action_dim' : action_dim,
            'vision_config': MyNameTuple({
                'shape_meta': {
                    'obs': {
                            'front_rgb': {
                                'shape': [3, 128, 128],
                                'type': 'rgb'
                            },
                            'wrist_rgb': {
                                'shape': [3, 128, 128],
                                'type': 'rgb'
                            },
                        }
                },
                'resize_shape': None,
                'random_crop': False,
                'crop_shape': None, 
                'use_group_norm': True,
                'share_rgb_model': False,
                'imagenet_norm': True,
                'rgb_model_cfg': {
                    'input_shape': [3, 128, 128],
                    'output_size': 64,
                }
            }),
            'diffuser_config': MyNameTuple({
                'num_train_timesteps': 30, # infer step, 50 for instance, the larger the more stable
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'beta_schedule': 'squaredcos_cap_v2',
                'variance_type': 'fixed_small',
                'clip_sample': True,
                'prediction_type': 'epsilon'
            }),
            'diffmodel_config': MyNameTuple({
                'input_dim': 8,  # equals to action dimension
                'local_cond_dim': None,
                'global_cond_dim': 64*2, # output size of the rgb_model x num_cams + propri_dim, 
                                         # observation as global dimension (plus proprioception feature)
                'diffusion_step_embed_dim': 128,
                'down_dims': [512, 1024, 2048],
                'kernel_size': 5,
                'n_groups': 8,
                'cond_predict_scale': True
            }),
            'optimizer_config': MyNameTuple({
                'lr': 1.0e-4,
                'betas': [0.95, 0.999],
                'eps': 1.0e-8,
                'weight_decay': 1.0e-6,
            }),
        }),
        'train_config': MyNameTuple({
            # dataset settings
            'seed': seed,
            'device': device,
            'train_ratio': train_ratio,
            'train_num': -1, 
            'num_worker' : num_worker,
            'pin_memory' : pin_memory,
            'prefetch_factor': 1,
            'train_batchsize': 8,
            'valid_batchsize': 8,
            'test_num'   : test_num,
            'use_indices': True,
            # training settings
            'num_epochs': 200, 
            'log_every_epoch': 10,
            'log_every_step' : 50,
            'lr_schduler_cfg': MyNameTuple({
                'name': 'cosine',
                'lr_warmup_steps': 500,
                'gradient_accumulate_every': 1,
            }),
            'use_ema': True,
            'ema_config': MyNameTuple({
                'update_after_step': 0,
                'inv_gamma': 1.0,
                'power': 0.75,
                'min_value': 0.0,
                'max_value': 0.9999
            }),
            'early_stop_cfg': MyNameTuple({
                'activate': False, 
                'patience': 80,
                'verbose': True, 
            })
        })
    },
    'D3P_LSTM': { 
        'policy_config': MyNameTuple({
            'action_type': action_type,
            'action_dim' : action_dim,
            'n_propri': n_propri,
            'query_every': 4, # tried 8 not good
            'chunking_size': chunking_size,
            'switch_prob': 0.4, # probability of selecting fuse feature as input for diffusion model
            'fuse_config': MyNameTuple({
                'fea_act_dim':n_propri*act_dim,  # n_propri x act_dim
                'fea_vis_dim':2*64, # n_cam x output_size, should fuse with latent action to avoid too frequent modification of the visual features
                'out_dim':64,
            }),
            'vision_config': MyNameTuple({
                'shape_meta': {
                    'obs': {
                            'front_rgb': {
                                'shape': [3, 128, 128],
                                'type': 'rgb'
                            },
                            'wrist_rgb': {
                                'shape': [3, 128, 128],
                                'type': 'rgb'
                            },
                        }
                },
                'resize_shape': None,
                'random_crop': False,
                'crop_shape': None, 
                'use_group_norm': True,
                'share_rgb_model': False, # each input camera view has it own ResNet18 model
                'imagenet_norm': True,
                'rgb_model_cfg': {
                    'input_shape': [3, 128, 128],
                    'output_size': 64,
                }
            }),
            'lstm_config': MyNameTuple({
                'in_dim': 128, # 64*2
                'mid_dim': 64,
                'num_layer': 1,
                'win_size': 8 # the same as num_obs in train_cfg
            }),
            'diffuser_config': MyNameTuple({
                'num_train_timesteps': 30, # infer step, 50 for instance, the larger the more stable
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'beta_schedule': 'squaredcos_cap_v2',
                'variance_type': 'fixed_small',
                'clip_sample': True,
                'prediction_type': 'epsilon'
            }),
            'diffmodel_config': MyNameTuple({
                'input_dim': act_dim,    # equals to action dimension
                'global_cond_dim': 64, # fea_act_dim / fea_fuse_dim, switch
                'local_cond_dim': None,
                'diffusion_step_embed_dim': 128,
                'down_dims': [512, 1024, 2048],
                'kernel_size': 5,
                'n_groups': 8,
                'cond_predict_scale': True
            }),
            'optimizer_config': MyNameTuple({
                'lr': 1.0e-4,
                'betas': [0.95, 0.999],
                'eps': 1.0e-8,
                'weight_decay': 1.0e-6,
            }),
        }),
        'train_config': MyNameTuple({
            # dataset settings
            'seed': seed,
            'device': device,
            'train_ratio': train_ratio,
            'train_num': 80,
            'num_worker' : num_worker,
            'pin_memory' : pin_memory,
            'prefetch_factor': 1,
            'train_batchsize': 8,
            'valid_batchsize': 8,
            'test_num'   : test_num,
            'use_indices': True,
            'num_obs'    : 8,
            'dataset_type': 'EpisodeMultiSeqDataset',
            # training settings
            'num_epochs': 200, 
            'log_every_epoch': 10,
            'log_every_step' : 50,
            'lr_schduler_cfg': MyNameTuple({
                'name': 'cosine',
                'lr_warmup_steps': 500,
                'gradient_accumulate_every': 1,
            }),
            'use_ema': True,
            'ema_config': MyNameTuple({
                'update_after_step': 0,
                'inv_gamma': 1.0,
                'power': 0.75,
                'min_value': 0.0,
                'max_value': 0.9999
            }),
            'early_stop_cfg': MyNameTuple({
                'activate': False, 
                'patience': 80,
                'verbose': True, 
            })
        })
    },
    'Default': {
        'policy_config': MyNameTuple({
            'action_type': action_type,
            'num_queries': chunking_size,
        }),
        'train_config': MyNameTuple({
            'seed': seed,
            'device': device,
            'train_ratio': train_ratio,
            'num_worker' : num_worker,
            'pin_memory' : pin_memory,
            'prefetch_factor': 1,
            'train_batchsize': 8,
            'valid_batchsize': 8,
            'test_num'   : test_num,
            'use_indices': False,
            'log_every_epoch': 20,
        }),
    }
}


# ===============================
#       Deployment Config
# ===============================
from itertools import product
from dataclasses import dataclass
@dataclass
class Run:
    """A class representing a single run configuration."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return "".join(f"  {key}: {value}" for key, value in vars(self).items())
    
    def __getitem__(self, key):  # allows run[key] 
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found in Run object.")
    
    def get_all_configs(self):
        # get all atrtributions
        return tuple(vars(self).values())  # use vars to get a dict and convert to tuple

class RunBuilder:
    '''
    Description: 
        Generate run parameters by computing the product of possible configurations.
        Each run corresponds to a unique parameter configuration.
    '''    
    @staticmethod
    def get_runs(params):
        '''
        @param params: dict -- A dictionary containing possible configurations for training.
        @return runs: list -- A list of Run objects representing different configurations.
        '''        
        runs = []
        for values in product(*params.values()):
            run_params = dict(zip(params.keys(), values))
            runs.append(Run(**run_params))
        return runs

# inference configuration (params for deploying the policy in simulation)
import os
_project_dir = os.path.dirname(os.path.abspath(__file__))
check_cfg = MyNameTuple({
    'max_episode_len': {
        'OpenDrawer':256,
        'PushButtons':128, 
        'StackWine':256, 
        'SlideBlockToTarget':256, 
        'SweepToDustpan':360, 
        'PlaceCups':360, 
        'TurnTap':360 
    }, 
    'query_freq': [4], # default
    'random_init': [False, True],
    'test_num': [-1],
    'save_info': True,
    'model_dir': {
        'OpenDrawer':{
            'ACT': f'{_project_dir}/baseline_model/OpenDrawer_joint/ACT_42_2025-02-16-22-42-35',
            'DiffusionPolicy': f'{_project_dir}/baseline_model/OpenDrawer_joint/DiffusionPolicy_42_2025-02-18-21-04-27',
            'DiffusionPolicy_visonly': f'{_project_dir}/logs/OpenDrawer_joint/DiffusionPolicy_visonly_42_2025-02-24-21-19-26',
            'D3P': f'{_project_dir}/logs/OpenDrawer_joint/D3P_42_2025-02-19-09-02-32', 
            'D3P_LSTM': f'{_project_dir}/logs/OpenDrawer_joint/D3P_LSTM_42_2025-08-29-23-11-12', 
            'DiffusionPolicy_switch': f'{_project_dir}/logs/OpenDrawer_joint/DiffusionPolicy_switch_42_2025-02-24-21-23-00',
            'DiffusionPolicy_wrec': f'{_project_dir}/logs/OpenDrawer_joint/DiffusionPolicy_wrec_42_2025-08-30-14-43-01', 
        },
        'PushButtons':{
            'ACT': f'{_project_dir}/baseline_model/PushButtons_joint/ACT_42_2025-02-17-03-56-13',
            'DiffusionPolicy': f'{_project_dir}/baseline_model/PushButtons_joint/DiffusionPolicy_42_2025-02-19-03-45-23',
            'DiffusionPolicy_visonly': f'{_project_dir}/logs/PushButtons_joint/DiffusionPolicy_visonly_42_2025-02-25-03-59-48',
            'D3P': f'{_project_dir}/logs/PushButtons_joint/D3P_42_2025-02-20-02-27-24', 
            'D3P_LSTM': f'{_project_dir}/logs/PushButtons_joint/D3P_LSTM_42_2025-08-30-09-42-32', 
            'DiffusionPolicy_switch': f'{_project_dir}/logs/PushButtons_joint/DiffusionPolicy_switch_42_2025-02-25-04-47-18',
            'DiffusionPolicy_wrec': f'{_project_dir}/logs/PushButtons_joint/DiffusionPolicy_wrec_42_2025-08-30-23-47-42', 
        },
        'StackWine':{
            'ACT':f'{_project_dir}/baseline_model/StackWine_joint/ACT_42_2025-02-17-17-12-19',
            'DiffusionPolicy': f'{_project_dir}/baseline_model/StackWine_joint/DiffusionPolicy_42_2025-02-19-15-07-51',
            'DiffusionPolicy_visonly': f'{_project_dir}/logs/StackWine_joint/DiffusionPolicy_visonly_42_2025-02-25-08-26-03',
            'D3P': f'{_project_dir}/logs/StackWine_joint/D3P_42_2025-02-22-21-38-54', 
            'D3P_LSTM': f'{_project_dir}/logs/StackWine_joint/D3P_LSTM_42_2025-08-30-15-00-07', 
            'DiffusionPolicy_switch': f'{_project_dir}/logs/StackWine_joint/DiffusionPolicy_switch_42_2025-02-25-09-37-21', 
            'DiffusionPolicy_wrec': f'{_project_dir}/logs/StackWine_joint/DiffusionPolicy_wrec_42_2025-08-31-05-43-21', 
        },
        'SlideBlockToTarget':{
            'ACT':f'{_project_dir}/baseline_model/SlideBlockToTarget_joint/ACT_42_2025-02-18-08-21-53',
            'DiffusionPolicy': f'{_project_dir}/baseline_model/SlideBlockToTarget_joint/DiffusionPolicy_42_2025-02-21-00-13-38',
            'DiffusionPolicy_visonly': f'{_project_dir}/logs/SlideBlockToTarget_joint/DiffusionPolicy_visonly_42_2025-02-26-03-53-55',
            'D3P': f'{_project_dir}/logs/SlideBlockToTarget_joint/D3P_42_2025-02-26-04-56-33', 
            'D3P_LSTM': f'{_project_dir}/logs/SlideBlockToTarget_joint/D3P_LSTM_42_2025-09-02-11-07-00', 
            'DiffusionPolicy_switch': f'{_project_dir}/logs/SlideBlockToTarget_joint/DiffusionPolicy_switch_42_2025-02-26-07-52-04', 
            'DiffusionPolicy_wrec': f'{_project_dir}/logs/SlideBlockToTarget_joint/DiffusionPolicy_wrec_42_2025-08-31-01-21-30', 
        },
        'SweepToDustpan':{
            'ACT':f'{_project_dir}/baseline_model/SweepToDustpan_joint/ACT_42_2025-02-18-02-20-34',
            'DiffusionPolicy': f'{_project_dir}/baseline_model/SweepToDustpan_joint/DiffusionPolicy_42_2025-02-20-01-31-08',
            'DiffusionPolicy_visonly': f'{_project_dir}/logs/SweepToDustpan_joint/DiffusionPolicy_visonly_42_2025-02-25-20-11-45',
            'D3P': f'{_project_dir}/logs/SweepToDustpan_joint/D3P_42_2025-02-24-06-48-43',
            'D3P_LSTM': f'{_project_dir}/logs/SweepToDustpan_joint/D3P_LSTM_42_2025-08-29-23-15-26', 
            'DiffusionPolicy_switch': f'{_project_dir}/logs/SweepToDustpan_joint/DiffusionPolicy_switch_42_2025-02-25-23-09-06', 
            'DiffusionPolicy_wrec': f'{_project_dir}/logs/SweepToDustpan_joint/DiffusionPolicy_wrec_42_2025-08-30-14-45-41', 
        },
        'TurnTap':{
            'ACT':f'{_project_dir}/baseline_model/TurnTap_joint/ACT_42_2025-02-18-13-35-59',
            'DiffusionPolicy': f'{_project_dir}/baseline_model/TurnTap_joint/DiffusionPolicy_42_2025-02-21-06-47-59',
            'DiffusionPolicy_visonly': f'{_project_dir}/logs/TurnTap_joint/DiffusionPolicy_visonly_42_2025-02-26-10-37-23',
            'D3P': f'{_project_dir}/logs/TurnTap_joint/D3P_42_2025-02-25-02-38-21', 
            'D3P_LSTM': f'{_project_dir}/logs/TurnTap_joint/D3P_LSTM_42_2025-09-02-11-06-14', 
            'DiffusionPolicy_switch': f'{_project_dir}/logs/TurnTap_joint/DiffusionPolicy_switch_42_2025-02-26-15-23-54', 
            'DiffusionPolicy_wrec': f'{_project_dir}/logs/TurnTap_joint/DiffusionPolicy_wrec_42_2025-08-31-10-33-26', 
        },
    }
})

# ===============================
#       Distillation Config
# ===============================
# distillation results
distill_cfg = MyNameTuple({
    'model_dir': {
        'OpenDrawer':{
            'D3P': f'{_project_dir}/logs_distill/OpenDrawer_joint/D3P_42_2025-09-09-22-35-03',
        }, 
        'PushButtons':{
            'D3P': f'{_project_dir}/logs_distill/PushButtons_joint/D3P_42_2025-09-09-22-38-53', 
        },  
        'StackWine':{
            'D3P': f'{_project_dir}/logs_distill/StackWine_joint/D3P_42_2025-09-09-22-51-35', 
        }, 
        'SlideBlockToTarget':{
            'D3P': f'{_project_dir}/logs_distill/SlideBlockToTarget_joint/D3P_42_2025-09-09-22-52-46', 
        },
        'SweepToDustpan':{
            'D3P': f'{_project_dir}/logs_distill/SweepToDustpan_joint/D3P_42_2025-09-10-15-43-57', 
        },
        'TurnTap':{
            'D3P': f'{_project_dir}/logs_distill/TurnTap_joint/D3P_42_2025-09-09-22-54-53', 
        },
    }
})
