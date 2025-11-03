# Conda env
d3p_env
do not forget to change the directory path of detr when deploying ACT to a new environment, in 'act_utils.py' -> 'sys.path.append('/home/camp/CampUsers/dianye/tabletop_rl/detr')'

# Configurations
## Dataset for training
1. observation: 
   - front_rgb: 128 x 128
   - wrist_rgb: 128 x 128
   - joint_position: 1 x 7
   - discrete_gripper_state:1 x 1 (0: close, 1: open)
2. action: 
   - [joint_position discrete_gripper_cmd] : 1 x 8

The low dimension data, either in observations or action will be normalized during the training, and the output will be denormalized during the deployment.

### Dataset structure
for each batch, we will have:
- image_data ([8, 2, 3, 128, 128]), batch size x camera_views x image channel x image height x image width
- jpos_data ([8, 8]), batch_size x dimension of low-dim proprioception data 
- act_data ([8, 16, 8]), batch_size x chunking size x dim of action, also defined as padded action
- pad_flag ([8, 16]), batch size x padding_flag (for label actions that is less than chunking size, will be padded by zeros, which will not be counted when computing the loss), also defined as padded flag

The data will first be loaded as an episode dataset, and then loaded as a Dataloader.

# Training
The method can be separated into two parts:
Policy: in charge of accepts the input signals and output loss items in the training mode, or output action in the evaluation mode. Thereby, governing the learning and deploy process. Including setting of an optimizer.
Model: integrated into the policy, using for approximation purpose, build different model structure etc.. 



# Evaluation


# Results Analysis


# Add new method
1. config_const.py 
   - Config_list -> 'ACT' for instance, the configuration is a dictionary, including a dict. of policy_config and a dict. of train_config. 


# Trouble shoot, qt failures
1. opencv 4.11.0
2. pip uninstall opencv-python opencv-python-headless opencv-contrib-python
3. pip install opencv-python opencv-contrib-python
4. pip install opencv-python-headless

pip uninstall opencv-python-headless # allows you to use cv2.imshow(*)


# Notes
check_config.py save the model path
config_const.py define the model configurations
2025.08.29
1. add "DKODPolicy_lstm" and "D3P_LSTM" to dko_policy_module_utils.py
2. add "EpisodeMultiSeqDataset" to dko_policy_dataset_utils.py
3. modify the load_data() in main_train_multi_methods.py
4. the window size of LSTM is set to 8
5. train the policy:
   1. 