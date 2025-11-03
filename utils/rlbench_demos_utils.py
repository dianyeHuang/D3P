"""
Module: rlbench_demos_utils.py
Description:
    Utilities for generating, parsing, and saving RLBench demonstration data.
    This script provides functions to:
      - Convert demonstration episodes from RLBench into structured dictionaries.
      - Store observations and actions into HDF5 files.
      - Load stored HDF5 files back into memory in various formats for training.
      - Retrieve specific timesteps or sequences for imitation learning.

    The stored dataset follows a hierarchical HDF5 structure resembling ROS topic naming:
        /observations/images/
        /observations/proprioception/
        /actions/

    The overall goal is to organize robot demonstrations for learning policies such as ACT.
"""

'''
    Generate the demonstration data
    and save the (observation, action) pairs into a hdf5 file
'''

''' Parsing the generated demonstration
------------
Observations obtained from RLBench:
1. visual signals
    - each camera has 4 types of modalities: _depth, _mask, _point_cloud, _rgb (follows the RGB sequence)
    - RLBench places cameras in 5 positions: front, overhead, left_shoulder, right_shoulder, wrist
2. joints
    - _positions, _velocities, _forces
3. gripper
    - _pose(position+quaternion), _open, _touch_forces, _joint_positions(usually None), _matrix (usually None)

The generated demos is a essentially the list of list: List(List(observations)), and should be 
processed one-by-one:
    Demos    -> multiple demonstration set
    Demos[0] -> get one episode data, Episode
    Episode._observations -> get a list of observations which stores the infos listed above
-------------
Demonstrations
    - Episodes
        - timestep
            - observations
            - actions
-------------

Data that can be recorded:
[
'left_shoulder_rgb', 'left_shoulder_depth', 'left_shoulder_mask', 'left_shoulder_point_cloud', 'right_shoulder_rgb', 
'right_shoulder_depth', 'right_shoulder_mask', 'right_shoulder_point_cloud', 'overhead_rgb', 'overhead_depth', 
'overhead_mask', 'overhead_point_cloud', 'wrist_rgb', 'wrist_depth', 'wrist_mask', 'wrist_point_cloud', 
'front_rgb', 'front_depth', 'front_mask', 'front_point_cloud', 'joint_velocities', 'joint_positions', 
'joint_forces', 'gripper_open', 'gripper_pose', 'misc', 'max_timesteps'
]
Action as command:
    - Joint position, gripper_discrete_state (0/1)

'''
import os
import h5py
import numpy as np

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config_const import *

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def episode2dict(episode):
    '''
        converting one demo episode into a dict consisting of list of signals
    '''
    # define the basic structure of the dictionary
    episode_dict = dict()
    obs_key_list = list()
    obs0 = episode[0].__dict__
    for key, value in obs0.items():
        if value is not None: 
            obs_key_list.append(key)
            episode_dict[key] = list()
    episode_dict['max_timesteps'] = len(episode)
    # appending the observation at each timestep
    for obs in episode:
        obs_dict = obs.__dict__
        for key in obs_key_list:
            episode_dict[key].append(obs_dict[key])
    return episode_dict

def dump_episode2hdf5(episode_dict:dict, save_path, sim=True):
    '''
    This function can convert the episode data from RLBench to a hdf5 file for ACT training, 
    and it is also a way to save the demonstrations into a hdf5 file

    hdf5 is a data format for storing the data hierarchically. we adopt hierarchical name as
    ros topic to record the demos, for each demo, it follows a structure like this:

        /observations/proprioception/xxx (can be joint position, velocity or pose of the ee)
        /observations/images/xxx (can be images from the cameras mounted at the wrist or shoulder
                                    of the robot, or fixed camera from the front, top or side view)
        /observations/3d_pointcloud/xxx (a rarely explored but currently hotspot modality)
        /actions (this is what the policy should learn)
    ------------
    Observations obtained from RLBench:
    1. visual signals
        - each camera has 4 types of modalities: _depth, _mask, _point_cloud, _rgb
        - RLBench places cameras in 5 positions: front, overhead, left_shoulder, right_shoulder, wrist
    2. joints
        - _positions, _velocities, _forces
    3. gripper
        - _pose, _open, _touch_forces, _joint_positions(usually None), _matrix (usually None)

    The generated demos is a essentially the list of list: List(List(observations)), and should be 
    processed one-by-one
    -------------
    '''
    '''
    To add the variables, do not forget to create_dataset in the following codes
        @episode_dict: dictionary that stores the observations and actions of the demos
        @save_path: file path for saving the demo as hdf5 file, suffix is not neccessary
        @sim: indicate the dataset is constructed based on the simulation results or not
    '''
    
    # print(episode_dict.keys())
    # "dict_keys(['left_shoulder_rgb', 'right_shoulder_rgb', 'overhead_rgb', 'wrist_rgb', 
    # 'front_rgb', 'joint_velocities', 'joint_positions', 'gripper_open', 'gripper_pose', 
    # 'misc', 'max_timesteps'])"

    image_rgb_res = episode_dict['front_rgb'][0].shape[:2] # rgb image resolution: height x width
    joint_dim   = len(episode_dict['joint_positions'][0])  # dimension of joint configuration
    data_dict = {
        'max_timesteps': episode_dict['max_timesteps'],
        
        '/observations/images/front_rgb': np.array(episode_dict['front_rgb']), # dtype: uint8
        '/observations/images/wrist_rgb': np.array(episode_dict['wrist_rgb']),
        '/observations/images/resolution_rgb': image_rgb_res,
        
        '/observations/proprioception/jpos': np.array(episode_dict['joint_positions']),
        '/observations/proprioception/jvel': np.array(episode_dict['joint_velocities']),
        '/observations/proprioception/joint_dim': joint_dim,
        '/observations/proprioception/gripper_pose': np.array(episode_dict['gripper_pose']),
        '/observations/proprioception/gripper_open': np.array(episode_dict['gripper_open']).reshape(-1, 1),
        
        '/actions/jpos': np.array(episode_dict['joint_positions']),
        '/actions/epos': np.array(episode_dict['gripper_pose']),
        '/actions/open': np.array(episode_dict['gripper_open']).reshape(-1, 1),
    }
    
    max_timesteps = data_dict['max_timesteps']
    with h5py.File(save_path+'.hdf5', 'w', rdcc_nbytes=1024**2*2) as h5f:
        # attribution
        h5f.attrs['max_timesteps'] = max_timesteps
        h5f.attrs['sim'] = sim  # can save the description of the dataset, like whether it is from sim or real
                                # or whether it is from RLBench or other benchmark dataset
        # observation
        obs = h5f.create_group('observations')
        
        image = obs.create_group('images')
        image_res = data_dict['/observations/images/resolution_rgb']
        _ = image.create_dataset('front_rgb', (max_timesteps, *image_res, 3), dtype='uint8', \
                                 chunks=(1, *image_res, 3))
        _ = image.create_dataset('wrist_rgb', (max_timesteps, *image_res, 3), dtype='uint8', \
                                 chunks=(1, *image_res, 3))

        propri = obs.create_group('proprioception')
        joint_dim = data_dict['/observations/proprioception/joint_dim']
        _ = propri.create_dataset('jpos', (max_timesteps, joint_dim))
        _ = propri.create_dataset('jvel', (max_timesteps, joint_dim))
        _ = propri.create_dataset('gripper_pose', (max_timesteps, 7)) # position 3 + quaternion 4 = 7
        _ = propri.create_dataset('gripper_open', (max_timesteps, 1)) # open=1, close=0
        
        # action
        action  = h5f.create_group('actions')
        _ = action.create_dataset('jpos', (max_timesteps, joint_dim)) # joint posisiton
        _ = action.create_dataset('epos', (max_timesteps, 7)) # pose, position + quaternion
        _ = action.create_dataset('open', (max_timesteps, 1)) # open/close
        
        # store the data
        del data_dict['max_timesteps']
        del data_dict['/observations/images/resolution_rgb']
        del data_dict['/observations/proprioception/joint_dim']
        for name, data_arr in data_dict.items():
            # print('name:', name)
            h5f[name][...] = data_arr    
        # print('episode data has been converted to the hdf5 file!')

def load_all_hdf52episode(filepath):
    '''
        Load the hd5f file, should be compatiable with the 'dump' function
    '''
    if not os.path.isfile(filepath):
        print(f'Dataset does not exist at \n{filepath}\n')
        exit()
    
    with h5py.File(filepath, 'r') as h5f:
        episode_dict = MyNameTuple({
            'front_rgb':h5f['/observations/images/front_rgb'][()],
            'wrist_rgb':h5f['/observations/images/wrist_rgb'][()],
            'joint_pos':h5f['/observations/proprioception/jpos'][()],
            'joint_vel':h5f['/observations/proprioception/jvel'][()],
            'ee_pose'  :h5f['/observations/proprioception/gripper_pose'][()],
            'ee_open'  :h5f['/observations/proprioception/gripper_open'][()],
            'act_jpos' :np.hstack((h5f['/actions/jpos'][()], h5f['/actions/open'][()])),
            'act_epos' :np.hstack((h5f['/actions/epos'][()], h5f['/actions/open'][()])),
        })
        # image_keys = h5f['/observations/images/'].keys()
        # print('length of episode: ', episode_dict.joint_pos.shape[0]) # get the length of episode
    return episode_dict

def get_hdf52length(filepath, act_joint):
    '''
        Load the hd5f file, should be compatiable with the 'dump' function
        get the length of the demonstrated episode
    '''
    if not os.path.isfile(filepath):
        print(f'Dataset does not exist at \n{filepath}\n')
        exit()
        
    act_key = '/actions/jpos' if act_joint else '/actions/epos'
    with h5py.File(filepath, 'r') as h5f:
        action = np.hstack((h5f[act_key][()], h5f['/actions/open'][()]))
        act_shape = action.shape
        eps_len = act_shape[0]
    return eps_len

def load_one_hdf52episode(filepath, act_joint, start_ts=None):
    '''
        Load the hd5f file, should be compatiable with the 'dump' function
    '''
    if not os.path.isfile(filepath):
        print(f'Dataset does not exist at \n{filepath}\n')
        exit()
    
    act_key = '/actions/jpos' if act_joint else '/actions/epos'
    with h5py.File(filepath, 'r') as h5f:
        action = np.hstack((h5f[act_key][()], h5f['/actions/open'][()]))
        act_shape = action.shape
        eps_len = act_shape[0]

        if start_ts is None:
            start_ts = np.random.choice(eps_len)
        
        act_len = eps_len - start_ts
        padded_action = np.zeros(act_shape, dtype=np.float32)
        padded_action[:act_len] = action[start_ts:]
        
        padded_flag = np.ones(eps_len)
        padded_flag[:act_len] = 0.0
        
        episode_dict = MyNameTuple({
            'front_rgb':h5f['/observations/images/front_rgb'][start_ts],
            'wrist_rgb':h5f['/observations/images/wrist_rgb'][start_ts],
            'joint_pos':h5f['/observations/proprioception/jpos'][start_ts],
            'joint_vel':h5f['/observations/proprioception/jvel'][start_ts],
            'ee_pose'  :h5f['/observations/proprioception/gripper_pose'][start_ts],
            'ee_open'  :h5f['/observations/proprioception/gripper_open'][start_ts],
            'padded_action':padded_action,
            'padded_flag'  :padded_flag,
        })
        # image_keys = h5f['/observations/images/'].keys()
    return episode_dict

def load_one_hdf52episode_pro(filepath, act_joint, start_ts=None, seq_len=1, n_propri=1):
    '''
        Load the hd5f file, should be compatiable with the 'dump' function
        n_propri data, counting with (n-1) previous data + one current proprioceptive data
    '''
    if not os.path.isfile(filepath):
        print(f'Dataset does not exist at \n{filepath}\n')
        exit()
    
    act_key = '/actions/jpos' if act_joint else '/actions/epos'
    with h5py.File(filepath, 'r') as h5f:
        action_all = np.hstack((h5f[act_key][()], h5f['/actions/open'][()]))
        propri_all = np.hstack((
            h5f['/observations/proprioception/jpos'][()], 
            h5f['/observations/proprioception/gripper_open'][()]
        ))
        propri = np.zeros((n_propri, propri_all.shape[1]), dtype=np.float32)
        preact = np.zeros((n_propri, action_all.shape[1]), dtype=np.float32)
        pre_padded_flag = np.ones((preact.shape[0],), dtype=np.float32)
        
        if start_ts is None:
            start_ts = np.random.choice(eps_len)
            
        replace = propri_all[max(start_ts-n_propri+1, 0):start_ts+1]
        propri[-len(replace):] = replace
        replace = action_all[max(start_ts-n_propri+1, 0):start_ts+1]
        preact[-len(replace):] = replace
        pre_padded_flag[-len(replace):] = 0.0
        
        # future actions
        act_shape = action_all.shape
        eps_len = act_shape[0]
        act_len = eps_len - start_ts
        padded_action = np.zeros(act_shape, dtype=np.float32)
        padded_action[:act_len] = action_all[start_ts:]

        padded_flag = np.ones(eps_len)
        padded_flag[:act_len] = 0.0
        
        # aggregate actions
        padded_action = np.vstack((
            preact[:-1, :], padded_action # exclude the overlapping action
        ))[:seq_len, :]
        padded_flag = np.hstack((
            pre_padded_flag[:-1], padded_flag
        ))[:seq_len]
        
        episode_dict = MyNameTuple({
            'front_rgb':h5f['/observations/images/front_rgb'][start_ts],
            'wrist_rgb':h5f['/observations/images/wrist_rgb'][start_ts],
            'joint_pos':h5f['/observations/proprioception/jpos'][start_ts],
            'joint_vel':h5f['/observations/proprioception/jvel'][start_ts],
            'ee_pose'  :h5f['/observations/proprioception/gripper_pose'][start_ts],
            'ee_open'  :h5f['/observations/proprioception/gripper_open'][start_ts],
            'propri': propri, # n_proprioception data
            'padded_action' : padded_action,
            'padded_flag'   : padded_flag,
        })
        # image_keys = h5f['/observations/images/'].keys()
    return episode_dict


if __name__ == '__main__':
    dataset_dir = 'xxx'

    # load demonstration data
    filepath = os.path.join(dataset_dir, 'OpenDrawer_joint', 'episode_'+str(1).zfill(3)+'.hdf5')
    episode_dict = load_all_hdf52episode(filepath=filepath)
    print('front rgb shape: ', episode_dict.front_rgb.shape)
    print('wrist rgb shape: ', episode_dict.wrist_rgb.shape)
    print('joint pos shape: ', episode_dict.joint_pos.shape)
    print('action    shape: ', episode_dict.act_jpos.shape)
    print('data keys:', episode_dict.keys())
    
    # import mediapy
    # mediapy.show_video(episode_dict.front_rgb, fps=30) # only used in IPython
    
