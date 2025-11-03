'''
This script defines utilities to load, normalize, and wrap robot demonstration 
episodes (stored in .hdf5 files) into a PyTorch-compatible Dataset for training.

--------------------
1. Dataset construction:
   - `EpisodeDataset` wraps each demonstration episode as a sample or as 
     per-step queries depending on `use_indices`.
   - Loads RGB observations from multiple cameras (front, wrist).
   - Loads proprioceptive states (joint positions, gripper status).
   - Loads and normalizes actions according to pre-computed statistics.

2. Normalization utilities:
   - `get_norm_stats()` computes per-dimension mean and std for joint states 
     and actions, with optional percentile-based clipping.
   - `get_norm_params()` computes normalization ranges using robust percentiles.

3. Supporting utilities:
   - `get_filelist_from_dir()` collects all episode .hdf5 file paths in a dataset.
   - `set_seed()` ensures deterministic behavior for reproducibility.
   
'''

import os
import torch
import numpy as np
from torch.utils.data  import Dataset

import random
def set_seed(seed=42):
    # there is a paper called: torch.manual seed(3047) is all you need 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
from natsort import natsorted
from utils.rlbench_demos_utils import load_one_hdf52episode, load_all_hdf52episode, get_hdf52length
import glob
def get_filelist_from_dir(dir_path, format='*.hdf5'):
    filepath_list = list()
    os.chdir(dir_path)
    h5_list = natsorted(glob.glob(format))
    for filename in h5_list:
        filepath_list.append(os.path.join(dir_path, filename))
    return filepath_list
    
class EpisodeDataset(Dataset):
    def __init__(self, idx_list, dataset_dir, norm_stats, num_queries, act_joint=True, use_indices=False):
        super().__init__() 
        '''
        Wrap the hdf5 data as a dataset for the imitation learning
            @idx_list   : list(), shuttled indices of the demonstraed episodes   
            @dataset_dir: string, directory of the *.hdf5 files 
            @norm_stats : dict(), statistical results of the proprioception and action data, exclude the image
            @act_joint  : string, True -> action is joint position, False -> action is ee pose
        '''
        self.idx_list    = idx_list
        self.dataset_dir = dataset_dir
        self.norm_stats  = norm_stats
        self.num_queries = int(num_queries)
        self.act_joint   = act_joint
        self.h5_filepath_list = get_filelist_from_dir(dataset_dir)
        
        self.use_indices = use_indices
        if self.use_indices:
            self.indices_list = list()
            for episode_id in self.idx_list:
                filepath = self.h5_filepath_list[episode_id]
                eps_len = get_hdf52length(filepath, act_joint=self.act_joint)
                # print(f'episode id: {episode_id}, length: {eps_len} ...')
                for start_ts in range(eps_len):
                    self.indices_list.append([episode_id, start_ts])

    def __len__(self):
        if self.use_indices:
            return len(self.indices_list)
        return len(self.idx_list)
    
    def __getitem__(self, index):
        if self.use_indices:
            episode_id, start_ts = self.indices_list[index]
        else:
            episode_id = self.idx_list[index]
            start_ts = None
        
        # - load and get observations from hdf5 file
        filepath = self.h5_filepath_list[episode_id]
        episode_dict = load_one_hdf52episode(filepath, act_joint=self.act_joint, start_ts=start_ts) 
        
        # - get observed data
        front_image = episode_dict.front_rgb
        wrist_image = episode_dict.wrist_rgb
        cam_images = np.stack(
            (front_image, wrist_image), axis=0
        )
        cam_images = cam_images / 255.0 # normalization
        image_data = torch.from_numpy(cam_images).float()
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        
        # proprioception
        jpos = np.hstack((episode_dict.joint_pos, episode_dict.ee_open))
        jpos_norm  = (jpos - self.norm_stats['jpos_mean']) / self.norm_stats['jpos_std']
        jpos_data = torch.from_numpy(jpos_norm).float() # joint position + gripper status
        
        # jvel_data  = torch.from_numpy(episode_dict.joint_vel).float()
        act_data = torch.from_numpy(episode_dict.padded_action[:self.num_queries]).float()
        pad_flag = torch.from_numpy(episode_dict.padded_flag[:self.num_queries]).bool()
        act_data = (act_data - self.norm_stats['act_mean']) / self.norm_stats['act_std']
        
        return image_data, jpos_data, act_data, pad_flag

def get_norm_stats(dataset_dir, act_joint=True, percentile=[2, 98]):
    h5_filepath_list = get_filelist_from_dir(dataset_dir)
    num_episodes = len(h5_filepath_list)
    
    all_act  = None
    all_jpos = None
    for filepath in h5_filepath_list:
        episode_dict = load_all_hdf52episode(filepath)
        if act_joint:
            act = episode_dict.act_jpos
        else:
            act = episode_dict.act_epos
        if all_act is None:
            all_act = act
            all_jpos = np.hstack((episode_dict.joint_pos, episode_dict.ee_open))
        else:
            all_act  = np.vstack((all_act, act))
            all_jpos = np.vstack((all_jpos, np.hstack((episode_dict.joint_pos, episode_dict.ee_open))))
    
    if percentile is not None:
        act_mean, act_std, _, _   = get_norm_params(inputs=all_act, percent_list=percentile)
        jpos_mean, jpos_std, _, _ = get_norm_params(inputs=all_jpos, percent_list=percentile)
    else:
        act_mean = np.mean(all_act, axis=0)
        act_std  = np.std(all_act, axis=0)
        act_std  = np.clip(act_std, 1e-2, np.inf)
        
        jpos_mean = np.mean(all_jpos, axis=0)
        jpos_std  = np.std(all_jpos, axis=0)
        jpos_std  = np.clip(jpos_std, 1e-2, np.inf)
        
    norm_stats = {
        'act_mean': act_mean, 'act_std': act_std,
        'jpos_mean': jpos_mean, 'jpos_std': jpos_std
    }
    return norm_stats, num_episodes

def get_norm_params(inputs:np.array, percent_list:list):
    '''
        np.percentile() ranks the inputs at ascending sequence
        and therefore usually i would choose [2, 98]
        the shape of inputs is: batch x input_dim
        to normalize the input, 
    '''
    dtype = inputs.dtype
    
    percent_low = np.percentile(inputs, percent_list[0], axis=0).astype(dtype)
    percent_up  = np.percentile(inputs, percent_list[1], axis=0).astype(dtype)
    norm_nomi   = (percent_low+percent_up)/2.0
    norm_deno   = percent_up - percent_low
    norm_deno[norm_deno<1.0e-8] = 1.0e-8
    
    return norm_nomi, norm_deno, percent_low, percent_up
