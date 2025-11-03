"""
Construct dataloader for training, should be compatible with the way of saving the demonstration.
"""

import torch
import numpy as np
from torch.utils.data import Dataset

from utils.data_utils import get_filelist_from_dir
from utils.rlbench_demos_utils import load_one_hdf52episode, load_one_hdf52episode_pro, get_hdf52length

class EpisodePairDataset(Dataset):
    def __init__(self, idx_list:list, dataset_dir:str, norm_stats:dict, query_every:int,
                 chunking_size:int, act_joint:bool=True, use_indices:bool=False):
        super().__init__()
        self.idx_list = idx_list
        self.dataset_dir = dataset_dir
        self.norm_stats  = norm_stats
        self.query_every = query_every
        self.act_joint   = act_joint
        self.use_indices = use_indices
        self.chunking_size = chunking_size
        
        self.h5_filepath_list = get_filelist_from_dir(dataset_dir)
        self.indices_list = list()
        self.eps_len_dict = dict()
        for episode_id in self.idx_list:
            filepath = self.h5_filepath_list[episode_id]
            eps_len = get_hdf52length(filepath, act_joint=self.act_joint)
            if self.use_indices:
                for start_ts in range(eps_len-self.query_every):
                    next_ts = start_ts+self.query_every
                    self.indices_list.append([episode_id, start_ts, next_ts])
            else:
                self.eps_len_dict[str(episode_id)] = eps_len
    
    def get_data_from_file(self, filepath, act_joint, ts):
        # - get episode data
        episode_dict = load_one_hdf52episode(
                        filepath, act_joint=act_joint, start_ts=ts)
        
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
        act_data = torch.from_numpy(episode_dict.padded_action[:self.chunking_size]).float()
        pad_flag = torch.from_numpy(episode_dict.padded_flag[:self.chunking_size]).bool()
        act_data = (act_data - self.norm_stats['act_mean']) / self.norm_stats['act_std']
        
        return image_data, jpos_data, act_data, pad_flag
    
    def __len__(self):
        if self.use_indices:
            return len(self.indices_list)
        return len(self.idx_list)

    def __getitem__(self, index):
        if self.use_indices:
            episode_id, start_ts, next_ts = self.indices_list[index]
        else:
            episode_id = self.idx_list[index]
            eps_len  = self.eps_len_dict[str(episode_id)]
            start_ts = np.random.choice(range(eps_len-self.query_every))
            next_ts  = start_ts+self.query_every
        
        # - load and get observations from hdf5 file
        filepath = self.h5_filepath_list[episode_id]
        
        # print(f'start timestep, next timestep: {start_ts}, {next_ts}')
        
        start_img, start_jpos, start_act, start_pad = self.get_data_from_file(
                                    filepath, act_joint=self.act_joint, ts=start_ts)
        next_img, next_jpos, next_act, next_pad = self.get_data_from_file(
                                    filepath, act_joint=self.act_joint, ts=next_ts)
        
        return  torch.stack((start_img, next_img), dim=0), \
                torch.stack((start_jpos, next_jpos), dim=0), \
                torch.stack((start_act, next_act), dim=0), \
                torch.stack((start_pad, next_pad), dim=0)

class EpisodePairDataset_pro(Dataset):
    def __init__(self, idx_list:list, dataset_dir:str, norm_stats:dict, query_every:int,
                 chunking_size:int, n_propri:int, act_joint:bool=True, use_indices:bool=False):
        super().__init__()
        self.idx_list = idx_list
        self.dataset_dir = dataset_dir
        self.norm_stats  = norm_stats
        self.query_every = query_every
        self.act_joint   = act_joint
        self.n_propri   = n_propri
        self.use_indices = use_indices
        self.chunking_size = chunking_size
        self.seq_len = chunking_size + n_propri - 1 # past act: n_propri; chunking size: current + future steps
        
        self.h5_filepath_list = get_filelist_from_dir(dataset_dir)
        self.indices_list = list()
        self.eps_len_dict = dict()
        for episode_id in self.idx_list:
            filepath = self.h5_filepath_list[episode_id]
            eps_len = get_hdf52length(filepath, act_joint=self.act_joint)
            if self.use_indices:
                for start_ts in range(eps_len-self.query_every):
                    next_ts = start_ts+self.query_every
                    self.indices_list.append([episode_id, start_ts, next_ts])
            else:
                self.eps_len_dict[str(episode_id)] = eps_len
        
        # print('indices list[0]: ', self.indices_list[0])
        # print('dataset size: ', len(self.indices_list))
    
    def get_data_from_file(self, filepath, act_joint, ts):
        # - get episode data
        episode_dict = load_one_hdf52episode_pro(
            filepath, act_joint=act_joint, start_ts=ts,
            seq_len=self.seq_len, n_propri=self.n_propri
        )
        
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
        # print('------------ shape of episode_dict.propri: ', episode_dict.propri.shape)
        # jpos = np.hstack((episode_dict.joint_pos_arr, episode_dict.ee_open_arr))
        propri = episode_dict.propri
        propri_norm  = (propri - self.norm_stats['jpos_mean']) / self.norm_stats['jpos_std']
        
        # print('propri shape: ', propri.shape)
        # print('propri mean shape: ', self.norm_stats['jpos_mean'].shape)
        
        propri_data = torch.from_numpy(propri_norm).float() # joint position + gripper status
        propri_data = propri_data.squeeze(0)
        
        # jvel_data  = torch.from_numpy(episode_dict.joint_vel).float()
        # act_data = torch.from_numpy(episode_dict.padded_action[:self.chunking_size]).float()
        # pad_flag = torch.from_numpy(episode_dict.padded_flag[:self.chunking_size]).bool()
        act_data = torch.from_numpy(episode_dict.padded_action[:self.seq_len]).float()
        pad_flag = torch.from_numpy(episode_dict.padded_flag[:self.seq_len]).bool()
        act_data = (act_data - self.norm_stats['act_mean']) / self.norm_stats['act_std']
        
        # print('act mean shape: ', self.norm_stats['act_mean'].shape)
        # print('act_data shape: ', act_data.shape)
        
        return image_data, propri_data, act_data, pad_flag
    
    def __len__(self):
        if self.use_indices:
            return len(self.indices_list)
        return len(self.idx_list)

    def __getitem__(self, index):
        if self.use_indices:
            episode_id, start_ts, next_ts = self.indices_list[index]
        else:
            episode_id = self.idx_list[index]
            eps_len  = self.eps_len_dict[str(episode_id)]
            start_ts = np.random.choice(range(eps_len-self.query_every))
            next_ts  = start_ts+self.query_every
        
        # - load and get observations from hdf5 file
        filepath = self.h5_filepath_list[episode_id]
        
        # print(f'start timestep, next timestep: {start_ts}, {next_ts}')
        
        start_img, start_jpos, start_act, start_pad = self.get_data_from_file(
                                    filepath, act_joint=self.act_joint, ts=start_ts)
        next_img, next_jpos, next_act, next_pad = self.get_data_from_file(
                                    filepath, act_joint=self.act_joint, ts=next_ts)
        
        return  torch.stack((start_img, next_img), dim=0), \
                torch.stack((start_jpos, next_jpos), dim=0), \
                torch.stack((start_act, next_act), dim=0), \
                torch.stack((start_pad, next_pad), dim=0)

class EpisodeMultiSeqDataset(Dataset):
    def __init__(self, idx_list:list, dataset_dir:str, norm_stats:dict, query_every:int, 
                 num_obs:int, chunking_size:int, n_propri:int, act_joint:bool=True):
        super().__init__()
        self.idx_list = idx_list
        self.dataset_dir = dataset_dir
        self.norm_stats  = norm_stats
        self.query_every = query_every
        self.num_obs     = num_obs
        self.act_joint   = act_joint
        self.n_propri    = n_propri
        self.chunking_size = chunking_size
        self.seq_len = chunking_size + n_propri - 1 # past act: n_propri; chunking size: current + future steps
        
        self.h5_filepath_list = get_filelist_from_dir(dataset_dir)
        self.indices_list = list()
        self.eps_len_dict = dict()
        for episode_id in self.idx_list:
            filepath = self.h5_filepath_list[episode_id]
            eps_len = get_hdf52length(filepath, act_joint=self.act_joint)
            for start_ts in range(eps_len-self.query_every*self.num_obs):
                ts_list = [start_ts+self.query_every*obs_i for obs_i in range(self.num_obs)]
                self.indices_list.append([episode_id]+ts_list)
                # next_ts = start_ts+self.query_every
                # self.indices_list.append([episode_id, start_ts, next_ts])
        
        # print('indices list[0]: ', self.indices_list[0])
        # print('dataset size: ', len(self.indices_list))
    
    def get_data_from_file(self, filepath, act_joint, ts):
        # - get episode data
        episode_dict = load_one_hdf52episode_pro(
            filepath, act_joint=act_joint, start_ts=ts,
            seq_len=self.seq_len, n_propri=self.n_propri
        )
        
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
        # print('------------ shape of episode_dict.propri: ', episode_dict.propri.shape)
        # jpos = np.hstack((episode_dict.joint_pos_arr, episode_dict.ee_open_arr))
        propri = episode_dict.propri
        propri_norm  = (propri - self.norm_stats['jpos_mean']) / self.norm_stats['jpos_std']
        
        # print('propri shape: ', propri.shape)
        # print('propri mean shape: ', self.norm_stats['jpos_mean'].shape)
        
        propri_data = torch.from_numpy(propri_norm).float() # joint position + gripper status
        propri_data = propri_data.squeeze(0)
        
        # jvel_data  = torch.from_numpy(episode_dict.joint_vel).float()
        # act_data = torch.from_numpy(episode_dict.padded_action[:self.chunking_size]).float()
        # pad_flag = torch.from_numpy(episode_dict.padded_flag[:self.chunking_size]).bool()
        act_data = torch.from_numpy(episode_dict.padded_action[:self.seq_len]).float()
        pad_flag = torch.from_numpy(episode_dict.padded_flag[:self.seq_len]).bool()
        act_data = (act_data - self.norm_stats['act_mean']) / self.norm_stats['act_std']
        
        # print('act mean shape: ', self.norm_stats['act_mean'].shape)
        # print('act_data shape: ', act_data.shape)
        
        return image_data, propri_data, act_data, pad_flag
    

    def __len__(self):
        return len(self.indices_list)

    def __getitem__(self, index):
        ret = self.indices_list[index]
        episode_id = ret[0]
        ts_list    = ret[1:]
        
        # - load and get observations from hdf5 file
        filepath = self.h5_filepath_list[episode_id]
        
        img_list, jops_list = list(), list()
        act_list, pad_list  = list(), list()
        for ts in ts_list:
            img, jpos, act, pad = self.get_data_from_file(
                    filepath, act_joint=self.act_joint, ts=ts)
            img_list.append(img)
            jops_list.append(jpos)
            act_list.append(act)
            pad_list.append(pad)
            
        return  torch.stack(img_list, dim=0), \
                torch.stack(jops_list, dim=0), \
                torch.stack(act_list,dim=0), \
                torch.stack(pad_list, dim=0)

