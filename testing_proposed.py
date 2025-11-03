'''
Script: testing_proposed.py
Description:
    Deploy and evaluate a trained policy on RLBench tasks, logging rich diagnostics
    and optional rollout videos. 
'''

import json
import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointPosition, EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import *
import collections
import mediapy as media

from collections import deque
from scipy.signal import savgol_filter

class TrajectoryFIFO:
    def __init__(self, action_chunking, action_dim, query, update_num: int = 1):
        """
        Initialize the TrajectoryFIFO buffer.
        
        Args:
            action_chunking (int): Number of action steps in each chunk.
            action_dim (int): Dimensionality of each action.
            query (int): Number of steps processed at a time.
            update_num (int): Number of updates applied at once (default is 1).
        
        Raises:
            AssertionError: If action_chunking is not divisible by query.
        """
        assert action_chunking % query == 0, 'The action_chunking should be divisible by query.'
        self.query = query
        self.action_dim = action_dim
        self.update_num = update_num
        self.action_chunking = action_chunking
        self.buffer_size = self.action_chunking // self.query  # Compute the number of action groups
        self.buffer_shape = (self.buffer_size * update_num, self.action_chunking, self.action_dim)  
        
        self.reset()  # Initialize/reset the buffers

    def reset(self):
        """
        Reset the buffers to store actions, weights, and reconstruction errors.
        All buffers are initialized with zeros.
        """
        self.action_buffer = np.zeros(self.buffer_shape, dtype=np.float32)      # Buffer for action sequences
        self.weight_buffer = np.zeros(self.buffer_shape[:2], dtype=np.float32)  # Buffer for weights
        self.recerr_buffer = np.zeros(self.buffer_shape[0], dtype=np.float32)   # Buffer for reconstruction errors
    
    def update(self, chunkings, weights, recerrs):
        """
        Update the FIFO buffer with new action chunkings, weights, and reconstruction errors.

        Args:
            chunkings (np.ndarray): New action sequences to be added to the buffer.
            weights (np.ndarray): Corresponding weights for the action sequences.
            recerrs (np.ndarray): Reconstruction errors corresponding to action sequences.
        """
        # 0. Roll out old data by shifting the buffer upward, making space for new updates.
        self.action_buffer = np.roll(self.action_buffer, -self.update_num, axis=0)
        self.weight_buffer = np.roll(self.weight_buffer, -self.update_num, axis=0)
        self.recerr_buffer = np.roll(self.recerr_buffer, -self.update_num, axis=0)

        # 1. Left shift each row by `query` positions and pad with zeros at the right.
        self.action_buffer[:, :-self.query] = self.action_buffer[:, self.query:]  # Shift left
        self.action_buffer[:, -self.query:] = 0  # Zero-padding on the right

        self.weight_buffer[:, :-self.query] = self.weight_buffer[:, self.query:]  # Shift left
        self.weight_buffer[:, -self.query:] = 0  # Zero-padding on the right

        # 2. Insert the new data into the last `update_num` rows of the buffer.
        self.action_buffer[-self.update_num:] = chunkings
        self.weight_buffer[-self.update_num:] = weights
        # self.recerr_buffer[-self.update_num:] = recerrs
        self.recerr_buffer[-self.update_num:] = recerrs

    def get_data(self):
        """
        Retrieve a copy of the action and weight buffers while computing weights based on reconstruction errors.

        Returns:
            tuple: A copy of (action_buffer, weighted_buffer), where the weights are adjusted based on reconstruction errors.
        """
        action_mask = np.where(self.weight_buffer < 1e-8, 0, 1)  # Mask: 0 where weight is small, 1 otherwise
        rec_weights = action_mask * self.compute_select_weights(self.recerr_buffer)  # Adjust weights using recerr
        res_weights = self.weight_buffer * rec_weights  # Compute final weighted buffer
        # return self.action_buffer.copy()[:, :self.query], res_weights[:, :self.query]
        return self.action_buffer.copy(), res_weights
    
    def compute_select_weights(self, generr_arr):
        """
        Compute selection weights based on the generative error values.
        
        Args:
            generr_arr (np.ndarray): Array containing generative error values.
        
        Returns:
            np.ndarray: A column vector of weights with shape (n,1).
        """
        _epsilon = 1e-8  # Small constant to prevent division by zero
        generr_arr = generr_arr.reshape(-1)  # Flatten the array
        generr_arr = generr_arr / np.sum(generr_arr)  # Normalize by sum
        
        # The smaller the generative errors, the higher the chances to be selected
        select_weights = 1 / (generr_arr + _epsilon)
        select_weights = select_weights / np.linalg.norm(select_weights, ord=2)  # Normalize using L2 norm
        return select_weights.reshape(-1, 1)  # Return as column vector (n,1)

class ActionAggregator:
    def __init__(self, beta=0.97, action_dim=8, chunking=16, query=8, smooth=5):
        '''
            The following codes are written based on the rules that 
            chunking -> the length of the prediced sequence
            query == chunking//2   # should be and even number, the power of 2
            smooth == query//2 + 1 # should ba an odd number
        '''
        assert smooth%2 != 0, '[class type: ActionAggregator] -> the smoothing window size should be an odd number!'
        self.action_dim = action_dim
        self.chunking = chunking
        self.query    = query
        self.smooth   = smooth
        
        # compute temporal weights
        c_weights = np.array([beta**i for i in range(self.chunking)]) 
        self.temporal_weights = (c_weights / np.sum(c_weights)).reshape(1, -1)
        
        # buffers to be updated
        self.generr_q = deque(maxlen=2)
        self.trafifo = TrajectoryFIFO(
            action_chunking=chunking, action_dim=self.action_dim, query=query, update_num=2 # update num includes two types of action sequence for current query steps
        )
        self.reset()
    
    def reset(self):
        self.trafifo.reset()
        self.generr_q.clear()
        self.prv_vis  = None
        self.prv_fuse = None
        self.cur_vis  = None
        self.cur_fuse = None
        self.prv_exec = None
        self.propri_act = None
        
    
    def update_buffers(self, act_dict:dict):
        '''
            all elements are of torch.tensor() variables
            act_dict = { # action chunking and the corresponding reconstruction errors from the diffusion model
                'cur_act': tensor of shape (1, act_dim)
                'vis' : [vis_act, act_err],
                'fuse': [fuse_act, fuse_err] 
            }
        '''
        # parse act_dict
        vis_acts,  vis_err = act_dict['vis']
        fuse_acts, fuse_err = act_dict['fuse']
        self.propri_act = act_dict['cur_act'].cpu().numpy().copy() # (1， act_dim)
        self.cur_vis  = vis_acts.squeeze(0).cpu().numpy().copy()   # (chunking, act_dim)
        self.cur_fuse = fuse_acts.squeeze(0).cpu().numpy().copy()  # (chunking, act_dim)
        
        # get action sequences
        self.cur_acts = np.stack((self.cur_vis, self.cur_fuse), axis=0)
        
        # compute weights
        self.generr_q.append(vis_err.item())
        self.generr_q.append(fuse_err.item())
        generr_arr = np.array(list(self.generr_q)).reshape(-1) # mind the sequence, corresponding to self.cur_acts
        
        self.trafifo.update(self.cur_acts, self.temporal_weights.copy(), generr_arr)
        
        
    
    def compute_select_weights(self, generr_arr):
        '''
            generr_arr: an (n, ) np.array, denoting the reconstruction errors of 
                        (prv_vis, prv_fuse, cur_vis, cur_fuse) action sequence
        '''
        _epsilon = 1e-8
        # ratio list
        generr_arr = generr_arr.reshape(-1)
        generr_arr = generr_arr/np.sum(generr_arr)
        # the smaller generative errors the higher chances to be selected
        select_weights = 1 / (generr_arr + _epsilon)
        select_weights = select_weights / np.linalg.norm(select_weights, ord=2)
        return select_weights.reshape(-1, 1) # nx1         
    
    def select_actions(self, select_weights:np.array, acts_arr:np.array=None, ret_indices=False):
        '''
            select_weights: (n, chunking)
        '''
        row_max_indices = np.argmax(select_weights, axis=0)
        if _DEBUG:
            print('row_max_indices: ', row_max_indices)
        # col_indices = range(self.query)
        col_indices = range(self.chunking)
        exec_list = list()
        for r, c in zip(row_max_indices, col_indices):
            exec_list.append(
                acts_arr[r, c]
            )
        if ret_indices:
            return np.stack(exec_list, axis=0), row_max_indices
        return np.stack(exec_list, axis=0)
    
    def smooth_trajectory_savgol(self, trajectory, window_size, poly_order=3, ignore_gripper=False):
        '''
            using Savitzky-Golay filter to smooth the high dimensional trajectories
        '''
        smoothed_trajectory = np.copy(trajectory)
        last_idx = trajectory.shape[1]-1
        for d in range(trajectory.shape[1]):
            if ignore_gripper and (d==last_idx): continue
            smoothed_trajectory[:, d] = savgol_filter(
                trajectory[:, d], 
                window_length=window_size, 
                polyorder=poly_order
            )
        return smoothed_trajectory
    
    
    def aggregate(self, act_dict:dict, poly_order=3, ignore_gripper=True, smooth=False):
        '''
            act_dict = { # action chunking and the corresponding reconstruction errors from the diffusion model
                'vis' : [vis_act, act_err],
                'fuse': [fuse_act, fuse_err] 
            }
        '''
        # act_dict = {
        #     'cur_act': xxx
        #     'vis': [
        #         np.random.randn(self.horizon, 8),
        #         0.012
        #     ],
        #     'fuse': [
        #         np.random.randn(self.horizon, 8),
        #         0.001
        #     ]
        # }
        self.update_buffers(act_dict)
        # initial actions
        if self.prv_exec is None:
            vis_err, fuse_err = np.array(list(self.generr_q))
            # select sequence based on the generative errors
            if vis_err < fuse_err: # choose the action sequence with the lowest reconstruction error, i.e., the highest confidence
                final_acts = self.cur_vis
                act_indices = np.array([-1])
            else:
                final_acts = self.cur_fuse
                act_indices = np.array([0])
                
            # augment final acts with the current single act for filtering, no smoothing for the initial actions
            final_acts = self.smooth_trajectory_savgol(
                trajectory=np.vstack((self.propri_act, final_acts)), # augment the proprio_action
                window_size=self.smooth, 
                poly_order=2,
                ignore_gripper=True
            )[1:]
            
            self.prv_exec = final_acts
            return torch.from_numpy(final_acts).unsqueeze(0), act_indices
        
        # select and smooth actions
        acts_arr, weights_arr = self.trafifo.get_data()
        final_acts, act_indices = self.select_actions(weights_arr, acts_arr, ret_indices=True)
        
        if _DEBUG:
            print('selection indices: ', act_indices.reshape(-1))
        
        # augment and smooth actions
        aug_exec_arr = np.vstack((
            self.prv_exec[-1:], final_acts
        ))
        final_acts = self.smooth_trajectory_savgol(
            trajectory=aug_exec_arr, window_size=self.smooth, 
            poly_order=poly_order, ignore_gripper=ignore_gripper
        )
        final_acts = final_acts[1:]
        self.prv_exec = final_acts.copy()
        
        return torch.from_numpy(final_acts).unsqueeze(0), act_indices #.cuda()

def obs2sample(obs, norm_stats, device, propri_deque:collections.deque=None):
    front_image = obs.front_rgb
    wrist_image = obs.wrist_rgb
    cam_images = np.stack(
        (front_image, wrist_image), axis=0
    )
    cam_images = cam_images / 255.0
    image_data = torch.from_numpy(cam_images).float()
    image_data = torch.einsum('k h w c -> k c h w', image_data)

    jpos = np.hstack((obs.joint_positions, obs.gripper_open))
    jpos_norm = (jpos - norm_stats['jpos_mean']) / norm_stats['jpos_std']
    
    if propri_deque is not None:
        if len(propri_deque) == 0:
            propri_dim = jpos_norm.shape[-1]
            for _ in range(propri_deque.maxlen-1):
                propri_deque.append(np.zeros(propri_dim))
        propri_deque.append(jpos_norm)
        propri_data = torch.tensor(list(propri_deque), dtype=torch.float32) 
    else:
        propri_data = torch.from_numpy(jpos_norm).float() # joint position + gripper status
    
    return image_data.to(device).unsqueeze(0), propri_data.to(device).unsqueeze(0)

def clip_joint_positions(joint_positions, safety_margin_deg=5.0):
    """
    Clips each element of a 7-dimensional joint positions vector so that 
    it is at least `safety_margin_deg` (in degrees) away from the mechanical limits.
    
    Parameters:
        joint_positions (array-like): A vector of 7 joint positions in radians.
        safety_margin_deg (float): The safety margin (in degrees) to stay away from the limits.
        
    Returns:
        np.ndarray: The joint positions vector after clipping to the safe limits.
    """
    if len(joint_positions) != 7:
        raise ValueError("The joint positions vector must have a length of 7.")
    
    # Convert safety margin from degrees to radians
    safety_margin_rad = np.deg2rad(safety_margin_deg)
    
    # Define the mechanical joint position limits (in radians) for the Franka Panda robot
    mechanical_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    mechanical_upper_limits = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    
    # Calculate the effective limits by adding/subtracting the safety margin
    effective_lower_limits = mechanical_lower_limits + safety_margin_rad
    effective_upper_limits = mechanical_upper_limits - safety_margin_rad
    
    # Use numpy.clip to restrict each joint position to the effective limits
    clipped_positions = np.clip(joint_positions, effective_lower_limits, effective_upper_limits)
    
    return clipped_positions

def policy_deployment(task_name, task, policy, norm_stats, check_cfg, 
                      test_demo_list, device, method, test_indices, query_freq,
                      info_dir=None, rand_init=True):
    '''
    Deploy and evaluate the learned policy:
        @task_name: name of the task, should be the same as the class name of the task
        @task     : task variable from the RLBench
        @policy   : learned polcy that accepts the observation and output action list
        @norm_stats: norm and unnorm of the observation and action
        @deploy_cfg: decides the maximum length of an episode, chunking size etc..
        @test_demo_list: previously, we split the demonstrations into trainlng and validation,
                         and this test_demo_list includes the random settings that has not
                         been seen during the policy training.
        @query_freq: if policy is 'ACT' then query_freq=1 and temporal_agg=True by default
    '''
    global _num_debug
    
    task_succ_list = list()
    episode_len   = check_cfg.max_episode_len[task_name]
    chunking_size = check_cfg.chunking_size
    flag_once = False
    
    act_agg = ActionAggregator(
        beta=0.97, 
        action_dim = check_cfg['action_dim'],
        chunking=chunking_size, 
        query=query_freq, # chunking_size//2, 
        smooth=query_freq//2 + 1, # chunking_size//4+1
    )
    
    info_all_dict = dict() # track all the information 
    for idx, demo in enumerate(tqdm(test_demo_list)):
        info_dict = {
            'planned_actions': dict(), # model output
            'execute_actions': dict(), # commands
            'joint_positions': dict(), # tracking results
            'vis_actions' : dict(),
            'fuse_actions': dict(),
            'vis_err' : dict(),
            'fuse_err': dict(),
            'act_indices': dict()
        }
        act_agg.reset() # reset the settings
        descriptions, obs = task.reset_to_demo(demo) # if wanna reproduce the scenario
        if not flag_once:
            print(f'\nTask descriptions: ', descriptions)
            flag_once = True
        
        # descriptions, obs = task.reset() # randomly reset the environment
        success = False
        if 'ACT' in method:
            k = 0.01
            check_cfg['temporal_agg'] = True  
            query_freq = 1 # defined by ACT
        else:
            k = 0.01
            
        if check_cfg['temporal_agg']:
            act_dim = check_cfg['action_dim']
            all_time_actions = torch.zeros((episode_len, episode_len+chunking_size, act_dim)) # .to(device)
            
        # randomize the initial position, each joints are set +-10 deg, gripper open
        if rand_init:
            rand_deg = 10
            rand_jpos = obs.joint_positions + np.deg2rad(np.random.uniform(-rand_deg, rand_deg, (7, )))
            rand_jpos = np.hstack((
                clip_joint_positions(rand_jpos), 
                np.array([1.0])
            ))  
            for _ in range(5):
                obs, _, _ = task.step(rand_jpos)
        jposx = np.hstack((obs.joint_positions, obs.gripper_open)) # get initial joint configuration
        
        # save initial joint configuration
        info_dict['planned_actions'][str(-1)] = jposx.tolist()
        info_dict['execute_actions'][str(-1)] = jposx.tolist()
        info_dict['joint_positions'][str(-1)] = jposx.tolist()
        info_dict['vis_actions'][str(-1)]  = jposx.tolist()
        info_dict['fuse_actions'][str(-1)] = jposx.tolist()
        info_dict['vis_err'][str(-1)]  = 0.0
        info_dict['fuse_err'][str(-1)] = 0.0
        info_dict['act_indices'][str(-1)] = 0.0

        # video frames
        vid_list = list()
        vid_list.append(np.hstack((obs.front_rgb, obs.wrist_rgb)))
        with torch.inference_mode():
            for ts in range(episode_len): # max_episode_len defines the maximum steps allowed to complete the task
                # get action from observation
                # if ts % chunking_size == 0:
                if ts % query_freq == 0:
                    # - proc observation
                    obs_img, obs_propri = obs2sample(obs, norm_stats, device, None) # no queue required for baseline
                    
                    # aggregate actions
                    act_dict = policy(obs_propri, obs_img, actions=None, is_pad=None, ret_generr=True) # during the inference, only jpos and img is needed   
                    act_dict['cur_act'] = obs_propri # get current action, abs: joint position/ee pose etc..
                    planned_actions, act_indices = act_agg.aggregate(act_dict, poly_order=2, ignore_gripper=False)
                    
                    fuse_actions, fuse_err = act_dict['fuse']
                    vis_actions, vis_err = act_dict['vis']
                    info_dict['fuse_err'][str(ts)] = fuse_err.item()
                    info_dict['vis_err'][str(ts)] = vis_err.item()
                    info_dict['vis_actions'][str(ts)]  = (vis_actions.squeeze(0).cpu().numpy()*norm_stats['act_std'] + norm_stats['act_mean']).tolist()
                    info_dict['fuse_actions'][str(ts)] = (fuse_actions.squeeze(0).cpu().numpy()*norm_stats['act_std'] + norm_stats['act_mean']).tolist()
                    info_dict['planned_actions'][str(ts)] = (planned_actions.squeeze(0).cpu().numpy()*norm_stats['act_std'] + norm_stats['act_mean']).tolist()
                    info_dict['act_indices'][str(ts)] = act_indices.tolist()
                    
                    if check_cfg['temporal_agg']:
                        all_time_actions[[ts], ts:ts+chunking_size] = planned_actions 
                    

                if check_cfg['temporal_agg']: # for ACT
                    # all_time_actions[[ts], ts:ts+chunking_size] = planned_actions
                    actions_for_curr_step = all_time_actions[:, ts]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    # exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                    exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).unsqueeze(dim=1)
                    act = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True).squeeze(0).cpu()
                else:
                    act = planned_actions[:, ts%query_freq].squeeze(0).cpu()
                act = act*norm_stats['act_std'] + norm_stats['act_mean'] # post processing, convert the output results to the original commands
                
                # step to get new observation
                obs, reward, _ = task.step(act)
                vid_list.append(np.hstack((obs.front_rgb, obs.wrist_rgb)))
                info_dict['execute_actions'][str(ts)] = act.numpy().tolist()
                info_dict['joint_positions'][str(ts)] = np.hstack((obs.joint_positions, obs.gripper_open)).tolist()

                # termination or not
                success, terminate = task._task.success()
                if terminate: break

        if not success:
            task_succ_list.append(False)
        else:
            task_succ_list.append(True)
            
        # save video
        if info_dir is not None:
            video_name = f'id_{test_indices[idx]:03d}_{success}.mp4'
            media.write_video(
                os.path.join(info_dir, video_name), np.array(vid_list), fps=30
            )
        
        info_all_dict[str(test_indices[idx])] = {
            'info': info_dict,
            'success': bool(success),
            'num_steps': ts,
        }
        
        if _DEBUG: 
            _num_debug -= 1
            if _num_debug==0: break
    
    if info_dir is not None:
        json_path = os.path.join(info_dir, f'{method}_{task_name}_log.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(info_all_dict, f, ensure_ascii=False, indent=4)

    return task_succ_list

import os
import pickle
import torch
from config_const import Obs_config, convert_dict_valuetype
from utils.baseline_utils import make_policy
from utils.data_utils import set_seed
from omegaconf import OmegaConf
def load_policy(log_dir, method, ret_policy_only=False):
    # load configurations
    yaml_path = os.path.join(log_dir, 'config.yaml')
    log_conf = OmegaConf.load(yaml_path)
    policy_config = log_conf['method_cfg']['policy_config']
    train_config  = log_conf['method_cfg']['train_config']
    norm_stats = convert_dict_valuetype(
        OmegaConf.to_container(log_conf['norm_stats'], resolve=True), 
        src_dtype=list, convert_func=lambda x: np.array(x)
    )
    
    act_type = policy_config['action_type']
    task_suffix = '_joint' if act_type == 'joint' else '_eepose'
    cur_filedir = os.path.dirname(os.path.abspath(__file__))
    
    device = train_config.device
    seed   = train_config.seed
    

    best_model_path = os.path.join(log_dir, f'last_{method}_{seed}.pth')
    
    # - load policy 
    policy = make_policy(method, policy_config)
    loading_status = policy.load_state_dict(torch.load(best_model_path, map_location=device))
    print('Policy loading status: ', loading_status)
    policy.to(device)
    policy.eval()
    
    if ret_policy_only: return policy
    return policy, norm_stats, task_suffix, cur_filedir, act_type, seed, device

from collections import OrderedDict
from config_const import check_cfg, RunBuilder, act_dim, chunking_size
from tqdm import tqdm

_DEBUG = False
_num_debug = 2

params = OrderedDict(
    method = [
        'D3P',
        # 'D3P_LSTM',
        # 'DiffusionPolicy_wrec'
        # 'DiffusionPolicy_switch'
    ],
    task   = [
        'OpenDrawer', 
        'PushButtons', 
        'StackWine',
        'SlideBlockToTarget',
        'SweepToDustpan',
        'TurnTap'
    ],
    query_freq  = [4],
    test_num    = check_cfg.test_num,
    random_init = check_cfg.random_init,
)
check_cfg['temporal_agg'] = False

def set_res_dir_name(results_dir, run):
    dir_str = f'{run.task}_randjoint_{run.random_init}_query_{run.query_freq}' 
    return os.path.join(results_dir, run.method, dir_str)

if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    res_suffix = 'proposed_res'
    results_dir = os.path.join(project_dir, res_suffix) # mind the path, which is different from the baseline settings
    check_cfg['action_dim'] = act_dim 
    check_cfg['chunking_size'] = chunking_size 
    seen_configs = set()
    
    for run in tqdm(RunBuilder.get_runs(params)):
        config_id = run.get_all_configs()
        if config_id in seen_configs: continue
        seen_configs.add(config_id)
        
        # create directories
        res_dir = None
        if check_cfg.save_info is True:
            res_dir = set_res_dir_name(results_dir, run)
            os.makedirs(res_dir, exist_ok=True)
            print(f'\nsave results to: {res_dir}.')
        
        print('----------------------------------------------')
        print(f'Infering task {run.task}, with method: {run.method} ...')
        print('----------------------------------------------')
        policy, norm_stats, task_suffix, cur_filedir, act_type, seed, device = load_policy(
            log_dir=check_cfg.model_dir[run.task][run.method], method=run.method, ret_policy_only=False
        )
        set_seed(seed)
    
        # '''
        #     @obs_cfg: defines the observation obtained from the simulation
        #     @render : show the simulation environment if render is True
        # '''
        # load demo list and demo indices
        test_demo_list  = list()
        Task_name = run.task+task_suffix
        dataset_dir = os.path.join(cur_filedir, 'dataset', Task_name)
        rec_demos_path  = os.path.join(dataset_dir, 'demos.pkl')
        eps_indices_path= os.path.join(dataset_dir, 'episode_indices.pkl')
        with open(rec_demos_path,   'rb') as f: demos = pickle.load(f)
        with open(eps_indices_path, 'rb') as f: 
            res_dict = pickle.load(f)
            test_indices  = res_dict['test_indices'] # ['valid_indices'], ['train_indices']
            valid_indices = res_dict['valid_indices']
        
        # concatenate valid and test indices, the valiation set does not participate the training
        if run.test_num < 0:
            test_demo_list = [demos[idx] for idx in test_indices] + [demos[idx] for idx in valid_indices]
            test_indices = test_indices.tolist() + valid_indices.tolist()
        else:
            test_demo_list = [demos[idx] for idx in test_indices][:run.test_num]

        # - launch env
        if act_type == 'joint':
            arm_action_mode = JointPosition()
        elif act_type == 'eepose':
            arm_action_mode = EndEffectorPoseViaIK()
        else:
            print(f'No action type: {act_type}.')
            raise ValueError
        env = Environment(
            action_mode=MoveArmThenGripper(
                arm_action_mode=arm_action_mode,
                gripper_action_mode=Discrete()
            ),
            obs_config=Obs_config,
            headless=True) 
        env.launch()
        
        # - launch task
        task = env.get_task(eval(run.task))
        task_succ_list = policy_deployment(
            task_name=run.task,
            task=env.get_task(eval(run.task)),
            policy=policy,
            norm_stats=norm_stats,
            check_cfg=check_cfg,
            test_demo_list=test_demo_list, # TODO: test 2 examples to save time for debugging
            device=device,
            method=run.method,
            test_indices=test_indices,
            query_freq=run.query_freq,
            info_dir=res_dir, 
            rand_init=run.random_init,
        )
        
        # save result as txt 
        # each except for the last element represents success (1) or failure (0), the last element represents the overall success rate (percentage)
        num_all   = float(len(task_succ_list))
        num_succ  = np.sum(np.array(task_succ_list).astype(np.float32))
        succ_rate = 0.0 if num_all == 0 else 100.0*num_succ/num_all
        
        print(f'Task {run.task} finished!')
        print(f'{run.task}-{run.method}-> task success rate: {succ_rate:.1f}% \n')
        print('-------------------------------------------------------')
        
        # - exit
        env.shutdown()
        
        if _DEBUG: 
            print('Debugging! break ...')
            break

    print('All done!')




