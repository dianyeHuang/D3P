'''
Script: testing_baseline.py
Description:
    Deploy and evaluate a baseline policy on RLBench tasks, logging rich diagnostics
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
    task_succ_list = list()
    episode_len   = check_cfg.max_episode_len[task_name]
    chunking_size = check_cfg.chunking_size
    flag_once = False
    
    info_all_dict = dict() # track all the information 
    for idx, demo in enumerate(tqdm(test_demo_list)):
        info_dict = {
            'planned_actions': dict(), # model output
            'execute_actions': dict(), # commands
            'joint_positions': dict(), # tracking results
        }
        descriptions, obs = task.reset_to_demo(demo) # if wanna reproduce the scenario
        if not flag_once:
            print(f'\nTask descriptions: ', descriptions)
            flag_once = True
        
        # descriptions, obs = task.reset() # randomly reset the environment
        success = False
        if 'ACT' in method:
            check_cfg['temporal_agg'] = True  
            query_freq = 1 # defined by the 
        else:
            check_cfg['temporal_agg'] = False
        if check_cfg['temporal_agg']:
            act_dim = check_cfg['action_dim']
            all_time_actions = torch.zeros((episode_len, episode_len+chunking_size, act_dim)).to(device)
            
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
                    planned_actions = policy(obs_propri, obs_img, actions=None, is_pad=None) # during the inference, only jpos and img is needed                       
                    info_dict['planned_actions'][str(ts)] = (planned_actions.squeeze(0).cpu().numpy()*norm_stats['act_std'] + norm_stats['act_mean']).tolist()

                if check_cfg['temporal_agg']: # for ACT
                    all_time_actions[[ts], ts:ts+chunking_size] = planned_actions
                    actions_for_curr_step = all_time_actions[:, ts]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
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
def load_policy(log_dir, method, last_model, ret_policy_only=False):
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
    
    if not last_model:
        best_model_path = os.path.join(log_dir, f'best_{method}_{seed}.pth')
    else:
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
from config_const import check_cfg, RunBuilder
from config_const import act_dim, chunking_size
from tqdm import tqdm

params = OrderedDict(
    method = [
        'ACT',
        'DiffusionPolicy',
        'DiffusionPolicy_visonly',
    ],
    task   = [
        'OpenDrawer', 
        'PushButtons', 
        'StackWine',
        'SlideBlockToTarget',
        'SweepToDustpan',
        'TurnTap'
    ],
    query_freq  = [8], # half the action horizon for diffusion model
    test_num    = check_cfg.test_num,
    random_init = check_cfg.random_init,
)

def set_res_dir_name(results_dir, run, suffix=''):
    dir_str = f'{run.task}_randjoint_{run.random_init}_query_{run.query_freq}' + suffix
    return os.path.join(results_dir, run.method, dir_str)

import os
if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    res_folder  = 'baseline_res' # the name of results
    results_dir = os.path.join(project_dir, res_folder)
    check_cfg['action_dim'] = act_dim 
    check_cfg['chunking_size'] = chunking_size 
    seen_configs = set()
    
    for run in tqdm(RunBuilder.get_runs(params)):
        if 'ACT' in run.method: run.query_freq = 1
        config_id = run.get_all_configs()
        if config_id in seen_configs: continue
        seen_configs.add(config_id)
        
        # create directories
        res_dir = None
        if check_cfg.save_info is True:
            res_dir = set_res_dir_name(results_dir, run)
            os.makedirs(res_dir, exist_ok=True)
            print(f'\nsave results to: {res_dir}.')
        
        print('--------------------------------------------------------')
        print(f'Infering task {run.task}, with method: {run.method} ...')
        print('--------------------------------------------------------')
        policy, norm_stats, task_suffix, cur_filedir, act_type, seed, device = load_policy(
            log_dir=check_cfg.model_dir[run.task][run.method], method=run.method, 
            last_model=True, ret_policy_only=False
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
        
        # concatenate valid and test indices
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
            headless= True) 
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
        
        # - exit
        env.shutdown()
        
    print('All done!')




