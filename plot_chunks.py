"""
Module: analyze_results.py
Description:
    Post-process D3P (and related) evaluation results:
      - Load per-run JSON logs (success flags, step counts, trajectories).
      - Plot summary bar charts of steps with success/failure coloring.
      - Plot per-trajectory action/joint evolution figures.
      - Aggregate metrics (success rate, avg steps overall / on successes)
        into a single CSV for easy comparison.

Inputs & Expectations:
    - Results are stored under a root folder (e.g., ./proposed_res/),
      with each run written into a subfolder produced by `set_res_dir_name`.
    - Each run folder contains exactly one *.json file with a dict:
        {
          "<test_idx>": {
              "success": bool,
              "num_steps": int,
              "info": {
                  "planned_actions": { ... },
                  "execute_actions": { ... },
                  "joint_positions": { ... },
                  "vis_actions": { ... },
                  "fuse_actions": { ... },
                  "vis_err": { ... },
                  "fuse_err": { ... },
                  "act_indices": { ... }
              }
          },
          ...
        }

Outputs:
    - Per-run plots saved to: <run_dir>/plotted_figs/
        * statistic_res.png            # bar chart of step counts (green=success/red=failure)
        * idx_{i}_succ_{b}_steps_{n}.png  # detailed action/joint evolution plots
    - Aggregated CSV saved to: <res_root_dir>/<folder_name>.csv
      Columns: method, task, test_num, query_freq, random_init,
               success_rate, succ_avg_step, total_avg_step

How to Use:
    - Adjust `folder_name`, and the `params` OrderedDict in `__main__`
      to enumerate methods/tasks/query_freq/test_num/random_init.
    - Set `check_image=True` to generate figures (can be slow for many runs).
    - Run the script directly: `python analyze_results.py`

Dependencies:
    - numpy, pandas, matplotlib, natsort, tqdm
    - Project-local utilities:
        * config_const.RunBuilder
        * testing_baseline.set_res_dir_name

Notes:
    - The plotting section uses a fixed subplot layout and color coding:
        blue/cyan for planned/fused signals, red for executed actions.
    - The script assumes each run directory contains exactly one JSON log.
      If multiple are present, the first globbed file is used.

"""


import json
import numpy as np
from pprint import pprint
from natsort import natsorted

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 15,            # tick and legend
    "axes.labelsize": 15,       # x/y label
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "legend.fontsize": 10,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.frameon": False
})

def parse_json_file(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        info_all_dict = json.load(file)
    test_indices = natsorted(info_all_dict.keys())
    succ_list = list()
    step_list = list()
    for idx in test_indices:
        info_dict = info_all_dict[idx]
        succ_list.append(info_dict['success'])
        step_list.append(info_dict['num_steps']+1)
    return test_indices, succ_list, step_list, info_all_dict

def plot_bar(steps, success, plt_title=None, save_path=None):
    colors = ['green' if flag else 'red' for flag in success]
    indices = np.arange(len(steps))
    
    plt.figure(figsize=(10, 5))
    plt.bar(indices, steps, color=colors)

    legend_elements = [
        Patch(facecolor='green', label='Success (True)'),
        Patch(facecolor='red', label='Failure (False)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.xlabel("Test Index")
    plt.ylabel("Number of Steps")
    if plt_title is None:
        plt.title("Execution Steps with Success Flags")
    else:
        plt.title(plt_title)
    plt.xticks(indices)
    
    if save_dir is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def plot_actions(info_all_dict:dict, idx=None, pre_title='', save_dir=None):
    '''
        plot the evolution of traj. 
        color code for visualization:
            -> planned
            -> execution
            -> joint states
    '''
    # parse info_all_dict
    if idx is None:
        idx_list = natsorted(info_all_dict.keys())
    else:
        idx_list = [idx] if not isinstance(idx, list) else idx
    
    y_list = [f'joint_{i}' for i in range(7)] + ['gripper'] + ['loss', 'select', '', '']
    for idx in idx_list:
        traj_info = info_all_dict[idx]
        success   = traj_info['success']
        num_step  = traj_info['num_steps']+2
        traj_info = traj_info['info']
        planned_actions = traj_info['planned_actions'] # selected actions
        execute_actions = traj_info['execute_actions'] # aggregated
        joint_positions = traj_info['joint_positions'] # actual joint states
        vision_actions  = traj_info['vis_actions']
        fusion_actions  = traj_info['fuse_actions']
        vision_recerrs  = traj_info['vis_err']
        fusion_recerrs  = traj_info['fuse_err']
        action_indices  = traj_info['act_indices']
        
        time_stampes = list(range(num_step))
        chunking_stamps = list(planned_actions.keys())

        exec_acts_arr = np.array(list(execute_actions.values()))
        joint_pos_arr = np.array(list(joint_positions.values()))
        
        # plot curves
        fig, axes = plt.subplots(4, 3, figsize=(18, 8))
        
        for i, ax in enumerate(axes.flat):
            # plot chunking actions
            ax.set_xlim(-3, 165)
            ax.set_xlabel('time step')
            
            for c_str in chunking_stamps:
                
                # print('c_str: ', c_str)
                if i > 9: continue
                if i == 9: 
                    if c_str == '-1': continue
                    if c_str == '0':
                        if vision_recerrs['0'] < fusion_recerrs['0']:
                            act_idxx = [6]*4
                        else:
                            act_idxx = [7]*4
                    else:
                        act_idxx = action_indices[c_str]
                    
                    start_idx = int(c_str)
                    for jdx, val in enumerate(act_idxx):
                        if val % 2 == 0:
                            color = 'blue'
                        else:
                            color = 'cyan'
                        ax.bar(start_idx+jdx, 1, color=color)
                    continue
                if i == 8:
                    if c_str == '-1': continue
                    c_stamps  = list(range(int(c_str), int(c_str)+16))
                    ax.plot(c_stamps, [vision_recerrs[c_str]]*16, '-b')
                    ax.scatter(int(c_str), vision_recerrs[c_str], s=10, color='b', marker='D')
                    ax.plot(c_stamps, [fusion_recerrs[c_str]]*16, '-c')
                    ax.scatter(int(c_str), fusion_recerrs[c_str], s=10, color='c', marker='D')
                    continue
                
                
                if c_str == '-1':
                    act = np.array(planned_actions[c_str])[i]
                    
                    vis_act = np.array(vision_actions[c_str])[i]
                    
                    fuse_act = np.array(fusion_actions[c_str])[i]
                    
                else:
                    vis_acts  = np.array(vision_actions[c_str])[:, i]
                    fuse_acts = np.array(fusion_actions[c_str])[:, i]
                    
                    c_stamps  = list(range(int(c_str), int(c_str)+len(vis_acts)))

                    ax.plot(c_stamps, vis_acts, '-b')
                    ax.scatter(int(c_str), vis_acts[0], s=8, color='b', marker='D')

                    ax.plot(c_stamps, fuse_acts, '-c')
                    ax.scatter(int(c_str), fuse_acts[0], s=8, color='c', marker='D')
                
                # plot executed actions
                ax.plot(time_stampes, exec_acts_arr[:, i], linestyle='-', color='r') # commands from the trained policy
                    
            ax.grid(True)
            ax.set_ylabel(y_list[i])
        
        plt_title = pre_title + f'_idx_{idx}_succ_{success}_steps_{num_step}'
        fig.suptitle(plt_title, fontsize=16, fontweight='bold')
        

        plt.tight_layout()
        
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, f'idx_{idx}_succ_{success}_steps_{num_step}.png'), dpi=300, bbox_inches='tight')
            # plt.savefig(os.path.join(save_dir, f'idx_{idx}_succ_{success}_steps_{num_step}.svg'), dpi=300, bbox_inches='tight')
            plt.close()
        
import os
import glob
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from config_const import RunBuilder
from testing_baseline import set_res_dir_name

_DEBUG = False

import argparse
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Set output folder for experiment results.")
    parser.add_argument(
        '--folder-name',
        type=str,
        default='proposed_res',
        help="Name of the result folder to be created under the project directory (default: 'proposed_res')"
    )
    args = parser.parse_args()
    
    folder_name = args.folder_name
    project_dir = os.path.dirname(os.path.abspath(__file__))
    res_root_dir = os.path.join(project_dir, folder_name)
    
    check_table = True
    check_image = True # Caution!!! it would take a lot of time to plot the figures
    params = OrderedDict(
        method = [
            'D3P'
        ],
        task = [
            'OpenDrawer', 
            'PushButtons', 
            'StackWine',
            'SlideBlockToTarget',
            'SweepToDustpan',
            'TurnTap'
        ],
        query_freq  = [4],
        test_num    = [40],
        random_init = [False, True],
    )
    suffix = ''
    
    seen_configs = set()
    
    if check_table:
        res_dict = dict()
        for key in params.keys() | {"success_rate", "total_avg_step", "succ_avg_step"}:
            res_dict[key] = list()
    
    for run in tqdm(RunBuilder.get_runs(params)):
        # avoid running the same configurations 
        if 'D3P' in run.method: run.query_freq = 4
        config_id = run.get_all_configs()
        if config_id in seen_configs: continue
        seen_configs.add(config_id)
        
        res_dir = set_res_dir_name(res_root_dir, run, suffix)
        json_path = glob.glob(os.path.join(res_dir, "*.json"))[0]
        test_indices, succ_list, step_list, info_all_dict = parse_json_file(json_path)
        succ_rate  = round(sum(list(map(float, succ_list)))/float(len(succ_list))*100, 1) # keep one decimal place
        avg_steps  = int(sum(step_list)/float(len(step_list)))
        succ_steps_np = np.array(step_list)[np.array(succ_list)]
        if len(succ_steps_np) == 0: 
            succ_steps = 0
        else:
            succ_steps = int(np.mean(succ_steps_np))
                
        if check_image:
            save_dir = os.path.join(res_dir, 'plotted_figs')
            os.makedirs(save_dir, exist_ok=True)
            
            # save image results:
            # plot bar
            plt_title = f'{run.method}_{run.task}_randinit_{run.random_init}_query_{run.query_freq}_{succ_rate:.1f}%_avg_step_{avg_steps}/{succ_steps}'
            save_path = os.path.join(save_dir, f'statistic_res.png')
            plot_bar(step_list, succ_list, plt_title, save_path)
            
            # plot curves
            pre_title = f'{run.method}_{run.task}_randinit_{run.random_init}_query_{run.query_freq}'
            plot_actions(info_all_dict, pre_title=pre_title, save_dir=save_dir)
    
        if check_table:
            for key in params.keys():
                res_dict[key].append(run[key])
            res_dict["success_rate"].append(succ_rate)
            res_dict["total_avg_step"].append(avg_steps)
            res_dict["succ_avg_step"].append(succ_steps)
        
        if _DEBUG: break
            
    res_df = pd.DataFrame(res_dict)
    res_df = res_df.drop_duplicates() # remove duplicate configuration and results
    res_df = res_df.reset_index(drop=True) # reset index
    res_df = res_df.reindex(columns=["method", "task", "test_num", "query_freq", "random_init", "success_rate", "succ_avg_step", "total_avg_step"])
    
    # save results
    csv_path = os.path.join(res_root_dir, folder_name+'.csv')
    res_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f'saving resutl to {csv_path}')
    print(res_df)
        
    print('done!!')
        
