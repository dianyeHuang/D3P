'''
Script: train_policy.py
Description:
    End-to-end training entrypoint for RLBench imitation-learning policies.
    Supports multiple methods (D3P, ACT, DiffusionPolicy variants, D3P_LSTM),
    dataset loaders, EMA, mixed-precision (AMP), cosine LR scheduler, and
    TensorBoard logging. Can run single or multiple (method, task) pairs via CLI.
'''

import sys
import pickle
import numpy as np
from torch.utils.data import DataLoader
from utils.data_utils import EpisodeDataset, get_norm_stats, set_seed
from utils.d3p_policy_dataset_utils import EpisodePairDataset, EpisodePairDataset_pro, EpisodeMultiSeqDataset

import copy
from utils.models.diffusion.ema_model import EMAModel
from diffusers.optimization import (
    Union, SchedulerType, Optional,
    Optimizer, TYPE_TO_SCHEDULER_FUNCTION
)

def load_data(dataset_dir, train_cfg, policy_cfg):
    '''
        The early stopping is disabled.
        Mind that the validation set does not join the training
    '''
    train_ratio = train_cfg.train_ratio
    num_worker  = train_cfg.num_worker
    pin_memory  = train_cfg.pin_memory
    prefetch_factor = train_cfg.prefetch_factor # save memory
    train_batchsize = train_cfg.train_batchsize
    valid_batchsize = train_cfg.valid_batchsize
    num_test = train_cfg.test_num
    use_indices = train_cfg.use_indices
    device = train_cfg.device
    # episode_indices_load = train_cfg.episode_indices_load # either automatically generated or load from file
    act_joint = True if policy_cfg['action_type'] == 'joint'else False
    norm_stats, num_eps = get_norm_stats(dataset_dir)
    
    # - save stats data
    stats_path = os.path.join(dataset_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f: pickle.dump(norm_stats, f)
    
    indices_filepath = os.path.join(dataset_dir, 'episode_indices.pkl')
    update_indices = False
    if os.path.exists(indices_filepath):
        with open(indices_filepath, 'rb') as f:
                res = pickle.load(f)
        train_indices = res['train_indices']
        valid_indices = res['valid_indices']
        test_indices  = res['test_indices']

        if not (len(train_indices) + len(valid_indices) + len(test_indices) == num_eps) :
            print('The indices splitting shall be updated!!!')
            update_indices = True
        else:
            print('Keep using the same indeices ...')
    else:
        print(f'No such file for indices splitting: {indices_filepath}')
        update_indices = True
    
    if update_indices:
        shuffled_indices = np.random.permutation(num_eps)
        test_indices  = shuffled_indices[:int(num_test)]
        train_indices = shuffled_indices[int(num_test):int(num_test+(num_eps-num_test)*train_ratio)] # TODO: cancel the train_ratio
        valid_indices = shuffled_indices[int(num_test+(num_eps-num_test)*train_ratio):]
        print('Update and save the indices!!!')
        with open(os.path.join(dataset_dir, 'episode_indices.pkl'), 'wb') as f: # record train and valid indices
            pickle.dump({'train_indices':train_indices, 'valid_indices':valid_indices, 'test_indices': test_indices}, f)
    
    if 'train_num' in train_cfg.data:
        train_indices = train_indices[:min(train_cfg.train_num, len(train_indices))] 
    print(f'number of demos for training: {len(train_indices)}')
    
    if 'dataset_type' in train_cfg.keys():
        if train_cfg.dataset_type == 'EpisodePairDataset':
            print('Loading EpisodePairDataset !!!!')
            train_dataset = EpisodePairDataset(
                                train_indices, dataset_dir, norm_stats, 
                                query_every=policy_cfg['query_every'], 
                                chunking_size=policy_cfg['chunking_size'], 
                                act_joint=act_joint, use_indices=use_indices)
            valid_dataset = EpisodePairDataset(valid_indices, dataset_dir, norm_stats,
                                query_every=policy_cfg['query_every'], 
                                chunking_size=policy_cfg['chunking_size'], 
                                act_joint=act_joint, use_indices=use_indices)
        elif train_cfg.dataset_type == 'EpisodePairDataset_pro':
            print('Loading EpisodePairDataset_pro !!!!') # speficying the n_propri actions
            train_dataset = EpisodePairDataset_pro(
                                train_indices, dataset_dir, norm_stats, 
                                query_every=policy_cfg['query_every'], 
                                chunking_size=policy_cfg['chunking_size'], 
                                act_joint=act_joint, use_indices=use_indices,
                                n_propri=policy_cfg['n_propri'])
            valid_dataset = EpisodePairDataset_pro(valid_indices, dataset_dir, norm_stats,
                                query_every=policy_cfg['query_every'], 
                                chunking_size=policy_cfg['chunking_size'], 
                                act_joint=act_joint, use_indices=use_indices,
                                n_propri=policy_cfg['n_propri'])
        elif train_cfg.dataset_type == 'EpisodeMultiSeqDataset': # for lstm module
            print('Loading EpisodeMultiSeqDataset !!!!') # speficying the n_propri actions
            train_dataset = EpisodeMultiSeqDataset(
                                train_indices, dataset_dir, norm_stats, 
                                query_every=policy_cfg['query_every'], 
                                chunking_size=policy_cfg['chunking_size'], 
                                act_joint=act_joint, 
                                num_obs=train_cfg['num_obs'],
                                n_propri=policy_cfg['n_propri'])
            valid_dataset = EpisodeMultiSeqDataset(valid_indices, dataset_dir, norm_stats,
                                query_every=policy_cfg['query_every'], 
                                chunking_size=policy_cfg['chunking_size'], 
                                act_joint=act_joint, 
                                num_obs=train_cfg['num_obs'],
                                n_propri=policy_cfg['n_propri'])
    else: 
        print('Loading EpisodeDataset ....')
        train_dataset = EpisodeDataset(train_indices, dataset_dir, norm_stats, 
                                    policy_cfg['num_queries'], act_joint=act_joint, 
                                    use_indices=use_indices)
        valid_dataset = EpisodeDataset(valid_indices, dataset_dir, norm_stats, 
                                    policy_cfg['num_queries'], act_joint=act_joint, 
                                    use_indices=use_indices)
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True,
                                  pin_memory=pin_memory, num_workers=num_worker, 
                                  prefetch_factor=prefetch_factor, 
                                  generator=torch.Generator(device=device)
                                )
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batchsize, shuffle=False,
                                  pin_memory=pin_memory, num_workers=num_worker, 
                                  prefetch_factor=prefetch_factor,
                                  generator=torch.Generator(device=device)
                                )
    
    return train_dataloader, valid_dataloader, norm_stats

def forward_pass(data, policy, device):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad) 

def item_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.item()
    return new_d

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
):
    """
    Added kwargs vs diffuser's original implementation

    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            Th SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)
e number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)

import os
from tqdm import tqdm
from copy import deepcopy

from config_const import Config_list
from utils.baseline_utils import make_policy, make_optimizer
from utils.rlbench_demos_utils import create_folder_if_not_exists

import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast_mode, grad_scaler

if torch.__version__ >= "2.1":
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")
else:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

import argparse
from typing import Iterable, List, Optional

def _split_csv_or_list(values: Optional[Iterable[str]]) -> List[str]:
    """Expand comma-separated or repeated CLI args into a single flat list."""
    if not values:
        return []
    out: List[str] = []
    for v in values:
        # Split on commas and trim whitespace
        out.extend([p.strip() for p in v.split(",") if p.strip()])
    return out

def _validate_choices(name: str, items: List[str], allowed: List[str]) -> List[str]:
    invalid = [x for x in items if x not in allowed]
    if invalid:
        allowed_str = ", ".join(allowed) if allowed else "(none)"
        raise argparse.ArgumentTypeError(
            f"Invalid {name}: {invalid}. Allowed {name}: {allowed_str}"
        )
    return items


# ----------------------------
# Script entry point
# ----------------------------
import gc
from omegaconf import OmegaConf
from config_const import convert_nametuple2dict, convert_dict_valuetype

METHOD_LIST = [
    "D3P", # proposed
    "ACT",
    "DiffusionPolicy",
    "DiffusionPolicy_visonly", 
    "DiffusionPolicy_switch",  # w/o DKO
    "DiffusionPolicy_wrec",    # w/ rec loss
    "D3P_LSTM",                # rp. DKO w/ LSTM
]

TASK_LIST = [
    "OpenDrawer",
    "PushButtons",
    "StackWine",
    "SweepToDustpan",
    "TurnTap",
    "SlideBlockToTarget",
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Select method(s) and task(s) to run."
    )
    parser.add_argument(
        "--method", "-m",
        action="append",
        metavar="NAME[,NAME...]",
        help=f"Method(s) to run (choices: {', '.join(METHOD_LIST) or 'N/A'}). "
             "You can repeat the flag or pass comma-separated names.",
        default= None,
    )
    parser.add_argument(
        "--task", "-t",
        action="append",
        metavar="NAME[,NAME...]",
        help=f"Task(s) to run (choices: {', '.join(TASK_LIST) or 'N/A'}). "
             "You can repeat the flag or pass comma-separated names.",
        default= None,
    )
    
    args = parser.parse_args()
    if args.method is None: args.method = [METHOD_LIST[0]]
    method_list = _split_csv_or_list(args.method)
    method_list = _validate_choices("method(s)", method_list, METHOD_LIST)
    
    if args.task is None: args.task = [TASK_LIST[0]]
    task_list = _split_csv_or_list(args.task)
    task_list = _validate_choices("task(s)", task_list, TASK_LIST)
    
    # start training
    root_dir = os.path.dirname(os.path.abspath(__file__))    
    for method in method_list:
        for task_name in task_list:
            # release the cache
            gc.collect()
            if torch.cuda.is_available():
                print('relasing cuda memory ...')
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            print(f'Learning the task of {task_name} ...')
            # Set taks 
            policy_config = Config_list[method]['policy_config']
            train_config  = Config_list[method]['train_config']
            
            act_type = policy_config['action_type']
            task_suffix = '_joint' if act_type == 'joint' else '_eepose'
            Task_name = task_name + task_suffix
            dataset_dir = os.path.join(root_dir, 'dataset', Task_name)
            
            # load data
            device = train_config.device
            use_indices = train_config.use_indices
            set_seed(train_config.seed) # fixed random settings
            train_dataloader, valid_dataloader, norm_stats = load_data(dataset_dir, train_cfg=train_config, 
                                                                    policy_cfg=policy_config)
        
            # - save stats data
            stats_path = os.path.join(dataset_dir, 'dataset_stats.pkl')
            with open(stats_path, 'wb') as f: pickle.dump(norm_stats, f)
            
            # start training (load model and optimizer)
            policy = make_policy(method, policy_config)
            policy.to(device)
            optimizer = make_optimizer(method, policy)
            
            # lr_scheduler
            epoch = 0
            global_step = 0
            lr_scheduler_cfg = train_config['lr_schduler_cfg']
            gradient_accumulate_every = lr_scheduler_cfg.gradient_accumulate_every
            lr_scheduler = get_scheduler(
                name=lr_scheduler_cfg.name,
                optimizer=optimizer,
                num_warmup_steps=lr_scheduler_cfg.lr_warmup_steps,
                num_training_steps=(
                    len(train_dataloader) * train_config.num_epochs) \
                        // gradient_accumulate_every,
                # pytorch assumes stepping LRScheduler every epoch
                # however huggingface diffusers steps it every batch
                last_epoch=global_step-1
            )
            scaler = grad_scaler.GradScaler()
            
            # ema_model
            use_ema = train_config.use_ema
            if use_ema:
                ema_policy = copy.deepcopy(policy)
                ema = EMAModel(model=ema_policy, **train_config['ema_config'])
            
            # '''
            #     Save model configurations and dataset statitics  
            # '''
            log_every_epoch = train_config.log_every_epoch
            log_every_step = train_config.log_every_step
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
            model_name = f"{method}_{train_config.seed}" 
            log_dir = os.path.join(root_dir, f'logs/{Task_name}/{model_name}_{timestamp}')
            # best_model_path = os.path.join(log_dir, f'best_{model_name}.pth')
            last_model_path = os.path.join(log_dir, f'last_{model_name}.pth')
            create_folder_if_not_exists(log_dir)
            tb_writer = SummaryWriter(log_dir=log_dir)
            
            # save configurations
            log_config = dict()
            log_config['method_cfg'] = convert_nametuple2dict(Config_list[method])
            log_config['norm_stats'] = convert_dict_valuetype(
                        norm_stats, src_dtype=np.ndarray, convert_func=lambda x: x.tolist())
            config_omega = OmegaConf.create(log_config)
            OmegaConf.save(config_omega, os.path.join(log_dir, 'config.yaml'))
            
            # Training
            # min_val_loss = np.inf
            # best_ckpt_info = None
            flag_stop = False
            pbar = tqdm(range(train_config.num_epochs), desc = 'description', file=sys.stdout)
            for epoch in pbar:
                if flag_stop is True: break
                # Training
                policy.train()
                train_history = []
                for batch_idx, data in enumerate(train_dataloader):
                    if flag_stop is True: break
                    
                    optimizer.zero_grad()
                    with autocast_mode.autocast():
                        forward_dict = forward_pass(data, policy, device) # return ['l1', 'kl', 'loss']
                        loss = forward_dict['loss'] / gradient_accumulate_every
                    scaler.scale(loss).backward() # loss.backward()
                    
                    # step optimizer and ema
                    if global_step % gradient_accumulate_every == 0:
                        scaler.step(optimizer) # optimizer.step()
                        scaler.update()
                        lr_scheduler.step()
                        
                    if use_ema:
                        ema.step(policy)
                    global_step += 1
                    
                    train_history.append(item_dict(forward_dict))
                    if use_indices:
                        if global_step % log_every_step == 0:
                            # - print training descriptions
                            train_summary = compute_dict_mean(train_history)
                            train_history = list()
                            summary_string = ''
                            for k, v in train_summary.items():
                                summary_string += f'{k}: {v:.3f} '
                            
                            for k in train_summary.keys():
                                tb_writer.add_scalar(f'Loss/Train/{k}', train_summary[k], global_step)
                            
                            percent = int(float(batch_idx)/float(len(train_dataloader))*100)
                            pbar.set_description(f'Training {percent}%: {epoch+1}/{train_config.num_epochs}, ' + summary_string)
                        
                        if global_step % (log_every_epoch*log_every_step) == 0:
                            # -- validation
                            if use_ema: 
                                policy_eval = ema_policy
                            else:
                                policy_eval = policy
                            policy_eval.eval()
                            with torch.inference_mode(): # more efficient than torch.no_grad()
                                validation_history = []
                                for batch_idx, data in enumerate(valid_dataloader):
                                    forward_dict = forward_pass(data, policy_eval, device)
                                    validation_history.append(forward_dict)
                                valid_summary = compute_dict_mean(validation_history)
                                
                                # save model
                                valid_loss = valid_summary['loss']
                                torch.save(policy_eval.state_dict(), last_model_path)
                                
                                summary_string = ''
                                for k, v in valid_summary.items():
                                    summary_string += f'{k}: {v:.3f} '
                                
                                # change the settings to adapt to the training models
                                for k in valid_summary.keys():
                                    tb_writer.add_scalar(f'Loss/Valid/{k}', valid_summary[k], global_step)
                                    
                                percent = int(float(batch_idx)/float(len(valid_dataloader))*100)
                                pbar.set_description(f'Validation {percent}%: {epoch+1}/{train_config.num_epochs}, ' + summary_string)
                    
                if not use_indices:
                    # - print training descriptions
                    train_summary = compute_dict_mean(train_history)
                    summary_string = ''
                    for k, v in train_summary.items():
                        summary_string += f'{k}: {v:.3f} '
                    
                    for k in train_summary.keys():
                        tb_writer.add_scalar(f'Loss/Train/{k}', train_summary[k], epoch)
                    pbar.set_description(f'Training: {epoch+1}/{train_config.num_epochs}, ' + summary_string)
                    
                    # - Tensorboard log
                    if epoch % log_every_epoch == 0:
                        # -- validation
                        if use_ema: 
                            policy_eval = ema_policy
                        else:
                            policy_eval = policy
                        policy_eval.eval()
                        with torch.inference_mode(): # more efficient than torch.no_grad()
                            validation_history = []
                            for batch_idx, data in enumerate(valid_dataloader):
                                forward_dict = forward_pass(data, policy_eval, device)
                                validation_history.append(forward_dict)
                            valid_summary = compute_dict_mean(validation_history)
                            
                            # save model
                            valid_loss = valid_summary['loss']
                            torch.save(policy_eval.state_dict(), last_model_path)
                            
                            summary_string = ''
                            for k, v in valid_summary.items():
                                summary_string += f'{k}: {v:.3f} '
                            
                            # change the settings to adapt to the training models
                            for k in valid_summary.keys():
                                tb_writer.add_scalar(f'Loss/Valid/{k}', valid_summary[k], epoch)
                            pbar.set_description(f'Validation: {epoch+1}/{train_config.num_epochs}, ' + summary_string)

            
            print(f'Training finished!')

            # clean 
            # del policy, optimizer, lr_scheduler, best_ckpt_info
            del train_dataloader, valid_dataloader, norm_stats
            if use_ema: del ema_policy
            tb_writer.close()
            print("SummaryWriter has been closed")
