'''
Script: train_d3pcm.py
Description:
    This script performs the *distillation training* of the D3P (Diffusion Policy with
    Deep Dual Perception) model using a compact diffusion-conditioned module (D3PCM).
    It prepares the training data, loads a pretrained teacher policy, and trains a 
    lightweight student model with EMA updates under diffusion-based supervision.

    The pipeline was referred and adapted from the official ManiCM implementation:
    https://github.com/ManiCM-fast/ManiCM
'''

import os
import time
from utils.data_utils import set_seed
from config_const import Config_list, check_cfg
from utils.d3p_cm_utils import load_data
from omegaconf import OmegaConf
from utils.baseline_utils import make_policy

def load_model_and_dataloader(train_shuffle=True, task_name='OpenDrawer'):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    method = 'D3P'

    policy_config = Config_list[method]['policy_config']
    train_config  = Config_list[method]['train_config']
    act_type = policy_config['action_type']
    task_suffix = '_joint' if act_type == 'joint' else '_eepose'
    Task_name = task_name + task_suffix
    dataset_dir = os.path.join(root_dir, 'dataset', Task_name)
    
    # load data
    device = train_config.device
    train_config.use_indices = True
    set_seed(train_config.seed) # fixed random settings
    train_dataloader, _, _ = load_data(dataset_dir, train_cfg=train_config, 
                                                    policy_cfg=policy_config,
                                                    train_shuffle=train_shuffle)
    print('length of train_dataloader: ', len(train_dataloader))
    
    # load policy
    log_dir = check_cfg.model_dir[task_name][method]
    yaml_path = os.path.join(log_dir, 'config.yaml')
    log_conf = OmegaConf.load(yaml_path)
    policy_config = log_conf['method_cfg']['policy_config']
    train_config  = log_conf['method_cfg']['train_config']
    seed   = train_config.seed
    best_model_path = os.path.join(log_dir, f'last_{method}_{seed}.pth')
    d3p_model = make_policy(method, policy_config)
    loading_status = d3p_model.load_state_dict(
        torch.load(best_model_path, map_location=device), strict=True)
    print('Teacher policy loading status: ', loading_status)
    
    # tbwriter
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) # datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{method}_42" # TODO: policy name to be defined
    log_dir = os.path.join(root_dir, f'logs_distill/{Task_name}/{model_name}_{timestamp}')
    last_model_path = os.path.join(log_dir, f'last_{model_name}.pth')
    create_folder_if_not_exists(log_dir)
    tb_writer = SummaryWriter(log_dir=log_dir)
    
    return train_dataloader, d3p_model, tb_writer, last_model_path

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

import sys
import torch
import torch.optim
from tqdm import tqdm
from utils.d3p_policy_module_utils import D3P
from utils.d3p_cm_utils import D3PCM, get_scheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_lcm import LCMScheduler
from torch.utils.data import DataLoader

import torch
from torch.utils.tensorboard import SummaryWriter
# from torch.cuda.amp import autocast_mode, grad_scaler
if torch.__version__ >= "2.1":
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")
else:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

from utils.d3p_cm_utils import update_ema
import torch.nn.functional as F
class TrainD3PCMWorkspace:
    def __init__(self, 
            train_dataloader:DataLoader, 
            d3p_model:D3P,
            tb_writer,
            last_model_path,
            max_grad_norm = 1.0,
            ema_decay = 0.95,
            num_inference_steps=10,
            num_inference_timesteps=3,
        ):
        self.last_model_path = last_model_path
        self.train_dataloader = train_dataloader
        self.tb_writer = tb_writer
        
        ddim_scheduler = DDIMScheduler( # same as the training settings
            num_train_timesteps=30, 
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='sample'
        )
        scheduler = LCMScheduler(
            num_train_timesteps=30,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='sample'
        )
        self.model = D3PCM(
            d3p_model=d3p_model,
            noise_scheduler=ddim_scheduler,
            scheduler=scheduler,
            num_inference_steps=num_inference_steps,
            num_inference_timesteps=num_inference_timesteps
        )
        
        self.optimizer = torch.optim.AdamW(
            lr=5.0e-5, betas=[0.05, 0.999],
            eps=1.0e-8, weight_decay=1.0e-6,
            params=self.model.student_model.parameters(),
        )
        self.max_grad_norm = max_grad_norm
        self.ema_decay = ema_decay
        
    def run(self, num_epochs=300):
        # training settings
        device = 'cuda'
        lr_scheduler = get_scheduler(
            'cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=(
                len(self.train_dataloader) * num_epochs)
        )
        
        # last_action = None
        global_step = 0
        log_every_step = 100
        log_every_epoch = 5
        pbar = tqdm(range(num_epochs), desc = 'description', file=sys.stdout)
        for epoch in pbar:
            train_history = []
            for batch_idx, data in enumerate(self.train_dataloader):                
                image_data, qpos_data, action_data, is_pad = data
                image_data, qpos_data = image_data.to(device), qpos_data.to(device)
                action_data, is_pad = action_data.to(device), is_pad.to(device)
                
                self.optimizer.zero_grad()
                # with autocast_mode.autocast():
                with torch.no_grad():
                    vision_dict = { 
                        'front_rgb': image_data[:, 0, 0, ...],
                        'wrist_rgb': image_data[:,0, 1, ...],
                    }
                    action_data  = action_data[:, 0, ...]
                    qpos_data    = qpos_data[:, 0, ...]
                    is_pad       = is_pad[:, 0, ...]

                    # compute global condition
                    vis_enc  = self.model.d3p_model.obs_encoder(vision_dict)
                    fuse_enc = self.model.d3p_model.fuse(qpos_data, vis_enc) # global_cond1
                    latent_act = self.model.d3p_model.dko.get_latnet_act(vis_enc) # global_cond2
                    switch_signal = self.model.d3p_model.generate_switch_signal( # 0.5
                                        self.model.d3p_model.fuse_prob, image_data.shape[0]) 
                    global_cond = switch_signal*latent_act + (1-switch_signal)*fuse_enc
                
                loss, loss_info = self.model.compute_loss(
                    action_data, global_cond, is_pad
                )                

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.student_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                update_ema(
                    self.model.ema_model.parameters(), 
                    self.model.student_model.parameters(), 
                    self.ema_decay
                )

                global_step += 1
                # log information
                train_history.append(loss_info)
                if global_step % log_every_step == 0:
                    train_summary = compute_dict_mean(train_history)
                    train_history = list()
                    summary_string = ''
                    for k, v in train_summary.items():
                        summary_string += f'{k}: {v:.3f} '
                    
                    for k in train_summary.keys():
                        self.tb_writer.add_scalar(f'Loss/Train/{k}', train_summary[k], global_step)
                    
                    percent = int(float(batch_idx)/float(len(self.train_dataloader))*100)
                    pbar.set_description(f'Training {percent}%: {epoch+1}/{num_epochs}, ' + summary_string)

            if epoch % log_every_epoch == 0:
                torch.save(self.model.student_model.state_dict(), self.last_model_path)

        # save the last model
        torch.save(self.model.student_model.state_dict(), self.last_model_path)
        self.tb_writer.close()
        print("SummaryWriter has been closed")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Distillate D3P model on a selected RLBench task.")
    parser.add_argument(
        "--task-name", "-t",
        type=str,
        default="OpenDrawer",
        choices=[
            "OpenDrawer",
            "PushButtons",
            "StackWine",
            "SlideBlockToTarget",
            "SweepToDustpan",
            "TurnTap",
        ],
        help="Select the RLBench task to train on (default: OpenDrawer)."
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=400,
        help="Number of training epochs (default: 400)."
    )
    return parser.parse_args()


# ----------------------------
# Script entry point
# ----------------------------

# task_name = 'OpenDrawer'
#     # 1. 'OpenDrawer'
#     # 2. 'PushButtons',
#     # 3. 'StackWine',
#     # 4. 'SlideBlockToTarget',
#     # 5. 'SweepToDustpan',
#     # 6. 'TurnTap',

if __name__ == '__main__':
    args = parse_args()
    task_name = args.task_name

    # Load model and dataloader
    train_dataloader, d3p_model, tb_writer, last_model_path = load_model_and_dataloader(
        train_shuffle=True, task_name=task_name
    )
    
    # Initialize training workspace
    workspace = TrainD3PCMWorkspace(
        d3p_model=d3p_model,
        train_dataloader=train_dataloader,
        tb_writer=tb_writer,
        last_model_path=last_model_path
    )
    
    # Run training
    workspace.run(num_epochs=args.epochs)


