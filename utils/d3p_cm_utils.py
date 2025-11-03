'''
    referred to and adapted from https://github.com/ManiCM-fast/ManiCM
'''

import torch
import numpy as np
from typing import Generator
import torch.nn.functional as F
from einops import reduce

from diffusers.optimization import (
    Union, SchedulerType, Optional,
    Optimizer, TYPE_TO_SCHEDULER_FUNCTION
)

import os, pickle
from torch.utils.data import DataLoader
from utils.data_utils import get_norm_stats
from utils.d3p_policy_dataset_utils import EpisodePairDataset
def load_data(dataset_dir, train_cfg, policy_cfg, train_shuffle=True):
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
    
    train_dataloader = DataLoader(train_dataset, batch_size=train_batchsize, shuffle=train_shuffle,
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
            The number of training steps to do. This is not required by all schedulers (hence the argument being
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

@torch.no_grad()
def update_ema(target_params: Generator, source_params: Generator, rate: float = 0.99) -> None:
    for tgt, src in zip(target_params, source_params):
        tgt.detach().mul_(rate).add_(src, alpha=1 - rate)

def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def predicted_origin(
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
        prediction_type: str,
        alphas: torch.Tensor,
        sigmas: torch.Tensor
) -> torch.Tensor:
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0

class DDIMSolver:
    def __init__(self, alpha_cumprods: np.ndarray, timesteps: int = 1000, ddim_timesteps: int = 50) -> None:
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device: torch.device) -> "DDIMSolver":
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0: torch.Tensor, pred_noise: torch.Tensor,
                  timestep_index: torch.Tensor) -> torch.Tensor:
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


import math
import torch.nn.functional as F
def Huber_Loss(pred, target, delta=0, weights = None):
    """
    Computes psuedo-huber loss of pred and target of shape (batch, time, dim)
    
    Delta is the boundary between l_1 and l_2 loss. At delta = 0, this is just MSE loss.
    Setting delta = -1 calculates iCT's recommended delta given data size.
    
    Also supports weighting of loss
    """
    if delta == -1:
        delta = math.sqrt(math.prod(pred.shape[1:])) * .00054
    mse = F.mse_loss(pred, target, reduction = "none")
    loss = torch.sqrt(mse**2 + delta**2) - delta
    if weights is not None:
        loss = torch.einsum("b T D, b -> b T D", loss, weights)
    return loss.mean()

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
def load_d3p_cm_diffmodel(ckpt_path, device):
    diffusion_model = ConditionalUnet1D(
        input_dim=8,
        local_cond_dim=None,
        global_cond_dim=64,
        diffusion_step_embed_dim=128,
        down_dims=[256,512,1024], # [512,1024,2048],
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True
    )
    diffusion_model.load_state_dict(
        torch.load(ckpt_path, map_location=device), strict=True
    )
    
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
    
    return diffusion_model, ddim_scheduler
    

import copy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_lcm import LCMScheduler
from utils.d3p_policy_module_utils import D3P
from models.diffusion.conditional_unet1d import ConditionalUnet1D

class D3PCM:
    def __init__(self,
        d3p_model:D3P,
        noise_scheduler: DDPMScheduler,
        scheduler: LCMScheduler,
        num_inference_steps=10,
        num_inference_timesteps=4,
        device='cuda'
    ):
        self.d3p_model = d3p_model
        # define new diffusion model
        self.student_model = ConditionalUnet1D(
            input_dim=8,
            local_cond_dim=None,
            global_cond_dim=64,
            diffusion_step_embed_dim=128,
            down_dims=[256,512,1024], # [512,1024,2048],
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True
        )
        self.ema_model = copy.deepcopy(self.student_model)
        
        # self.student_model = copy.deepcopy(self.d3p_model.diffusion_model)
        # self.ema_model = copy.deepcopy(self.d3p_model.diffusion_model)
        self.teacher_model = copy.deepcopy(self.d3p_model.diffusion_model)
        
        self.d3p_model.to(device)
        self.ema_model.to(device)
        self.student_model.to(device)
        self.teacher_model.to(device)
        
        self.d3p_model.requires_grad_(False)
        self.teacher_model.requires_grad_(False)
        self.ema_model.requires_grad_(False)
        self.student_model.requires_grad_(True)
        
        self.device = device
        self.solver = DDIMSolver(
            noise_scheduler.alphas_cumprod.cpu().numpy(),
            timesteps=noise_scheduler.config.num_train_timesteps,
            ddim_timesteps=num_inference_steps,
        )
        self.alpha_schedule = torch.sqrt(noise_scheduler.alphas_cumprod)
        self.sigma_schedule = torch.sqrt(1 - noise_scheduler.alphas_cumprod)
        self.alpha_schedule = self.alpha_schedule.to(device)
        self.sigma_schedule = self.sigma_schedule.to(device)
        self.solver = self.solver.to(device)

        self.noise_scheduler = noise_scheduler
        self.scheduler = scheduler
        self.num_inference_timesteps = num_inference_timesteps
        self.num_inference_steps = num_inference_steps
    
    def compute_loss_2_bak(self, trajectory:torch.Tensor, global_cond, is_pad):
        
        local_cond = None
        batch_size = trajectory.size(0) 
        
        # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
        # topk = self.noise_scheduler.config.num_train_timesteps // self.num_inference_steps
        num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        index = torch.randint(0, num_train_timesteps, (batch_size, 2), device=self.device).long()
        index, _ = torch.sort(index, dim=1)  # min and max for each row
        timestep_ah = index[:, 1]
        timestep_al = index[:, 0]
        
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        noise_ah = self.noise_scheduler.add_noise(trajectory, noise, timestep_ah)
        noise_al = self.noise_scheduler.add_noise(trajectory, noise, timestep_al)
        
        with torch.no_grad():
            target_al_a0 = self.ema_model(
                sample=noise_al, 
                timestep=timestep_al, 
                local_cond=local_cond, 
                global_cond=global_cond
            )
            target_ah_a0 = self.ema_model(
                sample=noise_ah, 
                timestep=timestep_ah, 
                local_cond=local_cond, 
                global_cond=global_cond
            )
            
        pred_ah_a0 = self.student_model(
            sample=noise_ah, 
            timestep=timestep_ah, 
            local_cond=local_cond, 
            global_cond=global_cond
        )
        pred_al_a0 = self.student_model(
            sample=noise_al, 
            timestep=timestep_al, 
            local_cond=local_cond, 
            global_cond=global_cond
        )
        
        alpha_bar = self.noise_scheduler.alphas_cumprod[timestep_ah]   
        sigma2    = 1.0 - alpha_bar
        weight    = (1.0 / (sigma2 + 1e-5)).clamp(max=1e2)
        weight_ah = weight / (weight.mean().detach() + 1e-8)
        loss_pair = Huber_Loss(pred=pred_ah_a0, target=target_al_a0, weights=weight_ah) # weight_ah*((pred_ah_a0 - target_al_a0)**2).mean()
        
        alpha_bar = self.noise_scheduler.alphas_cumprod[timestep_al]   
        sigma2    = 1.0 - alpha_bar
        weight    = (1.0 / (sigma2 + 1e-5)).clamp(max=1e2)
        weight_al = weight / (weight.mean().detach() + 1e-8)
        loss_pair += Huber_Loss(pred=pred_al_a0, target=target_ah_a0, weights=weight_al) # (weight_al.unsqueeze(-1)*(pred_al_a0 - target_ah_a0)**2).mean()
        
        # boundary conditions
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        noise_a0 = self.noise_scheduler.add_noise(trajectory, noise, timestep_ah*0)
        pred_a0 = self.student_model(
            sample=noise_a0, 
            timestep=timestep_ah*0, 
            local_cond=local_cond, 
            global_cond=global_cond
        )
        loss_bc = Huber_Loss(pred=pred_a0, target=trajectory, weights=None)

        # total loss
        eta = 5e-2
        gamma = 0.35
        loss_sup = eta * ((pred_ah_a0 - trajectory)**2).mean()
        loss = loss_pair + gamma*loss_bc + loss_sup
        
        loss_dict = {
            'loss_pair': loss_pair.item(),
            'loss_bc': loss_bc.item(),
            'loss_sup': loss_sup.item(),
            'loss': loss.item()
        }
        return loss, loss_dict
    
    def eval_loss(self, trajectory:torch.Tensor, global_cond, is_pad):
        # do inference 
        local_cond = None
        batch_size = trajectory.size(0) 
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        index = torch.randint(0, self.num_inference_steps, (batch_size,), device=self.device).long()
        start_timesteps = self.solver.ddim_timesteps[index]
        noisy_model_input = self.noise_scheduler.add_noise(trajectory, noise, start_timesteps)
        pred_x_0 = self.student_model(
            sample=noisy_model_input, 
            timestep=start_timesteps, 
            local_cond=local_cond, 
            global_cond=global_cond
        )
        pred_x_0 = pred_x_0*~is_pad.unsqueeze(-1)
        trajectory = trajectory*~is_pad.unsqueeze(-1)
        err_x0 = F.mse_loss(pred_x_0.float(), trajectory.float(), reduction="mean")
        return err_x0
        
    def compute_loss(self, trajectory:torch.Tensor, global_cond, is_pad):
        local_cond = None
        batch_size = trajectory.size(0) 
        
        # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
        topk = self.noise_scheduler.config.num_train_timesteps // self.num_inference_steps
        index = torch.randint(0, self.num_inference_steps, (batch_size,), device=self.device).long()
        start_timesteps = self.solver.ddim_timesteps[index]
        timesteps = start_timesteps - topk
        timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)
        
        # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
        c_skip_start, c_out_start = [append_dims(x, trajectory.ndim) for x in [c_skip_start, c_out_start]]
        c_skip, c_out = scalings_for_boundary_conditions(timesteps)
        c_skip, c_out = [append_dims(x, trajectory.ndim) for x in [c_skip, c_out]]
        
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        noisy_model_input = self.noise_scheduler.add_noise(trajectory, noise, start_timesteps)
        
        pred_x_0 = self.student_model(
            sample=noisy_model_input, 
            timestep=start_timesteps, 
            local_cond=local_cond, 
            global_cond=global_cond
        )
        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0
        loss_dsm = F.mse_loss(pred_x_0.float(), trajectory.float(), reduction="mean")
        
        with torch.no_grad():
            cond_teacher_output = self.teacher_model(
                sample=noisy_model_input, 
                timestep=start_timesteps, 
                local_cond=local_cond, 
                global_cond=global_cond)
            cond_pred_x0 = predicted_origin(
                cond_teacher_output,
                start_timesteps,
                noisy_model_input,
                'epsilon', # self.noise_scheduler.config.prediction_type,
                self.alpha_schedule,
                self.sigma_schedule
            )
            x_prev = self.solver.ddim_step(cond_pred_x0, cond_teacher_output, index)
        
        with torch.no_grad():
            pred_x_0 = self.ema_model(
                x_prev.float(),
                timesteps,
                local_cond=local_cond, 
                global_cond=global_cond)
            target = c_skip * x_prev + c_out * pred_x_0
        
        # loss 
        loss_cm = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        loss_total = 0.4*loss_cm + 0.6*loss_dsm
        loss_dict = {
            'loss_cm': loss_cm.item(),
            'loss_dsm': loss_dsm.item(),
            'loss_total': loss_total.item()
        }
        
        return loss_total, loss_dict
