# -*- coding: utf-8 -*-
'''
    Trainning the policy
'''
import sys
import pickle
import numpy as np
from torch.utils.data import DataLoader
from data_utils import EpisodeDataset, get_norm_stats, set_seed
from d3p_policy_dataset_utils import EpisodePairDataset, EpisodePairDataset_pro, EpisodeMultiSeqDataset

from diffusers.optimization import (
    Union, SchedulerType, Optional,
    Optimizer, TYPE_TO_SCHEDULER_FUNCTION
)

import os
import torch

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

from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

import math
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False,
            dropout_rate=.2):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

        self.scale_dropout = nn.Dropout(dropout_rate)
        self.bias_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        out = self.scale_dropout(out)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        out = self.bias_dropout(out)
        return out

logger = logging.getLogger(__name__)
class CTMConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        dropout_rate=.0,
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        self.dropout_rate = dropout_rate

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed * 2 # for both timestep and stoptime
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout_rate=dropout_rate),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout_rate=dropout_rate),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))



        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout_rate=dropout_rate),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout_rate=dropout_rate),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))


        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def prepare_drop_generators(self):
        dropout_generator = torch.Generator().manual_seed(42)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.generator = dropout_generator
                if self.dropout_rate == 0.0:
                    module.generator = None

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int],
            stoptime: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        encoded_times = self.diffusion_step_encoder(timestep)
        encoded_stops = self.diffusion_step_encoder(stoptime)
        
        global_feature = torch.cat([
            encoded_times, encoded_stops
        ], axis=-1)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)
            
        x = self.final_conv(x)
        
        x = einops.rearrange(x, 'b t h -> b h t')
        
        return x

from diffusers.optimization import (
    Union, SchedulerType, Optional,
    Optimizer, TYPE_TO_SCHEDULER_FUNCTION
)

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




