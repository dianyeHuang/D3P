import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.vision.model_getter import get_resnet
from models.vision.multi_image_obs_encoder import MultiImageObsEncoder
from models.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import numpy as np
from einops import reduce
import torch.nn.functional as F

import torch
from torch import nn
import torch.nn.functional as F

class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Dec(nn.Module):
    def __init__(self, last_size=4, resize_factor=1, latent_dim=64, out_chs=3, in_planes=512, num_Blocks=[2, 2, 2, 2]):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(latent_dim, in_planes*last_size**2),
            nn.Unflatten(1, (in_planes, last_size, last_size)),
            nn.Upsample(scale_factor=resize_factor)
        )
        
        self.in_planes = in_planes
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64,  num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 32,  num_Blocks[0], stride=2) 
        
        self.output_layer = nn.Sequential(
            ResizeConv2d(32, out_chs, kernel_size=3, scale_factor=2),
            nn.Sigmoid() # converge faster with sigmoid()
        ) 

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.input_layer(z)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.output_layer(x)
        return x

class AttentionFusion(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super(AttentionFusion, self).__init__()
        # dimension alignment
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)
        self.attention_weight = nn.Linear(hidden_dim, 1)
    
    def forward(self, feature1, feature2):
        f1 = self.fc1(feature1)  # [batch_size, hidden_dim]
        f2 = self.fc2(feature2)  # [batch_size, hidden_dim]
        
        features = torch.stack([f1, f2], dim=1)  # [batch_size, 2, hidden_dim]
        scores = self.attention_weight(features).squeeze(-1)  # [batch_size, 2]
        weights = torch.softmax(scores, dim=1)  # [batch_size, 2]
        
        fused_feature = torch.sum(weights.unsqueeze(-1) * features, dim=1)  # [batch_size, hidden_dim]
        
        return fused_feature

class AttFusion(nn.Module):
    def __init__(self, in_dim1=32, in_dim2=128, hidden_dim=64, out_dim=64, n_heads=4):
        super(AttFusion, self).__init__()
        self.linear_x = nn.Linear(in_dim1, hidden_dim)
        self.linear_f = nn.Linear(in_dim2, hidden_dim) 
        self.attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=n_heads)
        self.fc = nn.Linear(out_dim, out_dim)

    def forward(self, X, F):
        N = X.shape[0]
        X_flat = X.view(N, -1)
        X_proj = self.linear_x(X_flat)
        F_proj = self.linear_f(F)
        fusion, _ = self.attn(X_proj.unsqueeze(1), F_proj.unsqueeze(1), F_proj.unsqueeze(1))
        return self.fc(fusion.squeeze(1))

class DiffusionPolicy(nn.Module):
    def __init__(self, policy_config):
        super().__init__()
        # parse configs
        self.action_dim = policy_config.action_dim
        self.chunking_size = policy_config.num_queries
        vision_cfg = policy_config['vision_config']
        diffuser_cfg = policy_config['diffuser_config']
        diffmodel_cfg = policy_config['diffmodel_config']
        optimizer_cfg = policy_config['optimizer_config']
        
        # obs_encoder:MultiImageObsEncoder
        self.obs_encoder = MultiImageObsEncoder(
            shape_meta   = vision_cfg.shape_meta,
            rgb_model    = get_resnet(**vision_cfg.rgb_model_cfg),
            resize_shape    = vision_cfg.resize_shape,
            crop_shape      = vision_cfg.crop_shape,
            random_crop     = vision_cfg.random_crop,
            use_group_norm  = vision_cfg.use_group_norm,
            share_rgb_model = vision_cfg.share_rgb_model,
            imagenet_norm   = vision_cfg.imagenet_norm # image norm will be performed
        )
        
        # noise_scheduler:DDPMScheduler
        self.infer_steps = diffuser_cfg.num_train_timesteps
        self.noise_scheduler = DDPMScheduler(**diffuser_cfg) 
        
        # diffusion_model:ConditionalUnet1D
        self.diffusion_model = ConditionalUnet1D(**diffmodel_cfg)
        
        # optimizer
        self.optimizer = torch.optim.AdamW(
            params=self.parameters(), **optimizer_cfg                              
        )
    
    def forward(self, qpos, image, actions=None, is_pad=None):
        '''
        qpos -> prioperception data
        image -> observation images, multi-view cameras
        action -> actions to be predicted
        is_pad -> define which actions are not for computing
        '''
        # note that the keys of the observation are defined in the 
        # configuration script, and the images will be normed depends
        # on the vision_cfg settings when initializing the obs_encoder
        vision_dict = { 
            'front_rgb': image[:, 0, ...],
            'wrist_rgb': image[:, 1, ...],
        }
        vision_enc = self.obs_encoder(vision_dict) # (batch_size, dim)
        obs_enc = torch.cat((vision_enc, qpos), dim=-1)
        
        if actions is not None: 
            loss_dict = dict()
            loss = self.compute_loss(obs_enc, actions, is_pad)
            loss_dict['loss'] = loss
            return loss_dict
        else:
            action = self.predict_action(obs_enc)
            return action
    
    def configure_optimizers(self):
        return self.optimizer

    # ============= api for diffusion training
    def conditional_sample(self, traj_shape, global_cond):
        self.noise_scheduler.set_timesteps(self.infer_steps)
        traj = torch.randn(size=traj_shape, device=global_cond.device) 
        for t in self.noise_scheduler.timesteps:
            model_output = self.diffusion_model(
                sample=traj, 
                timestep=t, 
                local_cond=None, 
                global_cond=global_cond
            )
            traj = self.noise_scheduler.step(
                model_output, t, traj
            )['prev_sample']
        return traj

    def predict_action(self, obs_enc):
        batch_size = obs_enc.shape[0]
        actions = self.conditional_sample(
            traj_shape=(batch_size, self.chunking_size, self.action_dim),
            global_cond=obs_enc
        )
        return actions

    def compute_loss(self, obs_enc, actions, is_pad=None):
        batch_size = actions.shape[0]
        global_cond = obs_enc
        
        # get prediction
        noise = torch.randn(actions.shape, device=actions.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size, ), device=actions.device, dtype=torch.long
        ).long()
        noisy_traj = self.noise_scheduler.add_noise(actions, noise, timesteps)
        pred = self.diffusion_model(noisy_traj, timesteps,
            local_cond=None, global_cond=global_cond)
        
        # get loss, predict epsilon
        loss = F.mse_loss(pred, target=noise, reduction='none')
        loss = loss* ~is_pad.unsqueeze(-1) # exclude losses from the padding actions
        loss = reduce(loss, 'b ... -> b ', 'mean') # reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss

class DiffusionPolicy_visonly(nn.Module):
    def __init__(self, policy_config):
        super().__init__()
        # parse configs
        self.action_dim = policy_config.action_dim
        self.chunking_size = policy_config.num_queries
        vision_cfg = policy_config['vision_config']
        diffuser_cfg = policy_config['diffuser_config']
        diffmodel_cfg = policy_config['diffmodel_config']
        optimizer_cfg = policy_config['optimizer_config']
        
        # obs_encoder:MultiImageObsEncoder
        self.obs_encoder = MultiImageObsEncoder(
            shape_meta   = vision_cfg.shape_meta,
            rgb_model    = get_resnet(**vision_cfg.rgb_model_cfg),
            resize_shape    = vision_cfg.resize_shape,
            crop_shape      = vision_cfg.crop_shape,
            random_crop     = vision_cfg.random_crop,
            use_group_norm  = vision_cfg.use_group_norm,
            share_rgb_model = vision_cfg.share_rgb_model,
            imagenet_norm   = vision_cfg.imagenet_norm # image norm will be performed
        )
        
        # noise_scheduler:DDPMScheduler
        self.infer_steps = diffuser_cfg.num_train_timesteps
        self.noise_scheduler = DDPMScheduler(**diffuser_cfg) 
        
        # diffusion_model:ConditionalUnet1D
        self.diffusion_model = ConditionalUnet1D(**diffmodel_cfg)
        
        # optimizer
        self.optimizer = torch.optim.AdamW(
            params=self.parameters(), **optimizer_cfg                              
        )
    
    def forward(self, qpos, image, actions=None, is_pad=None):
        '''
        qpos -> prioperception data
        image -> observation images, multi-view cameras
        action -> actions to be predicted
        is_pad -> define which actions are not for computing
        '''
        # note that the keys of the observation are defined in the 
        # configuration script, and the images will be normed depends
        # on the vision_cfg settings when initializing the obs_encoder
        vision_dict = { 
            'front_rgb': image[:, 0, ...],
            'wrist_rgb': image[:, 1, ...],
        }
        vision_enc = self.obs_encoder(vision_dict) # (batch_size, dim)
        # obs_enc = torch.cat((vision_enc, qpos), dim=-1)
        obs_enc = vision_enc
        
        if actions is not None: 
            loss_dict = dict()
            loss = self.compute_loss(obs_enc, actions, is_pad)
            loss_dict['loss'] = loss
            return loss_dict
        else:
            action = self.predict_action(obs_enc)
            return action
    
    def configure_optimizers(self):
        return self.optimizer

    # ============= api for diffusion training
    def conditional_sample(self, traj_shape, global_cond):
        self.noise_scheduler.set_timesteps(self.infer_steps)
        traj = torch.randn(size=traj_shape, device=global_cond.device) 
        for t in self.noise_scheduler.timesteps:
            model_output = self.diffusion_model(
                sample=traj, 
                timestep=t, 
                local_cond=None, 
                global_cond=global_cond
            )
            traj = self.noise_scheduler.step(
                model_output, t, traj
            )['prev_sample']
        return traj

    def predict_action(self, obs_enc, ):
        batch_size = obs_enc.shape[0]
        actions = self.conditional_sample(
            traj_shape=(batch_size, self.chunking_size, self.action_dim),
            global_cond=obs_enc
        )
        return actions

    def compute_loss(self, obs_enc, actions, is_pad=None):
        batch_size = actions.shape[0]
        global_cond = obs_enc
        
        # get prediction
        noise = torch.randn(actions.shape, device=actions.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size, ), device=actions.device, dtype=torch.long
        ).long()
        noisy_traj = self.noise_scheduler.add_noise(actions, noise, timesteps)
        pred = self.diffusion_model(noisy_traj, timesteps,
            local_cond=None, global_cond=global_cond)
        
        # get loss, predict epsilon
        loss = F.mse_loss(pred, target=noise, reduction='none')
        loss = loss* ~is_pad.unsqueeze(-1) # exclude losses from the padding actions
        loss = reduce(loss, 'b ... -> b ', 'mean') # reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss

class DiffusionPolicy_switch(nn.Module):
    def __init__(self, policy_config):
        super().__init__()
        # parse configs
        self.action_dim = policy_config.action_dim
        self.chunking_size = policy_config.num_queries
        vision_cfg = policy_config['vision_config']
        diffuser_cfg = policy_config['diffuser_config']
        diffmodel_cfg = policy_config['diffmodel_config']
        optimizer_cfg = policy_config['optimizer_config']
        self.switch_prob = 0.4 # mind the 
        
        # obs_encoder:MultiImageObsEncoder
        self.obs_encoder = MultiImageObsEncoder(
            shape_meta   = vision_cfg.shape_meta,
            rgb_model    = get_resnet(**vision_cfg.rgb_model_cfg),
            resize_shape    = vision_cfg.resize_shape,
            crop_shape      = vision_cfg.crop_shape,
            random_crop     = vision_cfg.random_crop,
            use_group_norm  = vision_cfg.use_group_norm,
            share_rgb_model = vision_cfg.share_rgb_model,
            imagenet_norm   = vision_cfg.imagenet_norm # image norm will be performed
        )
        self.att_fuse = AttentionFusion(
            input_dim1=vision_cfg.rgb_model_cfg['output_size']*2, # visual signals
            input_dim2=policy_config.action_dim, # proprioceptive signals, normalized
            hidden_dim=vision_cfg.rgb_model_cfg['output_size']*2
        )
        
        # noise_scheduler:DDPMScheduler
        self.infer_steps = diffuser_cfg.num_train_timesteps
        self.noise_scheduler = DDPMScheduler(**diffuser_cfg) 
        
        # diffusion_model:ConditionalUnet1D
        self.diffusion_model = ConditionalUnet1D(**diffmodel_cfg)
        
        # optimizer
        self.optimizer = torch.optim.AdamW(
            params=self.parameters(), **optimizer_cfg                              
        )
    
    def generate_switch_signal(self, p0=0.3):
        return np.random.choice([0, 1], p=[p0, 1-p0])
    
    def forward(self, qpos, image, actions=None, is_pad=None, ret_generr=False):
        '''
        qpos -> prioperception data
        image -> observation images, multi-view cameras
        action -> actions to be predicted
        is_pad -> define which actions are not for computing
        '''
        # note that the keys of the observation are defined in the 
        # configuration script, and the images will be normed depends
        # on the vision_cfg settings when initializing the obs_encoder
        vision_dict = { 
            'front_rgb': image[:, 0, ...],
            'wrist_rgb': image[:, 1, ...],
        }
        vis_enc = self.obs_encoder(vision_dict) # (batch_size, dim)
        fuse_enc = self.att_fuse(vis_enc, qpos)
        
        if actions is not None: 
            loss_dict = dict()
            # select which feature to be used
            switch_signal = self.generate_switch_signal(self.switch_prob) 
            cond_enc = switch_signal*vis_enc + (1-switch_signal)*fuse_enc # switch to fuse_enc
            loss = self.compute_loss(cond_enc, actions, is_pad)
            loss_dict['loss'] = loss
            return loss_dict
        else:
            # cond_enc = self.switch_prob*vision_enc + (1-self.switch_prob)*fuse_enc # not good for tasks that require long horizon actions
            fuse_acts = self.predict_action(fuse_enc)
            vis_acts = self.predict_action(vis_enc)
            if not ret_generr:
                return fuse_acts, vis_acts
            else:
                # return diffusion model reconstruction errors (to test whether the output acts are of high confidence or not)
                fuse_err = self.get_generative_errs(self.diffusion_model, fuse_enc, fuse_acts, num_samples=10)
                vis_err  = self.get_generative_errs(self.diffusion_model, vis_enc, vis_acts, num_samples=10)
                return {
                    'fuse': [fuse_acts, fuse_err], 'vis': [vis_acts, vis_err]
                }
    
    def get_generative_errs(self, diff_model, cond:torch.Tensor, acts:torch.Tensor, num_samples:int):
        '''
            The errors are computed in batches, latest version
        '''
        # duplicate acts and cond #num_samples times
        repeat_factors = [1]*cond.ndimension()
        repeat_factors[0] = num_samples
        cond = cond.repeat(*repeat_factors)
        
        repeat_factors = [1]*acts.ndimension()
        repeat_factors[0] = num_samples
        acts = acts.repeat(*repeat_factors)
        
        # get prediction
        noise = torch.randn(acts.shape, device=acts.device)
        # if num_samples > self.infer_steps-1:
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (num_samples, ), device=acts.device, dtype=torch.long
        ).long()
            # print('timesteps: ', timesteps)
        # else:
        #     steps = torch.linspace(0, self.infer_steps-1, steps=num_samples, dtype=torch.long)
        #     timesteps = torch.floor(steps).to(device=acts.device).long()
        
        noisy_traj = self.noise_scheduler.add_noise(acts, noise, timesteps)
        pred = diff_model(noisy_traj, timesteps,
            local_cond=None, global_cond=cond)
        
        # get loss, predict epsilon
        loss = F.mse_loss(pred, target=noise, reduction='none')
        loss = reduce(loss, 'b ... -> b ', 'mean') # reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss
    
    def get_vis_action(self, image):
        vision_dict = { 
            'front_rgb': image[:, 0, ...],
            'wrist_rgb': image[:, 1, ...],
        }
        vision_enc = self.obs_encoder(vision_dict) # (batch_size, dim)
        action = self.predict_action(vision_enc)
        return action
    
    def get_fuse_action(self, qpos, image):
        vision_dict = { 
            'front_rgb': image[:, 0, ...],
            'wrist_rgb': image[:, 1, ...],
        }
        vision_enc = self.obs_encoder(vision_dict) # (batch_size, dim)
        fuse_enc = self.att_fuse(vision_enc, qpos)
        action = self.predict_action(fuse_enc)
        return action
    
    def get_hybrid_action(self, qpos, image):
        vision_dict = { 
            'front_rgb': image[:, 0, ...],
            'wrist_rgb': image[:, 1, ...],
        }
        vision_enc = self.obs_encoder(vision_dict) # (batch_size, dim)
        fuse_enc = self.att_fuse(vision_enc, qpos)
        cond_enc = self.switch_prob*vision_enc + (1-self.switch_prob)*fuse_enc 
        action = self.predict_action(cond_enc)
        return action
    
    def configure_optimizers(self):
        return self.optimizer

    # ============= api for diffusion training
    def conditional_sample(self, traj_shape, global_cond):
        self.noise_scheduler.set_timesteps(self.infer_steps)
        traj = torch.randn(size=traj_shape, device=global_cond.device) 
        for t in self.noise_scheduler.timesteps:
            model_output = self.diffusion_model(
                sample=traj, 
                timestep=t, 
                local_cond=None, 
                global_cond=global_cond
            )
            traj = self.noise_scheduler.step(
                model_output, t, traj
            )['prev_sample']
        return traj

    def predict_action(self, obs_enc, ):
        batch_size = obs_enc.shape[0]
        actions = self.conditional_sample(
            traj_shape=(batch_size, self.chunking_size, self.action_dim),
            global_cond=obs_enc
        )
        return actions

    def compute_loss(self, obs_enc, actions, is_pad=None):
        batch_size = actions.shape[0]
        global_cond = obs_enc
        
        # get prediction
        noise = torch.randn(actions.shape, device=actions.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size, ), device=actions.device, dtype=torch.long
        ).long()
        noisy_traj = self.noise_scheduler.add_noise(actions, noise, timesteps)
        pred = self.diffusion_model(noisy_traj, timesteps,
            local_cond=None, global_cond=global_cond)
        
        # get loss, predict epsilon
        loss = F.mse_loss(pred, target=noise, reduction='none') # TODO: try huber loss? deal with the gripper action separately?
        loss = loss* ~is_pad.unsqueeze(-1) # exclude losses from the padding actions
        loss = reduce(loss, 'b ... -> b ', 'mean') # reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss

class DiffusionPolicy_wrec(nn.Module):
    # diffusion policy with reconstruction loss
    # check the performance
    def __init__(self, policy_config):
        super().__init__()
        # parse configs
        self.action_dim = policy_config.action_dim
        self.chunking_size = policy_config.num_queries
        vision_cfg = policy_config['vision_config']
        diffuser_cfg = policy_config['diffuser_config']
        diffmodel_cfg = policy_config['diffmodel_config']
        optimizer_cfg = policy_config['optimizer_config']
        self.switch_prob = 0.4
        
        self.front_img_decoder = ResNet18Dec(
            last_size=4, resize_factor=1, latent_dim=64, 
            out_chs=3, in_planes=512, num_Blocks=[2, 2, 2, 2]
        )
        
        self.wrist_img_decoder = ResNet18Dec(
            last_size=4, resize_factor=1, latent_dim=64, 
            out_chs=3, in_planes=512, num_Blocks=[2, 2, 2, 2]
        )
        
        # obs_encoder:MultiImageObsEncoder
        self.obs_encoder = MultiImageObsEncoder(
            shape_meta   = vision_cfg.shape_meta,
            rgb_model    = get_resnet(**vision_cfg.rgb_model_cfg),
            resize_shape    = vision_cfg.resize_shape,
            crop_shape      = vision_cfg.crop_shape,
            random_crop     = vision_cfg.random_crop,
            use_group_norm  = vision_cfg.use_group_norm,
            share_rgb_model = vision_cfg.share_rgb_model,
            imagenet_norm   = vision_cfg.imagenet_norm # image norm will be performed
        )
        self.att_fuse = AttentionFusion(
            input_dim1=vision_cfg.rgb_model_cfg['output_size']*2, # visual signals
            input_dim2=policy_config.action_dim, # proprioceptive signals, normalized
            hidden_dim=vision_cfg.rgb_model_cfg['output_size']*2
        )
        
        # noise_scheduler:DDPMScheduler
        self.infer_steps = diffuser_cfg.num_train_timesteps
        self.noise_scheduler = DDPMScheduler(**diffuser_cfg) 
        
        # diffusion_model:ConditionalUnet1D
        self.diffusion_model = ConditionalUnet1D(**diffmodel_cfg)
        
        # optimizer
        self.optimizer = torch.optim.AdamW(
            params=self.parameters(), **optimizer_cfg                              
        )
    
    def generate_switch_signal(self, p0=0.3):
        return np.random.choice([0, 1], p=[p0, 1-p0])
    
    def forward(self, qpos, image, actions=None, is_pad=None, ret_generr=False):
        '''
        qpos -> prioperception data
        image -> observation images, multi-view cameras
        action -> actions to be predicted
        is_pad -> define which actions are not for computing
        '''
        # note that the keys of the observation are defined in the 
        # configuration script, and the images will be normed depends
        # on the vision_cfg settings when initializing the obs_encoder
        vision_dict = { 
            'front_rgb': image[:, 0, ...],
            'wrist_rgb': image[:, 1, ...],
        }
        vision_enc = self.obs_encoder(vision_dict) # (batch_size, dim)
        fuse_enc = self.att_fuse(vision_enc, qpos)
        
        if actions is not None: 
            loss_dict = dict()
            # reconstruction loss
            # reconstruct the image from the encoded feature
            # decode front image, shape of vision_enc is 128
            rec_front_img = self.front_img_decoder(vision_enc[:, :64])
            front_rec_loss = F.mse_loss(rec_front_img, image[:, 0, ...])
            # decode wrist image
            rec_wrist_img = self.wrist_img_decoder(vision_enc[:, 64:])
            wrist_rec_loss = F.mse_loss(rec_wrist_img, image[:, 1, ...])
            # compute reconstruction loss, weighted by 0.5 
            img_rec_loss = 0.5*front_rec_loss + 0.5*wrist_rec_loss
            loss_dict['img_rec_loss'] = img_rec_loss
            
            # select which feature to be used
            switch_signal = self.generate_switch_signal(self.switch_prob) 
            cond_enc = switch_signal*vision_enc + (1-switch_signal)*fuse_enc # switch to fuse_enc 
            diff_loss = self.compute_loss(cond_enc, actions, is_pad)
            loss_dict['diff_loss'] = diff_loss
            loss_dict['loss'] = diff_loss + img_rec_loss
            return loss_dict
        else:
            # cond_enc = self.switch_prob*vision_enc + (1-self.switch_prob)*fuse_enc # not good for tasks that require long horizon actions
            fuse_acts = self.predict_action(fuse_enc)
            vis_acts = self.predict_action(vision_enc)
            if not ret_generr:
                return fuse_acts, vis_acts
            else:
                # return diffusion model reconstruction errors (to test whether the output acts are of high confidence or not)
                fuse_err = self.get_generative_errs(self.diffusion_model, fuse_enc, fuse_acts, num_samples=10)
                vis_err  = self.get_generative_errs(self.diffusion_model, vision_enc, vis_acts, num_samples=10)
                return {
                    'fuse': [fuse_acts, fuse_err], 'vis': [vis_acts, vis_err]
                }
    
    def get_generative_errs(self, diff_model, cond:torch.Tensor, acts:torch.Tensor, num_samples:int):
        '''
            The errors are computed in batches, latest version
        '''
        # duplicate acts and cond #num_samples times
        repeat_factors = [1]*cond.ndimension()
        repeat_factors[0] = num_samples
        cond = cond.repeat(*repeat_factors)
        
        repeat_factors = [1]*acts.ndimension()
        repeat_factors[0] = num_samples
        acts = acts.repeat(*repeat_factors)
        
        # get prediction
        noise = torch.randn(acts.shape, device=acts.device)
        # if num_samples > self.infer_steps-1:
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (num_samples, ), device=acts.device, dtype=torch.long
        ).long()
        
        noisy_traj = self.noise_scheduler.add_noise(acts, noise, timesteps)
        pred = diff_model(noisy_traj, timesteps,
            local_cond=None, global_cond=cond)
        
        # get loss, predict epsilon
        loss = F.mse_loss(pred, target=noise, reduction='none')
        loss = reduce(loss, 'b ... -> b ', 'mean') # reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss
    
    def get_vis_action(self, image):
        vision_dict = { 
            'front_rgb': image[:, 0, ...],
            'wrist_rgb': image[:, 1, ...],
        }
        vision_enc = self.obs_encoder(vision_dict) # (batch_size, dim)
        action = self.predict_action(vision_enc)
        return action
    
    def get_fuse_action(self, qpos, image):
        vision_dict = { 
            'front_rgb': image[:, 0, ...],
            'wrist_rgb': image[:, 1, ...],
        }
        vision_enc = self.obs_encoder(vision_dict) # (batch_size, dim)
        fuse_enc = self.att_fuse(vision_enc, qpos)
        action = self.predict_action(fuse_enc)
        return action
    
    def get_hybrid_action(self, qpos, image):
        vision_dict = { 
            'front_rgb': image[:, 0, ...],
            'wrist_rgb': image[:, 1, ...],
        }
        vision_enc = self.obs_encoder(vision_dict) # (batch_size, dim)
        fuse_enc = self.att_fuse(vision_enc, qpos)
        cond_enc = self.switch_prob*vision_enc + (1-self.switch_prob)*fuse_enc 
        action = self.predict_action(cond_enc)
        return action
    
    def configure_optimizers(self):
        return self.optimizer

    # ============= api for diffusion training
    def conditional_sample(self, traj_shape, global_cond):
        self.noise_scheduler.set_timesteps(self.infer_steps)
        traj = torch.randn(size=traj_shape, device=global_cond.device) 
        for t in self.noise_scheduler.timesteps:
            model_output = self.diffusion_model(
                sample=traj, 
                timestep=t, 
                local_cond=None, 
                global_cond=global_cond
            )
            traj = self.noise_scheduler.step(
                model_output, t, traj
            )['prev_sample']
        return traj

    def predict_action(self, obs_enc, ):
        batch_size = obs_enc.shape[0]
        actions = self.conditional_sample(
            traj_shape=(batch_size, self.chunking_size, self.action_dim),
            global_cond=obs_enc
        )
        return actions

    def compute_loss(self, obs_enc, actions, is_pad=None):
        batch_size = actions.shape[0]
        global_cond = obs_enc
        
        # get prediction
        noise = torch.randn(actions.shape, device=actions.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size, ), device=actions.device, dtype=torch.long
        ).long()
        noisy_traj = self.noise_scheduler.add_noise(actions, noise, timesteps)
        pred = self.diffusion_model(noisy_traj, timesteps,
            local_cond=None, global_cond=global_cond)
        
        # get loss, predict epsilon
        loss = F.mse_loss(pred, target=noise, reduction='none')
        loss = loss* ~is_pad.unsqueeze(-1) # exclude losses from the padding actions
        loss = reduce(loss, 'b ... -> b ', 'mean') # reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss

import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from utils.detr.detr.main import build_ACT_model_and_optimizer

import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


from d3p_policy_module_utils import D3P, D3P_LSTM
def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'DiffusionPolicy':
        policy = DiffusionPolicy(policy_config)
    elif policy_class == 'DiffusionPolicy_visonly':
        policy = DiffusionPolicy_visonly(policy_config)
    elif policy_class == 'D3P':
        policy = D3P(policy_config)
    elif policy_class == 'DiffusionPolicy_switch':
        policy = DiffusionPolicy_switch(policy_config)
    elif policy_class == 'DiffusionPolicy_wrec':
        policy = DiffusionPolicy_wrec(policy_config)
    elif policy_class == 'D3P_LSTM':
        policy = D3P_LSTM(policy_config)
    else:
        raise ValueError(f'Unknown policy class: {policy_class}')
    return policy

def make_optimizer(policy_class, policy):
    if policy_class in [
            'ACT', 'DiffusionPolicy', 'D3P', \
            'DiffusionPolicy_visonly', 'DiffusionPolicy_wrec' \
            'DiffusionPolicy_switch', 'D3P_LSTM'
        ]:
        optimizer = policy.configure_optimizers()
    else:
        raise ValueError(f'Unknown policy class: {policy_class}')
    return optimizer 
