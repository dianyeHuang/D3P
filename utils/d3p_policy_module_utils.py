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
from einops import rearrange, reduce

import numpy as np
import imgaug.augmenters as iaa

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

def return_activiation_fcn(activation_type: str):
    # build the activation layer
    if activation_type == "sigmoid":
        act = torch.nn.Sigmoid()
    elif activation_type == "tanh":
        act = torch.nn.Sigmoid()
    elif activation_type == "ReLU":
        act = torch.nn.ReLU()
    elif activation_type == "PReLU":
        act = torch.nn.PReLU()
    elif activation_type == "softmax":
        act = torch.nn.Softmax(dim=-1)
    elif activation_type == "Mish":
        act = torch.nn.Mish()
    else:
        act = torch.nn.PReLU()
    return act

class TwoLayerPreActivationResNetLinear(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 100,
            activation: str = 'relu',
            dropout_rate: float = 0.25,
            spectral_norm: bool = False,
            use_norm: bool = False,
            norm_style: int = 'BatchNorm'
    ) -> None:
        super().__init__()
        if spectral_norm:
            self.l1 = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
            self.l2 = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        else:
            self.l1 = nn.Linear(hidden_dim, hidden_dim)
            self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_norm = use_norm
        self.act = return_activiation_fcn(activation)

        if use_norm:
            if norm_style == 'BatchNorm':
                self.normalizer = nn.BatchNorm1d(hidden_dim)
            elif norm_style == 'LayerNorm':
                self.normalizer = torch.nn.LayerNorm(hidden_dim, eps=1e-06)
            else:
                raise ValueError('not a defined norm type')

    def forward(self, x):
        x_input = x
        if self.use_norm:
            x = self.normalizer(x)
        x = self.l1(self.dropout(self.act(x)))
        if self.use_norm:
            x = self.normalizer(x)
        x = self.l2(self.dropout(self.act(x)))
        return x + x_input
    
class ResidualMLPNetwork(nn.Module):
    """
    Simple multi layer perceptron network with residual connections for
    benchmarking the performance of different networks. The resiudal layers
    are based on the IBC paper implementation, which uses 2 residual lalyers
    with pre-actication with or without dropout and normalization.
    """

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 100,
            num_hidden_layers: int = 1,
            output_dim=1,
            dropout: int = 0,
            activation: str = "Mish",
            use_spectral_norm: bool = False,
            use_norm: bool = False,
            norm_style: str = 'BatchNorm',
            device: str = 'cuda'
    ):
        super(ResidualMLPNetwork, self).__init__()
        self.network_type = "mlp"
        self._device = device

        assert num_hidden_layers % 2 == 0
        if use_spectral_norm:
            self.layers = nn.ModuleList([spectral_norm(nn.Linear(input_dim, hidden_dim))])
            spectral_norm_fn = spectral_norm
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            spectral_norm_fn = False
        self.layers.extend(
            [
                TwoLayerPreActivationResNetLinear(
                    hidden_dim=hidden_dim,
                    activation=activation,
                    dropout_rate=dropout,
                    spectral_norm=spectral_norm_fn,
                    use_norm=use_norm,
                    norm_style=norm_style
                )
                for i in range(1, num_hidden_layers, 2)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        # self.layers.to(self._device)

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x) # .to(torch.float32))
        return x

    def get_device(self, device: torch.device):
        self._device = device
        self.layers.to(device)

    def get_params(self):
        return self.layers.parameters()

import torch.nn.init as init
class DeepKoopmanModule(nn.Module):
    def __init__(self, dko_config):
        super().__init__()
        self.act_dim = dko_config.act_dim
        self.obs_dim = dko_config.obs_dim
        self.chunking_size = dko_config.chunking_size
        self.latent_act_dim = dko_config.latent_act_dim
        
        self.K = nn.Linear(self.obs_dim, self.obs_dim, bias=False)
        self.V = nn.Linear(self.latent_act_dim, self.obs_dim, bias=False)
        self._initialize_weights(self.K)
        self._initialize_weights(self.V)
        
        self.latent_policy = ResidualMLPNetwork(**dko_config.lp_cfg)
        self.action_header = ResidualMLPNetwork(**dko_config.act_header_cfg)
    
    def _initialize_weights(self, module:nn.Linear):
        init.xavier_uniform_(module.weight)  
        if module.bias is not None:
            init.zeros_(module.bias) 
    
    def enable_KV_grad(self, enable_flag):
        self.K.weight.requires_grad = enable_flag
        if self.K.bias is not None:
            self.K.bias.requires_grad = enable_flag
        
        self.V.weight.requires_grad = enable_flag
        if self.V.bias is not None:
            self.V.bias.requires_grad = enable_flag
    
    def forward(self, current_obs, latent_act=None):
        '''
            TODO: decide if K should segregate as Q*Q.T, !!!!! important !!!!!!!
        '''
        if latent_act is None:
            # TODO: should we stop the gradient of current_obs？ Nope, we need to update the visual feature
            latent_act = self.get_latnet_act(current_obs)
            # next_obs = self.K @ (current_obs.unsqueeze(2).detach() + self.V@latent_act.unsqueeze(2))
            # return next_obs.squeeze(2), latent_act
            self.enable_K_V_grads(True)
            next_obs = self.K(current_obs + self.V(latent_act))
            # next_obs = self.K(current_obs.detach() + self.V(latent_act))
            return next_obs, latent_act
        else:
            # next_obs = self.K.detach() @ (current_obs.unsqueeze(2) + self.V.detach()@latent_act.unsqueeze(2).detach())
            # return next_obs.squeeze(2)
            self.enable_KV_grad(False)
            next_obs = self.K(current_obs + self.V(latent_act))
            return next_obs
    
    def predict_action(self, current_obs):
        return self.action_header(current_obs).reshape(
                    -1, self.chunking_size, self.act_dim)
    
    def enable_K_V_grads(self, enable_flag):
        self.K.requires_grad_(enable_flag)
        self.V.requires_grad_(enable_flag)
        
    def gaussian_init_(self, row_dim, col_dim, std=1):    
        sampler = torch.distributions.Normal(
            torch.Tensor([0]), 
            torch.Tensor([std/float(max(row_dim, col_dim))])
        )
        Omega = sampler.sample((row_dim, col_dim))[..., 0]  
        return Omega
        
    def get_latnet_act(self, current_obs):
        return self.latent_policy(current_obs)

class AttentionFusion(nn.Module):
    '''
        more practical,
        out_dim is equal to hidden_dim
    '''
    def __init__(self, fea_act_dim, fea_vis_dim, out_dim):
        super(AttentionFusion, self).__init__()
        # dimension alignment
        self.fc_act = nn.Linear(fea_act_dim, out_dim)
        self.fc_vis = nn.Linear(fea_vis_dim, out_dim)
        self.attention_weight = nn.Linear(out_dim, 1)
    
    def forward(self, fea_act, fea_vis):
        bs = fea_act.shape[0]
        fea_act = fea_act.view(bs, -1)
        f1 = self.fc_act(fea_act)  # [batch_size, out_dim]
        f2 = self.fc_vis(fea_vis)  # [batch_size, out_dim]
        
        features = torch.stack([f1, f2], dim=1)  # [batch_size, 2, out_dim]
        scores = self.attention_weight(features).squeeze(-1)  # [batch_size, 2]
        weights = torch.softmax(scores, dim=1)  # [batch_size, 2]
        
        fused_feature = torch.sum(weights.unsqueeze(-1) * features, dim=1)  # [batch_size, out_dim]
        
        return fused_feature

class AttFusion(nn.Module):
    '''
        prone to overfitting
    '''
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

'''
1. emergent failure recovery behaviour (alleviate the causal problem), consider comparing feature / the resulting action chunking for policy switching
2. smoothness of the planned traj., consider predicting the b-previous action sequence and the n-future action sequence
'''
class DeepKoopmanModule(nn.Module):
    def __init__(self, dko_config):
        super().__init__()
        self.act_dim = dko_config.act_dim
        self.obs_dim = dko_config.obs_dim
        self.chunking_size = dko_config.chunking_size
        self.latent_act_dim = dko_config.latent_act_dim
        
        self.K = nn.Linear(self.obs_dim, self.obs_dim, bias=False)
        self.V = nn.Linear(self.latent_act_dim, self.obs_dim, bias=False)
        self._initialize_weights(self.K)
        self._initialize_weights(self.V)
        
        self.latent_policy  = ResidualMLPNetwork(**dko_config.lp_cfg)
        self.regress_header = ResidualMLPNetwork(**dko_config.reg_header_cfg)
        self.action_header  = ResidualMLPNetwork(**dko_config.act_header_cfg)
    
    def _initialize_weights(self, module:nn.Linear):
        init.xavier_uniform_(module.weight)  
        if module.bias is not None:
            init.zeros_(module.bias) 
    
    def enable_KV_grad(self, enable_flag):
        self.K.weight.requires_grad = enable_flag
        if self.K.bias is not None:
            self.K.bias.requires_grad = enable_flag
        
        self.V.weight.requires_grad = enable_flag
        if self.V.bias is not None:
            self.V.bias.requires_grad = enable_flag
    
    def forward(self, current_obs, latent_act=None):
        if latent_act is None:
            latent_act = self.get_latnet_act(current_obs)
            self.enable_K_V_grads(True)
            next_obs = self.K(current_obs.detach() + self.V(latent_act))
            return next_obs, latent_act 
        else:
            self.enable_KV_grad(False)
            next_obs = self.K(current_obs + self.V(latent_act.detach()))
            return next_obs
    
    def regress_action(self, curr_obs):
        '''
            predict fianl actions from the visual feature
        '''
        return self.regress_header(curr_obs).reshape(
                    -1, self.chunking_size, self.act_dim)
    
    def decode_action(self, latent_act):
        '''
            predict the final actions from the latent action
        '''
        return self.action_header(latent_act).reshape(
                    -1, self.chunking_size, self.act_dim)
    
    def enable_K_V_grads(self, enable_flag):
        self.K.requires_grad_(enable_flag)
        self.V.requires_grad_(enable_flag)
        
    def gaussian_init_(self, row_dim, col_dim, std=1):    
        sampler = torch.distributions.Normal(
            torch.Tensor([0]), 
            torch.Tensor([std/float(max(row_dim, col_dim))])
        )
        Omega = sampler.sample((row_dim, col_dim))[..., 0]  
        return Omega
        
    def get_latnet_act(self, enc_obs):
        '''
            using non-causal structure 
        '''
        lact = self.latent_policy(enc_obs)
        return lact 

class D3P(nn.Module):
    def __init__(self, policy_config): 
        super().__init__()
        # parse configs
        self.action_dim = policy_config.action_dim
        self.chunking_size = policy_config.chunking_size
        self.n_propri = policy_config.n_propri
        self.query_every = policy_config['query_every']
        self.pred_seqlen = self.n_propri + self.chunking_size - 1 # length of the predicted actions including n_propri acts and chunking_size -1 effective horizon
        dko_cfg = policy_config['dko_config']
        vision_cfg = policy_config['vision_config']
        diffuser_cfg = policy_config['diffuser_config']
        diffmodel_cfg = policy_config['diffmodel_config']
        optimizer_cfg = policy_config['optimizer_config']
        fuse_cfg = policy_config['fuse_config']
        self.fuse_prob = policy_config['switch_prob']
        
        # obs_encoder:MultiImageObsEncoder
        self.obs_encoder = MultiImageObsEncoder(
            shape_meta   = vision_cfg.shape_meta,
            rgb_model    = get_resnet(**vision_cfg.rgb_model_cfg),
            resize_shape    = vision_cfg.resize_shape,
            crop_shape      = vision_cfg.crop_shape,
            random_crop     = vision_cfg.random_crop,
            use_group_norm  = vision_cfg.use_group_norm,
            share_rgb_model = vision_cfg.share_rgb_model,
            imagenet_norm   = vision_cfg.imagenet_norm 
        )
        
        # deep koopman operator module
        self.dko = DeepKoopmanModule(dko_config=dko_cfg)
        
        # noise_scheduler:DDPMScheduler
        self.infer_steps = diffuser_cfg.num_train_timesteps
        self.noise_scheduler = DDPMScheduler(**diffuser_cfg) 
        
        # diffusion_model:ConditionalUnet1D
        self.fuse = AttentionFusion(**fuse_cfg)
        self.diffusion_model = ConditionalUnet1D(**diffmodel_cfg) 
        
        # optimizer
        self.huber_loss = nn.HuberLoss(delta=1.0, reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.optimizer  = torch.optim.AdamW(
            params=self.parameters(), **optimizer_cfg                              
        )
    
    def augment_images(self, images:torch.Tensor):
        '''
            image dimension: batch_size x img_t x img_view x img_channel x img_height x img_width
            iaa.aug : required dimension should be: batch_size x img_height x img_width x img_channel
        '''
        imgshape = images.shape
        device, datatype = images.device, images.dtype
        img_augmenter = iaa.Sequential([
            iaa.CropAndPad(percent=np.random.uniform(-0.10, 0.10)),  # crop and pad images
            iaa.Fliplr(np.random.choice([0, 1], p=[0.5, 0.5])),      # horizontal
            iaa.Affine(rotate=np.random.uniform(-15, 15)),           # rotation
            iaa.AdditiveGaussianNoise(scale=0.02),                   # additive gaussian noise, the intensity range is (0.0, 1.0)
            iaa.Multiply(np.random.uniform(0.9, 1.1)),               # change brightness of images
        ])

        # augment images
        input_images = rearrange(images, 'b t v c h w -> (b t v) h w c') # merge and change the channel position
        input_images = input_images.cpu().numpy()
        aug_images = img_augmenter(images=input_images)
        
        # restore the image shape, and cast it to cuda
        aug_images = rearrange(aug_images, 'x h w c -> x c h w')
        aug_images = aug_images.reshape(*imgshape)
        aug_images = torch.tensor(aug_images, dtype=datatype, device=device)
        
        curr_vis_dict = { 
            'front_rgb': aug_images[:, 0, 0, ...], 
            'wrist_rgb': aug_images[:, 0, 1, ...],
        }
        
        next_vis_dict = { 
            'front_rgb': aug_images[:, 1, 0, ...], 
            'wrist_rgb': aug_images[:, 1, 1, ...],
        }
        
        return curr_vis_dict, next_vis_dict

    def generate_switch_signal(self, p0=0.3, bs:int=None):
        if bs is None:
            return np.random.choice([0, 1], p=[p0, 1-p0])
        else:
            return ((torch.rand(bs, 1) > p0).int()).float() 

    def get_train_loss(self, qpos, image, actions=None, is_pad=None):
        # note that the keys of the observation are defined in the 
        # configuration script, and the images will be normed depends
        # on the vision_cfg settings when initializing the obs_encoder
        # print('image shape: ', image.shape) # batch_size x (2: start/next)
        # x (2: front/wrist) x channel x height x width
        curr_vis_dict = { 
            'front_rgb': image[:, 0, 0, ...], 
            'wrist_rgb': image[:, 0, 1, ...],
        }
        curr_vis_enc = self.obs_encoder(curr_vis_dict) # (batch_size, dim)
        curr_acts  = actions[:, 0, ...]
        curr_qpos = qpos[:, 0, ...]
        curr_pad  = is_pad[:, 0, ...]
        
        next_vis_dict = { 
            'front_rgb': image[:, 1, 0, ...], 
            'wrist_rgb': image[:, 1, 1, ...],
        }
        next_vis_enc = self.obs_encoder(next_vis_dict)
        next_acts  = actions[:, 1, ...]
        next_qpos = qpos[:, 1, ...]
        next_pad  = is_pad[:, 1, ...]
        
        #   - 1 koopman errors: koopman state consistency error, update K V and policy
        pred_next_vis_enc, curr_latent_acts = self.dko(curr_vis_enc, latent_act=None)
        koop_consistency_loss = F.mse_loss(next_vis_enc, pred_next_vis_enc, reduction='none')
        koop_consistency_loss = reduce(koop_consistency_loss, 'b ... -> b ', 'mean') 
        koop_consistency_loss = koop_consistency_loss.mean()
        
        #   - 2 latent action consistency error, update feature, augment image but using the same latent action
        aug_curr_vis_dict , aug_next_vis_dict = self.augment_images(image)
        aug_curr_vis_enc = self.obs_encoder(aug_curr_vis_dict)
        aug_next_vis_enc = self.obs_encoder(aug_next_vis_dict)
        pred_aug_next_vis_enc  = self.dko(aug_curr_vis_enc, latent_act=curr_latent_acts)
        koop_consistency_loss_ = F.mse_loss(aug_next_vis_enc, pred_aug_next_vis_enc, reduction='none')
        koop_consistency_loss_ = reduce(koop_consistency_loss_, 'b ... -> b ', 'mean') 
        koop_consistency_loss_ = koop_consistency_loss_.mean()
        
        #   - 3. diffusion loss, predict actions
        # 3.1 latent action feature to action sequence
        # loss 1
        switch_signal = self.generate_switch_signal(self.fuse_prob) 
        fea_fuse = self.fuse(curr_qpos, curr_vis_enc)
        cond_enc = switch_signal*curr_latent_acts + (1-switch_signal)*fea_fuse
        diffusion_loss1 = self.compute_diff_loss(
            diff_model=self.diffusion_model, global_cond=cond_enc, 
            actions=curr_acts, is_pad=curr_pad, ret_pred=False
        )
        # loss 2
        switch_signal = self.generate_switch_signal(self.fuse_prob) 
        fea_fuse2 = self.fuse(next_qpos, next_vis_enc)
        next_latent_acts = self.dko.get_latnet_act(next_vis_enc)
        cond_enc2 = switch_signal*next_latent_acts + (1-switch_signal)*fea_fuse2 
        diffusion_loss2 = self.compute_diff_loss(
            diff_model=self.diffusion_model, global_cond=cond_enc2, 
            actions=next_acts, is_pad=next_pad, ret_pred=False
        )
        
        # sum up losses
        loss_dict = dict()
        loss_dict['koop_consis_kvp'] = koop_consistency_loss
        loss_dict['koop_consis_fea'] = koop_consistency_loss_
        loss_dict['diffusion']  = 0.5*diffusion_loss1 + 0.5*diffusion_loss2
        loss_dict['loss'] = koop_consistency_loss*0.3 + koop_consistency_loss_*0.7 + loss_dict['diffusion']
        return loss_dict
     
    def get_generative_errs(self, diff_model, cond:torch.Tensor, acts:torch.Tensor, num_samples:int):
        '''
            The errors are computed in batches
        '''
        batch_size = num_samples
        
        # duplicate acts and cond #num_samples times
        repeat_factors = [1]*cond.ndimension()
        repeat_factors[0] = num_samples
        cond = cond.repeat(*repeat_factors)
        
        repeat_factors = [1]*acts.ndimension()
        repeat_factors[0] = num_samples
        acts = acts.repeat(*repeat_factors)
        
        # get prediction
        noise = torch.randn(acts.shape, device=acts.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size, ), device=acts.device, dtype=torch.long
        ).long()
        noisy_traj = self.noise_scheduler.add_noise(acts, noise, timesteps)
        pred = diff_model(noisy_traj, timesteps, local_cond=None, global_cond=cond)
        
        # get loss, predict epsilon
        loss = F.mse_loss(pred, target=noise, reduction='none')
        loss = reduce(loss, 'b ... -> b ', 'mean') # reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss
    
    def get_action_uncertainty_pairs(self, qpos, image, num_samples:int):
        with torch.inference_mode():
            vision_dict = { 
                'front_rgb': image[:, 0, ...],
                'wrist_rgb': image[:, 1, ...],
            }
            vis_enc = self.obs_encoder(vision_dict) # (batch_size, dim)
            
            # get features
            batch_size = qpos.shape[0]
            fea_fuse = self.fuse(qpos, vis_enc) # fused feature
            fea_act = self.dko.get_latnet_act(vis_enc) # latent feature
            
            # get conditions
            global_fuse_cond=torch.cat((torch.zeros_like(fea_fuse), fea_fuse), dim=-1)
            global_vis_cond=torch.cat((fea_act, torch.zeros_like(fea_act)), dim=-1)
            global_all_cond=torch.cat((fea_act, fea_act), dim=-1)
            
            # get actions
            acts_fuse = self.conditional_sample(
                model=self.diffusion_model,
                traj_shape=(batch_size, self.pred_seqlen, self.action_dim),
                global_cond=global_fuse_cond
            )[:, self.n_propri-1:]
            
            acts_vis = self.conditional_sample(
                model=self.diffusion_model,
                traj_shape=(batch_size, self.pred_seqlen, self.action_dim),
                global_cond=global_vis_cond
            )[:, self.n_propri-1:]
            
            acts_all = self.conditional_sample(
                model=self.diffusion_model,
                traj_shape=(batch_size, self.pred_seqlen, self.action_dim),
                global_cond=global_all_cond
            )[:, self.n_propri-1:]
            
            # compute generative errors
            err_fuse = self.get_generative_errs(
                diff_model=self.diffusion_model, 
                cond=global_fuse_cond, acts=acts_fuse, 
                num_samples=num_samples
            )
            err_vis = self.get_generative_errs(
                diff_model=self.diffusion_model, 
                cond=global_vis_cond, acts=acts_vis, 
                num_samples=num_samples
            )
            err_all = self.get_generative_errs(
                diff_model=self.diffusion_model, 
                cond=global_all_cond, acts=acts_all, 
                num_samples=num_samples
            )
            
        return {
            'fuse': [acts_fuse, err_fuse], 'vis': [acts_vis, err_vis], 'all': [acts_all, err_all]
        }
    
    def forward(self, qpos, image, actions=None, is_pad=None, ret_generr=False):
        '''
        qpos -> prioperception data
        image -> observation images, multi-view cameras
        action -> actions to be predicted
        is_pad -> define which actions are not for computing
        tti_flag -> test time inference, to refine the final output action
        '''
        # twisted immage input
        if actions is not None:
            return self.get_train_loss(qpos, image, actions, is_pad)
        else:
            vision_dict = { 
                'front_rgb': image[:, 0, ...],
                'wrist_rgb': image[:, 1, ...],
            }
            vis_enc = self.obs_encoder(vision_dict) # (batch_size, dim)
            fuse_enc = self.fuse(qpos, vis_enc) # fused feature
            
            # get actions
            # batch_size = fuse_enc.shape[0]
            batch_size = fuse_enc.shape[0]
            fuse_acts = self.conditional_sample(
                model=self.diffusion_model,
                traj_shape=(batch_size, self.pred_seqlen, self.action_dim),
                global_cond=fuse_enc
            )[:, self.n_propri-1:]
            
            latent_act = self.dko.get_latnet_act(vis_enc)
            vis_acts = self.conditional_sample(
                model=self.diffusion_model,
                traj_shape=(batch_size, self.pred_seqlen, self.action_dim),
                global_cond=latent_act
            )[:, self.n_propri-1:]

            if not ret_generr:
                return fuse_acts, vis_acts
            else:
                # return diffusion model reconstruction errors (to test whether the output acts are of high confidence or not)
                fuse_err = self.get_generative_errs(self.diffusion_model, fuse_enc, fuse_acts, num_samples=10)
                vis_err  = self.get_generative_errs(self.diffusion_model, latent_act, vis_acts, num_samples=10) # action presentation can not reflect the task progress
                return {
                    'fuse': [fuse_acts, fuse_err], 'vis': [vis_acts, vis_err]
                }
    
    def get_generative_errs(self, diff_model, cond:torch.Tensor, acts:torch.Tensor, num_samples:int):
        '''
            The errors are computed in batches
        '''
        batch_size = num_samples
        
        # duplicate acts and cond #num_samples times
        repeat_factors = [1]*cond.ndimension()
        repeat_factors[0] = num_samples
        cond = cond.repeat(*repeat_factors)
        
        repeat_factors = [1]*acts.ndimension()
        repeat_factors[0] = num_samples
        acts = acts.repeat(*repeat_factors)
        
        # get prediction
        noise = torch.randn(acts.shape, device=acts.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size, ), device=acts.device, dtype=torch.long
        ).long()
        noisy_traj = self.noise_scheduler.add_noise(acts, noise, timesteps)
        pred = diff_model(noisy_traj, timesteps,
            local_cond=None, global_cond=cond)
        
        # get loss, predict epsilon
        loss = F.mse_loss(pred, target=noise, reduction='none')
        loss = reduce(loss, 'b ... -> b ', 'mean') # reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss
    
    def configure_optimizers(self):
        return self.optimizer

    def conditional_sample(self, model, traj_shape, global_cond):
        self.noise_scheduler.set_timesteps(self.infer_steps)
        traj = torch.randn(size=traj_shape, device=global_cond.device) 
        for t in self.noise_scheduler.timesteps:
            model_output = model(
                sample=traj, 
                timestep=t, 
                local_cond=None, 
                global_cond=global_cond
            )
            traj = self.noise_scheduler.step(
                model_output, t, traj
            )['prev_sample']
        return traj

    def compute_action_loss(self, gt_acts, pred_acts, is_pad):
        gt_acts = gt_acts*~is_pad.unsqueeze(-1)
        pred_acts = pred_acts*~is_pad.unsqueeze(-1) 
        
        loss = self.huber_loss(input=gt_acts, target=pred_acts)
        loss = reduce(loss, 'b ... -> b ', 'mean') 
        loss = loss.mean()
        return loss
    
    def compute_align_loss(self, feature_1, feature_2):
        loss = self.mse_loss(feature_1, feature_2)
        loss = reduce(loss, 'b ... -> b ', 'mean') 
        loss = loss.mean()
        return loss

    def compute_diff_loss(self, diff_model, global_cond, actions, is_pad=None, ret_pred=False):
        batch_size = actions.shape[0]
        # get prediction
        noise = torch.randn(actions.shape, device=actions.device)

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size, ), device=actions.device, dtype=torch.long
        ).long()
        noisy_traj = self.noise_scheduler.add_noise(actions, noise, timesteps)
        pred = diff_model(noisy_traj, timesteps,
            local_cond=None, global_cond=global_cond)
        
        # get loss, predict epsilon
        loss = F.mse_loss(pred, target=noise, reduction='none')
        loss = loss* ~is_pad.unsqueeze(-1) # exclude losses from the padding actions
        loss = reduce(loss, 'b ... -> b ', 'mean') # reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        if ret_pred: 
            return loss, pred
        return loss

class LstmModule(nn.Module):
    def __init__(self, lstm_config):
        super().__init__()
        self.in_dim  = lstm_config.in_dim
        self.mid_dim = lstm_config.mid_dim
        self.num_layer = lstm_config.num_layer
        self.win_size  = lstm_config.win_size
        self.lstm = nn.LSTM(self.in_dim, self.mid_dim, self.num_layer, batch_first=False)
    
    def forward(self, input):
        _, (h_n, c_n) = self.lstm(input)
        return h_n[0]
        
class D3P_LSTM(nn.Module):
    def __init__(self, policy_config): 
        super().__init__()
        # parse configs
        self.action_dim = policy_config.action_dim
        self.chunking_size = policy_config.chunking_size
        self.n_propri = policy_config.n_propri
        self.query_every = policy_config['query_every']
        self.pred_seqlen = self.n_propri + self.chunking_size - 1 # length of the predicted actions including n_propri acts and chunking_size -1 effective horizon
        lstm_config = policy_config['lstm_config']
        vision_cfg = policy_config['vision_config']
        diffuser_cfg = policy_config['diffuser_config']
        diffmodel_cfg = policy_config['diffmodel_config']
        optimizer_cfg = policy_config['optimizer_config']
        fuse_cfg = policy_config['fuse_config']
        # rnc_cfg = policy_config['rnc_loss_cfg']
        self.fuse_prob = policy_config['switch_prob']
        
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
        
        # lstm for comparison
        self.lstm = LstmModule(lstm_config)
        
        # noise_scheduler:DDPMScheduler
        self.infer_steps = diffuser_cfg.num_train_timesteps
        self.noise_scheduler = DDPMScheduler(**diffuser_cfg) 
        
        # diffusion_model:ConditionalUnet1D
        # self.fuse = AttFusion(**fuse_cfg) # prone to overfitting
        self.fuse = AttentionFusion(**fuse_cfg)
        self.diffusion_model = ConditionalUnet1D(**diffmodel_cfg) # for latent to detect OOD data
        
        # optimizer
        self.huber_loss = nn.HuberLoss(delta=1.0, reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.optimizer  = torch.optim.AdamW(
            params=self.parameters(), **optimizer_cfg                              
        )

    def generate_switch_signal(self, p0=0.3, bs:int=None):
        if bs is None:
            return np.random.choice([0, 1], p=[p0, 1-p0])
        else:
            return ((torch.rand(bs, 1) > p0).int()).float() 

    def get_train_loss(self, qpos, image, actions=None, is_pad=None):
        # note that the keys of the observation are defined in the 
        # configuration script, and the images will be normed depends
        # on the vision_cfg settings when initializing the obs_encoder
        # print('image shape: ', image.shape) # batch_size x (2: start/next)
        # x (2: front/wrist) x channel x height x width
        # no image augmentation
        lstm_input_list = list()
        for i in range(self.lstm.win_size-1):
            vis_dict = {
                'front_rgb': image[:, -1, 0, ...], 
                'wrist_rgb': image[:, -1, 1, ...],
            }
            lstm_input_list.append(self.obs_encoder(vis_dict))
        curr_vis_dict = { 
            'front_rgb': image[:, -1, 0, ...], 
            'wrist_rgb': image[:, -1, 1, ...],
        }
        curr_vis_enc = self.obs_encoder(curr_vis_dict)   # (batch_size, dim)
        lstm_input_list.append(curr_vis_enc)
        
        curr_acts  = actions[:, -1, ...]
        curr_qpos = qpos[:, -1, ...]
        curr_pad  = is_pad[:, -1, ...]
        
        lstm_input = torch.stack(lstm_input_list, dim=0)
        lstm_enc = self.lstm(lstm_input)
        
        #   - diffusion loss, predict actions
        #     latent action feature to action sequence
        loss_dict = dict()
        switch_signal = self.generate_switch_signal(self.fuse_prob) 
        fea_fuse = self.fuse(curr_qpos, curr_vis_enc)
        cond_enc = switch_signal*lstm_enc + (1-switch_signal)*fea_fuse # switch to fuse_enc 
        loss_dict['diffusion'] = self.compute_diff_loss(
            diff_model=self.diffusion_model, global_cond=cond_enc, 
            actions=curr_acts, is_pad=curr_pad, ret_pred=False
        )
        loss_dict['loss'] = loss_dict['diffusion']
        return loss_dict
    
    def get_generative_errs(self, diff_model, cond:torch.Tensor, acts:torch.Tensor, num_samples:int):
        '''
            The errors are computed in batches
        '''
        batch_size = num_samples
        
        # duplicate acts and cond #num_samples times
        repeat_factors = [1]*cond.ndimension()
        repeat_factors[0] = num_samples
        cond = cond.repeat(*repeat_factors)
        
        repeat_factors = [1]*acts.ndimension()
        repeat_factors[0] = num_samples
        acts = acts.repeat(num_samples)
        
        # get prediction
        noise = torch.randn(acts.shape, device=acts.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size, ), device=acts.device, dtype=torch.long
        ).long()
        noisy_traj = self.noise_scheduler.add_noise(acts, noise, timesteps)
        pred = diff_model(noisy_traj, timesteps, local_cond=None, global_cond=cond)
        
        # get loss, predict epsilon
        loss = F.mse_loss(pred, target=noise, reduction='none')
        loss = reduce(loss, 'b ... -> b ', 'mean') # reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss
    
    def get_action_uncertainty_pairs(self, qpos, image, num_samples:int):
        with torch.inference_mode():
            vision_dict = { 
                'front_rgb': image[:, 0, ...],
                'wrist_rgb': image[:, 1, ...],
            }
            vis_enc = self.obs_encoder(vision_dict) # (batch_size, dim)
            
            # get features
            batch_size = qpos.shape[0]
            fea_fuse = self.fuse(qpos, vis_enc) # fused feature
            fea_act = self.dko.get_latnet_act(vis_enc) # latent feature
            
            # get conditions
            global_fuse_cond=torch.cat((torch.zeros_like(fea_fuse), fea_fuse), dim=-1)
            global_vis_cond=torch.cat((fea_act, torch.zeros_like(fea_act)), dim=-1)
            global_all_cond=torch.cat((fea_act, fea_act), dim=-1)
            
            # get actions
            acts_fuse = self.conditional_sample(
                model=self.diffusion_model,
                traj_shape=(batch_size, self.pred_seqlen, self.action_dim),
                global_cond=global_fuse_cond
            )[:, self.n_propri-1:]
            
            acts_vis = self.conditional_sample(
                model=self.diffusion_model,
                traj_shape=(batch_size, self.pred_seqlen, self.action_dim),
                global_cond=global_vis_cond
            )[:, self.n_propri-1:]
            
            acts_all = self.conditional_sample(
                model=self.diffusion_model,
                traj_shape=(batch_size, self.pred_seqlen, self.action_dim),
                global_cond=global_all_cond
            )[:, self.n_propri-1:]
            
            # compute generative errors
            err_fuse = self.get_generative_errs(
                diff_model=self.diffusion_model, 
                cond=global_fuse_cond, acts=acts_fuse, 
                num_samples=num_samples
            )
            err_vis = self.get_generative_errs(
                diff_model=self.diffusion_model, 
                cond=global_vis_cond, acts=acts_vis, 
                num_samples=num_samples
            )
            err_all = self.get_generative_errs(
                diff_model=self.diffusion_model, 
                cond=global_all_cond, acts=acts_all, 
                num_samples=num_samples
            )
            
        return {
            'fuse': [acts_fuse, err_fuse], 'vis': [acts_vis, err_vis], 'all': [acts_all, err_all]
        }
    
    def forward(self, qpos, image, actions=None, is_pad=None, ret_generr=False):
        '''
        qpos -> prioperception data
        image -> observation images, multi-view cameras
        action -> actions to be predicted
        is_pad -> define which actions are not for computing
        tti_flag -> test time inference, to refine the final output action
        '''
        # twisted immage input
        if actions is not None:
            return self.get_train_loss(qpos, image, actions, is_pad)
        else:
            vision_dict = { 
                'front_rgb': image[:, 0, ...],
                'wrist_rgb': image[:, 1, ...],
            }            
            vis_enc = self.obs_encoder(vision_dict) # (batch_size, dim)
            fuse_enc = self.fuse(qpos, vis_enc) # fused feature
            
            # get actions
            batch_size = fuse_enc.shape[0]
            fuse_acts = self.conditional_sample(
                model=self.diffusion_model,
                traj_shape=(batch_size, self.pred_seqlen, self.action_dim),
                global_cond=fuse_enc
            )[:, self.n_propri-1:]
            
            
            latent_act = self.lstm(vis_enc.unsqueeze(0))
            
            vis_acts = self.conditional_sample(
                model=self.diffusion_model,
                traj_shape=(batch_size, self.pred_seqlen, self.action_dim),
                global_cond=latent_act
            )[:, self.n_propri-1:]

            if not ret_generr:
                return fuse_acts, vis_acts
            else:
                # return diffusion model reconstruction errors (to test whether the output acts are of high confidence or not)
                fuse_err = self.get_generative_errs(self.diffusion_model, fuse_enc, fuse_acts, num_samples=10)
                vis_err  = self.get_generative_errs(self.diffusion_model, latent_act, vis_acts, num_samples=10) # action presentation can not reflect the task progress
                return {
                    'fuse': [fuse_acts, fuse_err], 'vis': [vis_acts, vis_err]
                }

            # return fuse_acts, act_acts # use the future actions for execution, batch size x num acts x act_dim
    
    def get_generative_errs(self, diff_model, cond:torch.Tensor, acts:torch.Tensor, num_samples:int):
        '''
            The errors are computed in batches
        '''
        batch_size = num_samples
        
        # duplicate acts and cond #num_samples times
        repeat_factors = [1]*cond.ndimension()
        repeat_factors[0] = num_samples
        cond = cond.repeat(*repeat_factors)
        
        repeat_factors = [1]*acts.ndimension()
        repeat_factors[0] = num_samples
        acts = acts.repeat(*repeat_factors)
        
        # get prediction
        noise = torch.randn(acts.shape, device=acts.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size, ), device=acts.device, dtype=torch.long
        ).long()
        noisy_traj = self.noise_scheduler.add_noise(acts, noise, timesteps)
        pred = diff_model(noisy_traj, timesteps,
            local_cond=None, global_cond=cond)
        
        # get loss, predict epsilon
        loss = F.mse_loss(pred, target=noise, reduction='none')
        loss = reduce(loss, 'b ... -> b ', 'mean') # reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss
    
    def configure_optimizers(self):
        return self.optimizer

    def conditional_sample(self, model, traj_shape, global_cond):
        self.noise_scheduler.set_timesteps(self.infer_steps)
        traj = torch.randn(size=traj_shape, device=global_cond.device) 
        for t in self.noise_scheduler.timesteps:
            model_output = model(
                sample=traj, 
                timestep=t, 
                local_cond=None, 
                global_cond=global_cond
            )
            traj = self.noise_scheduler.step(
                model_output, t, traj
            )['prev_sample']
        return traj

    def compute_action_loss(self, gt_acts, pred_acts, is_pad):
        gt_acts = gt_acts*~is_pad.unsqueeze(-1)
        pred_acts = pred_acts*~is_pad.unsqueeze(-1) 
        
        loss = self.huber_loss(input=gt_acts, target=pred_acts)
        loss = reduce(loss, 'b ... -> b ', 'mean') 
        loss = loss.mean()
        return loss
    
    def compute_align_loss(self, feature_1, feature_2):
        loss = self.mse_loss(feature_1, feature_2)
        loss = reduce(loss, 'b ... -> b ', 'mean') 
        loss = loss.mean()
        return loss

    def compute_diff_loss(self, diff_model, global_cond, actions, is_pad=None, ret_pred=False):
        batch_size = actions.shape[0]
        # get prediction
        noise = torch.randn(actions.shape, device=actions.device)

        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size, ), device=actions.device, dtype=torch.long
        ).long()
        noisy_traj = self.noise_scheduler.add_noise(actions, noise, timesteps)
        pred = diff_model(noisy_traj, timesteps,
            local_cond=None, global_cond=global_cond)
        
        # get loss, predict epsilon
        loss = F.mse_loss(pred, target=noise, reduction='none')
        loss = loss* ~is_pad.unsqueeze(-1) # exclude losses from the padding actions
        loss = reduce(loss, 'b ... -> b ', 'mean') # reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        if ret_pred: 
            return loss, pred
        return loss
    
    
    