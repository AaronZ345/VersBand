import os
from pytorch_memlab import LineProfiler,profile
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps
from torchvision.utils import make_grid
try:
    from pytorch_lightning.utilities.distributed import rank_zero_only
except:
    from pytorch_lightning.utilities import rank_zero_only # torch2
from torchdyn.core import NeuralODE
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.models.diffusion.ddpm_audio import LatentDiffusion_audio, disabled_train
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from omegaconf import ListConfig
import math

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


class CFM(LatentDiffusion_audio):

    def __init__(self, **kwargs):

        super(CFM, self).__init__(**kwargs)
        self.sigma_min = 1e-4

    def p_losses(self, x_start, cond, t, noise=None):
        x1 = x_start
        x0 = default(noise, lambda: torch.randn_like(x_start))
        ut = x1 - (1 - self.sigma_min) * x0  # 和ut的梯度没关系
        t_unsqueeze = t.unsqueeze(1).unsqueeze(1).float() / self.num_timesteps
        x_noisy = t_unsqueeze * x1 + (1. - (1 - self.sigma_min) * t_unsqueeze) * x0

        model_output,lb_loss = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'
        target = ut

        mean_dims = list(range(1,len(target.shape)))
        loss_simple = self.get_loss(model_output, target, mean=False).mean(dim=mean_dims)

        
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        loss_dict.update({f'{prefix}/lb_loss': lb_loss})
        

        loss = loss_simple
        loss = self.l_simple_weight * loss.mean()+lb_loss
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    @torch.no_grad()
    def sample(self, cond, batch_size=16, timesteps=None, shape=None, x_latent=None, t_start=None, **kwargs):
        if shape is None:
            mel_length = math.ceil(cond['acoustic']['acousitc'].shape[2] * 1 / 2)
            shape = (self.channels, self.mel_dim, mel_length) if self.channels > 0 else (self.mel_dim, mel_length)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        neural_ode = NeuralODE(self.ode_wrapper(cond), solver='euler', sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        t_span = torch.linspace(0, 1, 25 if timesteps is None else timesteps)
        if t_start is not None:
            t_span = t_span[t_start:]

        x0 = torch.randn(shape, device=self.device) if x_latent is None else x_latent
        eval_points, traj = neural_ode(x0, t_span)

        return traj[-1], traj

    def ode_wrapper(self, cond):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper(self, cond)

    @torch.no_grad()
    def sample_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning, batch_size=16, timesteps=None, shape=None, x_latent=None, t_start=None, **kwargs):
        if shape is None:
            # if self.channels > 0:
            #     shape = (batch_size, self.channels, self.mel_dim, self.mel_length)
            # else:
            #     shape = (batch_size, self.mel_dim, self.mel_length)
            mel_length = math.ceil(cond['acoustic']['acoustic'].shape[2] * 1 / 2)
            shape = (self.channels, self.mel_dim, mel_length) if self.channels > 0 else (self.mel_dim, mel_length)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        neural_ode = NeuralODE(self.ode_wrapper_cfg(cond, unconditional_guidance_scale, unconditional_conditioning), solver='euler', sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        t_span = torch.linspace(0, 1, 25 if timesteps is None else timesteps)

        if t_start is not None:
            t_span = t_span[t_start:]

        x0 = torch.randn(shape, device=self.device) if x_latent is None else x_latent
        eval_points, traj = neural_ode(x0, t_span)

        return traj[-1], traj

    def ode_wrapper_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning)


    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
        sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas
        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)


class Wrapper(nn.Module):
    def __init__(self, net, cond):
        super(Wrapper, self).__init__()
        self.net = net
        self.cond = cond

    def forward(self, t, x, args):
        t = torch.tensor([t * 1000] * x.shape[0], device=t.device).long()
        results,loss= self.net.apply_model(x, t, self.cond)
        return results


class Wrapper_cfg(nn.Module):

    def __init__(self, net, cond, unconditional_guidance_scale, unconditional_conditioning):
        super(Wrapper_cfg, self).__init__()
        self.net = net
        self.cond = cond
        self.unconditional_conditioning = unconditional_conditioning
        self.unconditional_guidance_scale = unconditional_guidance_scale

    def forward(self, t, x, args):
        # x_in = torch.cat([x] * 2)
        t = torch.tensor([t * 1000] * x.shape[0], device=t.device).long()
        # t_in = torch.cat([t] * 2)
        e_t,loss= self.net.apply_model(x, t, self.cond)
        e_t_uncond,loss= self.net.apply_model(x, t, self.unconditional_conditioning)
        e_t = e_t_uncond + self.unconditional_guidance_scale * (e_t - e_t_uncond)        
        
        return e_t
