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
from ldm.models.diffusion.cfm1_audio import Wrapper, Wrapper_cfg
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from omegaconf import ListConfig

from ldm.util import log_txt_as_img, exists, default

class CFMSampler(object):

    def __init__(self, model, num_timesteps, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.num_timesteps = num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def stochastic_encode(self, x_start, t, noise=None):
        x1 = x_start
        x0 = default(noise, lambda: torch.randn_like(x_start))
        t_unsqueeze = 1 - t.unsqueeze(1).unsqueeze(1).float() / self.num_timesteps
        x_noisy = t_unsqueeze * x1 + (1. - (1 - self.model.sigma_min) * t_unsqueeze) * x0
        return x_noisy

    @torch.no_grad()
    def sample(self, cond, batch_size=16, timesteps=None, shape=None, x_latent=None, t_start=None, **kwargs):
        
        # print(shape)
        
        if shape is None:
            if self.model.channels > 0:
                shape = (batch_size, self.model.channels, self.model.mel_dim, self.model.mel_length)
            else:
                shape = (batch_size, self.model.mel_dim, self.model.mel_length)
        # if cond is not None:
        #     if isinstance(cond, dict):
        #         cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
        #         list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
        #     else:
        #         cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if len(shape)==3:
            C, H, W = shape
            shape = (batch_size, C, H, W)
        else:
            C, T = shape
            shape = (batch_size, C, T) 

        neural_ode = NeuralODE(self.ode_wrapper(cond), solver='euler', sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        t_span = torch.linspace(0, 1, 25 if timesteps is None else timesteps)
        if t_start is not None:
            t_span = t_span[t_start:]

        x0 = torch.randn(shape, device=self.model.device) if x_latent is None else x_latent
        eval_points, traj = neural_ode(x0, t_span)

        return traj[-1], traj

    def ode_wrapper(self, cond):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper(self.model, cond)

    @torch.no_grad()
    def sample_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning, batch_size=16, timesteps=None, shape=None, x_latent=None, t_start=None, **kwargs):
        if shape is None:
            if self.model.channels > 0:
                shape = (batch_size, self.model.channels, self.model.mel_dim, self.model.mel_length)
            else:
                shape = (batch_size, self.model.mel_dim, self.model.mel_length)
        # if cond is not None:
            # if isinstance(cond, dict):
            #     cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
            #     list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            # else:
            #     cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if len(shape)==3:
            C, H, W = shape
            shape = (batch_size, C, H, W)
        else:
            C, T = shape
            shape = (batch_size, C, T) 

        neural_ode = NeuralODE(self.ode_wrapper_cfg(cond, unconditional_guidance_scale, unconditional_conditioning), solver='euler', sensitivity="adjoint", atol=1e-4, rtol=1e-4)
        t_span = torch.linspace(0, 1, 25 if timesteps is None else timesteps)

        if t_start is not None:
            t_span = t_span[t_start:]

        x0 = torch.randn(shape, device=self.model.device) if x_latent is None else x_latent
        eval_points, traj = neural_ode(x0, t_span)

        return traj[-1], traj

    def ode_wrapper_cfg(self, cond, unconditional_guidance_scale, unconditional_conditioning):
        # self.estimator receives x, mask, mu, t, spk as arguments
        return Wrapper_cfg(self.model, cond, unconditional_guidance_scale, unconditional_conditioning)

