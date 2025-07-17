# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import math
try:
    from flash_attn import flash_attn_func
    is_flash_attn = True
except:
    is_flash_attn = False
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
from einops import rearrange
from ldm.modules.diffusionmodules.flag_large_dit_moe import Attention, FeedForward, RMSNorm, modulate, TimestepEmbedder

#############################################################################
#                               Core DiT Model                              #
#############################################################################

class Conv1DFinalLayer(nn.Module):
    """
    The final layer of CrossAttnDiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.GroupNorm(16,hidden_size)
        self.conv1d = nn.Conv1d(hidden_size, out_channels,kernel_size=1)

    def forward(self, x): # x:(B,C,T)
        x = self.norm_final(x)
        x = self.conv1d(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int,
                 multiple_of: int, ffn_dim_multiplier: float, norm_eps: float,
                 qk_norm: bool, y_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, y_dim)
        self.feed_forward = FeedForward(
            dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                dim, 6 * dim, bias=True
            ),
        )
        self.attention_y_norm = RMSNorm(y_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention.
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and
                feedforward layers.

        """
        if adaln_input is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
                self.adaLN_modulation(adaln_input).chunk(6, dim=1)

            h = x + gate_msa.unsqueeze(1) * self.attention(
                modulate(self.attention_norm(x), shift_msa, scale_msa),
                x_mask,
                freqs_cis,
                self.attention_y_norm(y), y_mask,
            )
            out = h + gate_mlp.unsqueeze(1) * self.feed_forward(
                modulate(self.ffn_norm(h), shift_mlp, scale_mlp),
            )

        else:
            h = x + self.attention(
                self.attention_norm(x), x_mask, freqs_cis, self.attention_y_norm(y), y_mask,
            )
            out = h + self.feed_forward(self.ffn_norm(h))

        return out

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6,
        )
        self.linear = nn.Linear(
            hidden_size, out_channels, bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                hidden_size, 2 * hidden_size, bias=True
            ),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x



class TxtFlagLargeDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        in_channels,
        context_dim,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        max_len = 1000,
        n_kv_heads=None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps=1e-5,
        qk_norm=None,
        rope_scaling_factor: float = 1.,
        ntk_factor: float = 1.
    ):
        super().__init__()
        self.in_channels = in_channels # vae dim
        self.out_channels = in_channels
        self.num_heads = num_heads
        kernel_size = 5
        self.hidden_size = hidden_size
        
        self.t_embedder = TimestepEmbedder(hidden_size)

        # self.proj_in = nn.Linear(in_channels, hidden_size//2, bias=True)
        self.proj_in = nn.Conv1d(in_channels, hidden_size//2, kernel_size=kernel_size, padding=kernel_size//2)

        # self.code_num = 1024
        # self.codebook_num = 3
        # self.unit_upsample_rate = 1
        # self.code_embed = nn.Embedding(self.code_num * self.codebook_num + 5, hidden_size//2//self.codebook_num)
        # # self.code_proj = nn.Conv1d(hidden_size//2, hidden_size//2, kernel_size=kernel_size, padding=kernel_size//2)
        # acoustic
        self.code_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_size//2, kernel_size=kernel_size, padding=kernel_size//2),
            # nn.Linear(80, hidden_size//2, bias=True)
            nn.LeakyReLU(),
            nn.AvgPool1d(2),    # vae downsample 2
        )

        # Embeddings for midi and beats
        self.midi_embedding = nn.Embedding(130, hidden_size//2)  # MIDI range is 0-100
        self.beats_embedding = nn.Embedding(3, hidden_size//2)  # Beats are 0 or 1

        # Conv1d layer for f0 (continuous)
        self.f0_proj = nn.Sequential(
            nn.Conv1d(1, hidden_size//2, kernel_size=kernel_size, padding=kernel_size//2),
            # nn.Linear(80, hidden_size//2, bias=True)
            nn.LeakyReLU(),
            nn.AvgPool1d(2),    # vae downsample 2
        )
        
        self.midi_proj = nn.Sequential(
            nn.Conv1d(hidden_size//2, hidden_size//2, kernel_size=kernel_size, padding=kernel_size//2),
            # nn.Linear(80, hidden_size//2, bias=True)
            nn.LeakyReLU(),
            nn.AvgPool1d(2),    # vae downsample 2
        )
        
        self.beats_proj = nn.Sequential(
            nn.Conv1d(hidden_size//2, hidden_size//2, kernel_size=kernel_size, padding=kernel_size//2),
            # nn.Linear(80, hidden_size//2, bias=True)
            nn.LeakyReLU(),
            nn.AvgPool1d(2),    # vae downsample 2
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(layer_id, hidden_size, num_heads, n_kv_heads, multiple_of,
                             ffn_dim_multiplier, norm_eps, qk_norm, context_dim)
            for layer_id in range(depth)
        ])
        
        self.final_proj = nn.Conv1d(hidden_size//2, hidden_size//2, kernel_size=1)

        self.freqs_cis = TxtFlagLargeDiT.precompute_freqs_cis(hidden_size // num_heads, max_len,
                       rope_scaling_factor=rope_scaling_factor, ntk_factor=ntk_factor)

        self.final_layer = FinalLayer(hidden_size//2, self.out_channels)
        self.rope_scaling_factor = rope_scaling_factor
        self.ntk_factor = ntk_factor

        self.cap_embedder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_size, bias=True),
        )
        
        self.ada_final=nn.Linear(hidden_size, hidden_size//2, bias=True)
        # self.final_layer2 = Conv1DFinalLayer(hidden_size//2, self.out_channels)


    def forward(self, x, t, context):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of temporal inputs (latent representations of melspec)
        t: (N,) tensor of diffusion timesteps
        y: (N,max_tokens_len=77, context_dim)
        """
        self.freqs_cis = self.freqs_cis.to(x.device)

        acoustic = context['c_concat']['acoustic']    # (B, 80, T)
        f0 = context['c_concat']['f0']    # (B, 1, T)
        midi = context['c_concat']['midi']    # (B, 1, T)
        beats = context['c_concat']['beats']    # (B, 1, T)
        caption = context['c_crossattn']  # (B, T, 1024)
        name = context['name']


        acoustic = self.code_proj(acoustic).transpose(1, 2)     # [B, C, T] -> [B, T, C]
        # print('acoustic.shape 3', acoustic.shape)
        f0=self.f0_proj(f0).transpose(1, 2)
        midi=self.midi_embedding(midi.squeeze(1)).transpose(1, 2)   
        beats=self.beats_embedding(beats.squeeze(1)).transpose(1, 2)   
        midi=self.midi_proj(midi).transpose(1, 2)
        beats=self.beats_proj(beats).transpose(1, 2)
        

        acoustic+=f0+midi+beats #[B, T, C]
        acoustic=self.final_proj(acoustic.transpose(1, 2)).transpose(1, 2) #[B, T, C]

        x = self.proj_in(x).transpose(1, 2)     # [B, C, T] -> [B, T, C]
        
        if abs(x.shape[1] - acoustic.shape[1]) <= 2:
            if x.shape[1] > acoustic.shape[1]:
                acoustic = torch.concat([acoustic, acoustic[:, -1, :].unsqueeze(1).repeat(1, x.shape[1] - acoustic.shape[1], 1)], dim=1)
            else:
                acoustic = acoustic[:, :x.shape[1], :]

        cap_mask = torch.ones((caption.shape[0], caption.shape[1]), dtype=torch.int32, device=x.device)  # [B, T] video时一直用非mask
        mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.int32, device=x.device)

        t = self.t_embedder(t)  # [B, 768]

        # get pooling feature
        cap_mask_float = cap_mask.float().unsqueeze(-1)
        cap_feats_pool = (caption * cap_mask_float).sum(dim=1) / cap_mask_float.sum(dim=1)
        cap_feats_pool = cap_feats_pool.to(caption) # [B, 768]
        cap_emb = self.cap_embedder(cap_feats_pool)  # [B, 768]

        x = torch.concat([acoustic, x], dim=2)      # channel-wise concat  [B, T, C]

        adaln_input = t + cap_emb   # [B, 768]
        cap_mask = cap_mask.bool()
        for block in self.blocks:
            x = block(
                x, mask, caption, cap_mask, self.freqs_cis[:x.size(1)],
                adaln_input=adaln_input
            )# (N,T,out_channels)


        x = x[:, :, self.hidden_size//2:]
        adaln_input=self.ada_final(adaln_input)
        x = self.final_layer(x, adaln_input)                # (N, out_channels,T)
        x = rearrange(x, 'b t c -> b c t')            # (B, C, T)
        # x = x[:, self.hidden_size//2:, :]
        # x = self.final_layer2(x)
        
        return x

    @staticmethod
    def precompute_freqs_cis(
        dim: int,
        end: int,
        theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        ntk_factor: float = 1.0
    ):
        """
        Precompute the frequency tensor for complex exponentials (cis) with
        given dimensions.

        This function calculates a frequency tensor with complex exponentials
        using the given dimension 'dim' and the end index 'end'. The 'theta'
        parameter scales the frequencies. The returned tensor contains complex
        values in complex64 data type.

        Args:
            dim (int): Dimension of the frequency tensor.
            end (int): End index for precomputing frequencies.
            theta (float, optional): Scaling factor for frequency computation.
                Defaults to 10000.0.

        Returns:
            torch.Tensor: Precomputed frequency tensor with complex
                exponentials.
        """

        theta = theta * ntk_factor

        print(f"theta {theta} rope scaling {rope_scaling_factor} ntk {ntk_factor}")

        freqs = 1.0 / (theta ** (
            torch.arange(0, dim, 2)[: (dim // 2)].float().cuda() / dim
        ))
        t = torch.arange(end, device=freqs.device, dtype=torch.float)  # type: ignore
        t = t / rope_scaling_factor
        freqs = torch.outer(t, freqs).float()  # type: ignore
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        return freqs_cis




class TxtFlagLargeImprovedDiTV2(TxtFlagLargeDiT):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        in_channels,
        context_dim,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        max_len = 1000,
    ):
        super().__init__(in_channels, context_dim, hidden_size, depth, num_heads, max_len)

        self.initialize_weights()


    def initialize_weights(self):
        # Initialize transformer layers and proj_in:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        print('-------------------------------- successfully init! --------------------------------')
