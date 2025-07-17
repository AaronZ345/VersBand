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
from ldm.modules.diffusionmodules.flag_large_dit_moe import Attention, FeedForward, RMSNorm, modulate, TimestepEmbedder,ConditionEmbedder

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


class MoE(nn.Module):
    LOAD_BALANCING_LOSSES = []

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        temperature: float = 2.0,  # Gumbel-Softmax 的温度参数
    ):
        super().__init__()

        self.num_experts = num_experts
        self.temperature = temperature
        self.local_experts = [str(i) for i in range(num_experts)]

        # 高层门控网络，用于选择激活哪些专家组
        self.high_level_gating_network = nn.Linear(dim, 2)  # 3 表示 time, caption, acoustic

        # caption维度的 experts
        self.caption_experts = nn.ModuleDict({
            i: FeedForward(dim, hidden_dim, multiple_of=multiple_of,
                           ffn_dim_multiplier=ffn_dim_multiplier,) for i in self.local_experts
        })
        
        # acoustic维度的 experts
        self.acoustic_experts = nn.ModuleDict({
            i: FeedForward(dim, hidden_dim, multiple_of=multiple_of,
                           ffn_dim_multiplier=ffn_dim_multiplier,) for i in self.local_experts
        })

        # 门控网络（gating network）用于各自的专家选择
        self.caption_gating_network = nn.Linear(dim, num_experts)  
        self.acoustic_gating_network = nn.Linear(dim , num_experts)  

        self.freq_experts = nn.ModuleDict({
            i: FeedForward(dim, hidden_dim, multiple_of=multiple_of,
                           ffn_dim_multiplier=ffn_dim_multiplier, ) for i in self.local_experts
        })
        
        # 新增的 cross attention 层，caption 和 x 的交叉注意力
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)

    def gumbel_softmax(self, logits, temperature, hard=False):
        gumbels = -torch.empty_like(logits).exponential_().log()  
        gumbels = (logits + gumbels) / temperature  
        y_soft = gumbels.softmax(dim=-1)

        if hard:
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft

        return ret

    def load_balancing_loss(self, expert_probs_list, mask_list):
        expanded_masks = []
        for mask in mask_list:
            expanded_mask = mask.repeat(1, self.num_experts)  # 扩展mask到与专家数量相同的维度
            expanded_masks.append(expanded_mask)
        
        # 将所有专家的概率和对应的mask拼接
        all_expert_probs = torch.cat(expert_probs_list, dim=1)  # shape [N, num_experts * 3]
        all_masks = torch.cat(expanded_masks, dim=1)  # shape [N, num_experts * 3]

        # 计算每个专家的使用情况
        expert_usage = (all_expert_probs * all_masks).sum(dim=0)  # 对所有样本求和，shape [num_experts * 3]

        # 对每个专家的使用概率进行全局归一化
        total_samples = all_masks.sum()  # 计算所有激活的样本数
        expert_usage = expert_usage / (total_samples + 1e-10)  # 归一化使用情况
        
        # 计算负载均衡损失
        loss = torch.mean(expert_usage * torch.log(expert_usage + 1e-10))  # 计算损失，加上1e-10以避免log(0)

        return loss

    def forward(self, x, time, caption, acoustic):
        
        cross_attn_output, _ = self.cross_attention(x, caption, caption)  # cross-attention, query 是 x，key 和 value 是 caption

        # 将转换后的 caption 代入后续的计算中
        caption = cross_attn_output  # [B, T, 768]        
        caption = caption.reshape(-1, acoustic.shape[-1])  # [N, C]

        orig_shape = x.shape  # [B, T, 768]
        x = x.reshape(-1, x.shape[-1])  # [N, 768] N = B * T

        # acoustic 的 shape 是 [B, T, C]，需要拍平成 [N, C]
        acoustic = acoustic.reshape(-1, acoustic.shape[-1])  # [N, C]

        # 高层门控机制，选择激活哪些专家组
        high_level_logits = self.high_level_gating_network(time)  # [b, 3]
        high_level_logits=high_level_logits.repeat_interleave(orig_shape[1], dim=0)
        high_level_probs =self.gumbel_softmax(high_level_logits, 1.0, hard=False)# [N, 3] 概率值
        
        # 构建蒙版，确保至少一个专家组激活
        caption_mask = (high_level_probs[:, 0] ).float().unsqueeze(1)  # [N, 1]
        acoustic_mask = (high_level_probs[:, 1]).float().unsqueeze(1)  # [N, 1]

        # 时间维度的门控机制
        caption_logits = self.caption_gating_network(caption)  # [N, num_experts]
        acoustic_logits = self.acoustic_gating_network(acoustic)  # [N, num_experts]

        hard = not self.training  # 推理模式时使用硬采样

        if self.training and self.temperature > 0.3:
            self.temperature*=0.9999
            
        # 使用 Gumbel-Softmax 进行稀疏专家选择
        caption_expert_probs = self.gumbel_softmax(caption_logits, self.temperature, hard=hard)  # [N, num_experts]
        acoustic_expert_probs = self.gumbel_softmax(acoustic_logits, self.temperature, hard=hard)  # [N, num_experts]

        # 使用蒙版计算每个专家组的输出
        z_caption = torch.zeros_like(x)
        z_acoustic = torch.zeros_like(x)

        for str_i, expert in self.caption_experts.items():
            idx = int(str_i)
            expert_weight = caption_expert_probs[:, idx].unsqueeze(1).expand_as(x)  # [N, 768]
            z_caption += expert(x) * expert_weight * caption_mask  # [N, 768]

        for str_i, expert in self.acoustic_experts.items():
            idx = int(str_i)
            expert_weight = acoustic_expert_probs[:, idx].unsqueeze(1).expand_as(x)
            z_acoustic += expert(x) * expert_weight * acoustic_mask  # [N, 768]

        y = z_caption + z_acoustic
        
        y = y.view(*orig_shape).to(x)
        
        z = torch.zeros_like(y)
        # frequency-moe
        range = orig_shape[-1] // self.num_experts
        for str_i, expert in self.freq_experts.items(): # 找到需要用哪个expert算
            idx = int(str_i)
            region = torch.zeros_like(z)
            region[:, :, range * idx: range * (idx+1)] = True
            z[:, :, range * idx: range * (idx+1)] = expert(y * region)[:, :, range * idx: range * (idx+1)]

        z = z.view(*orig_shape).to(x)

        # 计算所有专家的负载均衡损失
        lb_loss = self.load_balancing_loss([caption_expert_probs, acoustic_expert_probs], [caption_mask, acoustic_mask])
        
        return z, lb_loss

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim: int, n_heads: int, n_kv_heads: int,
                 multiple_of: int, ffn_dim_multiplier: float, norm_eps: float,
                 qk_norm: bool, y_dim: int,num_experts) -> None:
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, n_kv_heads, qk_norm, y_dim)
        self.feed_forward = FeedForward(
            dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )
        self.feed_forward = MoE(
            dim=dim, hidden_dim= dim, multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier, num_experts=num_experts,
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
        time=None,
        caption=None,
        acoustic=None
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
            out,loss = self.feed_forward(
                modulate(self.ffn_norm(h), shift_mlp, scale_mlp),time,caption,acoustic
            )
            # out = self.feed_forward(
            #     modulate(self.ffn_norm(h), shift_mlp, scale_mlp)
            # )
            out=h + gate_mlp.unsqueeze(1) * out

        else:
            h = x + self.attention(
                self.attention_norm(x), x_mask, freqs_cis, self.attention_y_norm(y), y_mask
            )
            out,loss = self.feed_forward(self.ffn_norm(h),time,caption,acoustic)
            out=h + out

        return out,loss

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
        ntk_factor: float = 1.,
        num_experts=4,
        ori_dim=1024
    ):
        super().__init__()
        self.in_channels = in_channels # vae dim
        self.out_channels = in_channels
        self.num_heads = num_heads
        kernel_size = 5
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.ori_dim = ori_dim
        
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.proj_in = nn.Conv1d(in_channels, hidden_size, kernel_size=kernel_size, padding=kernel_size//2)

        # acoustic
        self.code_proj = nn.Sequential(
            nn.Conv1d(in_channels, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(),
            nn.AvgPool1d(2),    # vae downsample 2
        )

        # Embeddings for midi and beats
        self.midi_embedding = nn.Embedding(130, hidden_size)  # MIDI range is 0-100
        self.beats_embedding = nn.Embedding(3, hidden_size)  # Beats are 0 or 1
        
        self.midi_proj = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(),
            nn.AvgPool1d(2),    # vae downsample 2
        )
        
        self.beats_proj = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(),
            nn.AvgPool1d(2),    # vae downsample 2
        )
        
        self.blocks = nn.ModuleList([
            TransformerBlock(layer_id, hidden_size, num_heads, n_kv_heads, multiple_of,
                             ffn_dim_multiplier, norm_eps, qk_norm, context_dim,num_experts=self.num_experts)
            for layer_id in range(depth)
        ])
        
        self.final_proj = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)

        self.freqs_cis = TxtFlagLargeDiT.precompute_freqs_cis(hidden_size // num_heads, max_len,
                       rope_scaling_factor=rope_scaling_factor, ntk_factor=ntk_factor)

        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.rope_scaling_factor = rope_scaling_factor
        self.ntk_factor = ntk_factor

        self.cap_embedder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_size, bias=True),
        )

        self.loss_w=1
        self.c_embedder = ConditionEmbedder(hidden_size, ori_dim)

    def forward(self, x, t, context):
        """
        Forward pass of DiT.
        x: (N, C, T) tensor of temporal inputs (latent representations of melspec)
        t: (N,) tensor of diffusion timesteps
        y: (N,max_tokens_len=77, context_dim)
        """
        self.freqs_cis = self.freqs_cis.to(x.device)

        midi = context['c_concat']['midi']    # (B, 1, T)
        beats = context['c_concat']['beats']    # (B, 1, T)
        caption = context['c_crossattn']  # (B, T, 1024)

        midi=self.midi_embedding(midi.squeeze(1)).transpose(1, 2)   
        beats=self.beats_embedding(beats.squeeze(1)).transpose(1, 2)   
        midi=self.midi_proj(midi).transpose(1, 2)
        beats=self.beats_proj(beats).transpose(1, 2)
        acoustic=midi+beats #[B, T, C]
        acoustic=self.final_proj(acoustic.transpose(1, 2)).transpose(1, 2) #[B, T, C]

        x = self.proj_in(x).transpose(1, 2)     # [B, C, T] -> [B, T, C]
        
        if abs(x.shape[1] - acoustic.shape[1]) <= 2:
            if x.shape[1] > acoustic.shape[1]:
                acoustic = torch.concat([acoustic, acoustic[:, -1, :].unsqueeze(1).repeat(1, x.shape[1] - acoustic.shape[1], 1)], dim=1)
            else:
                acoustic = acoustic[:, :x.shape[1], :]

        cap_mask = torch.ones((caption.shape[0], caption.shape[1]), dtype=torch.int32, device=x.device)  # [B, T] video时一直用非mask
        mask = torch.ones((x.shape[0], x.shape[1]), dtype=torch.int32, device=x.device)

        t_emb = self.t_embedder(t)  # [B, 768]
        caption = self.c_embedder(caption)  # [B, T, 768]

        # get pooling feature
        cap_mask_float = cap_mask.float().unsqueeze(-1)
        cap_feats_pool = (caption * cap_mask_float).sum(dim=1) / cap_mask_float.sum(dim=1)
        cap_feats_pool = cap_feats_pool.to(caption) # [B, 768]
        cap_emb = self.cap_embedder(cap_feats_pool)  # [B, 768]

        x=acoustic+x
        adaln_input = t_emb + cap_emb   # [B, 768]
        cap_mask = cap_mask.bool()
        loss=0
        for block in self.blocks:
            x,loss_tmp = block(
                x, mask, caption, cap_mask, self.freqs_cis[:x.size(1)],
                adaln_input=adaln_input,time=t_emb,caption=caption,acoustic=acoustic
            )# (N,T,out_channels)
            loss+=loss_tmp
        loss/=len(self.blocks)
        
        if self.loss_w>0.01:
            self.loss_w*=0.9999
        loss*=self.loss_w #加系数防止影响过大

        x = self.final_layer(x, adaln_input)                # (N, out_channels,T)
        x = rearrange(x, 'b t c -> b c t')            # (B, C, T)
        
        return x,loss

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
        max_len = 1000,num_experts=4,ori_dim=1024
    ):
        super().__init__(in_channels, context_dim, hidden_size, depth, num_heads,max_len= max_len,num_experts=num_experts)

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
