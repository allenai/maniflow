# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# RDT: https://github.com/thu-ml/RoboticsDiffusionTransformer
# --------------------------------------------------------

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import Mlp, use_fused_attn

logger = logging.getLogger(__name__)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class AdaptiveLayerNorm(nn.Module):
    def __init__(
        self,
        dim,
        dim_cond,
    ):
        super().__init__()

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
 
        self.cond_linear = nn.Linear(dim_cond, dim * 2)

        self.cond_modulation = nn.Sequential(
            Rearrange('b d -> b 1 d'),
            nn.SiLU(),
            self.cond_linear
        )

        # Initialize the weights and biases of the conditional linear layer
        nn.init.zeros_(self.cond_linear.weight)
        nn.init.constant_(self.cond_linear.bias[:dim], 1.)
        nn.init.zeros_(self.cond_linear.bias[dim:])

    def forward(
        self,
        x,
        cond = None
    ):
        x = self.ln(x)
        gamma, beta = self.cond_modulation(cond).chunk(2, dim = -1)
        x = x * gamma + beta

        return x

class CrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention.
    """
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0,
            proj_drop: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                mask: None) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Prepare attn mask (B, L) to mask the conditioion
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x

class DiTXBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, p_drop_attn=0., qkv_bias=False, qk_norm=False, 
                 **block_kwargs):
        super().__init__()
        
        self.hidden_size = hidden_size

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, 
                                            dropout=p_drop_attn,
                                            )
        
        # Cross-Attention
        self.cross_attn = CrossAttention(
            dim=hidden_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_norm=qk_norm,
            norm_layer=nn.LayerNorm, **block_kwargs
        )
       
        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")  # Standard GELU
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # For self-attention
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # For cross-attention
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # For MLP

        # AdaLN modulation
        modulation_size = 9 * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, modulation_size, bias=True)
        )
        
    def forward(self, x, time_c, context_c, attn_mask=None):
        """
        Forward pass of the DiTX block.
        x: action, input tensor of shape (batch_size, seq_length, hidden_size)
        time_c: time embedding, global context tensor of shape (batch_size, hidden_size)
        context_c: visual and language tokens, context tensor of shape (batch_size, seq_length, hidden_size)
        attn_mask: Optional attention mask of shape (batch_size, seq_length, seq_length)
        """

        # adaLN modulation for self-attention, cross-attention, and MLP
        modulation = self.adaLN_modulation(time_c)

        # Split into 9 chunks of hidden_size each
        chunks = modulation.chunk(9, dim=-1)
        
        # Self-Attention parameters
        shift_msa, scale_msa, gate_msa = chunks[0], chunks[1], chunks[2]
        
        # Cross-Attention parameters  
        shift_cross, scale_cross, gate_cross = chunks[3], chunks[4], chunks[5]
        
        # MLP parameters
        shift_mlp, scale_mlp, gate_mlp = chunks[6], chunks[7], chunks[8]


        # Self-Attention with adaLN conditioning
        normed_x = modulate(self.norm1(x), shift_msa, scale_msa)  # Shape: (batch_size, seq_length, hidden_size)
        self_attn_output, _ = self.self_attn(normed_x, normed_x, normed_x, attn_mask=attn_mask)  # Shape: (batch_size, seq_length, hidden_size)
        x = x + gate_msa.unsqueeze(1) * self_attn_output  # Apply gating and residual connection
        

        # Cross-Attention with adaLN conditioning
        normed_x_cross = modulate(self.norm2(x), shift_cross, scale_cross)  # Apply adaLN to x before cross-attn
        cross_attn_output = self.cross_attn(normed_x_cross, context_c, mask=None)  # Shape: (batch_size, seq_length, hidden_size)
        x = x + gate_cross.unsqueeze(1) * cross_attn_output  # Apply gating and residual connection
       

        # MLP with adaLN conditioning
        normed_x_mlp = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(normed_x_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_output  # Apply gating and residual connection

        return x