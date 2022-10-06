import torch
import torch.nn as nn 
import numpy as np 
from timm.models.layers import DropPath, trunc_normal_




def generate_relative_distance(number_size):
    """return relative distance, (number_size**2, number_size**2, 2)
    """
    indices = torch.tensor(np.array([[x, y] for x in range(number_size) for y in range(number_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    distances = distances + number_size - 1   # shift the zeros postion
    return distances



class LightMutilHeadSelfAttention(nn.Module):
    """calculate the self attention with down sample the resolution for k, v, add the relative position bias before softmax
    Args:
        dim (int) : features map channels or dims 
        num_heads (int) : attention heads numbers
        relative_pos_embeeding (bool) : relative position embeeding 
        no_distance_pos_embeeding (bool): no_distance_pos_embeeding
        features_size (int) : features shape
        qkv_bias (bool) : if use the embeeding bias
        qk_scale (float) : qk scale if None use the default 
        attn_drop (float) : attention dropout rate
        proj_drop (float) : project linear dropout rate
        sr_ratio (float)  : k, v resolution downsample ratio
    Returns:
        x : LMSA attention result, the shape is (B, H, W, C) that is the same as inputs.
    """
    def __init__(self, dim, num_heads=8, features_size=56, 
                relative_pos_embeeding=False, no_distance_pos_embeeding=False, qkv_bias=False, qk_scale=None, 
                attn_drop=0., proj_drop=0., sr_ratio=1.):
        super(LightMutilHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}"
        self.dim = dim 
        self.num_heads = num_heads
        head_dim = dim // num_heads   # used for each attention heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_pos_embeeding = relative_pos_embeeding
        self.no_distance_pos_embeeding = no_distance_pos_embeeding

        self.features_size = features_size

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim) 
        
        if self.relative_pos_embeeding:
            self.relative_indices = generate_relative_distance(self.features_size)
            self.position_embeeding = nn.Parameter(torch.randn(2 * self.features_size - 1, 2 * self.features_size - 1))
        elif self.no_distance_pos_embeeding:
            self.position_embeeding = nn.Parameter(torch.randn(self.features_size ** 2, self.features_size ** 2))
        else:
            self.position_embeeding = None

        if self.position_embeeding is not None:
            trunc_normal_(self.position_embeeding, std=0.2)

    def forward(self, x):
        B, C, H, W = x.shape 
        N = H*W
        x_q = rearrange(x, 'B C H W -> B (H W) C')  # translate the B,C,H,W to B (H X W) C
        q = self.q(x_q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # B,N,H,DIM -> B,H,N,DIM
        
        # conv for down sample the x resoution for the k, v
        if self.sr_ratio > 1:
            x_reduce_resolution = self.sr(x)
            x_kv = rearrange(x_reduce_resolution, 'B C H W -> B (H W) C ')
            x_kv = self.norm(x_kv)
        else:
            x_kv = rearrange(x, 'B C H W -> B (H W) C ')
        
        kv_emb = rearrange(self.kv(x_kv), 'B N (dim h l ) -> l B h N dim', h=self.num_heads, l=2)         # 2 B H N DIM
        k, v = kv_emb[0], kv_emb[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale    # (B H Nq DIM) @ (B H DIM Nk) -> (B H NQ NK)
        
        # TODO: add the relation position bias, because the k_n != q_n, we need to split the position embeeding matrix
        q_n, k_n = q.shape[1], k.shape[2]
       
        if self.relative_pos_embeeding:
            attn = attn + self.position_embeeding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]][:, :k_n]
        elif self.no_distance_pos_embeeding:
            attn = attn + self.position_embeeding[:, :k_n]

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B H NQ NK) @ (B H NK dim)  -> (B NQ H*DIM)
        x = self.proj(x)
        x = self.proj_drop(x)
            
        x = rearrange(x, 'B (H W) C -> B C H W ', H=H, W=W)
        return x 

