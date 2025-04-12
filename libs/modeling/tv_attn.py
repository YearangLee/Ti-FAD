# ------------------------------------------------------------------------
# Modified from Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import  nn


class TVMultiHeadAttention(nn.Module):
    def __init__(self, v_dim, t_dim, embed_dim, num_heads, dropout=0.1, cfg=None):
        super(TVMultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim              
        self.num_heads = num_heads              
        self.head_dim = embed_dim // num_heads  
        self.v_dim = v_dim
        self.t_dim = t_dim

        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)    
        self.dropout = dropout                

        self.query_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.query_t_proj = nn.Linear(self.t_dim, self.embed_dim)
        self.value_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.value_t_proj = nn.Linear(self.t_dim, self.embed_dim)
        self.key_v_proj = nn.Linear(self.v_dim, self.embed_dim)
        self.key_t_proj = nn.Linear(self.t_dim, self.embed_dim)

        self.out_v_proj = nn.Linear(self.embed_dim, self.v_dim)
        self.out_t_proj = nn.Linear(self.embed_dim, self.t_dim)


        self.stable_softmax_2d = True
        self.clamp_min_for_underflow = True
        self.clamp_max_for_overflow = True

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.query_v_proj.weight)
        self.query_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.query_t_proj.weight)
        self.query_t_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.value_v_proj.weight)
        self.value_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.value_t_proj.weight)
        self.value_t_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_v_proj.weight)
        self.out_v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_t_proj.weight)
        self.out_t_proj.bias.data.fill_(0)

    def forward(self, v, t, importance_weights, attention_mask_v=None, attention_mask_t=None):
        """_summary_

        Args:
            v (_type_): bs, n_vid, dim
            t (_type_): bs, n_text, dim
            attention_mask_v (_type_, optional): _description_. bs, n_img

        Returns:
            _type_: _description_
        """
        bsz, tgt_len, d = v.size()   

        # video
        query_v_states = self.query_v_proj(v) * self.scale                            
        key_t_states = self._shape(self.key_t_proj(t), -1, bsz)                       
        value_v_states = self._shape(self.value_v_proj(v), -1, bsz)                   
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_v_states = self._shape(query_v_states, tgt_len, bsz).view(*proj_shape)  
        key_t_states = key_t_states.view(*proj_shape)                                
        value_v_states = value_v_states.view(*proj_shape)                             
        src_len = key_t_states.size(1)

        attn_weights = torch.bmm(query_v_states, key_t_states.transpose(1, 2))      
        

        # text 
        query_t_states = self.query_t_proj(t) * self.scale                            
        key_v_states = self._shape(self.key_v_proj(v), -1, bsz)                       
        value_t_states = self._shape(self.value_t_proj(t), -1, bsz)                    
        query_t_states = self._shape(query_t_states, t.size(1), bsz).view(*proj_shape)
        key_v_states = key_v_states.view(*proj_shape)                                  
        value_t_states = value_t_states.view(*proj_shape)                             

        attn_weights_2 = torch.bmm(query_t_states, key_v_states.transpose(1, 2))       
        
        if importance_weights is not None:
            importance_weights = importance_weights.repeat(self.num_heads, 1, 1)
            attn_weights = attn_weights * importance_weights  
            
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.stable_softmax_2d:
            attn_weights = attn_weights - attn_weights.max()
            attn_weights_2 = attn_weights_2 - attn_weights_2.max()

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(
                attn_weights, min=-50000
            )  
            attn_weights_2 = torch.clamp(
                attn_weights_2, min=-50000
            ) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(
                attn_weights, max=50000
            )  
            attn_weights_2 = torch.clamp(
                attn_weights_2, max=50000
            )# Do not increase 50000, data type half has quite limited range

        attn_weights_t = attn_weights_2.softmax(dim=-1)
        attn_weights_v = attn_weights.softmax(dim=-1) 

        # video mask
        if attention_mask_v is not None:
            attn_weights = attn_weights.masked_fill(
                torch.logical_not(attention_mask_v[:, :, None]).repeat(self.num_heads, 1, 1), 0.0)

        attn_probs_v = F.dropout(attn_weights_v, p=self.dropout, training=self.training) 
        attn_probs_t = F.dropout(attn_weights_t, p=self.dropout, training=self.training) 

        attn_output_v = torch.bmm(attn_probs_v, value_t_states) 
        attn_output_t = torch.bmm(attn_probs_t, value_v_states)  

        if attn_output_v.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output_v` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output_v.size()}"
            )

        if attn_output_t.size() != (bsz * self.num_heads, src_len, self.head_dim):
            raise ValueError(
                f"`attn_output_t` should be of size {(bsz, self.num_heads, src_len, self.head_dim)}, but is {attn_output_t.size()}"
            )

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2)
        attn_output_v = attn_output_v.reshape(bsz, tgt_len, self.embed_dim) 

        attn_output_t = attn_output_t.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_t = attn_output_t.transpose(1, 2)
        attn_output_t = attn_output_t.reshape(bsz, src_len, self.embed_dim) 

        attn_output_v = self.out_v_proj(attn_output_v)
        attn_output_t = self.out_t_proj(attn_output_t)

        return attn_output_v, attn_output_t


class TVAttentionBlock(nn.Module):
    def __init__(
        self,
        v_dim,
        t_dim,
        embed_dim,
        num_heads,
        dropout=0.0
    ):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(TVAttentionBlock, self).__init__()

        # pre layer norm
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_t = nn.LayerNorm(t_dim)

        self.attn = TVMultiHeadAttention(
            v_dim=v_dim, t_dim=t_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout
        )

        # add layer scale for training stability
        self.dropout_v = nn.Dropout(dropout)
        self.dropout_t = nn.Dropout(dropout)

    def forward(self, v, t, importance_weights, attention_mask_v=None, attention_mask_t=None):
        delta_v, delta_t = self.attn(
            self.layer_norm_v(v),
            self.layer_norm_t(t),
            importance_weights,
            attention_mask_v=attention_mask_v,
            attention_mask_t=attention_mask_t
        )
        v = v + self.dropout_v(delta_v)
        t = t + self.dropout_t(delta_t)
        return v, t

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class TVCA(nn.Module):
    def __init__(
        self,
        v_dim,
        t_dim,
        embed_dim,
        num_heads,
        dropout=0.1,
        drop_path=0.0,
        init_values=1e-4,
        mlp_ratio=4.0,
        cfg=None,
    ):
        super().__init__()

        self.tv_attn_block = TVAttentionBlock(
            v_dim,
            t_dim,
            embed_dim,
            num_heads,
            dropout
        )

        self.ln_v = nn.LayerNorm(v_dim)
        self.ln_t = nn.LayerNorm(t_dim)

        self.linear_v1 = nn.Linear(v_dim, int(v_dim * mlp_ratio))
        self.linear_v2 = nn.Linear(int(v_dim * mlp_ratio), v_dim)
        self.linear_t1 = nn.Linear(t_dim, int(t_dim * mlp_ratio))
        self.linear_t2 = nn.Linear(int(t_dim * mlp_ratio), t_dim)

        self.dropout_v_ffn = nn.Dropout(dropout)
        self.dropout_t_ffn = nn.Dropout(dropout)
        self.dropout_v_mlp = nn.Dropout(dropout)
        self.dropout_t_mlp = nn.Dropout(dropout)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def mlp_v(self, v):
        return self.linear_v2(self.dropout_v_ffn(F.relu(self.linear_v1(v))))

    def mlp_l(self, l):
        return self.linear_t2(self.dropout_t_ffn(F.relu(self.linear_t1(l))))

    def forward(
        self, v, t, importance_weights, attention_mask_v=None, attention_mask_t=None,
    ):
        v, t = self.tv_attn_block(v, t, importance_weights, attention_mask_v, attention_mask_t) 

        # MLP 
        v = self.dropout_v_mlp(self.mlp_v(self.ln_v(v))) + v
        t = self.dropout_t_mlp(self.mlp_l(self.ln_t(t))) + t

        return v, t
