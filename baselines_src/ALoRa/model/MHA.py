
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt, pi
import pandas as pd

class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ALoRaAtt(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=True):
        super(ALoRaAtt, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size

    def forward(self, queries, keys, values, attn_mask):
        device = queries.device  # Ensure all computations are on the same device
        
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Self-attention scores
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # dot product using Einstein Summation
        
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        window_size = attn.shape[-1]        
        
        SA = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", SA, values)

        if self.output_attention:
            return V.contiguous(), SA
        else:
            return V.contiguous(), None

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        # self.key_projection = self.query_projection # Symetric Self-attention matrix
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads
        self.d_model = d_model
 
    def forward(self, queries, keys, values, attn_mask):
        device = queries.device  # Ensure all computations are on the same device

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1).to(device)
        # print(f'the quiries shape is {queries.shape}')
        keys = self.key_projection(keys).view(B, S, H, -1).to(device)
        values = self.value_projection(values).view(B, S, H, -1).to(device)


        out, SA = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)  # Concatenate heads

        return self.out_projection(out), SA
    
