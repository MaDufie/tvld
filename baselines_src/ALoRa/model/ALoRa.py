import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .MHA import ALoRaAtt, AttentionLayer
from .LMTSembed import DataEmbedding, TokenEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, win_size, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        # INITIALIZATIONS
        self.attention = attention
        self.norm1 = nn.LayerNorm(win_size) # For window size win_size
        self.norm2 = nn.LayerNorm(win_size) # For window size 20
        self.dropout = nn.Dropout(dropout) # Dropout layer WITH dropout rate with default value 0.1
        self.activation = F.relu if activation == "relu" else F.gelu
    # FORWARD PASS
    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        #Normalize across time for each feature ---
        x_permuted = x.permute(0, 2, 1)  # [B, D, T]
        x_norm = self.norm1(x_permuted)  
        x = x_norm.permute(0, 2, 1)      # [B, T, D]
        y = x
        # Optional activation fucntion
        # y = self.dropout(self.activation(y)) # (Optional)
        # Final normalization (again, per time series across time)
        y_permuted = (x + y).permute(0, 2, 1)
        y_norm = self.norm2(y_permuted)
        out = y_norm.permute(0, 2, 1)
        return out, attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        SA_list = []
        for attn_layer in self.attn_layers:
            x, SA = attn_layer(x, attn_mask=attn_mask)
            SA_list.append(SA)

        if self.norm is not None:
            x = self.norm(x)

        return x, SA_list

#Modified for the top512 cobination
class ALoRaT(nn.Module):
    def __init__(self,
                 win_size,
                 enc_in,
                 c_out,
                 d_model=512,
                 n_heads=8,
                 e_layers=3,
                 dropout=0.0,
                 activation='gelu',
                 output_attention=True,
                 *,
                 # optional: only needed if you want to use precomputed top-512 pairs
                 dataset=None,
                 precomputed_dir="./data_factory/Preprocess",
                 top_k_limit=512,
                 token_kernel_size=3):
        super(ALoRaT, self).__init__()
        self.output_attention = output_attention

        # Encoding (DataEmbedding may cap to top_k_limit and/or use precomputed pairs)
        self.embedding = DataEmbedding(
            c_in=enc_in,
            d_model=d_model,
            dropout=dropout,
            dataset=dataset,                 # pass if you want precomputed ordering
            precomputed_dir=precomputed_dir,
            top_k_limit=top_k_limit,
            kernel_size=token_kernel_size,
        )

        # Use the effective d_model from the embedding everywhere below
        d_eff = self.embedding.d_model
        self.d_model = d_eff
        # assert d_eff % n_heads == 0, f"d_model ({d_eff}) must be divisible by n_heads ({n_heads})"

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ALoRaAtt(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_eff, n_heads
                    ),
                    d_eff, win_size,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_eff)
        )

        # Projection (reconstruction MLP)
        self.projection = nn.Linear(d_eff, c_out, bias=True)

    def forward(self, x):
        Emb_data = self.embedding(x)          # [B, T, d_eff]
        enc_out, SA = self.encoder(Emb_data)  # [B, T, d_eff], attention list
        enc_out = self.projection(enc_out)    # [B, T, c_out]

        if self.output_attention:
            return enc_out, SA, Emb_data
        else:
            return enc_out
 
