import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):      
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float() 
        pe.require_grad = False 
        position = torch.arange(0, max_len).float().unsqueeze(1) 
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp() 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0)            
        self.register_buffer('pe', pe)  
    def forward(self, x):
        return self.pe[:, :x.size(1)]


#============ LigthMTSEmbeding =============================
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
from typing import Optional, List, Tuple

class TokenEmbedding(nn.Module):
    """
    - If total_pairs = C choose 2 <= 512: use ALL pairs (or fewer if you requested smaller d_model).
    - If total_pairs  > 512: load precomputed ordered pairs (pairs_idx_{dataset}.npy) and
      take the first min(d_model, 512).
    """

    def __init__(self,
                 c_in: int,
                 d_model: int,
                 kernel_size: int = 3,
                 top_k_limit: int = 512,
                 dataset: Optional[str] = None,
                 precomputed_dir: str = "./data_factory/Preprocess"):
        super().__init__()
        assert kernel_size % 2 == 1, "Use an odd kernel_size for symmetric circular padding."
        self.c_in = c_in
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.top_k_limit = top_k_limit

        total_pairs = c_in * (c_in - 1) // 2
        # Effective d_model: never exceed top_k_limit nor total available pairs
        d_eff = min(d_model, top_k_limit, total_pairs)
        if d_eff != d_model:
            print(f"[Top-K pairs selection] Adjusting d_model from {d_model} -> {d_eff} "
                  f"(TOP-K={top_k_limit}, total_pairs={total_pairs}).")
        self.d_model = d_eff

        selected_pairs: List[Tuple[int, int]] = []
        used_precomputed = False

        if total_pairs <= top_k_limit:
            # Use all possible pairs (or the first d_eff if you requested even smaller)
            possible_pairs = list(itertools.combinations(range(c_in), 2))
            selected_pairs = possible_pairs[:d_eff]
        else:
            # total_pairs > 512: prefer precomputed ordered pairs
            if dataset is None:
                print("[TokenEmbedding] Warning: dataset=None; falling back to unordered combinations.")
                possible_pairs = list(itertools.combinations(range(c_in), 2))
                selected_pairs = possible_pairs[:d_eff]
            else:
                pairs_idx_path = f"{precomputed_dir.rstrip('/')}/pairs_idx_{dataset}.npy"
                try:
                    pairs_pre = np.load(pairs_idx_path)
                    if pairs_pre.ndim != 2 or pairs_pre.shape[1] != 2:
                        raise ValueError(f"Invalid pairs file shape: {pairs_pre.shape}, expected (K,2).")
                    # keep only valid indices
                    valid = (pairs_pre[:, 0] < c_in) & (pairs_pre[:, 1] < c_in)
                    pairs_pre = pairs_pre[valid]

                    # take as many as available up to d_eff
                    take = min(len(pairs_pre), d_eff)
                    selected_pairs = [tuple(map(int, p)) for p in pairs_pre[:take].tolist()]
                    used_precomputed = True

                    # If precomputed has fewer than needed, fill remaining from combinations (no duplicates)
                    if take < d_eff:
                        need = d_eff - take
                        print(f"[TokenEmbedding] Precomputed pairs={take} < needed={d_eff}; "
                              f"filling {need} from combinations.")
                        used = set(selected_pairs)
                        for p in itertools.combinations(range(c_in), 2):
                            if p not in used:
                                selected_pairs.append(p)
                                used.add(p)
                                if len(selected_pairs) == d_eff:
                                    break
                except Exception as e:
                    print(f"[TokenEmbedding] Warning: could not load {pairs_idx_path}; "
                          f"falling back to combinations. Error: {e}")
                    possible_pairs = list(itertools.combinations(range(c_in), 2))
                    selected_pairs = possible_pairs[:d_eff]

        # Final checks
        assert len(selected_pairs) == d_eff, f"Selected {len(selected_pairs)} pairs, expected {d_eff}."
        self.pairs = selected_pairs
        self.register_buffer("pairs_idx", torch.tensor(self.pairs, dtype=torch.long))  # [d_eff, 2]

        # Two learnable weights per pair
        self.weights = nn.Parameter(torch.randn(d_eff, 2))  # [d_eff, 2]

        # Fixed averaging kernel for depthwise conv
        avg_kernel = torch.ones(c_in, 1, kernel_size) / kernel_size      # [C,1,K]
        self.register_buffer('avg_kernel', avg_kernel)

        msg_src = "precomputed" if used_precomputed else "combinations"
        print(f"Spearman-Correlation Pairs selection: [Top - {self.d_model} | total_pairs={total_pairs}")

    @torch.no_grad()
    def get_weight_contribution_matrix(self) -> torch.Tensor:
        d_model = len(self.pairs)
        contrib = torch.zeros(d_model, self.c_in, device=self.weights.device, dtype=self.weights.dtype)
        contrib.scatter_(1, self.pairs_idx[:, 0:1], self.weights[:, 0:1])
        contrib.scatter_(1, self.pairs_idx[:, 1:2], self.weights[:, 1:2])
        return contrib

    @torch.no_grad()
    def save_weight_contribution_matrix(self, save_path: str = "filter_weight_matrix.npy") -> None:
        matrix = self.get_weight_contribution_matrix()
        np.save(save_path, matrix.cpu().numpy())
        print(f"Saved filter weight matrix to {save_path}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C] -> [B,C,T]
        x = x.permute(0, 2, 1)
        B, C, T = x.shape
        assert C == self.c_in, f"Input channel count mismatch: got {C}, expected {self.c_in}."

        # Smooth all channels once (depthwise conv with circular padding)
        x_pad = F.pad(x, (self.padding, self.padding), mode='circular')
        kernel = self.avg_kernel.to(dtype=x.dtype)
        x_smooth = F.conv1d(x_pad, kernel, groups=C)   # [B,C,T]

        # Gather the two channels per pair
        ch1 = self.pairs_idx[:, 0]                     # [d_eff]
        ch2 = self.pairs_idx[:, 1]                     # [d_eff]
        x1 = x_smooth[:, ch1, :]                       # [B,d_eff,T]
        x2 = x_smooth[:, ch2, :]                       # [B,d_eff,T]

        # Apply learnable scalars
        w1 = self.weights[:, 0].view(1, -1, 1).to(dtype=x.dtype)
        w2 = self.weights[:, 1].view(1, -1, 1).to(dtype=x.dtype)
        out = w1 * x1 + w2 * x2                        # [B,d_eff,T]

        return out.permute(0, 2, 1)                    # [B,T,d_eff]


## DataEmeding that works with the top-512
class DataEmbedding(nn.Module):
    def __init__(self,
                 c_in: int,
                 d_model: int,
                 dropout: float = 0.0,
                 *,
                 dataset: str = None,
                 precomputed_dir: str = "./data_factory/Preprocess",
                 top_k_limit: int = 512,
                 kernel_size: int = 3):
        super().__init__()
        self.value_embedding = TokenEmbedding(
            c_in=c_in,
            d_model=d_model,                
            kernel_size=kernel_size,
            top_k_limit=top_k_limit,
            dataset=dataset,                
            precomputed_dir=precomputed_dir
        )
        d_eff = self.value_embedding.d_model
        self.position_embedding = PositionalEmbedding(d_model=d_eff)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_eff
    def forward(self, x):
        ve = self.value_embedding(x)
        pe = self.position_embedding(ve)
        return self.dropout(ve + pe)
    

#region: standartEmb

# class TokenEmbedding(nn.Module):
#     def __init__(self, c_in, d_model):
#         super(TokenEmbedding, self).__init__()
#         padding = 1 if torch.__version__ >= '1.5.0' else 2
#         self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
#                                    kernel_size=3, padding=padding, padding_mode='circular', bias=False)
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
#     def save_weight_contribution_matrix(self, save_path):
#         # Save the convolution weights
#         W = self.tokenConv.weight.detach().cpu().numpy()
#         np.save(save_path, W)

#     def forward(self, x):
#         x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
#         return x

#endregion



