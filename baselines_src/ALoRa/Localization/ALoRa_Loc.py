import argparse
import os
import torch
import numpy as np
from Projects.AnomalyDetection.tvld.baselines_src.ALoRa.Localization.Loc_Metrics import compute_hit_rate, extract_anomalous_ranges_and_features, compute_ips_range_level, ndcg
import torch
import numpy as np
import os
import pandas as pd
import math

# === Parse arguments ===
parser = argparse.ArgumentParser(description="Localization matrix computation")
parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., HAI, SMD, SWAT)')
args = parser.parse_args()

# === Set d_model per dataset ===
d_model_dict = {
    "SMD"  :512,
    "MSDS" :44,
    "SWAT" :512, #1274

}

if args.dataset not in d_model_dict:
    raise ValueError(f"Unknown dataset: {args.dataset}. Add it to d_model_dict.")

d_model = d_model_dict[args.dataset]

# === Build paths ===
localization_path = "./Localization/PATHS"  # Base folder
dataset_dir = os.path.join(localization_path, args.dataset)

embedding_path = os.path.join(dataset_dir, f"filter_weight_matrix_{args.dataset}.npy")
b_save_dir = dataset_dir
checkpoint_path = os.path.join(
    "./Localization/PATHS", args.dataset, f"{args.dataset}_checkpoint.pth"
)

print(checkpoint_path)


import torch
import numpy as np
import os
import os


# === STEP 1: Construct B with residual connections per layer ===
def compute_B_with_residuals(checkpoint_path, save_dir, d_model):
    os.makedirs(save_dir, exist_ok=True)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    layer_count = 0
    B = torch.eye(d_model)

    while True:
        wv_key = f"encoder.attn_layers.{layer_count}.attention.value_projection.weight"
        out_key = f"encoder.attn_layers.{layer_count}.attention.out_projection.weight"

        if wv_key not in state_dict:
            break

        Wv = state_dict[wv_key].T    # [dmodel, dk], dk=dmodel/H
        Wout = state_dict[out_key].T # [dk, dmodel]

        Wvl = torch.matmul(Wv, Wout)  # [dmodel, dmodel]
        residual = torch.eye(d_model) + Wvl  # Residual block: I + Wvl

        B = torch.matmul(B, residual)
        layer_count += 1

    save_path = os.path.join(save_dir, "B_matrix.pt")
    torch.save(B, save_path)
    print(f"Final B with residuals shape: {B.shape}")
    # print(f"Saved to: {save_path}")
    return B


# === STEP 2: Compute E using filter weights and B ===
def compute_E_with_embedding(embedding_path, B_path):
    W = np.load(embedding_path)  # shape: [d_model, D, k] or [d_model, D]
    B = torch.load(B_path).numpy()  # shape: [d_model, d_model]

    if W.ndim == 3:
        print(f"📐 W shape: {W.shape} (conv filters with kernel)")
        W_sum = W.sum(axis=-1)
    elif W.ndim == 2:
        print(f"📐 W shape: {W.shape} (linear projection)")
        W_sum = W
    else:
        raise ValueError("Unexpected embedding shape.")

    E = np.matmul(W_sum.T, B)
    print(f"E shape: {E.shape}")
    return E


# === STEP 3: Compute C from E and final projection ===
def compute_C_from_E(E, checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    W_proj = state_dict["projection.weight"].T.numpy()  # [d_model, D]

    C = np.matmul(E, W_proj)  # [D, D]
    print(f"C matrix shape: {C.shape}")
    return C


# === STEP 4: Save and optionally inspect ===
def save_C(C, path="checkpoints/SWATMinute"):
    os.makedirs(path,exist_ok=True)
    full_path = os.path.join(path,"C_matrix.npy")
    np.save(full_path, C)
    print(f"💾 Saved C matrix to: {path}")



# === File paths (based on dataset_dir) ===
rec_loc_path = os.path.join(dataset_dir, "RecLoc_matrix.npy")
c_path = os.path.join(dataset_dir, "C_matrix.npy")
label_path = os.path.join(dataset_dir, "output.csv")  # Localization Labels

# === Load data ===
RecLoc = np.load(rec_loc_path)
C = np.load(c_path)
ground_truth = pd.read_csv(label_path, index_col=0).to_numpy()


# === Matrix processing ===
C = np.abs(C)
# ALoRa-Loc (top-K)
k = 2
rows = np.arange(C.shape[0])[:, None]
topk_indices = np.argpartition(C, -k, axis=1)[:, -k:]
mask = np.zeros_like(C, dtype=bool)
mask[rows, topk_indices] = True
C[~mask] = 0

#region: Normalize:
C_plus = C + np.eye(C.shape[0])

RecLoc = np.abs(RecLoc)
C_normalized = C / np.sum(C, axis=1, keepdims=True)
C_plus_normalized = C_plus / np.sum(C_plus, axis=1, keepdims=True)
#endregion

# === LAS SCORE ===
AS = np.matmul(RecLoc, C_normalized)

np.save(os.path.join(dataset_dir, "AS_matrix_normalized.npy"), AS)

# === Evaluate method 1: ours ===
print("\n=== Evaluation: ALoRa-Loc ===")
print(f"📐 RecLoc shape: {RecLoc.shape}")
print(f"📐 C shape: {C.shape}")
hit_rate = compute_hit_rate(AS, ground_truth)
print("🎯 Hit Rate @100%:", hit_rate[100])
print("🎯 Hit Rate @150%:", hit_rate[150])

ranges, gt_features = extract_anomalous_ranges_and_features(ground_truth)
ips = compute_ips_range_level(AS, ranges, gt_features)
print("🧠 IPS @100%:", ips[100])
print("🧠 IPS @150%:", ips[150])

ndcg_scores = ndcg(AS, ground_truth)
print("📊 NDCG @100%:", ndcg_scores.get("NDCG@100%", "N/A"))
print("📊 NDCG @150%:", ndcg_scores.get("NDCG@150%", "N/A"))

# === Evaluate method 2: baseline (RecLoc only) ===
print("\n=== Evaluation: ALoRa-Loc WITHOUT the Contibution effect weights ===")
hit_rate = compute_hit_rate(RecLoc, ground_truth)
print("🎯 Hit Rate @100%:", hit_rate[100])
print("🎯 Hit Rate @150%:", hit_rate[150])

