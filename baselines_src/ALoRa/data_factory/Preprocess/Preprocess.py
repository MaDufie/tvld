import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

OUTDIR = Path("./data_factory/Preprocess")

def compute_spearman_matrix_from_array(X: np.ndarray) -> np.ndarray:
    """Compute Spearman correlation matrix (C,C) from array (N,C)."""
    Xr = np.apply_along_axis(rankdata, 0, X)  # rank transform each column
    Xr = (Xr - Xr.mean(axis=0)) / Xr.std(axis=0, ddof=1)  # z-score ranks
    R = (Xr.T @ Xr) / (Xr.shape[0] - 1)  # Pearson on ranks
    return R

def top_k_pairs_by_abs_rho(R: np.ndarray, K: int):
    C = R.shape[0]
    iu, ju = np.triu_indices(C, k=1)
    rho = R[iu, ju]
    order = np.argsort(-np.abs(rho))  # descending by |rho|
    iu_top = iu[order][:K]
    ju_top = ju[order][:K]
    rho_top = rho[order][:K]
    pairs_idx = np.stack([iu_top, ju_top], axis=1).astype(np.int32)
    return pairs_idx, rho_top

def main():
    parser = argparse.ArgumentParser(
        description="Precompute Spearman correlation matrix and top-K pairs."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., SMD, MSL, PSM, HAI, SWAT)")
    parser.add_argument("--file", required=True, help="Full path to training file (CSV or NPY)")
    parser.add_argument("--top_k", type=int, default=512, help="Top-K pairs to keep")
    parser.add_argument("--dropna", action="store_true", help="Drop rows with NaNs if CSV")
    parser.add_argument("--drop_first_col", action="store_true",
                        help="Drop first column (e.g., time index) before computing Spearman")
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    dataset = args.dataset.strip()
    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist")

    # --- load training data ---
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
        if args.drop_first_col:
            df = df.iloc[:, 1:]  # drop first column (time index)
        df_num = df.select_dtypes(include=[np.number]).copy()
        if args.dropna:
            df_num = df_num.dropna(axis=0)
        else:
            df_num = df_num.fillna(df_num.median(numeric_only=True))
        X = df_num.to_numpy(dtype=np.float32)
    elif file_path.suffix == ".npy":
        X = np.load(file_path)
        if X.ndim == 3:  # (N,T,C)
            N, T, C = X.shape
            X = X.reshape(N*T, C)
        elif X.ndim != 2:
            raise ValueError(f"Unexpected shape {X.shape} in {file_path}")
        X = X.astype(np.float32)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    N, C = X.shape
    print(f"[{dataset}] Loaded {file_path} -> N={N}, C={C}")

    # --- Spearman matrix ---
    R = compute_spearman_matrix_from_array(X)
    spearman_path = OUTDIR / f"spearman_corr_{dataset}.npy"
    np.save(spearman_path, R)
    print(f"[{dataset}] Saved Spearman matrix -> {spearman_path}")

    # --- Top-K pairs ---
    K = min(args.top_k, C * (C - 1) // 2)
    pairs_idx, rho_top = top_k_pairs_by_abs_rho(R, K)
    pairs_path = OUTDIR / f"pairs_idx_{dataset}.npy"
    rho_path = OUTDIR / f"rho_top_{dataset}.npy"
    np.save(pairs_path, pairs_idx)
    np.save(rho_path, rho_top)
    print(f"[{dataset}] Saved top-{K} pairs -> {pairs_path}")
    print(f"[{dataset}] Saved rho values -> {rho_path}")

if __name__ == "__main__":
    main()
