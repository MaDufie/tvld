"""TVL-D model: Shape + Level + Structure, POT/GPD-calibrated,
weighted fusion + smoothing. One fixed statistical pipeline.

-------------------------------------------------------------------------
THE MODEL, IN ONE PARAGRAPH
-------------------------------------------------------------------------
Three independent severity signals are computed per variable, per timestep:
  Shape     -- AB-join matrix-profile discord (stumpy.stump, z-normalized):
               "does this recent pattern look like anything seen in the normal 
               reference data?"
  Level     -- |x - train_mean| / train_std: "is the raw magnitude unusual?"
  Structure -- |x - PCA_reconstruction(x)|: "does this break the learned
               cross-channel correlation structure?"
Each is independently calibrated onto a common -log(tail_probability) scale via
POT/GPD extreme-value tail fitting (Siffer et al. 2017), placing the three
signals on a comparable transformed tail-probability scale. Their weighted
additive fusion is related to Fisher-style evidence aggregation, but is not
treated here as an exact Fisher test. Cells that are unremarkable in all three
signals contribute approximately 0, while extreme observations receive larger
severity values.

The three signals are combined with fixed weights. The same weights and
smoothing window are used unchanged across all evaluation datasets. TVL-D is
training-free in the sense used in the paper: it requires no gradient-based
model optimization, while statistical reference quantities are estimated from
the provided normal training split.

"""
import ast
import time
from pathlib import Path

import numpy as np
import pandas as pd
import stumpy
from scipy import stats
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="stumpy")

TAIL_FRAC = 0.05
N_COMPONENTS = 10

# ----------------------------------------------------------------------
# THE HYPERPARAMETERS
# ----------------------------------------------------------------------
W_SHAPE = 0.15
W_LEVEL = 0.75
W_STRUCT = 1.0
SMOOTH_WINDOW = 35


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- calibration --
def pot_severity(x, tail_frac=TAIL_FRAC):
    """Raw 1-D score array -> POT/GPD tail severity. 0 = not in the upper tail."""
    thresh = np.quantile(x, 1 - tail_frac)
    exceed = x[x > thresh] - thresh
    if len(exceed) >= 20 and exceed.std() > 1e-9:
        try:
            shape, loc, scale = stats.genpareto.fit(exceed, floc=0)
            tail_p = np.where(
                x > thresh, tail_frac * stats.genpareto.sf(x - thresh, shape, loc=0, scale=scale), 1.0)
        except Exception:
            tail_p = 1 - stats.rankdata(x) / len(x)
    else:
        tail_p = 1 - stats.rankdata(x) / len(x)
    tail_p = np.clip(tail_p, 1e-8, 1.0)
    return -np.log(tail_p)


def calibrate_map(raw_map, tail_frac=TAIL_FRAC):
    severity = np.zeros_like(raw_map, dtype=float)
    for v in range(raw_map.shape[1]):
        severity[:, v] = pot_severity(raw_map[:, v], tail_frac=tail_frac)
    return severity


def window_scores_to_point_scores(window_scores, m, n_points):
    n_windows = len(window_scores)
    padded = np.full(n_points, np.nan)
    padded[:n_windows] = window_scores
    point = maximum_filter1d(np.nan_to_num(
        padded, nan=-np.inf), size=m, origin=-(m // 2))
    valid = ~np.isnan(padded)
    last_valid_idx = np.maximum.accumulate(
        np.where(valid, np.arange(n_points), 0))
    point = np.where(point == -np.inf, padded[last_valid_idx], point)
    return point


def pick_window(train_len, test_len, default=100):
    """SMD uses the fixed default (100). Shorter series (SMAP/MSL) shrink m so
    both train and test can still form enough windows."""
    return min(default, max(8, train_len // 4), max(8, test_len // 4))


# ---------------------------------------------------------------- three signals --
def shape_severity(train, test, m, tail_frac=TAIL_FRAC):
    """Per-variable AB-join matrix-profile discord (z-normalized): 'does this
    recent window look like anything seen in the normal reference data, for THIS variable?'
    Constant columns (common one-hot command flags) are skipped - no meaningful
    shape signal, left at severity 0."""
    n_points, D = test.shape
    severity = np.zeros((n_points, D))
    for v in range(D):
        T_A = np.ascontiguousarray(train[:, v])
        T_B = np.ascontiguousarray(test[:, v])
        if np.std(T_A) < 1e-9 and np.std(T_B) < 1e-9:
            continue
        profile = stumpy.stump(T_B, m, T_A, ignore_trivial=False, normalize=True)[
            :, 0].astype(float)
        point_scores = window_scores_to_point_scores(profile, m, n_points)
        severity[:, v] = pot_severity(point_scores, tail_frac=tail_frac)
    return severity


def level_severity(train, test, tail_frac=TAIL_FRAC):
    """Per-variable magnitude deviation: |x - train_mean| / train_std, POT-calibrated.
    Catches level-shift anomalies that z-normalized Shape is blind to by design."""
    mu, sd = train.mean(axis=0), train.std(axis=0) + 1e-9
    dev = np.abs((test - mu) / sd)
    return calibrate_map(dev, tail_frac=tail_frac)


def pca_map(train, test, n_components=N_COMPONENTS):
    n_components = min(n_components, train.shape[1] - 1, train.shape[0] - 1)
    n_components = max(n_components, 1)
    pca = PCA(n_components=n_components).fit(train)
    recon = pca.inverse_transform(pca.transform(test))
    return np.abs(test - recon)


def structure_severity(train, test, n_components=N_COMPONENTS, tail_frac=TAIL_FRAC):
    """PCA-reconstruction residual, POT-calibrated: does this break the learned
    cross-channel correlation structure? Fit on the FULL feature vector (so
    context columns still condition it) even when only a subset gets scored."""
    raw = pca_map(train, test, n_components=n_components)
    return calibrate_map(raw, tail_frac=tail_frac)


def zscore_map(train, test):
    mu = train.mean(axis=0)
    sd = train.std(axis=0) + 1e-9
    return np.abs((test - mu) / sd)


# ---------------------------------------------------------------- fusion --
def point_score_from_map(the_map, primary_cols=None):
    """Sum calibrated severity across the scored columns.
    primary_cols=None -> score every column (SMD/PSM).
    primary_cols=[0] -> score only the telemetry column for SMAP/MSL.
    """
    cols = the_map if primary_cols is None else the_map[:, primary_cols]
    return cols.sum(axis=1)


def fuse(shape_sev, level_sev, struct_sev, primary_cols=None,
         w_shape=W_SHAPE, w_level=W_LEVEL, w_struct=W_STRUCT, smooth_window=SMOOTH_WINDOW):
    """The final model's score: weighted sum of the three calibrated severities
    (scored on primary_cols only), then moving-average smoothed."""
    weighted_map = w_shape * shape_sev + w_level * level_sev + w_struct * struct_sev
    v2 = point_score_from_map(weighted_map, primary_cols=primary_cols)
    v3 = uniform_filter1d(v2, size=smooth_window, mode="nearest")
    return v2, v3


# ---------------------------------------------------------------- metrics --

def get_ranges(a):
    ranges, in_r, start = [], False, 0
    for i, v in enumerate(a):
        if v == 1 and not in_r:
            start, in_r = i, True
        elif v == 0 and in_r:
            ranges.append((start, i))
            in_r = False
    if in_r:
        ranges.append((start, len(a)))
    return ranges


def _ov(a_s, a_e, b_s, b_e):
    return max(0, min(a_e, b_e) - max(a_s, b_s))


def _overlaps_for(ra, rb):
    j, out = 0, []
    for a_s, a_e in ra:
        while j < len(rb) and rb[j][1] <= a_s:
            j += 1
        k, hits = j, []
        while k < len(rb) and rb[k][0] < a_e:
            if _ov(a_s, a_e, rb[k][0], rb[k][1]) > 0:
                hits.append(rb[k])
            k += 1
        out.append(hits)
    return out


def range_recall(tr, pr, alpha=0.0):
    if not tr:
        return 1.0
    tot = 0.0
    for (rs, re), ov in zip(tr, _overlaps_for(tr, pr)):
        ex = 1.0 if ov else 0.0
        card = 1.0 if len(ov) <= 1 else 1.0 / len(ov)
        frac = sum(_ov(rs, re, a, b) for a, b in ov) / (re - rs)
        tot += alpha * ex + (1 - alpha) * card * frac
    return tot / len(tr)


def range_precision(tr, pr):
    if not pr:
        return 0.0 if tr else 1.0
    tot = 0.0
    for (ps, pe), ov in zip(pr, _overlaps_for(pr, tr)):
        card = 1.0 if len(ov) <= 1 else 1.0 / len(ov)
        frac = sum(_ov(rs, re, ps, pe) for rs, re in ov) / (pe - ps)
        tot += card * frac
    return tot / len(pr)


def range_pr_auc(scores, y, n_thresh=80):
    tr = get_ranges(y)
    lo, hi = np.percentile(scores, 1), np.percentile(scores, 99.9)
    pts = [(0.0, 1.0)]
    for t in np.linspace(lo, hi, n_thresh):
        pr = get_ranges((scores > t).astype(int))
        pts.append((range_recall(tr, pr), range_precision(tr, pr)))
    pts.sort(key=lambda p: p[0])
    r = np.array([p[0] for p in pts])
    pv = np.array([p[1] for p in pts])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(pv, r))


def best_threshold_f1(scores, y, n_thresh=150):
    lo, hi = np.percentile(scores, 30), np.percentile(scores, 99.99)
    best = (0.0, 0.0, 0.0)
    for t in np.linspace(lo, hi, n_thresh):
        preds = (scores > t).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best[0]:
            best = (f1, precision_score(y, preds, zero_division=0),
                    recall_score(y, preds, zero_division=0))
    return best


def full_metrics(scores, y):
    if y.sum() == 0 or y.sum() == len(y):
        return None
    out = {"pr_auc": average_precision_score(y, scores), "auc_roc": roc_auc_score(y, scores),
           "auc_pr_t": range_pr_auc(scores, y)}
    f1, p, r = best_threshold_f1(scores, y)
    out.update({"f1": f1, "precision": p, "recall": r})
    return out


# ---------------------------------------------------------------- SMD loader --
def load_smd_machine(mid, data_dir):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train" /
                        f"{mid}.txt", header=None).to_numpy(dtype=float)
    test = pd.read_csv(data_dir / "test" /
                       f"{mid}.txt", header=None).to_numpy(dtype=float)
    label = pd.read_csv(data_dir / "test_label" /
                        f"{mid}.txt", header=None).to_numpy(dtype=int).ravel()
    return train, test, label


# ---------------------------------------------------------------- SMAP/MSL loader --


def load_smap_msl_meta(csv_path):
    df = pd.read_csv(csv_path)
    meta = {}
    for _, row in df.iterrows():
        cid = row["chan_id"]
        windows = ast.literal_eval(row["anomaly_sequences"])
        if cid not in meta:
            meta[cid] = {"spacecraft": row["spacecraft"], "windows": list(
                windows), "num_values": int(row["num_values"])}
        else:
            meta[cid]["windows"].extend(windows)
    return meta


def load_smap_msl_channel(cid, meta, data_dir):
    data_dir = Path(data_dir)
    train = np.load(data_dir / "train" / f"{cid}.npy").astype(float)
    test = np.load(data_dir / "test" / f"{cid}.npy").astype(float)
    label = np.zeros(test.shape[0], dtype=int)
    for s, e in meta[cid]["windows"]:
        label[s:e] = 1
    return train, test, label
