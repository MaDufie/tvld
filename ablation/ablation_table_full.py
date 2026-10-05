"""Leave-one-component-out ablation for TVL-D's three severity signals plus
the smoothing step: starting from the full model
(W_SHAPE=0.15, W_LEVEL=0.75, W_STRUCT=1.0, smoothing window 35), zero out one
weight (or disable smoothing) at a time and re-measure all six  metrics,
macro-averaged per dataset then across the three datasets (SMD, SMAP/MSL,
PSM). This answers "how much does each component matter," holding everything
else about the model fixed.

This is a leave-one-COMPONENT-out ablation (Shape vs. Level vs. Structure vs.
Smoothing).

Needs series_cache.pkl (run build_series_cache.py first, next to this
script, if it isn't already there) - the per-timestep Shape/Level/Structure
severity SCALARS (already column-aggregated per dataset's own convention)
plus labels for every SMD machine, SMAP/MSL channel, and PSM, so every
variant below is a numpy re-weighting + re-threshold.

Usage:
    python build_series_cache.py   # once, or reuses main_benchmark/'s checkpoints if present
    python ablation_table_full.py
Writes: ablation_results.json, and prints the summary table.
"""
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
CACHE_PATH = HERE / "series_cache.pkl"
OUT_JSON = HERE / "ablation_results.json"

if not CACHE_PATH.exists():
    raise SystemExit(f"{CACHE_PATH} not found - run build_series_cache.py first (next to this script).")

with open(CACHE_PATH, "rb") as f:
    series = pickle.load(f)

DATASETS = ["SMD", "SMAP/MSL", "PSM"]
by_dataset = {d: [s for s in series if s["dataset"] == d] for d in DATASETS}
for d in DATASETS:
    if not by_dataset[d]:
        print(f"WARNING: no series loaded for {d} -- its macro-average below will be meaningless "
              f"(empty-mean). Check build_series_cache.py found that dataset's data.")


# ---- range-based PR-AUC, same semantics as tvld_final.py's / metrics.py's ----
def get_ranges(a):
    a = np.asarray(a)
    d = np.diff(np.concatenate(([0], a.astype(int), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


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
    r = np.array([p[0] for p in pts]); pv = np.array([p[1] for p in pts])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(pv, r))


def best_threshold_f1(scores, y, n_thresh=150):
    lo, hi = np.percentile(scores, 30), np.percentile(scores, 99.99)
    thresholds = np.linspace(lo, hi, n_thresh)
    y_bool = y.astype(bool)
    n_pos = y_bool.sum()
    best = (0.0, 0.0, 0.0)
    for t in thresholds:
        pred = scores > t
        tp = np.count_nonzero(pred & y_bool)
        fp = np.count_nonzero(pred) - tp
        fn = n_pos - tp
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best[0]:
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            best = (f1, p, r)
    return best


def score_variant(w_shape, w_level, w_struct, smooth_window, subset):
    out = {k: [] for k in ["pr_auc", "auc_roc", "auc_pr_t", "f1", "precision", "recall"]}
    for s in subset:
        raw = w_shape * s["shape"] + w_level * s["level"] + w_struct * s["struct"]
        sm = uniform_filter1d(raw, size=smooth_window, mode="nearest") if smooth_window > 1 else raw
        y = s["label"]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        out["pr_auc"].append(average_precision_score(y, sm))
        out["auc_roc"].append(roc_auc_score(y, sm))
        out["auc_pr_t"].append(range_pr_auc(sm, y))
        f1, p, r = best_threshold_f1(sm, y)
        out["f1"].append(f1); out["precision"].append(p); out["recall"].append(r)
    return {k: float(np.mean(v)) if v else float("nan") for k, v in out.items()}


VARIANTS = [
    ("Full TVL-D",  0.15, 0.75, 1.0, 35),
    ("- Smoothing", 0.15, 0.75, 1.0, 1),
    ("- Shape",     0.0,  0.75, 1.0, 35),
    ("- Level",     0.15, 0.0,  1.0, 35),
    ("- Structure", 0.15, 0.75, 0.0, 35),
]

METRICS = ["pr_auc", "auc_roc", "auc_pr_t", "f1", "precision", "recall"]


def main():
    results = {}
    for name, ws, wl, wst, win in VARIANTS:
        per_ds = {d: score_variant(ws, wl, wst, win, by_dataset[d]) for d in DATASETS}
        macro = {m: float(np.nanmean([per_ds[d][m] for d in DATASETS])) for m in METRICS}
        results[name] = {"per_ds": per_ds, "macro": macro}
        print(f"done: {name}", flush=True)

    print(flush=True)
    header = f"{'Variant':<14} | " + " ".join(f"{m:>10}" for m in METRICS)
    print(header, flush=True)
    for name, _, _, _, _ in VARIANTS:
        macro = results[name]["macro"]
        print(f"{name:<14} | " + " ".join(f"{macro[m]:>10.4f}" for m in METRICS), flush=True)

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
