"""Metrics : pr_auc, auc_roc, auc_pr_t (range-based PR-AUC), and f1/precision/recall at the single best-F1 threshold.
"""
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def get_ranges(a):
    ranges, in_r, start = [], False, 0
    for i, v in enumerate(a):
        if v == 1 and not in_r:
            start, in_r = i, True
        elif v == 0 and in_r:
            ranges.append((start, i)); in_r = False
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
    r = np.array([p[0] for p in pts]); pv = np.array([p[1] for p in pts])
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(trapz(pv, r))


def best_threshold_f1(scores, y, n_thresh=150):
    lo, hi = np.percentile(scores, 30), np.percentile(scores, 99.99)
    best = (0.0, 0.0, 0.0)
    for t in np.linspace(lo, hi, n_thresh):
        preds = (scores > t).astype(int)
        f1 = f1_score(y, preds, zero_division=0)
        if f1 > best[0]:
            best = (f1, precision_score(y, preds, zero_division=0), recall_score(y, preds, zero_division=0))
    return best


def full_metrics(scores, y):
    """None if the label is degenerate (no anomaly, or all anomaly) -- AUCs are
    undefined in that case, matching tvld_final.py's own full_metrics()."""
    if y.sum() == 0 or y.sum() == len(y):
        return None
    out = {"pr_auc": average_precision_score(y, scores), "auc_roc": roc_auc_score(y, scores),
           "auc_pr_t": range_pr_auc(scores, y)}
    f1, p, r = best_threshold_f1(scores, y)
    out.update({"f1": f1, "precision": p, "recall": r})
    return out
