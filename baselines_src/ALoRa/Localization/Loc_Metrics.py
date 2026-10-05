
import numpy as np
import numpy as np
import pandas as pd
import math
from sklearn.metrics import ndcg_score

def compute_hit_rate(anomaly_scores, ground_truth, p_values=[100, 150]):
    T, D = anomaly_scores.shape
    hit_rates = {p: [] for p in p_values}

    for t in range(T):
        gt = ground_truth[t]
        if np.sum(gt) == 0:
            continue

        G_t = set(np.where(gt == 1)[0])
        k_base = len(G_t)

        scores = anomaly_scores[t]
        ranked_indices = np.argsort(scores)[::-1]

        for p in p_values:
            k = math.ceil(k_base * p / 100.0)
            top_k = set(ranked_indices[:k])
            overlap = G_t.intersection(top_k)
            hr = len(overlap) / k_base
            hit_rates[p].append(hr)

    avg_hit_rates = {p: np.mean(hit_rates[p]) if hit_rates[p] else 0.0 for p in p_values}
    return avg_hit_rates

def extract_anomalous_ranges_and_features(ground_truth):
    T, D = ground_truth.shape
    binary_seq = (np.sum(ground_truth, axis=1) > 0).astype(int)

    ranges = []
    gt_features = []
    in_range = False
    start = 0

    for t in range(T):
        if binary_seq[t] == 1 and not in_range:
            in_range = True
            start = t
        elif binary_seq[t] == 0 and in_range:
            end = t - 1
            ranges.append((start, end))
            G_i = set(np.where(np.any(ground_truth[start:end+1] == 1, axis=0))[0])
            gt_features.append(G_i)
            in_range = False

    if in_range:
        end = T - 1
        ranges.append((start, end))
        G_i = set(np.where(np.any(ground_truth[start:end+1] == 1, axis=0))[0])
        gt_features.append(G_i)

    return ranges, gt_features

def compute_ips_range_level(anomaly_scores, ranges, gt_features, p_values=[100, 150]):
    ips_results = {p: [] for p in p_values}

    for (start, end), G_i in zip(ranges, gt_features):
        if not G_i:
            continue

        scores_in_range = anomaly_scores[start:end+1]
        max_scores = np.max(scores_in_range, axis=0)
        # max_scores = np.mean(scores_in_range, axis=0)
        ranked_features = np.argsort(max_scores)[::-1]

        for p in p_values:
            k = math.ceil(len(G_i) * p / 100.0)
            top_k = set(ranked_features[:k])
            overlap = top_k.intersection(G_i)
            ips = len(overlap) / len(G_i)
            ips_results[p].append(ips)

    avg_ips = {p: np.mean(ips_results[p]) if ips_results[p] else 0.0 for p in p_values}
    return avg_ips


# === NDCG computation ===
def ndcg(ascore, labels, ps=[100, 150]):
    res = {}
    for p in ps:
        ndcg_scores = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            labs = list(np.where(l == 1)[0])
            if labs:
                k_p = round(p * len(labs) / 100)
                try:
                    hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k=k_p)
                except Exception:
                    return {}
                ndcg_scores.append(hit)
        res[f'NDCG@{p}%'] = np.mean(ndcg_scores)
    return res
