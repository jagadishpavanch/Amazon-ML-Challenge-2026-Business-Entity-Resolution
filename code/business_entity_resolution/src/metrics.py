"""Exact challenge metric (macro F0.5 per S1 entity) and blocking diagnostics."""
import numpy as np


def f05(pred: set, truth: set) -> float:
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_dict: dict, truth_dict: dict, s1_ids) -> float:
    return float(np.mean([f05(set(pred_dict.get(s, ())), truth_dict.get(s, set())) for s in s1_ids]))


def macro_f05_breakdown(pred_dict: dict, truth_dict: dict, s1_ids) -> dict:
    """Overall / singleton / non-singleton macro F0.5."""
    sing, non = [], []
    for s in s1_ids:
        t = truth_dict.get(s, set())
        (non if t else sing).append(f05(set(pred_dict.get(s, ())), t))
    allv = sing + non
    return dict(overall=float(np.mean(allv)),
                singleton=float(np.mean(sing)) if sing else float("nan"),
                non_singleton=float(np.mean(non)) if non else float("nan"),
                n_singleton=len(sing), n_non_singleton=len(non))


def blocking_recall(cands: dict, truth: dict) -> float:
    """Fraction of true (s1, id) pairs present in the candidate sets."""
    hit = tot = 0
    for s, t in truth.items():
        if t:
            c = cands.get(s, ())
            c = c if isinstance(c, set) else set(c)
            hit += len(t & c)
            tot += len(t)
    return hit / max(tot, 1)


def reduction_ratio(n_candidates: int, n_s1: int, n_pool: int) -> float:
    return 1.0 - n_candidates / float(n_s1 * n_pool)
