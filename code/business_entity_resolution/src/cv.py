"""Decision-knob tuning on OOF probabilities and the leave-country-out check."""
import lightgbm as lgb
import numpy as np

from config import LGB_PARAMS, SEED
from decide import one_owner, select_expected_f, select_threshold, to_pred_dict
from metrics import macro_f05_breakdown
from model import fit_calibrator


def _score(sel, s1n, pooln, truth, s_ids):
    pred = to_pred_dict(sel, s1n.entity_id.values, pooln.entity_id.values)
    return macro_f05_breakdown(pred, truth, s_ids)


def apply_decision(df, prob_col, knobs):
    pr = one_owner(df, prob_col, knobs["alpha"])
    if knobs["method"] == "expected_f":
        return select_expected_f(df, pr, knobs["p_min"], knobs["temp"])
    return select_threshold(df, pr, knobs["thr"])


def tune_decision(tr, s1n, pooln, truth, s_ids, n_tune=200_000, log=print):
    """Grid-search the decision layer on OOF probs; returns (best knobs, results table)."""
    rng = np.random.default_rng(SEED)
    sub_ids = s_ids if len(s_ids) <= n_tune else rng.choice(s_ids, n_tune, replace=False)
    pos = {s: i for i, s in enumerate(s1n.entity_id.values)}
    keep = np.zeros(len(s1n), bool)
    keep[[pos[s] for s in sub_ids]] = True
    df = tr[keep[tr.s1.values]][["s1", "p", "prob"]].reset_index(drop=True)
    table = []

    def run(knobs):
        r = _score(apply_decision(df, "prob", knobs), s1n, pooln, truth, sub_ids)
        table.append({**knobs, **r})
        log(f"    {knobs} -> {r['overall']:.5f} (sing {r['singleton']:.4f} / non {r['non_singleton']:.4f})")
        return r["overall"]

    for alpha in (0.0, 0.3, 1.0):
        for thr in np.round(np.arange(0.3, 0.91, 0.05), 2):
            run(dict(method="threshold", thr=float(thr), alpha=alpha))
    best_alpha = max((t for t in table), key=lambda t: t["overall"])["alpha"]
    for temp in (0.8, 1.0, 1.2, 1.5):
        for p_min in (0.05, 0.1, 0.2, 0.3):
            run(dict(method="expected_f", temp=temp, p_min=p_min, alpha=best_alpha))
    best = max(table, key=lambda t: t["overall"])
    knobs = {k: best[k] for k in ("method", "alpha", "thr", "temp", "p_min") if k in best}
    # confirm on the whole training sample
    full = _score(apply_decision(tr, "prob", knobs), s1n, pooln, truth, s_ids)
    log(f"  OOF macro F0.5 on full sample with best knobs: {full}")
    knobs["oof_f05"] = full
    return knobs, table


def leave_country_out(tr, feats, s1n, pooln, truth, knobs, n_rounds=400, log=print):
    """Train on all-but-one country, evaluate on the held-out country; compare to in-country OOF."""
    country = s1n.country.values[tr.s1.values]
    out = {}
    for c in np.unique(country):
        te_m = country == c
        tr_m = ~te_m
        if tr_m.sum() == 0:
            continue
        rng = np.random.default_rng(SEED)
        s1_tr = np.unique(tr.s1.values[tr_m])
        cal_s1 = set(rng.choice(s1_tr, len(s1_tr) // 5, replace=False))
        cal_m = tr_m & np.isin(tr.s1.values, list(cal_s1))
        fit_m = tr_m & ~cal_m
        X = tr[feats].values.astype(np.float32)
        m = lgb.train(LGB_PARAMS, lgb.Dataset(X[fit_m], tr.label.values[fit_m], feature_name=feats), n_rounds)
        iso = fit_calibrator(m.predict(X[cal_m]), tr.label.values[cal_m])
        held = tr[te_m][["s1", "p"]].reset_index(drop=True)
        held["prob_lco"] = iso.predict(m.predict(X[te_m]))
        held["prob_in"] = tr.prob.values[te_m]
        s_ids = np.unique(s1n.entity_id.values[held.s1.values])
        # entities of this country in the sample that have no candidates also count
        r_lco = _score(apply_decision(held, "prob_lco", knobs), s1n, pooln, truth, s_ids)
        r_in = _score(apply_decision(held, "prob_in", knobs), s1n, pooln, truth, s_ids)
        out[str(c)] = dict(held_out=r_lco, in_country_oof=r_in, gap=r_in["overall"] - r_lco["overall"])
        log(f"  LCO held-out={c}: F0.5={r_lco['overall']:.5f} vs in-country OOF {r_in['overall']:.5f}")
    return out
