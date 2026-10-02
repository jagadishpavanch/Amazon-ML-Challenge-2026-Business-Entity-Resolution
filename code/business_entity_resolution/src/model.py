"""LightGBM pair classifier: grouped OOF training, isotonic calibration, final fit."""
import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from config import LGB_EARLY_STOP, LGB_MAX_ROUNDS, LGB_PARAMS, N_FOLDS

NON_FEATURES = {"s1", "p", "label", "prob", "oof", "fold", "r_f", "cos_f"}   # r_f/cos_f: stage-2 only


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def _params(seed):
    return LGB_PARAMS if seed is None else {**LGB_PARAMS, "seed": seed}


def train_oof(df, feats, log=print, weight=None, seed=None):
    """5-fold GroupKFold by S1; returns (oof raw probs, list of best iterations).
    weight: optional per-row training weights; seed: optional LightGBM seed override."""
    X = df[feats].values.astype(np.float32)
    y = df.label.values
    oof = np.zeros(len(df), dtype=np.float32)
    iters = []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, groups=df.s1.values)):
        dtr = lgb.Dataset(X[tr], y[tr], weight=None if weight is None else weight[tr], feature_name=feats,
                          free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], weight=None if weight is None else weight[va], reference=dtr)
        m = lgb.train(_params(seed), dtr, LGB_MAX_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(LGB_EARLY_STOP, verbose=False)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        log(f"  fold {k}: best_iter={m.best_iteration} auc={roc_auc_score(y[va], oof[va]):.5f}")
    log(f"  OOF AUC={roc_auc_score(y, oof):.5f} PR-AUC={average_precision_score(y, oof):.5f}")
    return oof, iters


def fit_final(df, feats, n_rounds, weight=None, seed=None):
    X = df[feats].values.astype(np.float32)
    return lgb.train(_params(seed), lgb.Dataset(X, df.label.values, weight=weight, feature_name=feats), n_rounds)


def fit_calibrator(raw, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(raw, y)
    return iso


def feature_importance(model, feats, top=30):
    imp = model.feature_importance("gain")
    order = np.argsort(-imp)[:top]
    return [(feats[i], float(imp[i] / imp.sum())) for i in order]
