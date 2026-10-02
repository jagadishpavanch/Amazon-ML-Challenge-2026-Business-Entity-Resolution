"""Central configuration: paths, seeds, blocking K, model names."""
import os
import random

import numpy as np

SEED = 42

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(SRC_DIR)                      # code/business_entity_resolution
ROOT_DIR = os.path.dirname(os.path.dirname(PKG_DIR))    # student_resource
CACHE_DIR = os.environ.get("ER_CACHE_DIR", os.path.join(ROOT_DIR, "cache"))
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
MODELS_DIR = os.environ.get("ER_MODELS_DIR", os.path.join(os.path.dirname(ROOT_DIR), "models"))

# Pretrained encoders (both multilingual, both permissive licenses).
EMB_MODEL = os.environ.get(
    "ER_EMB_MODEL", os.path.join(MODELS_DIR, "paraphrase-multilingual-MiniLM-L12-v2"))
EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0)"
EMB_BATCH = 1024

# Blocking: top-K per source (S2, S3) for each nearest-neighbour pass (tuned by tune_blocking.py).
K_PASSES = {"w": 10, "e1": 3, "e2": 0, "c": 3, "a": 5}
# Sibling-based blocking (pass "s"): siblings = top SIB_TOP candidates per S1 with combo >= SIB_MIN.
SIB_MIN = 0.75
SIB_TOP = 3
# If a previous run's model is available, siblings are chosen by its stage-1 probability instead of the
# cheap combo score (probe: recall 0.9813 vs 0.9743). Set ER_SIB_PRIOR to that run's model.pkl.
SIB_PRIOR_MODEL = os.environ.get("ER_SIB_PRIOR", os.path.join(CACHE_DIR, "prior_model.pkl"))
SIB_MIN_MODEL = 0.5
SIB_TOP_MODEL = 5
_KFILE = os.path.join(CACHE_DIR, "blocking_k.json")
if os.path.exists(_KFILE):
    import json as _json
    with open(_KFILE) as _f:
        K_PASSES = _json.load(_f)["K"]
B5_CAP = 30

# Number of train S1 entities (with all their candidates) used for CV / model fitting.
TRAIN_S1_SAMPLE = int(os.environ.get("ER_TRAIN_SAMPLE", 600_000))

# Stage-2 stacking on neighbour stage-1 probabilities (see stage2.py).
USE_STAGE2 = True
# Stage-2 extras: sibling (cluster-support) features and the fine-tuned cross-encoder score.
USE_SIBLINGS = True
USE_TWINS = os.environ.get("ER_TWINS", "1") == "1"   # per-pool-record nearest-twin similarity (distractor detection)
USE_CE = True
CE_P1_MIN = 0.01            # only pairs with stage-1 prob >= this are scored by the cross-encoder
# fine-tuned bi-encoder for retrieval pass "f" (train_biencoder.py); empty -> pass disabled
BIENC_DIR = os.environ.get("ER_BIENC", "")
CE_MODEL_DIR = os.environ.get("ER_CE_MODEL", os.path.join(CACHE_DIR, "cross_encoder_minilm"))
# Second cross-encoder (XLM-R base, MIT), added to stage-2 by stage2_refit.py (see reproduce.sh)
XLMR_BACKBONE = os.environ.get("ER_XLMR", os.path.join(MODELS_DIR, "xlm-roberta-base"))
XLMR_DIR = os.path.join(CACHE_DIR, "cross_encoder_xlmr")

N_FOLDS = 5
LGB_PARAMS = dict(
    objective="binary", num_leaves=int(os.environ.get("ER_LGB_LEAVES", 63)), learning_rate=0.05, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, min_child_samples=int(os.environ.get("ER_LGB_MCS", 20)), seed=SEED,
    lambda_l2=float(os.environ.get("ER_LGB_L2", 0.0)),
    verbose=-1, num_threads=int(os.environ.get("ER_LGB_THREADS", max(1, (os.cpu_count() or 4) - 2))),
)
LGB_MAX_ROUNDS = 2000
LGB_EARLY_STOP = 100


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
