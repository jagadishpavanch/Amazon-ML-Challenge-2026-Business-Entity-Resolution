"""Loading source TSVs / ground truth and writing submission files."""
import csv
import os

import pandas as pd

COLS = ["entity_id", "business_name", "business_address", "country"]


def read_tsv(path: str) -> pd.DataFrame:
    """Read a challenge TSV as all-string columns with empty strings for missing values."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       quoting=csv.QUOTE_NONE, na_filter=False, engine="c")


def load_split(data_dir: str, split: str):
    """Return (s1, pool) where pool = S2 ∪ S3 with a `source` column from the ID prefix."""
    s1 = read_tsv(os.path.join(data_dir, f"{split}_source1.tsv"))[COLS]
    parts = [read_tsv(os.path.join(data_dir, f"{split}_source{k}.tsv"))[COLS] for k in (2, 3)]
    pool = pd.concat(parts, ignore_index=True)
    pool["source"] = pool["entity_id"].str.slice(0, 2)
    s1["source"] = "S1"
    return s1, pool


def load_truth(data_dir: str) -> dict:
    """Parse ground truth into {s1_id: set(match ids)} (empty set for singletons)."""
    gt = read_tsv(os.path.join(data_dir, "train_ground_truth.tsv"))
    return {s: set(m.split(",")) if m else set()
            for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}


def _dedup(ids):
    seen, out = set(), []
    for i in ids:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def write_id_lists(path: str, header: list, s1_ids, mapping: dict) -> None:
    """Write one row per S1 id; ids comma-joined, de-duplicated in order, no quoting."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(header) + "\n")
        for s in s1_ids:
            f.write(s + "\t" + ",".join(_dedup(mapping.get(s, ()))) + "\n")
