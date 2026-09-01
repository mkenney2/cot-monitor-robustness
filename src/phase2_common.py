"""Shared Phase 2 protocol: dataset loading, splits, AUC, score saving.

Every arm (04-07) uses exactly this so results are comparable:
- load_dataset(): labels.parquet joined with hinted_items.jsonl, plus a 'split'
  column — 60/40 train/test stratified by (label, hint_type), seeded.
- Arms output one scalar suspicion score per transcript and call
  save_scores(arm_name, df) which upserts into results/scores_clean.parquet.
- AUC is POS vs NEG-inert on the test split (POS vs NEG-clean reported
  separately), with 1000-resample bootstrap CIs.
"""

import json

import numpy as np
import pandas as pd

from common import ROOT, load_config, read_jsonl, rng

SCORES = ROOT / "results" / "scores_clean.parquet"


def load_dataset():
    """Labeled transcripts with hint/question text and train/test split.

    NEG-inert is subsampled (seeded, stratified to match the POS hint-type
    distribution) down to POS n — the protocol's equal-n comparison, and it
    stops arms from keying on hint-type composition instead of hint-use.
    The full table stays in labels.parquet.
    """
    cfg = load_config()
    df = pd.read_parquet(ROOT / "data" / "labels.parquet")
    pos_by_type = df[df.label == "POS"]["hint_type"].value_counts()
    keep = df.label != "NEG-inert"
    for ht, n in pos_by_type.items():
        pool = df[(df.label == "NEG-inert") & (df.hint_type == ht)]
        picked = pool.sample(n=min(n, len(pool)),
                             random_state=cfg["seeds"]["neg_subsample"])
        keep |= df.index.isin(picked.index)
    df = df[keep].reset_index(drop=True)
    items = {it["qid"]: it for it in read_jsonl(ROOT / "data" / "hinted_items.jsonl")}
    df["hint_text"] = df["qid"].map(lambda q: items[q]["hint_text"])
    df["question_block"] = df["qid"].map(lambda q: items[q]["question_block"])
    df["split"] = assign_splits(df, cfg["seeds"]["split"])
    return df


def assign_splits(df, seed, train_frac=0.6):
    """60/40 stratified by (label, hint_type). Deterministic given the seed."""
    split = pd.Series(index=df.index, dtype=object)
    for _, group in df.groupby(["label", "hint_type"], dropna=False):
        idx = list(group.index)
        rng(f"{seed}:{group['label'].iloc[0]}:{group['hint_type'].iloc[0]}").shuffle(idx)
        n_train = round(len(idx) * train_frac)
        split.loc[idx[:n_train]] = "train"
        split.loc[idx[n_train:]] = "test"
    return split


def auc_with_ci(y_true, scores, seed, n_boot=1000):
    """AUC + percentile bootstrap CI. y_true: 1=POS, 0=NEG. Returns dict."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    keep = ~np.isnan(s)
    y, s = y[keep], s[keep]
    point = roc_auc_score(y, s)
    r = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = r.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            "n_dropped_nan": int((~keep).sum()), "n_boot": len(boots)}


def report_arm(arm, df):
    """Standard eval printout + json artifact. df needs: label, split, score."""
    cfg = load_config()
    test = df[df.split == "test"]
    out = {"arm": arm}
    for negclass in ("NEG-inert", "NEG-clean"):
        sub = test[test.label.isin(["POS", negclass])]
        if sub.label.nunique() < 2:
            out[f"auc_vs_{negclass}"] = None
            continue
        out[f"auc_vs_{negclass}"] = auc_with_ci(
            (sub.label == "POS").astype(int), sub.score, cfg["seeds"]["bootstrap"])
    path = ROOT / "results" / f"auc_{arm}_clean.json"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    return out


def save_scores(arm, df):
    """Upsert this arm's rows into the shared clean-condition score table.

    df must have columns: qid, label, split, score. Extra columns are kept.
    """
    df = df.copy()
    df["arm"] = arm
    if SCORES.exists():
        existing = pd.read_parquet(SCORES)
        existing = existing[existing.arm != arm]
        df = pd.concat([existing, df], ignore_index=True)
    SCORES.parent.mkdir(exist_ok=True)
    df.to_parquet(SCORES, index=False)
    print(f"scores for arm '{arm}' saved -> {SCORES}", flush=True)
