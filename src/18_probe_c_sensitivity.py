"""Post-hoc sensitivity analysis: probe AUC vs logistic-regression C.

The pre-registered probe (06) used sklearn's default C=1.0, untuned. This
script asks: would any reasonable regularization strength have changed the
story? It is explicitly POST-HOC — test AUCs across the C grid are reported
for sensitivity only, never to pick a new C (test was already touched).

Run on the pod (needs data/activations/, CPU only, ~seconds):
  python src/18_probe_c_sensitivity.py

Outputs:
  results/probe_c_sensitivity.json   train-CV AUC + test AUC per C, per pooling
  data/pooled_features.parquet       mean-pool + last-token vectors per layer
                                     (~40 MB) — scp this down so activation-level
                                     analyses no longer need the pod:
                                     one row per transcript (qid + tkey), columns {pooling}_L{layer}_{dim}
                                     stored as flat float32 lists per layer.

Protocol matches 06 train() exactly: same splits (phase2_common), same
StratifiedKFold(5, seed 109), same pipeline (StandardScaler -> balanced
logistic regression), same fixed layer per pooling (the pre-registered choice
from results/probe_layer_selection.json) — only C varies.
"""

import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ROOT, load_config
from phase2_common import load_dataset


def _load(mod):
    spec = importlib.util.spec_from_file_location(mod, ROOT / "src" / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


probe_mod = _load("06_arm_probe")

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
OUT = ROOT / "results" / "probe_c_sensitivity.json"
POOLED = ROOT / "data" / "pooled_features.parquet"


def make_probe_c(C):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced",
                                            max_iter=2000))


def export_pooled(feats, layers):
    """Flatten pooled vectors to one parquet row per qid (float32 lists)."""
    rows = []
    for tkey, d in feats.items():
        row = {"qid": tkey.replace("__clean", ""), "tkey": tkey}
        for pooling in ("meanpool", "lasttoken"):
            for L in layers:
                row[f"{pooling}_L{L}"] = d[pooling][L].astype(np.float32)
        rows.append(row)
    pd.DataFrame(rows).to_parquet(POOLED, index=False)
    print(f"pooled features -> {POOLED}", flush=True)


def main():
    cfg = load_config()
    cv_folds = cfg["phase2"]["probe"]["cv_folds"]
    cv_seed = cfg["seeds"].get("probe_cv", 109)

    with open(ROOT / "results" / "probe_layer_selection.json") as f:
        selection = json.load(f)

    df = load_dataset()
    feats = probe_mod.load_features(df.tkey)
    layers = sorted(next(iter(feats.values()))["meanpool"])
    print(f"{len(feats)}/{len(df)} transcripts with activations", flush=True)
    export_pooled(feats, layers)

    has = df.tkey.isin(feats)
    pn = df.label.isin(["POS", "NEG-inert"]) & has
    train_rows = df[(df.split == "train") & pn]
    test_rows = df[(df.split == "test") & pn]

    result = {"c_grid": C_GRID, "cv_folds": cv_folds, "cv_seed": cv_seed,
              "n_train": int(len(train_rows)), "n_test": int(len(test_rows)),
              "note": ("post-hoc sensitivity only; pre-registered probe is "
                       "C=1.0 at the layer in probe_layer_selection.json. "
                       "Do NOT use this to select a new C.")}
    for arm, pooling in (("probe", "meanpool"), ("probe_lasttoken", "lasttoken")):
        L = selection[arm]["chosen_layer"]
        X_tr, y_tr = probe_mod.stack(train_rows, feats, pooling, L)
        X_te, y_te = probe_mod.stack(test_rows, feats, pooling, L)
        per_c = {}
        for C in C_GRID:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                                  random_state=cv_seed)
            cv_auc = float(np.mean(cross_val_score(
                make_probe_c(C), X_tr, y_tr, cv=skf, scoring="roc_auc")))
            clf = make_probe_c(C).fit(X_tr, y_tr)
            test_auc = float(roc_auc_score(
                y_te, clf.predict_proba(X_te)[:, 1]))
            per_c[str(C)] = {"train_cv_auc": cv_auc, "test_auc": test_auc}
            print(f"[{arm} L{L}] C={C:>7}: train-CV {cv_auc:.3f}  "
                  f"test {test_auc:.3f}", flush=True)
        result[arm] = {"layer": L, "by_c": per_c}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
