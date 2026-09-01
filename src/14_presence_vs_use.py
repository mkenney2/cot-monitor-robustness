"""Phase 4 — presence-vs-use disentanglement for the linear probe.

The probe separates POS vs NEG-inert at 0.735, but also separates NEG-inert vs
NEG-clean at 0.698 (a hint-PRESENCE axis). Question: is the probe's use-signal the
SAME direction as the presence axis, or independent of it?

At the probe's layer (L48 meanpool), in one shared standardized feature space:
  w_use   = logistic direction for POS vs NEG-inert   (the probe itself)
  w_pres  = logistic direction for NEG-inert vs NEG-clean  (pure presence)
Then report:
  1. cosine(w_use, w_pres)  — are they the same direction? (cos^2 = shared variance)
  2. presence-only AUC on the USE task (project onto w_pres, classify POS vs NEG-inert)
  3. use AUC after RESIDUALIZING the presence direction out of the features, refit
     (if it stays ~0.735, the use-signal is largely independent of presence)
All labels only used to fit directions on the TRAIN split; AUCs on TEST.

Reads cached activations (pod). Output: results/presence_vs_use.json
Usage: python src/14_presence_vs_use.py
"""

import importlib.util
import json

import numpy as np

from common import ROOT, load_config
from phase2_common import auc_with_ci, load_dataset


def _load(mod):
    spec = importlib.util.spec_from_file_location(mod, ROOT / "src" / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _dir(X, y):
    """Unit logistic-regression weight vector (class-balanced) for labels y."""
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X, y)
    w = lr.coef_[0]
    return w / np.linalg.norm(w)


def main():
    from sklearn.preprocessing import StandardScaler

    cfg = load_config()
    m06 = _load("06_arm_probe")
    layer = json.load(open(ROOT / "results" / "probe_layer_selection.json",
                           encoding="utf-8"))["probe"]["chosen_layer"]
    seed = cfg["seeds"]["bootstrap"]

    df = load_dataset()
    feats = m06.load_features(df.qid)
    has = df.qid.isin(feats)
    df = df[has].reset_index(drop=True)

    def mat(rows):
        return np.stack([feats[q]["meanpool"][layer] for q in rows.qid])

    # Shared standardized space: scaler fit on ALL train rows (all 3 classes) so the
    # two directions live in the same geometry and cosine is meaningful.
    train = df[df.split == "train"]
    test = df[df.split == "test"]
    scaler = StandardScaler().fit(mat(train))
    def X(rows):
        return scaler.transform(mat(rows))

    tr_use = train[train.label.isin(["POS", "NEG-inert"])]
    tr_pres = train[train.label.isin(["NEG-inert", "NEG-clean"])]
    w_use = _dir(X(tr_use), (tr_use.label == "POS").astype(int).to_numpy())
    w_pres = _dir(X(tr_pres), (tr_pres.label == "NEG-inert").astype(int).to_numpy())

    cos = float(np.dot(w_use, w_pres))
    print(f"layer L{layer}  cosine(w_use, w_pres) = {cos:+.3f}  "
          f"(shared variance = {cos**2:.3f})", flush=True)

    te_use = test[test.label.isin(["POS", "NEG-inert"])]
    y_use = (te_use.label == "POS").astype(int).to_numpy()

    # (0) baseline probe in this shared space (sanity vs the 0.735 headline)
    base = auc_with_ci(y_use, X(te_use) @ w_use, seed)

    # (2) presence direction alone, applied to the use task
    pres_on_use = auc_with_ci(y_use, X(te_use) @ w_pres, seed)

    # (3) residualize the presence axis out of the features, refit the use probe
    def resid(Xr):
        return Xr - np.outer(Xr @ w_pres, w_pres)
    w_use_resid = _dir(resid(X(tr_use)), (tr_use.label == "POS").astype(int).to_numpy())
    resid_auc = auc_with_ci(y_use, resid(X(te_use)) @ w_use_resid, seed)

    out = {
        "layer": layer,
        "cosine_use_vs_presence": cos,
        "shared_variance": cos**2,
        "baseline_use_auc": base,
        "presence_dir_only_on_use_task_auc": pres_on_use,
        "use_auc_after_removing_presence_axis": resid_auc,
        "interpretation": (
            "Low |cosine| + residual AUC ~= baseline => the probe's hint-USE "
            "direction is largely INDEPENDENT of the hint-PRESENCE axis (presence "
            "is a confound the probe does not primarily ride on). High cosine + "
            "residual AUC collapse => the probe is mostly a presence detector."),
    }
    op = ROOT / "results" / "presence_vs_use.json"
    json.dump(out, open(op, "w", encoding="utf-8"), indent=2)
    print(f"\nbaseline use AUC (shared space)      : {base['auc']:.3f} "
          f"[{base['ci_lo']:.3f},{base['ci_hi']:.3f}]", flush=True)
    print(f"presence-dir ALONE on use task       : {pres_on_use['auc']:.3f} "
          f"[{pres_on_use['ci_lo']:.3f},{pres_on_use['ci_hi']:.3f}]", flush=True)
    print(f"use AUC after removing presence axis : {resid_auc['auc']:.3f} "
          f"[{resid_auc['ci_lo']:.3f},{resid_auc['ci_hi']:.3f}]", flush=True)
    print(f"-> {op}", flush=True)


if __name__ == "__main__":
    main()
