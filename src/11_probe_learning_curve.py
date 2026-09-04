"""Phase 4 — probe learning curve: does the linear probe's AUC still climb with
more training data, or has it plateaued at n_train~80?

Metric = the headline metric (probe AUC POS vs NEG-inert on the FIXED test split).
We grow a stratified subsample of the CLEAN train pool from a floor up to 100%,
refit the same probe (meanpool, frozen primary layer) at each size over many
seeds, and record mean +/- std test AUC. Reuses 06_arm_probe's exact feature
loader and probe so there is no reimplementation drift.

Reads cached activations (pod). Writes:
  results/probe_learning_curve.json
  figures/probe_learning_curve.png
Usage: python src/11_probe_learning_curve.py
"""

import importlib.util
import json

import numpy as np

from common import ROOT, load_config
from phase2_common import load_dataset


def _load(mod):
    spec = importlib.util.spec_from_file_location(mod, ROOT / "src" / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def subsample(rows, frac, rng):
    """Stratified per-class subsample of `rows` (a POS/NEG-inert train frame)."""
    keep = []
    for lab, g in rows.groupby("label"):
        k = max(2, int(round(frac * len(g))))
        k = min(k, len(g))
        idx = rng.choice(g.index.to_numpy(), size=k, replace=False)
        keep.extend(idx.tolist())
    return rows.loc[sorted(keep)]


def main():
    from sklearn.metrics import roc_auc_score

    cfg = load_config()
    m06 = _load("06_arm_probe")
    layer = json.load(open(ROOT / "results" / "probe_layer_selection.json",
                           encoding="utf-8"))["probe"]["chosen_layer"]

    df = load_dataset()
    feats = m06.load_features(df.tkey)
    has = df.tkey.isin(feats)
    train_pool = df[(df.split == "train") & df.label.isin(["POS", "NEG-inert"]) & has]
    test = df[(df.split == "test") & df.label.isin(["POS", "NEG-inert"]) & has]
    X_test, y_test = m06.stack(test, feats, "meanpool", layer)
    print(f"layer {layer}  train pool={train_pool.label.value_counts().to_dict()}  "
          f"test={test.label.value_counts().to_dict()}", flush=True)

    fracs = [0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
    n_seeds = 40
    base_seed = cfg["seeds"].get("probe_cv", 109)

    curve = []
    for frac in fracs:
        aucs, ntr = [], None
        for s in range(n_seeds):
            rng = np.random.default_rng(base_seed + s)
            sub = subsample(train_pool, frac, rng)
            if sub.label.nunique() < 2:
                continue
            ntr = len(sub)
            X_tr, y_tr = m06.stack(sub, feats, "meanpool", layer)
            probe = m06.make_probe()
            probe.fit(X_tr, y_tr)
            aucs.append(roc_auc_score(y_test, probe.predict_proba(X_test)[:, 1]))
        curve.append({"frac": frac, "n_train": ntr, "n_seeds": len(aucs),
                      "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                      "auc_min": float(np.min(aucs)), "auc_max": float(np.max(aucs))})
        print(f"  frac={frac:.2f} n_train={ntr:3d}  AUC {np.mean(aucs):.3f} "
              f"+/- {np.std(aucs):.3f}  [{np.min(aucs):.3f},{np.max(aucs):.3f}]",
              flush=True)

    # Full-pool slope: linear fit of mean AUC vs n_train over the top half of the
    # curve — a rough "is it still climbing?" readout (positive => not plateaued).
    ns = np.array([c["n_train"] for c in curve], float)
    ys = np.array([c["auc_mean"] for c in curve], float)
    half = ns >= np.median(ns)
    slope_per_100 = float(np.polyfit(ns[half], ys[half], 1)[0] * 100) if half.sum() >= 2 else None

    out = {"layer": layer, "metric": "AUC POS vs NEG-inert, fixed test split",
           "n_seeds": n_seeds, "base_seed": base_seed, "curve": curve,
           "top_half_slope_auc_per_100_train": slope_per_100,
           "note": ("positive slope over the top half => AUC still rising with data "
                    "(probe is data-limited); ~0 => plateaued")}
    op = ROOT / "results" / "probe_learning_curve.json"
    json.dump(out, open(op, "w", encoding="utf-8"), indent=2)
    print(f"\ntop-half slope = {slope_per_100:+.3f} AUC per +100 train examples", flush=True)
    print(f"-> {op}", flush=True)

    plot(curve, layer, n_seeds)


def plot(curve, layer, n_seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [c["n_train"] for c in curve]
    ms = [c["auc_mean"] for c in curve]
    sd = [c["auc_std"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(ns, ms, yerr=sd, marker="o", capsize=3, color="#2b6cb0")
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.set_xlabel("training examples (POS + NEG-inert)")
    ax.set_ylabel("test AUC (POS vs NEG-inert)")
    ax.set_title(f"Probe learning curve (meanpool L{layer}, {n_seeds} seeds/point)")
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fp = ROOT / "figures" / "probe_learning_curve.png"
    fp.parent.mkdir(exist_ok=True)
    fig.savefig(fp, dpi=150)
    print(f"-> {fp}", flush=True)


def plot_from_artifact():
    """Redraw the figure from results/probe_learning_curve.json without refitting."""
    op = ROOT / "results" / "probe_learning_curve.json"
    d = json.load(open(op, encoding="utf-8"))
    plot(d["curve"], d["layer"], d["n_seeds"])


if __name__ == "__main__":
    import sys
    if "--figure-only" in sys.argv:
        plot_from_artifact()
    else:
        main()
