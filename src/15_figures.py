"""Phase 5 — exec-summary figures beyond the degradation heatmap.

score_distributions: for the honest text arm (judge_hintblind) and the activation
arm (probe), show POS vs NEG-inert suspicion-score distributions on the clean test
split, and the same under the recruit attack — visually: the judge's clean
separation collapses under recruit while the probe's holds.

Reads results/scores_clean.parquet + results/scores_attacked.parquet (local).
Writes figures/score_distributions.png
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT

POS_C, NEG_C = "#c0392b", "#2b6cb0"   # POS red, NEG-inert blue


def panel(ax, s, title, xlabel):
    pos = s[s.label == "POS"].score.dropna()
    neg = s[s.label == "NEG-inert"].score.dropna()
    lo = float(min(pos.min(), neg.min())); hi = float(max(pos.max(), neg.max()))
    bins = np.linspace(lo, hi, 16)
    ax.hist(neg, bins=bins, color=NEG_C, alpha=0.55, label="NEG-inert (hint present, not used)")
    ax.hist(pos, bins=bins, color=POS_C, alpha=0.55, label="POS (silent hint-use)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("transcripts", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)


def main():
    clean = pd.read_parquet(ROOT / "results" / "scores_clean.parquet")
    att = pd.read_parquet(ROOT / "results" / "scores_attacked.parquet")

    def clean_test(arm):
        s = clean[(clean.arm == arm) & (clean.split == "test")]
        return s[s.label.isin(["POS", "NEG-inert"])]

    def att_test(arm, attack):
        s = att[(att.arm == arm) & (att.split == "test") & (att.attack == attack)]
        return s[s.label.isin(["POS", "NEG-inert"])]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panel(axes[0, 0], clean_test("judge_hintblind"),
          "Text judge — CLEAN (AUC 0.895)", "suspicion score (0-100)")
    panel(axes[0, 1], att_test("judge_hintblind", "recruit"),
          "Text judge — RECRUIT attack (AUC 0.635)", "suspicion score (0-100)")
    panel(axes[1, 0], clean_test("probe"),
          "Activation probe — CLEAN (AUC 0.735)", "probe P(hint-use)")
    panel(axes[1, 1], att_test("probe", "recruit"),
          "Activation probe — RECRUIT attack (AUC 0.650)", "probe P(hint-use)")
    axes[0, 0].legend(fontsize=8, loc="upper center")
    fig.suptitle("Recruit attack collapses the text judge's separation; the probe's holds",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ROOT / "figures" / "score_distributions.png"
    fig.savefig(out, dpi=150)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
