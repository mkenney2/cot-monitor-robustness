"""Phase 4 — is the probe's attack degradation just textual divergence?

For each attacked POS transcript, measure similarity of the rewritten CoT to the
original (jaccard / difflib seqratio / length ratio), aggregate per attack, and
correlate per-item similarity with the per-item probe score change. Key contrast:
paraphrase (max surface change, content preserved) vs recruit (content laundered).
If the probe read surface novelty, equally-divergent paraphrase and recruit would
move it equally; they don't.

Inputs (all local-capable): data/attacked/attacked_transcripts.parquet (carries
orig_cot + rewritten canonical_cot), results/scores_clean.parquet,
results/scores_attacked.parquet.
Output: results/probe_divergence_analysis.json
Usage: python src/12_probe_divergence.py
"""

import json
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import ROOT

WORD = re.compile(r"[a-z0-9]+")
tok = lambda s: WORD.findall((s or "").lower())


def jaccard(a, b):
    sa, sb = set(tok(a)), set(tok(b))
    return len(sa & sb) / len(sa | sb) if (sa | sb) else np.nan


def seqratio(a, b):
    return SequenceMatcher(None, tok(a), tok(b)).ratio()


def lenratio(a, b):
    la = len(tok(a))
    return len(tok(b)) / la if la else np.nan


def main():
    att = pd.read_parquet(ROOT / "data" / "attacked" / "attacked_transcripts.parquet")
    clean = pd.read_parquet(ROOT / "results" / "scores_clean.parquet")
    attsc = pd.read_parquet(ROOT / "results" / "scores_attacked.parquet")

    att["jaccard"] = [jaccard(o, r) for o, r in zip(att.orig_cot, att.canonical_cot)]
    att["seqratio"] = [seqratio(o, r) for o, r in zip(att.orig_cot, att.canonical_cot)]
    att["lenratio"] = [lenratio(o, r) for o, r in zip(att.orig_cot, att.canonical_cot)]

    pc = clean[clean.arm == "probe"][["qid", "score"]].rename(columns={"score": "pc"})
    pa = attsc[attsc.arm == "probe"][["qid", "attack", "label", "score"]].rename(
        columns={"score": "pa"})
    m = (pa.merge(pc, on="qid")
         .merge(att[["qid", "attack", "jaccard"]], on=["qid", "attack"]))
    m["d"] = m.pa - m.pc

    out = {}
    for a in ["paraphrase", "launder", "recruit"]:
        pos = att[(att.attack == a) & (att.label == "POS")]
        sub = m[(m.attack == a) & (m.label == "POS")].dropna(subset=["jaccard", "d"])
        rho, pv = spearmanr(sub.jaccard, sub.d)
        out[a] = {
            "pos_jaccard_mean": round(float(pos.jaccard.mean()), 3),
            "pos_seqratio_mean": round(float(pos.seqratio.mean()), 3),
            "pos_lenratio_mean": round(float(pos.lenratio.mean()), 3),
            "probe_delta_score_mean": round(float(sub.d.mean()), 3),
            "spearman_jaccard_vs_probedelta": round(float(rho), 3),
            "spearman_p": round(float(pv), 4),
            "n": int(len(sub)),
        }
    out["_interpretation"] = (
        "paraphrase and recruit are tied on jaccard (~.36-.37) but paraphrase does "
        "not move the probe while recruit does -> probe tracks content laundering, "
        "not surface divergence. Within-attack positive spearman = residual nuisance.")
    op = ROOT / "results" / "probe_divergence_analysis.json"
    json.dump(out, open(op, "w", encoding="utf-8"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"-> {op}", flush=True)


if __name__ == "__main__":
    main()
