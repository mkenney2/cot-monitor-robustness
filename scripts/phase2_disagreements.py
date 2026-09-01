"""Phase 2 output: review dump of the 10 transcripts where arms maximally
disagree (|percentile rank(judge_hintblind) - percentile rank(probe)|).

Run on the pod after all arms have scored:
  python scripts/phase2_disagreements.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from phase2_common import load_dataset  # noqa: E402

ARMS = ["judge", "judge_hintblind", "textfeat", "probe", "jlens", "jlens_llm"]


def main():
    scores = pd.read_parquet(ROOT / "results" / "scores_clean.parquet")
    meta = load_dataset().set_index("qid")
    wide = scores.pivot_table(index="qid", columns="arm", values="score")
    ranks = wide.rank(pct=True)
    gap = (ranks["judge_hintblind"] - ranks["probe"]).abs()
    top = gap.sort_values(ascending=False).head(10)

    lines = ["# Phase 2: transcripts where arms maximally disagree\n",
             "Top 10 by |pct rank(judge_hintblind) - pct rank(probe)|.",
             "Raw scores from results/scores_clean.parquet; pct rank in parens.\n"]
    for qid in top.index:
        m = meta.loc[qid]
        lines += [f"\n---\n\n## {qid}  [{m.label}]  hint_type={m.hint_type} "
                  f"hint_letter={m.hint_letter} correct={m.correct} split={m.split}\n",
                  "| arm | score (pct rank) |", "|---|---|"]
        for arm in ARMS:
            if arm in wide.columns and pd.notna(wide.loc[qid, arm]):
                lines.append(f"| {arm} | {wide.loc[qid, arm]:.3f} "
                             f"({ranks.loc[qid, arm]:.2f}) |")
        lines += [f"\n**Hint:**\n```\n{m.hint_text}\n```\n",
                  f"**CoT:**\n```\n{m.canonical_cot}\n```\n"]
    out = ROOT / "review" / "phase2_disagreements.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} with {len(top)} entries")


if __name__ == "__main__":
    main()
