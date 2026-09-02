"""Arm 4 rerun: J-Lens at LATE layers (56 / 60 / 62), after 21_jlens_anchor_check.py
showed Qwen 3.6 27B is undecodable at 16/32/48 (next-token top-10 hit 4-8%)
and only becomes readable from ~56 (40%) up to the lens anchor 62 (75%).

Usage (pod):
  python src/22_jlens_late.py cache            # layers 56/60/62, clean + attacked
  python src/22_jlens_late.py score            # 07's scoring -> arms jlens_late, jlens_llm_late
  python src/22_jlens_late.py score_attacked   # 09's jlens scoring -> jlens_late in scores_attacked
  python src/22_jlens_late.py analysis         # 10 with jlens_late added, outputs under results/jlens_late_analysis/
  python src/22_jlens_late.py all

Nothing published is overwritten: new activation dirs data/activations_late{,_attacked};
new arm names jlens_late / jlens_llm_late (upserted alongside the old rows in
scores_clean.parquet / scores_attacked.parquet); token sets re-frozen on the
train split at the new layers -> results/jlens_late_token_sets.json; review dump
review/22_jlens_late_sample.md; LLM cache data/rollouts/jlens_llm_late_raw.jsonl;
degradation matrix + phi with the late arm under results/jlens_late_analysis/.

Caches are keyed by the transcript key `tkey` (phase2_common.transcript_key):
qid for hinted transcripts, qid__clean for unhinted NEG-clean ones — the
duplicate-qid fix of 2026-09-02 (src/23_dupfix_migrate.py).
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

LAYERS_LATE = [56, 60, 62]
ACT_LATE = ROOT / "data" / "activations_late"
ACT_LATE_ATT = ROOT / "data" / "activations_late_attacked"
SUFFIX = "_late"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def patched_m07():
    """07_arm_jlens with every path/arm-name redirected to the *_late variants."""
    from phase2_common import report_arm, save_scores

    m07 = load("07_arm_jlens")
    m07.ACTS = ACT_LATE
    m07.TOKEN_SETS = ROOT / "results" / "jlens_late_token_sets.json"
    m07.LLM_CACHE = ROOT / "data" / "rollouts" / "jlens_llm_late_raw.jsonl"
    m07.SAMPLE_MD = ROOT / "review" / "22_jlens_late_sample.md"
    m07.save_scores = lambda arm, df: save_scores(arm + SUFFIX, df)
    m07.report_arm = lambda arm, df: report_arm(arm + SUFFIX, df)
    return m07


def cmd_cache():
    from phase2_common import load_dataset

    m06 = load("06_arm_probe")
    cfg = m06.load_config()
    tokenizer, model, _ = m06.load_model(cfg)
    print(f"caching layers {LAYERS_LATE}", flush=True)
    m06.run_cache(cfg, tokenizer, model, LAYERS_LATE, load_dataset(), ACT_LATE,
                  lambda r: r["tkey"])
    att = pd.read_parquet(m06.ATTACKED)
    m06.run_cache(cfg, tokenizer, model, LAYERS_LATE, att, ACT_LATE_ATT,
                  lambda r: f"{r['qid']}__{r['attack']}")


def cmd_score():
    m07 = patched_m07()
    m07.cmd_score()


def cmd_score_attacked():
    m09 = load("09_score_attacked")
    m07 = patched_m07()
    m09._load = lambda mod: m07 if mod == "07_arm_jlens" else load(mod)
    m09.ACT_ATTACKED = ACT_LATE_ATT
    _save, _report = m09.save, m09.report
    m09.save = lambda arm, rows: _save(arm + SUFFIX, rows)
    m09.report = lambda arm, att: _report(arm + SUFFIX, att)
    cfg = m09.load_config()
    att = pd.read_parquet(m09.ATTACKED)
    m09.score_jlens(cfg, att)


def cmd_analysis():
    m10 = load("10_analysis")
    m10.ACT_ARMS = ["probe", "jlens", "jlens" + SUFFIX]
    m10.ARMS = m10.TEXT_ARMS + m10.ACT_ARMS
    out = ROOT / "results" / "jlens_late_analysis"
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    m10.ROOT = out          # outputs go under results/jlens_late_analysis/{results,figures}/
    m10.main()
    print(f"late-arm analysis -> {out}", flush=True)


if __name__ == "__main__":
    cmds = {"cache": cmd_cache, "score": cmd_score, "score_attacked": cmd_score_attacked,
            "analysis": cmd_analysis}
    args = sys.argv[1:]
    if args == ["all"]:
        args = list(cmds)
    if not args or any(a not in cmds for a in args):
        raise SystemExit(f"usage: python src/22_jlens_late.py {{{'|'.join(cmds)}|all}}")
    for a in args:
        print(f"===== {a} =====", flush=True)
        cmds[a]()
