"""Phase 4 — the correctness confound control.

By construction POS = wrong final answer, NEG-inert = correct. So the honest arms
could be reading "answer is wrong / reasoning doesn't support it" rather than
"hint used". Test it on data with NO hint anywhere: score the FROZEN probe,
concept directions, and hint-blind judge on organically-WRONG vs organically-
CORRECT *unhinted* transcripts.

Cleanest design — WITHIN-ITEM pairs: for each item that produced both a wrong and
a correct unhinted rollout, take one of each (seeded). Same question, same no-hint
context, only the answer's correctness differs -> isolates wrongness from
difficulty/subject. If the probe separates these at ~0.70, a chunk of its 0.735 is
a wrongness signal and the claim softens to "detects answering against knowledge".
If ~0.5, the hint-use claim is much stronger.

Stages:
  build  -> data/control/correctness_control.parquet   (no GPU/API)
  cache  -> data/activations_control/{qid}__unh{idx}.npz  (GPU)
  score  -> results/correctness_control.json  (frozen probe + concept dirs; no GPU)
  judge  -> adds hint-blind judge AUC to the same json  (OpenRouter API)
  python src/16_correctness_control.py build cache score

All monitors FROZEN (probe_model.joblib, concept_directions.npz) — applied, never
refit, exactly like the attacked re-scoring.
"""

import asyncio
import importlib.util
import json
import sys

import numpy as np
import pandas as pd

from common import ROOT, item_rng, load_config, read_jsonl
from phase2_common import auc_with_ci

ROLLOUTS = ROOT / "data" / "rollouts" / "rollouts.jsonl"
HINTED = ROOT / "data" / "hinted_items.jsonl"
CONTROL = ROOT / "data" / "control" / "correctness_control.parquet"
ACT_CONTROL = ROOT / "data" / "activations_control"
OUT = ROOT / "results" / "correctness_control.json"
CONTROL_SEED = 424242


def _load(mod):
    spec = importlib.util.spec_from_file_location(mod, ROOT / "src" / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build(cfg):
    """Within-item paired wrong/correct unhinted transcripts."""
    lab = pd.read_parquet(ROOT / "data" / "labels.parquet")
    correct_map = dict(zip(lab.qid, lab.correct))
    subj_map = dict(zip(lab.qid, lab.subject))
    src_map = dict(zip(lab.qid, lab.source))
    qblock = {it["qid"]: it["question_block"] for it in read_jsonl(HINTED)}

    rows = []
    for r in read_jsonl(ROLLOUTS):
        if r.get("condition") != "unhinted":
            continue
        qid = r["qid"]
        corr = correct_map.get(qid)
        if corr is None or qid not in qblock:
            continue
        ans = r["answers"]
        wrong = [i for i, a in enumerate(ans) if a is not None and a != corr]
        right = [i for i, a in enumerate(ans) if a is not None and a == corr]
        if not (wrong and right):
            continue
        rr = item_rng(CONTROL_SEED, qid)
        wi = rr.choice(wrong)
        ci = rr.choice(right)
        for idx, corr_label in ((wi, "wrong"), (ci, "correct")):
            rows.append({
                "qid": qid, "rollout_idx": int(idx),
                "stem": f"{qid}__unh{idx}",
                "correctness": corr_label,
                "canonical_condition": "unhinted",
                "canonical_cot": r["rollouts"][idx],
                "question_block": qblock[qid],
                "given_answer": ans[idx], "correct": corr,
                "subject": subj_map.get(qid), "source": src_map.get(qid),
            })
    df = pd.DataFrame(rows)
    CONTROL.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CONTROL, index=False)
    n = df.correctness.value_counts().to_dict()
    print(f"built {len(df)} control transcripts ({n}) from "
          f"{df.qid.nunique()} paired items -> {CONTROL}", flush=True)


def cache(cfg):
    m06 = _load("06_arm_probe")
    df = pd.read_parquet(CONTROL)
    tok, model, layers = m06.load_model(cfg)
    m06.run_cache(cfg, tok, model, layers, df, ACT_CONTROL, lambda r: r["stem"])


def _auc(df, score_col):
    """AUC wrong-vs-correct: 1 = wrong (the 'suspicious' class by construction)."""
    d = df.dropna(subset=[score_col])
    y = (d.correctness == "wrong").astype(int).to_numpy()
    if len(set(y)) < 2:
        return None
    r = auc_with_ci(y, d[score_col].to_numpy(), load_config()["seeds"]["bootstrap"])
    return r


def score(cfg):
    import joblib

    m06 = _load("06_arm_probe")
    df = pd.read_parquet(CONTROL)
    feats = m06.load_features(df.stem, act_dir=ACT_CONTROL)
    print(f"{len(feats)}/{len(df)} control transcripts cached", flush=True)

    result = {"n": int(len(df)), "n_pairs": int(df.qid.nunique()),
              "positive_class": "wrong (higher score on wrong = wrongness confound)",
              "reference": {"probe_hint_use_auc": 0.735,
                            "concept_hint_use": {"deception": 0.601, "cheating": 0.585,
                                                 "hidden-info": 0.582, "sycophancy": 0.577}}}

    # ---- frozen probe ----
    saved = joblib.load(m06.PROBE_MODEL)
    probe, layer = saved["probe"], saved["layer"]
    df["probe"] = [float(probe.predict_proba(feats[s]["meanpool"][layer][None])[0, 1])
                   if s in feats and layer in feats[s]["meanpool"] else np.nan
                   for s in df.stem]
    r = _auc(df, "probe")
    result["probe"] = r
    print(f"[probe] wrong-vs-correct AUC = {r['auc']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}] "
          f"(hint-use ref 0.735)", flush=True)

    # ---- frozen concept directions ----
    dz = np.load(ROOT / "results" / "concept_directions.npz")
    sel = json.load(open(ROOT / "results" / "concept_layer_selection.json", encoding="utf-8"))
    result["concept"] = {}
    for c in ["deception", "cheating", "hidden-info", "sycophancy"]:
        L = sel[c]["chosen_layer"]
        d = dz[f"{c}_L{L}"]
        df[f"concept_{c}"] = [float(np.dot(feats[s]["meanpool"][L], d))
                              if s in feats and L in feats[s]["meanpool"] else np.nan
                              for s in df.stem]
        r = _auc(df, f"concept_{c}")
        result["concept"][c] = r
        print(f"[concept_{c}] wrong-vs-correct AUC = {r['auc']:.3f} "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  (hint-use ref "
              f"{result['reference']['concept_hint_use'][c]})", flush=True)

    df.to_parquet(CONTROL, index=False)  # persist scores for the judge stage / audit
    json.dump(_clean(result), open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"-> {OUT}", flush=True)


def judge(cfg):
    from common import append_jsonl, chat_batch, openrouter_client

    m04 = _load("04_arm_judge")
    df = pd.read_parquet(CONTROL)
    jcfg = cfg["phase2"]["judge"]
    n = jcfg["samples_per_transcript"]
    cache_path = ROOT / "data" / "control" / "judge_control_raw.jsonl"
    cached = {(r["stem"], r["k"]): r for r in read_jsonl(cache_path)}

    todo = [(row, k) for _, row in df.iterrows() for k in range(n)
            if (row["stem"], k) not in cached]
    print(f"judge control: {len(cached)} cached, {len(todo)} to fetch", flush=True)
    if todo:
        client = openrouter_client(cfg)

        def on_result(key, texts):
            stem, k = key
            if texts is None:
                return
            append_jsonl(cache_path, {"stem": stem, "k": k,
                                      "score": m04.parse_score(texts[0]), "text": texts[0]})

        # unhinted transcripts -> judge_prompt uses just the question block (no hint,
        # no redaction marker) — exactly what a deployed monitor sees with no hint.
        jobs = [((row["stem"], k),
                 [{"role": "user", "content": m04.judge_prompt(cfg, row, "judge_hintblind")}])
                for row, k in todo]
        asyncio.run(chat_batch(client, cfg["models"]["judge"], jobs,
                               temperature=jcfg["temperature"], max_tokens=jcfg["max_tokens"],
                               n=1, concurrency=cfg["serving"]["max_concurrency"],
                               on_result=on_result))
        cached = {(r["stem"], r["k"]): r for r in read_jsonl(cache_path)}

    def mean_score(stem):
        vals = [cached[(stem, k)]["score"] for k in range(n)
                if (stem, k) in cached and cached[(stem, k)]["score"] is not None]
        return float(np.mean(vals)) if vals else np.nan
    df["judge_hintblind"] = [mean_score(s) for s in df.stem]
    r = _auc(df, "judge_hintblind")
    result = json.load(open(OUT, encoding="utf-8")) if OUT.exists() else {}
    result["judge_hintblind"] = r
    result["reference"]["judge_hint_use_auc"] = 0.895
    json.dump(_clean(result), open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"[judge_hintblind] wrong-vs-correct AUC = {r['auc']:.3f} "
          f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  (hint-use ref 0.895)", flush=True)
    print(f"-> {OUT}", flush=True)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


if __name__ == "__main__":
    stages = [a for a in sys.argv[1:]] or ["build", "cache", "score"]
    cfg = load_config()
    for st in stages:
        {"build": build, "cache": cache, "score": score, "judge": judge}[st](cfg)
