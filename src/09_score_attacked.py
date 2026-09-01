"""Phase 3 — re-score the four monitors on attacked transcripts, FROZEN.

No retraining on attacked data. Trainable arms (textfeat, probe) are fit on the
CLEAN train split (deterministic — identical to the frozen clean model) and
applied to the rewritten transcripts. Judge is stateless. J-Lens uses its frozen
token sets. Activations for probe/J-Lens come from data/activations_attacked/
(06 cache_attacked), keyed {qid}__{attack}.

Usage (judge needs OPENROUTER_API_KEY; probe/jlens need attacked activations):
  python src/09_score_attacked.py            # all arms
  python src/09_score_attacked.py judge textfeat   # subset

Output: results/scores_attacked.parquet  (qid, attack, arm, label, split, score)
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (ROOT, append_jsonl, chat_batch, load_config, openrouter_client,
                    read_jsonl)
from phase2_common import auc_with_ci

ATTACKED = ROOT / "data" / "attacked" / "attacked_transcripts.parquet"
SCORES = ROOT / "results" / "scores_attacked.parquet"
JUDGE_CACHE = ROOT / "data" / "attacked" / "judge_attacked_raw.jsonl"
JLENS_CACHE = ROOT / "data" / "attacked" / "jlens_llm_attacked_raw.jsonl"
ACT_ATTACKED = ROOT / "data" / "activations_attacked"


def _load(mod):
    """Import a numeric-prefixed arm module (can't be a normal import)."""
    spec = importlib.util.spec_from_file_location(mod, ROOT / "src" / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def clean_df():
    from phase2_common import load_dataset
    return load_dataset()


def save(arm, rows):
    df = pd.DataFrame(rows)
    df["arm"] = arm
    if SCORES.exists():
        prev = pd.read_parquet(SCORES)
        prev = prev[prev.arm != arm]
        df = pd.concat([prev, df], ignore_index=True)
    SCORES.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SCORES, index=False)


def report(arm, att):
    """Attacked AUC per attack (POS vs NEG-inert, test split). The saved score
    rows already carry label/split — no merge needed."""
    cfg = load_config()
    s = pd.read_parquet(SCORES)
    s = s[s.arm == arm]
    print(f"\n=== {arm} (attacked) ===", flush=True)
    for attack in sorted(att.attack.unique()):
        sub = s[(s.attack == attack) & (s.split == "test")
                & s.label.isin(["POS", "NEG-inert"])].dropna(subset=["score"])
        if sub.label.nunique() < 2:
            print(f"  {attack:11s}: n/a", flush=True)
            continue
        r = auc_with_ci((sub.label == "POS").astype(int), sub.score,
                        cfg["seeds"]["bootstrap"])
        print(f"  {attack:11s}: AUC {r['auc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
              f"  (n={r['n_pos']}+{r['n_neg']})", flush=True)


# ------------------------------------------------------------------- judge arm

def score_judge(cfg, att):
    m04 = _load("04_arm_judge")
    judge = cfg["models"]["judge"]
    n = cfg["phase2"]["judge"]["samples_per_transcript"]
    clean = clean_df().set_index("qid")
    cached = {(r["qid"], r["variant"], r["attack"], r["k"]): r
              for r in read_jsonl(JUDGE_CACHE)}

    # Build one row-view per attacked transcript carrying the rewritten CoT so
    # 04's judge_prompt (which reads canonical_cot / hint_text / question_block)
    # scores the ATTACKED text.
    def row_for(a):
        base = clean.loc[a["qid"]]
        r = base.to_dict()
        r["canonical_cot"] = a["canonical_cot"]
        return r

    todo = []
    for _, a in att.iterrows():
        for variant in m04.VARIANTS:
            for k in range(n):
                if (a["qid"], variant, a["attack"], k) not in cached:
                    todo.append((a, variant, k))
    print(f"judge attacked: {len(cached)} cached samples, {len(todo)} to fetch",
          flush=True)
    if todo:
        client = openrouter_client(cfg)

        def on_result(key, texts):
            qid, variant, attack, k = key
            if texts is None:
                return
            append_jsonl(JUDGE_CACHE, {"qid": qid, "variant": variant, "attack": attack,
                                       "k": k, "score": m04.parse_score(texts[0]),
                                       "text": texts[0]})

        jobs = [((a["qid"], variant, a["attack"], k),
                 [{"role": "user", "content": m04.judge_prompt(cfg, row_for(a), variant)}])
                for a, variant, k in todo]
        asyncio.run(chat_batch(
            client, judge, jobs, temperature=cfg["phase2"]["judge"]["temperature"],
            max_tokens=cfg["phase2"]["judge"]["max_tokens"], n=1,
            concurrency=cfg["serving"]["max_concurrency"], on_result=on_result))
        cached = {(r["qid"], r["variant"], r["attack"], r["k"]): r
                  for r in read_jsonl(JUDGE_CACHE)}

    for variant in m04.VARIANTS:
        rows = []
        for _, a in att.iterrows():
            vals = [cached.get((a["qid"], variant, a["attack"], k), {}).get("score")
                    for k in range(n)]
            vals = [v for v in vals if v is not None]
            rows.append({"qid": a["qid"], "attack": a["attack"], "label": a["label"],
                         "split": a["split"],
                         "score": float(np.mean(vals)) if vals else np.nan})
        save(variant, rows)
        report(variant, att)


# ---------------------------------------------------------------- textfeat arm

def score_textfeat(cfg, att):
    m05 = _load("05_arm_textfeat")
    model = m05.fit_textfeat(cfg, clean_df())          # frozen on clean train
    scores = m05.score_textfeat(model, att["canonical_cot"].fillna("").tolist())
    rows = [{"qid": a.qid, "attack": a.attack, "label": a.label, "split": a.split,
             "score": float(s)} for (_, a), s in zip(att.iterrows(), scores)]
    save("textfeat", rows)
    report("textfeat", att)


# ------------------------------------------------------------------- probe arm

def score_probe(cfg, att):
    import joblib

    m06 = _load("06_arm_probe")
    if not m06.PROBE_MODEL.exists():
        raise SystemExit("run 06_arm_probe.py train first (persists probe_model.joblib)")
    saved = joblib.load(m06.PROBE_MODEL)
    probe, layer = saved["probe"], saved["layer"]
    stems = [f"{a.qid}__{a.attack}" for _, a in att.iterrows()]
    feats = m06.load_features(stems, act_dir=ACT_ATTACKED)
    rows = []
    for _, a in att.iterrows():
        stem = f"{a.qid}__{a.attack}"
        f = feats.get(stem)
        score = (float(probe.predict_proba(f["meanpool"][layer][None])[0, 1])
                 if f and layer in f["meanpool"] else np.nan)
        rows.append({"qid": a.qid, "attack": a.attack, "label": a.label,
                     "split": a.split, "score": score})
    n_missing = sum(1 for r in rows if np.isnan(r["score"]))
    if n_missing:
        print(f"probe attacked: {n_missing} transcripts missing activations (NaN)",
              flush=True)
    save("probe", rows)
    report("probe", att)


# ------------------------------------------------------------------ jlens arm

def score_jlens(cfg, att):
    m07 = _load("07_arm_jlens")
    import json

    import torch
    from transformers import AutoTokenizer

    lens = m07.load_lens(cfg)
    w_u, norm_w, eps, _ = m07.load_unembed(cfg)
    if torch.cuda.is_available():
        w_u, norm_w = w_u.cuda(), norm_w.cuda()
    tok = AutoTokenizer.from_pretrained(cfg["models"]["policy"])
    k = m07.jcfg(cfg)["top_k"]
    letter_window = m07.jcfg(cfg).get("letter_window", 5)

    with open(m07.TOKEN_SETS, encoding="utf-8") as f:  # FROZEN on clean train
        frozen = json.load(f)["frozen_type_words"]
    all_words = sorted({w for ws in m07.CANDIDATE_WORDS.values() for w in ws})
    word_ids = {w: m07.word_token_ids(tok, w) for w in all_words}
    dec_cache = {}

    rows = []
    for _, a in att.iterrows():
        path = ACT_ATTACKED / f"{a.qid}__{a.attack}.npz"
        if not path.exists():
            rows.append({"qid": a.qid, "attack": a.attack, "label": a.label,
                         "split": a.split, "score": np.nan})
            continue
        data = np.load(path)
        layers = sorted(int(x.split("_")[1]) for x in data.files if x.startswith("layer_"))
        topk = np.stack([m07.lens_topk_ids(m07.lens_row(lens, L), data[f"layer_{L}"],
                                           norm_w, eps, w_u, k) for L in layers])
        word_w = {w: m07.set_weight(topk, word_ids[w]) for w in all_words}
        if isinstance(a.hint_letter, str) and a.hint_letter:
            lids = m07.letter_token_ids(tok, a.hint_letter)
            decoded = m07.decode_ids(tok, data["token_ids"], dec_cache)
            mask = m07.letter_mention_mask(decoded, a.hint_letter, letter_window)
            letter_w = m07.set_weight(topk, lids, pos_mask=mask)
        else:
            letter_w = 0.0
        score = letter_w + sum(word_w[w] for w in frozen.get(a.hint_type, []))
        rows.append({"qid": a.qid, "attack": a.attack, "label": a.label,
                     "split": a.split, "score": float(score)})
    save("jlens", rows)
    report("jlens", att)


ARMS = {"judge": score_judge, "textfeat": score_textfeat,
        "probe": score_probe, "jlens": score_jlens}


def main():
    cfg = load_config()
    if not ATTACKED.exists():
        raise SystemExit(f"{ATTACKED} missing — run 08_attacks.py build first")
    att = pd.read_parquet(ATTACKED)
    which = [a for a in sys.argv[1:] if a in ARMS] or list(ARMS)
    print(f"scoring attacked arms: {which} over {len(att)} transcripts", flush=True)
    for arm in which:
        ARMS[arm](cfg, att)


if __name__ == "__main__":
    main()
