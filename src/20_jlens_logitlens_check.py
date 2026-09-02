"""Diagnostic: is the J-Lens readout decoding anything at layers 16/32/48?

Compares, on the same cached activations, the top-10 next-token hit rate of
  (a) the J-Lens readout used by the arm:  topk(W_U @ rmsnorm(J_L @ h))
  (b) a plain logit lens (no Jacobian):    topk(W_U @ rmsnorm(h))
and records what each shows at the position right before the CoT's final
answer letter (where the model is about to emit "Answer: X").

Usage: python src/20_jlens_logitlens_check.py [--acts DIR] [--n N]
Writes results/jlens_logitlens_check.json. Needs torch + the lens/unembed
downloads (same as 19). Seeded sample of N transcripts per class (seed 106).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, default=ROOT / "data" / "hf" / "activations-full" / "activations")
    ap.add_argument("--n", type=int, default=2, help="transcripts per class")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    spec = importlib.util.spec_from_file_location("m07", HERE / "07_arm_jlens.py")
    m07 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m07)
    cfg = m07.load_config()
    k = m07.jcfg(cfg)["top_k"]
    lens = m07.load_lens(cfg)
    w_u, norm_w, eps, _ = m07.load_unembed(cfg)
    tok = AutoTokenizer.from_pretrained(cfg["models"]["policy"])
    letter_ids = set(sum((m07.letter_token_ids(tok, L) for L in "ABCD"), []))

    scores = pd.read_parquet(ROOT / "data" / "hf" / "core" / "scores_clean.parquet")
    roster = scores[scores.arm == "jlens"]
    have = roster[roster.qid.map(lambda q: (args.acts / f"{q}.npz").exists())]
    sample = pd.concat([have[have.label == c].sample(n=args.n, random_state=106)
                        for c in ["POS", "NEG-inert", "NEG-clean"]])

    eye = torch.eye(int(lens["d_model"]))
    out = {"n_transcripts": len(sample), "per_transcript": [], "by_layer": {}}
    agg = {}
    for _, row in sample.iterrows():
        data = np.load(args.acts / f"{row.qid}.npz")
        tok_ids = data["token_ids"]
        nxt = tok_ids[1:]
        letter_pos = [int(p) for p in np.where(np.isin(tok_ids, list(letter_ids)))[0] if p > 0]
        q = letter_pos[-1] if letter_pos else None
        rec = {"qid": row.qid, "label": row.label, "span": int(len(tok_ids)),
               "answer_context": tok.decode(tok_ids[max(0, q - 6):q + 1]) if q else None, "layers": {}}
        layers = sorted(int(key.split("_")[1]) for key in data.files if key.startswith("layer_"))
        for L in layers:
            h = data[f"layer_{L}"]
            res = {}
            for name, J in (("jlens", m07.lens_row(lens, L)), ("logitlens", eye)):
                ids = m07.lens_topk_ids(J, h, norm_w, eps, w_u, k)          # [span, k]
                hit10 = float((ids[:-1] == nxt[:, None]).any(-1).mean())
                hit1 = float((ids[:-1, 0] == nxt).mean())
                res[name] = {"next_top1": round(hit1, 4), "next_top10": round(hit10, 4)}
                if q:
                    res[name]["readout_before_answer"] = [tok.decode([int(t)]) for t in ids[q - 1]]
                    res[name]["letter_in_top10_before_answer"] = bool(np.isin(ids[q - 1], list(letter_ids)).any())
                a = agg.setdefault((L, name), [])
                a.append((hit1, hit10, res[name].get("letter_in_top10_before_answer")))
            rec["layers"][str(L)] = res
        out["per_transcript"].append(rec)
        print(row.qid, row.label, {L: {n: rec["layers"][str(L)][n]["next_top10"] for n in ("jlens", "logitlens")} for L in layers}, flush=True)
    for (L, name), vals in agg.items():
        out["by_layer"].setdefault(str(L), {})[name] = {
            "mean_next_top1": round(float(np.mean([v[0] for v in vals])), 4),
            "mean_next_top10": round(float(np.mean([v[1] for v in vals])), 4),
            "frac_letter_in_top10_before_answer": round(float(np.mean([bool(v[2]) for v in vals])), 3)}
    path = ROOT / "results" / "jlens_logitlens_check.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["by_layer"], indent=1))
    print("->", path)


if __name__ == "__main__":
    main()
