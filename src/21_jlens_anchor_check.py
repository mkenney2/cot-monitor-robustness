"""Pod-side: does the J-Lens readout decode at all, and which layer-index
convention is right?  (Closes the two checks 07_arm_jlens.py `verify` left open.)

Loads the policy model (bf16, GPU) and, for a few seeded canonical transcripts,
runs the SAME forced forward pass 06_arm_probe.py used for caching. Then, for
every hidden_states index h in H_IDX, compares against the model's OWN logits:

  logitlens[h]      = topk(W_U @ rmsnorm(hidden_states[h]))          (no Jacobian)
  jlens[h, J=h]     = topk(W_U @ rmsnorm(J[h]   @ hidden_states[h]))  (07's convention)
  jlens[h, J=h-1]   = topk(W_U @ rmsnorm(J[h-1] @ hidden_states[h]))  (off-by-one alt.)

Metrics per readout, over the CoT span: top-1 agreement with the model's own
argmax, top-10 hit rate of the model's argmax, top-10 hit rate of the ACTUAL
next token, and whether an A-D letter sits in the top-10 right before the
final "Answer: X".

Decisive expectations:
  - logitlens at h = num_hidden_layers (final residual) must agree ~100% with
    the model (it IS the model's head). If not, the readout formula is wrong.
  - the lens anchor (J[target_layer] = I) should behave like the logit lens at
    whichever h the lens means by "layer target_layer" -> that fixes the
    convention (h == L or h == L+1).
  - the next-token hit rate should rise smoothly toward the top as h grows.

Usage (pod):
  cd /workspace/mats-application && source /workspace/venv/bin/activate
  python src/21_jlens_anchor_check.py [--n 3]
Writes results/jlens_anchor_check.json and prints the table.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="seeded transcripts (one per class, cycling)")
    args = ap.parse_args()

    import torch
    from phase2_common import load_dataset

    m06, m07 = load("06_arm_probe"), load("07_arm_jlens")
    cfg = m06.load_config()
    k = m07.jcfg(cfg)["top_k"]
    tokenizer, model, _ = m06.load_model(cfg)
    n_layers = model.config.num_hidden_layers
    lens = m07.load_lens(cfg)
    J = lens["J"]
    w_u, norm_w, eps, _ = m07.load_unembed(cfg)
    w_u, norm_w = w_u.cuda(), norm_w.cuda()
    letter_ids = set(sum((m07.letter_token_ids(tokenizer, L) for L in "ABCD"), []))
    tgt = lens.get("provenance", {}).get("target_layer")

    df = load_dataset()
    picks = []
    for i, c in enumerate(["POS", "NEG-inert", "NEG-clean"] * args.n):
        if len(picks) >= args.n:
            break
        picks.append(df[df.label == c].sample(n=1, random_state=106 + i).iloc[0])

    H_IDX = sorted({16, 32, 48, 56, 60, 61, 62, 63, n_layers} | ({tgt, tgt + 1} if tgt is not None else set()))
    H_IDX = [h for h in H_IDX if 0 <= h <= n_layers]
    eye = torch.eye(int(lens["d_model"]))

    def metrics(ids, model_top1, nxt, q):
        return {"top1_agree_model": round(float((ids[:, 0] == model_top1).mean()), 4),
                "top10_hit_model_argmax": round(float((ids == model_top1[:, None]).any(-1).mean()), 4),
                "top10_hit_actual_next": round(float((ids[:-1] == nxt[:, None]).any(-1).mean()), 4),
                "letter_in_top10_before_answer": (bool(np.isin(ids[q - 1], list(letter_ids)).any()) if q else None)}

    out = {"num_hidden_layers": n_layers, "lens_target_layer": tgt, "lens_J_keys": sorted(int(x) for x in J),
           "h_idx": H_IDX, "per_transcript": []}
    for row in picks:
        chat = m06.build_chat(cfg, row)
        prompt_ids = tokenizer.apply_chat_template(chat[:2], tokenize=True, add_generation_prompt=True,
                                                   enable_thinking=cfg["prompting"]["enable_thinking"],
                                                   return_dict=False)
        cot_ids = tokenizer.encode(row["canonical_cot"], add_special_tokens=False)
        full = prompt_ids + cot_ids
        s0, s1 = len(prompt_ids), len(full)
        with torch.no_grad():
            o = model(torch.tensor([full], device=model.device), output_hidden_states=True)
        logits = o.logits[0, s0:s1].float()
        model_top1 = logits.argmax(-1).cpu().numpy()
        span_tok = np.array(full[s0:s1])
        nxt = span_tok[1:]
        lp = [int(p) for p in np.where(np.isin(span_tok, list(letter_ids)))[0] if p > 0]
        q = lp[-1] if lp else None
        rec = {"qid": row["qid"], "label": row["label"], "span": int(s1 - s0),
               "answer_context": tokenizer.decode(span_tok[max(0, (q or 6) - 6):(q or 0) + 1]) if q else None,
               "model_top10_before_answer": [tokenizer.decode([int(t)]) for t in logits[q - 1].topk(10).indices] if q else None,
               "readouts": {}}
        for h in H_IDX:
            hs = o.hidden_states[h][0, s0:s1].to(torch.float16).cpu().numpy()   # exactly what 06 cached
            r = {}
            ids = m07.lens_topk_ids(eye, hs, norm_w, eps, w_u, k)
            r["logitlens"] = metrics(ids, model_top1, nxt, q)
            for name, key in (("jlens_J=h", h), ("jlens_J=h-1", h - 1)):
                if key in J:
                    ids = m07.lens_topk_ids(J[key], hs, norm_w, eps, w_u, k)
                    r[name] = metrics(ids, model_top1, nxt, q)
                    if h in (48, n_layers) and q:
                        r[name]["readout_before_answer"] = [tokenizer.decode([int(t)]) for t in ids[q - 1]]
            rec["readouts"][str(h)] = r
        out["per_transcript"].append(rec)
        del o
        torch.cuda.empty_cache()
        print(f"{row['qid']} [{row['label']}] span={s1 - s0}", flush=True)
        for h in H_IDX:
            r = rec["readouts"][str(h)]
            print(f"  h={h:2d} " + "  ".join(f"{n}: agree={v['top1_agree_model']:.2f} next10={v['top10_hit_actual_next']:.2f}"
                                             for n, v in r.items()), flush=True)

    # mean table
    summary = {}
    for h in H_IDX:
        for name in ("logitlens", "jlens_J=h", "jlens_J=h-1"):
            vals = [t["readouts"][str(h)].get(name) for t in out["per_transcript"]]
            vals = [v for v in vals if v]
            if vals:
                summary.setdefault(str(h), {})[name] = {
                    m: round(float(np.mean([v[m] for v in vals])), 4)
                    for m in ("top1_agree_model", "top10_hit_model_argmax", "top10_hit_actual_next")}
                summary[str(h)][name]["frac_letter_before_answer"] = round(float(np.mean(
                    [bool(v["letter_in_top10_before_answer"]) for v in vals if v["letter_in_top10_before_answer"] is not None] or [0])), 3)
    out["summary"] = summary
    path = ROOT / "results" / "jlens_anchor_check.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\nMEAN over transcripts (agree = top-1 agreement with the model's own argmax; next10 = actual next token in top-10):")
    for h, d in summary.items():
        print(f"  h={h:>2s} " + "  ".join(f"{n}: agree={v['top1_agree_model']:.2f} next10={v['top10_hit_actual_next']:.2f} "
                                          f"letter@ans={v['frac_letter_before_answer']:.2f}" for n, v in d.items()))
    print("->", path)


if __name__ == "__main__":
    main()
