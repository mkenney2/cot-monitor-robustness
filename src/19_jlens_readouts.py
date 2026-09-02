"""Save the J-Lens per-position top-k readouts as an artifact.

07_arm_jlens.py collapses the lens readout to one scalar per transcript and
throws the top-k token lists away (only a 10-transcript review dump survives).
This stage recomputes exactly the same readout — same lens, same unembedding,
same `lens_topk_ids` — and SAVES it, so the readouts can be audited offline
(notebooks/jlens_readout_audit.ipynb) without a GPU, the lens, or the model.

Usage:
  python src/19_jlens_readouts.py [--acts DIR] [--out DIR] [--qids a,b,c]
                                  [--limit N] [--force]

  --acts   directory of {stem}.npz per-token activations (default: the
           `activations-full/activations` config of the HF dataset under
           data/hf/, falling back to data/activations/ in the study repo).
  --out    output directory (default results/jlens_readouts/).

Per input {stem}.npz it writes {out}/{stem}.npz with
  topk       [n_layers, span, k] int32  top-k vocab ids per layer x position
  layers     [n_layers]          int32  residual layers (16/32/48)
  token_ids  [span]              int32  the CoT token at each position
  meta       json string (stem, k, lens repo/variant, source npz meta)
and maintains {out}/vocab.json — {token_id: decoded string} for every id that
appears in any saved readout or CoT span, so the audit notebook needs no
tokenizer.

Cost: ~1e13 flops per transcript (the vocab-sized matmul) — seconds on a GPU,
~30 s per transcript on a 14-thread CPU. Resume-safe: existing outputs are
skipped unless --force.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))  # `common`, `phase2_common` for 07's imports


def load_m07():
    spec = importlib.util.spec_from_file_location("m07", HERE / "07_arm_jlens.py")
    m07 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m07)
    return m07


def default_acts_dir():
    cands = [ROOT / "data" / "hf" / "activations-full" / "activations",
             ROOT / "data" / "activations"]
    for c in cands:
        if c.exists() and any(c.glob("*.npz")):
            return c
    raise SystemExit("no activations found; pass --acts DIR (download the "
                     "`activations-full/activations` config of the HF dataset)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "jlens_readouts")
    ap.add_argument("--qids", default=None, help="comma-separated stems to process")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    m07 = load_m07()
    cfg = m07.load_config()
    j = m07.jcfg(cfg)
    k = j["top_k"]
    acts = args.acts or default_acts_dir()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in acts.glob("*.npz"))
    if args.qids:
        want = set(args.qids.split(","))
        stems = [s for s in stems if s in want]
    if not args.force:
        stems = [s for s in stems if not (out / f"{s}.npz").exists()]
    if args.limit:
        stems = stems[:args.limit]
    print(f"{len(stems)} transcripts to read out from {acts} -> {out}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["models"]["policy"])
    lens_meta = {"lens_repo": j["lens_repo"],
                 "lens_variant": j.get("lens_variant", "j-lens"),
                 "top_k": k, "unembed_model": cfg["models"]["policy"]}

    # Token ids for the arm's candidate words and answer letters, computed with
    # the same helpers 07 uses, so the notebook can re-derive the published
    # hint-salience score without a tokenizer.
    all_words = sorted({w for ws in m07.CANDIDATE_WORDS.values() for w in ws})
    ids_map = {"words": {w: m07.word_token_ids(tok, w) for w in all_words},
               "letters": {L: m07.letter_token_ids(tok, L) for L in "ABCD"},
               "candidates": m07.CANDIDATE_WORDS}
    (out / "token_ids_map.json").write_text(json.dumps(ids_map, indent=1), encoding="utf-8")

    vocab_path = out / "vocab.json"
    vocab = {}
    if vocab_path.exists():
        vocab = {int(a): b for a, b in json.loads(vocab_path.read_text(encoding="utf-8")).items()}

    def flush_vocab():
        vocab_path.write_text(json.dumps({str(a): b for a, b in sorted(vocab.items())},
                                         ensure_ascii=False), encoding="utf-8")

    # Self-heal: make sure every id in every existing readout is in vocab.json
    # (an interrupted earlier run can leave readouts written but vocab unflushed).
    healed = 0
    for p in sorted(out.glob("*.npz")):
        d = np.load(p)
        for tid in set(np.unique(d["topk"]).tolist()) | set(np.unique(d["token_ids"]).tolist()):
            if tid not in vocab:
                vocab[tid] = tok.decode([tid]); healed += 1
    if healed:
        flush_vocab()
        print(f"vocab.json: added {healed} ids missing from existing readouts", flush=True)
    if not stems:
        flush_vocab()
        return

    lens = m07.load_lens(cfg)
    w_u, norm_w, eps, _ = m07.load_unembed(cfg)
    if torch.cuda.is_available():
        w_u, norm_w = w_u.cuda(), norm_w.cuda()

    t_start = time.time()
    for i, stem in enumerate(stems, 1):
        t0 = time.time()
        data = np.load(acts / f"{stem}.npz")
        layers = sorted(int(key.split("_")[1]) for key in data.files if key.startswith("layer_"))
        topk = np.stack([
            m07.lens_topk_ids(m07.lens_row(lens, L), data[f"layer_{L}"], norm_w, eps, w_u, k)
            for L in layers
        ]).astype(np.int32)                                   # [n_layers, span, k]
        token_ids = data["token_ids"].astype(np.int32)
        src_meta = json.loads(str(data["meta"][0])) if "meta" in data.files else {}
        meta = {"stem": stem, **lens_meta, "layers": layers,
                "span_len": int(token_ids.shape[0]), "source_meta": src_meta}
        tmp = out / f"{stem}.tmp.npz"
        np.savez_compressed(tmp, topk=topk, layers=np.array(layers, dtype=np.int32),
                            token_ids=token_ids, meta=np.array([json.dumps(meta)]))
        tmp.replace(out / f"{stem}.npz")

        new_ids = set(np.unique(topk).tolist()) | set(np.unique(token_ids).tolist())
        for tid in new_ids - vocab.keys():
            vocab[tid] = tok.decode([tid])
        flush_vocab()      # every transcript: readouts on disk must never outrun the vocab
        el = time.time() - t_start
        print(f"[{i}/{len(stems)}] {stem} span={token_ids.shape[0]} "
              f"{time.time() - t0:.0f}s (elapsed {el / 60:.1f} min, "
              f"eta {el / i * (len(stems) - i) / 60:.0f} min)", flush=True)
    flush_vocab()
    print(f"done: {len(stems)} readouts -> {out}; vocab {len(vocab)} ids", flush=True)


if __name__ == "__main__":
    main()
