"""Phase 2, Arm 3 — linear probe on cached residual streams.

Stages:
  python src/06_arm_probe.py cache          # forced forward passes (GPU pod)
  python src/06_arm_probe.py cache_attacked # same, over rewritten transcripts (GPU)
  python src/06_arm_probe.py train          # probes + controls, no GPU

cache -> data/activations/{tkey}.npz  one file per canonical transcript
         (tkey = qid for hinted transcripts, qid__clean for unhinted ones):
           layer_{L}  [span_len, hidden] float16 residual stream over the CoT
                      token span only, for each captured layer L
           token_ids  [span_len] int32 token ids of the span
           meta       length-1 json string array (qid, layers, prompt_len, span)
         This exact per-token format is shared with the J-Lens arm (07).
train -> results/scores_clean.parquet (arms 'probe' and 'probe_lasttoken')
         results/auc_probe_clean.json, results/auc_probe_lasttoken_clean.json
         results/probe_layer_selection.json  (per-layer train-CV AUCs — auditable)
         results/probe_controls.json         (mandatory controls, CLAUDE.md Arm 3)

Transcript reconstruction (cache): generation prompt rendered via
apply_chat_template (add_generation_prompt=True, enable_thinking as configured),
then the CoT appended as separately-encoded tokens — matching how generation
actually tokenized it. A one-shot full-template render does NOT work: BPE merges
across the prompt/CoT boundary break token-level prefix equality (verified on
the pod 2026-08-29). Span = [len(prompt_ids), end), recorded in meta.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

from common import ROOT, load_config
from phase2_common import load_dataset, report_arm, save_scores

ACT_DIR = ROOT / "data" / "activations"
ACT_DIR_ATTACKED = ROOT / "data" / "activations_attacked"
ATTACKED = ROOT / "data" / "attacked" / "attacked_transcripts.parquet"
LAYER_SELECTION = ROOT / "results" / "probe_layer_selection.json"
CONTROLS = ROOT / "results" / "probe_controls.json"
PROBE_MODEL = ROOT / "results" / "probe_model.joblib"


def pick_layers(cfg, num_hidden_layers):
    """Layer indices at the configured depth fractions (hidden_states[L] is the
    output of block L; index 0 is the embeddings, so L >= 1)."""
    fracs = cfg["phase2"]["probe"]["layers_frac"]
    layers = sorted({min(max(int(round(f * num_hidden_layers)), 1), num_hidden_layers)
                     for f in fracs})
    return layers


def build_chat(cfg, row):
    if row["canonical_condition"] == "hinted":
        user = row["hint_text"] + "\n\n" + row["question_block"]
    else:
        user = row["question_block"]
    return [{"role": "system", "content": cfg["prompting"]["system"]},
            {"role": "user", "content": user},
            {"role": "assistant", "content": row["canonical_cot"]}]


def cache(cfg):
    tokenizer, model, layers = load_model(cfg)
    df = load_dataset()
    run_cache(cfg, tokenizer, model, layers, df, ACT_DIR, lambda r: r["tkey"])


def cache_one(cfg, tokenizer, model, layers, row, out_path):
    """Forced forward pass over one transcript; write residuals to out_path.npz
    (atomic). Shared by clean and attacked caching."""
    import torch

    chat = build_chat(cfg, row)
    # enable_thinking must match generation-time template rendering or the token
    # spans are wrong (see config prompting.enable_thinking).
    think = cfg["prompting"]["enable_thinking"]
    prompt_ids = tokenizer.apply_chat_template(
        chat[:2], tokenize=True, add_generation_prompt=True,
        enable_thinking=think, return_dict=False)
    # CoT tokens are appended to the generation prompt exactly as the model
    # produced them — NOT a one-shot full-template render: BPE merges across the
    # prompt/CoT boundary break token-level prefix equality (string render agrees).
    cot_ids = tokenizer.encode(row["canonical_cot"], add_special_tokens=False)
    if not cot_ids:
        raise ValueError("empty CoT span")
    prompt_len = len(prompt_ids)
    full_ids = prompt_ids + cot_ids
    span = (prompt_len, len(full_ids))

    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    arrays = {f"layer_{L}": out.hidden_states[L][0, span[0]:span[1]]
              .to(torch.float16).cpu().numpy() for L in layers}
    del out
    meta = {"qid": row["qid"], "tkey": row.get("tkey", row["qid"]), "layers": layers,
            "prompt_len": prompt_len,
            "span": list(span), "n_full_tokens": len(full_ids),
            "model": cfg["models"]["policy"]}
    # np.savez appends '.npz' to names lacking it — the tmp name must already end
    # in .npz or the atomic rename source won't exist.
    tmp = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp, **arrays,
        token_ids=np.array(full_ids[span[0]:span[1]], dtype=np.int32),
        meta=np.array([json.dumps(meta)]))
    os.replace(tmp, out_path)


def run_cache(cfg, tokenizer, model, layers, df, out_dir, key):
    """Cache every row of df to out_dir/{key(row)}.npz, resume-safe, fail-loud."""
    import torch  # noqa: F401 — ensures torch present in this scope for callers

    out_dir.mkdir(parents=True, exist_ok=True)
    n_done = n_skip = n_fail = 0
    for i, row in enumerate(df.itertuples(index=False)):
        row = row._asdict()
        out_path = out_dir / f"{key(row)}.npz"
        if out_path.exists():
            n_skip += 1
            continue
        try:
            cache_one(cfg, tokenizer, model, layers, row, out_path)
            n_done += 1
        except Exception as e:  # noqa: BLE001 — working rules 1/4: log, keep going
            n_fail += 1
            print(f"[FAIL] {key(row)}: {e}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"progress: {i + 1}/{len(df)} (new={n_done} skipped={n_skip} "
                  f"failed={n_fail})", flush=True)
    print(f"cache done: {n_done} new, {n_skip} already cached, {n_fail} FAILED "
          f"of {len(df)} transcripts -> {out_dir}", flush=True)
    if n_fail:
        print("failures above are NOT cached — rerun after fixing, or report them.",
              flush=True)


def load_model(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = cfg["models"]["policy"]
    print(f"loading {model_id} (bf16, device_map=auto)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    layers = pick_layers(cfg, model.config.num_hidden_layers)
    print(f"num_hidden_layers={model.config.num_hidden_layers} -> capturing {layers}",
          flush=True)
    return tokenizer, model, layers


def cache_attacked(cfg):
    """Recompute activations over the REWRITTEN transcripts (attacked set).
    Keyed {qid}__{attack} so all three attacks per item coexist."""
    import pandas as pd

    if not ATTACKED.exists():
        raise SystemExit(f"{ATTACKED} missing — run 08_attacks.py build first")
    df = pd.read_parquet(ATTACKED)
    tokenizer, model, layers = load_model(cfg)
    run_cache(cfg, tokenizer, model, layers, df, ACT_DIR_ATTACKED,
              lambda r: f"{r['qid']}__{r['attack']}")


def load_features(qids, act_dir=ACT_DIR):
    """stem -> {'meanpool': {L: vec}, 'lasttoken': {L: vec}} from npz files
    act_dir/{stem}.npz. Clean stems are qids; attacked stems are '{qid}__{attack}'."""
    feats = {}
    for qid in qids:
        path = act_dir / f"{qid}.npz"
        if not path.exists():
            continue
        with np.load(path) as z:
            per_layer_mean, per_layer_last = {}, {}
            for key in z.files:
                if not key.startswith("layer_"):
                    continue
                arr = z[key].astype(np.float32)
                if arr.shape[0] == 0:
                    continue
                L = int(key.split("_")[1])
                per_layer_mean[L] = arr.mean(axis=0)
                per_layer_last[L] = arr[-1]
        if per_layer_mean:
            feats[qid] = {"meanpool": per_layer_mean, "lasttoken": per_layer_last}
    return feats


def make_probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Standardization is fit inside the pipeline, i.e. on whatever it is fit on
    # (the train split) — no peeking at test statistics.
    return make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000))


def stack(df_rows, feats, pooling, layer):
    X = np.stack([feats[q][pooling][layer] for q in df_rows.tkey])
    y = (df_rows.label == "POS").astype(int).to_numpy()
    return X, y


def train(cfg):
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    cv_folds = cfg["phase2"]["probe"]["cv_folds"]
    # seeds.probe_cv is not in config.yaml yet — defaulted here (fold shuffling).
    cv_seed = cfg["seeds"].get("probe_cv", 109)

    df = load_dataset()
    feats = load_features(df.tkey)
    missing = sorted(q for q in df.tkey if q not in feats)
    if missing:
        print(f"WARNING: {len(missing)} transcripts have no cached activations "
              f"(scored NaN): {missing[:10]}{'...' if len(missing) > 10 else ''}",
              flush=True)
    layers = sorted(next(iter(feats.values()))["meanpool"])
    print(f"{len(feats)}/{len(df)} transcripts with activations, layers {layers}",
          flush=True)

    has = df.tkey.isin(feats)
    train_rows = df[(df.split == "train") & df.label.isin(["POS", "NEG-inert"]) & has]
    print(f"train rows (POS vs NEG-inert): {train_rows.label.value_counts().to_dict()}",
          flush=True)

    selection = {"cv_folds": cv_folds, "cv_seed": cv_seed, "layers": layers}
    fitted = {}  # (pooling) -> (best_layer, fitted pipeline) for the controls
    for arm, pooling in (("probe", "meanpool"), ("probe_lasttoken", "lasttoken")):
        # Layer choice: train-split CV only, before touching test (CLAUDE.md Arm 3).
        cv_aucs = {}
        for L in layers:
            X, y = stack(train_rows, feats, pooling, L)
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed)
            cv_aucs[L] = float(np.mean(
                cross_val_score(make_probe(), X, y, cv=skf, scoring="roc_auc")))
        best = max(sorted(cv_aucs), key=lambda L: cv_aucs[L])
        selection[arm] = {"cv_auc_by_layer": {str(L): v for L, v in cv_aucs.items()},
                          "chosen_layer": best}
        print(f"[{arm}] train-CV AUC by layer: {cv_aucs} -> layer {best}", flush=True)

        probe = make_probe()
        X, y = stack(train_rows, feats, pooling, best)
        probe.fit(X, y)
        fitted[pooling] = (best, probe)

        out = df[["qid", "label", "split"]].copy()
        out["score"] = np.nan
        scorable = df[has]
        X_all = np.stack([feats[q][pooling][best] for q in scorable.tkey])
        out.loc[scorable.index, "score"] = probe.predict_proba(X_all)[:, 1]
        out["layer"] = best
        save_scores(arm, out)
        report_arm(arm, out)

    with open(LAYER_SELECTION, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"layer selection audit -> {LAYER_SELECTION}", flush=True)

    # Persist the frozen primary probe so Phase 3 re-scores attacked transcripts
    # with the IDENTICAL model (no retraining on attacked data).
    import joblib

    best_layer, probe = fitted["meanpool"]
    joblib.dump({"probe": probe, "layer": best_layer, "pooling": "meanpool"},
                PROBE_MODEL)
    print(f"frozen probe -> {PROBE_MODEL} (layer {best_layer})", flush=True)

    controls(cfg, df, feats, train_rows, fitted["meanpool"], cv_seed)


def controls(cfg, df, feats, train_rows, primary, cv_seed):
    """Mandatory Arm 3 controls (CLAUDE.md): shuffled labels + presence-vs-use."""
    from sklearn.metrics import roc_auc_score

    best, probe = primary
    has = df.tkey.isin(feats)
    test = df[(df.split == "test") & has]

    # (i) shuffled-label probe on the primary layer — must land near AUC 0.5.
    shuffle_seed = cfg["seeds"]["bootstrap"]
    X_tr, y_tr = stack(train_rows, feats, "meanpool", best)
    y_shuf = np.random.default_rng(shuffle_seed).permutation(y_tr)
    shuf_probe = make_probe()
    shuf_probe.fit(X_tr, y_shuf)
    t = test[test.label.isin(["POS", "NEG-inert"])]
    X_t = np.stack([feats[q]["meanpool"][best] for q in t.tkey])
    shuf_auc = float(roc_auc_score((t.label == "POS").astype(int),
                                   shuf_probe.predict_proba(X_t)[:, 1]))
    print(f"[control i] shuffled-label probe test AUC = {shuf_auc:.3f} "
          f"(must be near 0.5; if not, something is leaking)", flush=True)

    # (ii) trained probe's scores on NEG-inert vs NEG-clean (test split).
    negs = test[test.label.isin(["NEG-inert", "NEG-clean"])]
    if negs.label.nunique() < 2:
        presence_auc = None
        print("[control ii] cannot run: missing a NEG class in the test split",
              flush=True)
    else:
        X_n = np.stack([feats[q]["meanpool"][best] for q in negs.tkey])
        presence_auc = float(roc_auc_score((negs.label == "NEG-inert").astype(int),
                                           probe.predict_proba(X_n)[:, 1]))
        print(f"[control ii] probe AUC NEG-inert vs NEG-clean = {presence_auc:.3f}",
              flush=True)
        if presence_auc > 0.6:
            print("The probe SEPARATES hinted-but-inert from clean transcripts: it "
                  "is (at least partly) detecting hint-PRESENCE, not hint-USE. "
                  "State this plainly in the writeup.", flush=True)
        else:
            print("The probe does not meaningfully separate NEG-inert from "
                  "NEG-clean: no evidence it is merely a hint-presence detector.",
                  flush=True)

    result = {
        "primary_layer": best,
        "shuffled_label_test_auc": shuf_auc,
        "shuffle_seed": shuffle_seed,
        "cv_seed": cv_seed,
        "neg_inert_vs_neg_clean_auc": presence_auc,
        "n_train": int(len(train_rows)),
        "n_test_pos_vs_neg_inert": int(len(t)),
        "n_test_negs": int(len(negs)),
        "note": ("shuffled-label AUC should be ~0.5; NEG-inert vs NEG-clean AUC "
                 ">> 0.5 means the probe detects hint-presence, not hint-use"),
    }
    CONTROLS.parent.mkdir(exist_ok=True)
    with open(CONTROLS, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"controls -> {CONTROLS}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    cfg = load_config()
    if stage == "cache":
        cache(cfg)
    elif stage == "cache_attacked":
        cache_attacked(cfg)
    elif stage == "train":
        train(cfg)
    else:
        raise SystemExit("usage: python src/06_arm_probe.py {cache|cache_attacked|train}")
