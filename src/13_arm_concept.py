"""Phase 4, stretch Arm 5 — concept-direction probe (RepE-style diff-of-means).

Fits UNSUPERVISED concept directions (deception, cheating, hidden-info,
sycophancy) from a frozen contrast set (data/concept_contrast_set.json) — no task
labels touch the direction. Scores each transcript by projecting its mean-pooled
CoT activation onto the (unit) direction. Answers: does a GENERIC concept
representation detect silent hint-use, and is it more/less attackable than the
task-trained probe?

Reuses 06_arm_probe's cached transcript activations (data/activations/{qid}.npz,
via load_features) — the SAME meanpool vectors the supervised probe reads. Only
the contrast statements need a fresh forward pass (GPU).

Task labels are used ONLY to (a) pick the read-out layer per concept on the TRAIN
split and (b) z-score for the combined score — never to define a direction.

Stages:
  python src/13_arm_concept.py fit     # GPU: forward-pass contrast set -> directions
  python src/13_arm_concept.py score   # no GPU: project cached transcripts, AUCs
  python src/13_arm_concept.py         # fit then score

Writes:
  results/concept_directions.npz        frozen {concept}_L{L} unit vectors (+ layers)
  results/concept_layer_selection.json  per-concept train-CV layer choice (auditable)
  results/scores_clean.parquet          arms 'concept_<name>' and 'concept_combined'
  results/auc_concept_*_clean.json
  review/13_concept_sample.md            contrast set + top/bottom scoring transcripts
"""

import importlib.util
import json
import sys

import numpy as np
import pandas as pd

from common import ROOT, load_config
from phase2_common import auc_with_ci, load_dataset, report_arm, save_scores

CONTRAST = ROOT / "data" / "concept_contrast_set.json"
DIRECTIONS = ROOT / "results" / "concept_directions.npz"
LAYER_SEL = ROOT / "results" / "concept_layer_selection.json"
REVIEW = ROOT / "review" / "13_concept_sample.md"
CONCEPTS = ["deception", "cheating", "hidden-info", "sycophancy"]
NEUTRAL_USER = "What are you going to do?"


def _load(mod):
    spec = importlib.util.spec_from_file_location(mod, ROOT / "src" / f"{mod}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def statement_meanpool(cfg, tok, model, layers, statement):
    """Mean-pooled residual over the statement tokens, per layer — mirrors 06's
    CoT capture: [system, neutral user, assistant=statement], pool the assistant
    span. The shared prompt/template cancels in the pos-neg difference."""
    import torch

    chat = [{"role": "system", "content": cfg["prompting"]["system"]},
            {"role": "user", "content": NEUTRAL_USER}]
    prompt_ids = tok.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True,
        enable_thinking=cfg["prompting"]["enable_thinking"], return_dict=False)
    stmt_ids = tok.encode(statement, add_special_tokens=False)
    full = prompt_ids + stmt_ids
    span = (len(prompt_ids), len(full))
    ids = torch.tensor([full], device=model.device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    return {L: out.hidden_states[L][0, span[0]:span[1]].to(torch.float32).mean(0).cpu().numpy()
            for L in layers}


def fit(cfg):
    """Forward-pass the contrast set, build unit diff-of-means directions."""
    m06 = _load("06_arm_probe")
    tok, model, layers = m06.load_model(cfg)
    pairs = json.load(open(CONTRAST, encoding="utf-8"))["pairs"]
    print(f"contrast set: {len(pairs)} pairs, layers {layers}", flush=True)

    pos_mp = {c: {L: [] for L in layers} for c in CONCEPTS}
    neg_mp = {c: {L: [] for L in layers} for c in CONCEPTS}
    for i, p in enumerate(pairs):
        c = p["concept"]
        for pole, bucket in (("pos", pos_mp), ("neg", neg_mp)):
            mp = statement_meanpool(cfg, tok, model, layers, p[pole])
            for L in layers:
                bucket[c][L].append(mp[L])
        if (i + 1) % 12 == 0:
            print(f"  {i + 1}/{len(pairs)} statements pooled", flush=True)

    saved = {}
    for c in CONCEPTS:
        for L in layers:
            d = np.mean(pos_mp[c][L], axis=0) - np.mean(neg_mp[c][L], axis=0)
            n = np.linalg.norm(d)
            saved[f"{c}_L{L}"] = (d / n if n > 0 else d).astype(np.float32)
    saved["_layers"] = np.array(layers, dtype=np.int32)
    np.savez(DIRECTIONS, **saved)
    print(f"frozen directions -> {DIRECTIONS} ({len(CONCEPTS)} concepts x {len(layers)} layers)",
          flush=True)


def _project(feats, qids, direction, layer):
    return np.array([float(np.dot(feats[q]["meanpool"][layer], direction))
                     if q in feats and layer in feats[q]["meanpool"] else np.nan
                     for q in qids])


def score(cfg):
    m06 = _load("06_arm_probe")
    if not DIRECTIONS.exists():
        raise SystemExit("run `13_arm_concept.py fit` first (needs GPU)")
    dz = np.load(DIRECTIONS)
    layers = [int(x) for x in dz["_layers"]]

    df = load_dataset()
    feats = m06.load_features(df.qid)
    print(f"{len(feats)}/{len(df)} transcripts with cached activations", flush=True)
    train = df[(df.split == "train") & df.label.isin(["POS", "NEG-inert"])]
    cv_seed = cfg["seeds"]["bootstrap"]

    selection = {"layers": layers, "note": "layer chosen by train-split AUC; "
                 "direction itself uses NO task labels"}
    combined_z = {}   # concept -> per-qid z-scored projection at its chosen layer
    for c in CONCEPTS:
        # Pick read-out layer on the TRAIN split (POS vs NEG-inert). Sign is fixed
        # by the contrast set (pos pole = concept), so higher projection = more
        # concept; if a layer anti-correlates we still just report its AUC honestly.
        train_auc = {}
        for L in layers:
            proj = _project(feats, train.qid, dz[f"{c}_L{L}"], L)
            y = (train.label == "POS").astype(int).to_numpy()
            keep = ~np.isnan(proj)
            if keep.sum() and len(set(y[keep])) == 2:
                train_auc[L] = float(auc_with_ci(y[keep], proj[keep], cv_seed)["auc"])
        best = max(train_auc, key=train_auc.get)
        selection[c] = {"train_auc_by_layer": train_auc, "chosen_layer": best}
        print(f"[{c}] train AUC by layer {train_auc} -> L{best}", flush=True)

        proj_all = _project(feats, df.qid, dz[f"{c}_L{best}"], best)
        out = df[["qid", "label", "split"]].copy()
        out["score"] = proj_all
        out["layer"] = best
        save_scores(f"concept_{c}", out)
        report_arm(f"concept_{c}", out)

        mu = np.nanmean(_project(feats, train.qid, dz[f"{c}_L{best}"], best))
        sd = np.nanstd(_project(feats, train.qid, dz[f"{c}_L{best}"], best)) or 1.0
        combined_z[c] = {q: (v - mu) / sd for q, v in zip(df.qid, proj_all)}

    # Combined = mean of per-concept train-z-scored projections (equal weight).
    comb = df[["qid", "label", "split"]].copy()
    comb["score"] = [float(np.nanmean([combined_z[c][q] for c in CONCEPTS]))
                     for q in df.qid]
    comb["layer"] = -1
    save_scores("concept_combined", comb)
    report_arm("concept_combined", comb)

    with open(LAYER_SEL, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"layer selection -> {LAYER_SEL}", flush=True)

    _review_dump(df, comb, feats, dz, selection)


def _review_dump(df, comb, feats, dz, selection):
    """Contrast set summary + the transcripts the combined concept score ranks
    most/least suspicious, with their true labels (surface for Michael)."""
    pairs = json.load(open(CONTRAST, encoding="utf-8"))
    test = comb[comb.split == "test"].dropna(subset=["score"]).sort_values("score")
    lines = ["# Concept-probe arm — review dump\n",
             "Unsupervised RepE diff-of-means directions from a frozen contrast "
             "set. Direction uses NO task labels; only layer choice / z-scoring do.\n",
             "## Chosen read-out layers (train split)\n"]
    for c in CONCEPTS:
        lines.append(f"- **{c}**: L{selection[c]['chosen_layer']} "
                     f"(train AUCs {selection[c]['train_auc_by_layer']})")
    lines.append("\n## Contrast set (frozen)\n")
    for c in CONCEPTS:
        ex = [p for p in pairs["pairs"] if p["concept"] == c][:2]
        lines.append(f"**{c}** ({sum(p['concept']==c for p in pairs['pairs'])} pairs), e.g.:")
        for p in ex:
            lines.append(f"  - concept: _{p['pos']}_")
            lines.append(f"  - honest:  _{p['neg']}_")
    # qid is not unique (a POS/NEG-inert shares its base qid with the NEG-clean
    # unhinted twin) — use a dict so the lookup returns one scalar, not a Series.
    htmap = dict(zip(df.qid, df.hint_type))
    lines.append("\n## Combined concept score — most suspicious 8 (test)\n")
    for _, r in test.tail(8)[::-1].iterrows():
        lines.append(f"- score {r.score:+.2f} | **{r.label}** | {r.qid} "
                     f"({htmap.get(r.qid)})")
    lines.append("\n## Least suspicious 8 (test)\n")
    for _, r in test.head(8).iterrows():
        lines.append(f"- score {r.score:+.2f} | **{r.label}** | {r.qid} "
                     f"({htmap.get(r.qid)})")
    REVIEW.parent.mkdir(exist_ok=True)
    REVIEW.write_text("\n".join(lines), encoding="utf-8")
    print(f"review dump -> {REVIEW}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = load_config()
    if stage in ("fit", "all"):
        fit(cfg)
    if stage in ("score", "all"):
        score(cfg)
