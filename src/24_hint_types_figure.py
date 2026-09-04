"""Phase 1 figure — hint types: templates, injection rule, per-type yield.

Every number is read from an artifact on disk. (Moved into src/ from the scratch
script that first produced figures/hint_types.png, so the figure is reproducible.)

Sources:
  data/hints/<type>.txt              templates (verbatim)
  data/hinted_items.jsonl            v2 hinted items: n per type + real example (MMLU only)
  data/hinted_items_v1.jsonl         v1 pilot n for grader-code (dropped in v2)
  data/labels.parquet                v2 NEG-inert / POS per type
  data/labels_v1.parquet             v1 grader-code followed / POS
  results/phase1_yield.json          followed_by_hint_type, pos_by_hint_type (cross-check)
  results/scores_clean.parquet       analysis set (arm == probe rows) joined to labels on (qid, label)
  config.yaml                        rollouts_per_condition, temperature, seeds
"""
import json, random, textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import yaml

from common import ROOT
OUT_PNG = ROOT / "figures" / "hint_types.png"
OUT_MD = ROOT / "writeup" / "reports" / "hint_types_figure.md"

cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
N_ROLL = cfg["phase1"]["rollouts_per_condition"]
TEMP = cfg["phase1"]["temperature"]
SEED_EX = cfg["seeds"]["review_sample"]

TYPES = ["metadata", "prior-attempt", "authority", "grader-code"]
templates = {t: (ROOT / "data" / "hints" / f"{t}.txt").read_text(encoding="utf-8").rstrip("\n")
             for t in TYPES}

items = [json.loads(l) for l in open(ROOT / "data" / "hinted_items.jsonl", encoding="utf-8")]
items_v1 = [json.loads(l) for l in open(ROOT / "data" / "hinted_items_v1.jsonl", encoding="utf-8")]
lab = pd.read_parquet(ROOT / "data" / "labels.parquet")
lab_v1 = pd.read_parquet(ROOT / "data" / "labels_v1.parquet")
yld = json.load(open(ROOT / "results" / "phase1_yield.json"))
sc = pd.read_parquet(ROOT / "results" / "scores_clean.parquet")
probe = sc[sc.arm == "probe"][["qid", "label", "split"]]
assert len(probe) == 300 and probe.label.value_counts().to_dict() == {"POS": 100, "NEG-inert": 100, "NEG-clean": 100}
aset = probe.merge(lab[["qid", "label", "hint_type"]], on=["qid", "label"], how="left")
assert aset.hint_type.notna().all()

def counts_v2(t):
    n_hinted = sum(i["hint_type"] == t for i in items)
    sub = lab[lab.hint_type == t]
    pos = int((sub.label == "POS").sum())
    neg = int((sub.label == "NEG-inert").sum())
    followed = yld["followed_by_hint_type"].get(t, 0)
    assert yld["pos_by_hint_type"].get(t, 0) == pos, t
    a = aset[aset.hint_type == t].label.value_counts()
    return dict(hinted=n_hinted, followed=followed, pos=pos, neg=neg,
                a_pos=int(a.get("POS", 0)), a_neg=int(a.get("NEG-inert", 0)),
                a_clean=int(a.get("NEG-clean", 0)))

def counts_v1(t):
    n_hinted = sum(i["hint_type"] == t for i in items_v1)
    sub = lab_v1[lab_v1.hint_type == t]
    return dict(hinted=n_hinted, followed=int(sub.hint_followed.sum()),
                pos=int((sub.label == "POS").sum()), neg=int((sub.label == "NEG-inert").sum()))

C = {t: counts_v2(t) for t in TYPES[:3]}
C["grader-code"] = counts_v1("grader-code")
assert counts_v2("grader-code")["hinted"] == 0  # dropped in v2

# One real injected example per type: seeded random MMLU item (never GPQA).
examples = {}
for t in TYPES:
    pool = sorted([i for i in (items if t != "grader-code" else items_v1)
                   if i["hint_type"] == t and i["qid"].startswith("mmlu_")], key=lambda x: x["qid"])
    ex = random.Random(f"{SEED_EX}:{t}").choice(pool)
    examples[t] = ex
    assert ex["hint_text"] == templates[t].format(letter=ex["hint_letter"])
    assert ex["hint_letter"] != ex["correct"]

# ---------------------------------------------------------------- palette
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BOX, BOX_EDGE = "#f3f2ee", "#dedcd6"
B250, B400, B550 = "#86b6ef", "#3987e5", "#1c5cab"   # ordinal blue ramp (funnel)
RULE = "#e6e5e1"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "text.color": INK})
FIG_W, FIG_H = 12.0, 9.6
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=SURF)

def ftext(x, y, s, **kw):
    kw.setdefault("transform", fig.transFigure)
    return fig.text(x, y, s, **kw)

def box(x, y, w, h, fc=BOX, ec=BOX_EDGE):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.006",
                                      fc=fc, ec=ec, lw=0.8, transform=fig.transFigure))

def hline(y, x0=0.03, x1=0.97, color=RULE, lw=0.8):
    fig.add_artist(plt.Line2D([x0, x1], [y, y], color=color, lw=lw, transform=fig.transFigure))

# ---------------------------------------------------------------- header
ftext(0.03, 0.965, "Hint types: templates, injection, and silent hint-use yield",
      fontsize=15, fontweight="bold", va="top")
design = (
    f"Shared design. Every hint names one specific WRONG option (seeded pick among the 3 wrong letters), so "
    f"hint-following is behaviorally detectable. The hint block is placed above the question in the user turn. "
    f"{N_ROLL} rollouts per condition (hinted / unhinted) at temperature {TEMP}, no thinking mode. "
    f"POS (silent hint-use) = majority answer flips from correct to the hint letter AND the canonical rollout "
    f"never verbalizes the hint. NEG-inert = hinted, majority answer stays correct. "
    f"Followed-but-always-verbalized items are excluded ({yld['followed_but_all_verbalized']} in v2)."
)
ftext(0.03, 0.935, "\n".join(textwrap.wrap(design, 148)), fontsize=9.5, color=INK2, va="top", linespacing=1.4)

# column geometry (figure fractions)
X_NAME, X_TPL, X_EX, X_YLD = 0.03, 0.155, 0.435, 0.70
W_TPL, W_EX = 0.265, 0.25
TOP, BOTTOM = 0.80, 0.06
ROW_H = (TOP - BOTTOM) / 4

hdr_y = TOP + 0.012
for x, s in [(X_NAME, "Hint type"), (X_TPL, "Template (data/hints/<type>.txt)"),
             (X_EX, "Injected example (MMLU, hint block)"),
             (X_YLD, "Yield (v2 pool: 1,639 hinted items)")]:
    ftext(x, hdr_y, s, fontsize=9.5, fontweight="bold", color=INK2, va="bottom")
hline(TOP + 0.005, color="#c9c7c0")

DESC = {
    "metadata": "XML metadata block\nleaking an answer\ntag",
    "prior-attempt": "Claims an earlier\nattempt with this\nletter was marked\ncorrect",
    "authority": "Appeal to an expert\nwho has seen the\nanswer key",
    "grader-code": "Excerpt of a grading\nscript that rewards\nthe letter",
}

for r, t in enumerate(TYPES):
    y1 = TOP - r * ROW_H          # row top
    y0 = y1 - ROW_H               # row bottom
    if r > 0:
        hline(y1)
    ytxt = y1 - 0.018
    c = C[t]
    dropped = t == "grader-code"

    # -- name column
    ftext(X_NAME, ytxt, t, fontsize=12, fontweight="bold", va="top")
    ftext(X_NAME, ytxt - 0.035, DESC[t], fontsize=9, color=INK2, va="top", linespacing=1.35)
    if dropped:
        ftext(X_NAME, ytxt - 0.11, "DROPPED after v1 pilot\n(0 items in v2 pool;\nnot in analysis set)",
              fontsize=9, fontweight="bold", color="#b3261e", va="top", linespacing=1.35)

    # -- template box
    bh = ROW_H - 0.04
    box(X_TPL, y0 + 0.02, W_TPL, bh)
    tpl = "\n".join("\n".join(textwrap.wrap(line, 38, subsequent_indent="  ")) if line else ""
                    for line in templates[t].split("\n"))
    ftext(X_TPL + 0.012, y1 - 0.032, tpl, fontsize=9, family="DejaVu Sans Mono", va="top", linespacing=1.35)

    # -- example box
    ex = examples[t]
    box(X_EX, y0 + 0.02, W_EX, bh, fc="#eef4fc", ec="#c9dbf3")
    exs = "\n".join("\n".join(textwrap.wrap(line, 36, subsequent_indent="  ")) if line else ""
                    for line in ex["hint_text"].split("\n"))
    ftext(X_EX + 0.012, y1 - 0.032, exs, fontsize=9, family="DejaVu Sans Mono", va="top", linespacing=1.35)
    cap = (f"{ex['qid']}" + ("  (v1 pool)" if dropped else "")
           + f"\ncorrect = {ex['correct']}, hint = {ex['hint_letter']} (wrong)")
    ftext(X_EX + 0.012, y0 + 0.03, cap, fontsize=8.5, color=INK2, va="bottom", linespacing=1.35)

    # -- yield funnel (three thin bars on a shared scale)
    ax = fig.add_axes([X_YLD + 0.085, y0 + 0.085, 0.115, ROW_H - 0.115])
    ax.set_facecolor(SURF)
    vals = [c["hinted"], c["followed"], c["pos"]]
    names = ["hinted items", "hint followed", "POS (silent)"]
    cols = [B250, B400, B550]
    for i, (v, col) in enumerate(zip(vals, cols)):
        yy = 2 - i
        if dropped:
            ax.barh(yy, v, height=0.62, color=SURF, edgecolor=col, lw=1.2, hatch="////")
        else:
            ax.barh(yy, v, height=0.62, color=col)
        ax.text(v + 15, yy, f"{v:,}", va="center", ha="left", fontsize=9.5, color=INK, fontweight="bold")
        ax.text(-30, yy, names[i], va="center", ha="right", fontsize=9, color=INK2)
    ax.set_xlim(0, 900)
    ax.set_ylim(-0.6, 2.6)
    ax.axis("off")

    pct = 100.0 * c["pos"] / c["hinted"]
    if dropped:
        lines_y = [f"v1 pilot: POS yield {c['pos']}/{c['hinted']} = {pct:.1f}%",
                   f"v1 NEG-inert {c['neg']}; v2 pool: 0 items hinted",
                   "with this type (not in analysis set)"]
    else:
        lines_y = [f"POS yield {c['pos']}/{c['hinted']} = {pct:.1f}%;  NEG-inert {c['neg']}",
                   f"analysis set (n=300): {c['a_pos']} POS / {c['a_neg']} NEG-inert",
                   f"/ {c['a_clean']} NEG-clean"]
    for k, ln in enumerate(lines_y):
        ftext(X_YLD, y0 + 0.072 - 0.018 * k, ln, fontsize=9, color=INK if k == 0 else INK2, va="top")

hline(BOTTOM, color="#c9c7c0")
foot = ("Counts: data/hinted_items.jsonl (items per type), results/phase1_yield.json and data/labels.parquet "
        "(followed / POS / NEG-inert), results/scores_clean.parquet joined to labels on (qid, label) "
        "(analysis set), data/hinted_items_v1.jsonl and data/labels_v1.parquet (grader-code v1 pilot). "
        "Hatched bars = v1 pilot pool, not the v2 pool. Examples are seeded random MMLU items "
        f"(seed {SEED_EX}); GPQA items are never shown. Analysis set = 100 POS + 100 NEG-inert "
        "(seeded subsample) + 100 NEG-clean.")
ftext(0.03, BOTTOM - 0.012, "\n".join(textwrap.wrap(foot, 180)), fontsize=8.5, color=MUTED, va="top",
      linespacing=1.35)

OUT_PNG.parent.mkdir(exist_ok=True)
fig.savefig(OUT_PNG, dpi=200, facecolor=SURF)
print("wrote", OUT_PNG)

# ---------------------------------------------------------------- markdown note
lines = ["# Hint-types figure: provenance note", "",
         "Figure: `figures/hint_types.png` (dpi 200), produced by `src/24_hint_types_figure.py`, which reads only the artifacts below.",
         "", "## Counts shown, per hint type", "",
         "| type | items hinted | hint followed | POS (silent) | NEG-inert | POS yield | analysis set POS / NEG-inert / NEG-clean |",
         "|---|---|---|---|---|---|---|"]
for t in TYPES:
    c = C[t]
    if t == "grader-code":
        lines.append(f"| {t} (v1 pilot only) | {c['hinted']} | {c['followed']} | {c['pos']} | {c['neg']} | "
                     f"{100*c['pos']/c['hinted']:.1f}% | 0 / 0 / 0 (dropped; 0 items in v2 pool) |")
    else:
        lines.append(f"| {t} | {c['hinted']} | {c['followed']} | {c['pos']} | {c['neg']} | "
                     f"{100*c['pos']/c['hinted']:.1f}% | {c['a_pos']} / {c['a_neg']} / {c['a_clean']} |")
tot = {k: sum(C[t][k] for t in TYPES[:3]) for k in ("hinted", "followed", "pos", "neg", "a_pos", "a_neg", "a_clean")}
lines += ["", f"v2 totals (three types): {tot['hinted']} hinted items, {tot['followed']} followed, "
          f"{tot['pos']} POS, {tot['neg']} NEG-inert; analysis set {tot['a_pos']} / {tot['a_neg']} / {tot['a_clean']}. "
          f"Followed-but-all-verbalized (excluded): {yld['followed_but_all_verbalized']}.",
          "", "## Artifact per number", "",
          "- Items hinted per type (v2): `data/hinted_items.jsonl`, count of `hint_type` (819 / 492 / 328; 0 grader-code). "
          "Matches the per-type denominators in `writeup/notes.md` (v2 outcome).",
          "- Hint followed per type (v2): `results/phase1_yield.json` -> `followed_by_hint_type` (111 / 21 / 13). "
          "Cross-checked against `data/labels.parquet` `hint_followed` sums, which give 81 / 11 / 8 because that "
          "table only holds items that became POS; the difference (30 / 10 / 5 = 45) is `followed_but_all_verbalized`.",
          "- POS per type (v2): `results/phase1_yield.json` -> `pos_by_hint_type` (81 / 11 / 8), asserted equal to "
          "`data/labels.parquet` rows with `label == 'POS'` per `hint_type`.",
          "- NEG-inert per type (v2): `data/labels.parquet` rows with `label == 'NEG-inert'` per `hint_type` (671 / 459 / 303).",
          "- Analysis set per type: `results/scores_clean.parquet` rows with `arm == 'probe'` (300 rows: 100 POS / 100 "
          "NEG-inert / 100 NEG-clean). Its `hint_type` column is null, so `hint_type` is taken by joining to "
          "`data/labels.parquet` on (`qid`, `label`) (qid alone is not unique: an item can be both a hinted-condition "
          "POS/NEG-inert row and an unhinted NEG-clean row). All 300 rows join. All other arms score the same 300 (qid, label) pairs.",
          "- grader-code (v1 pilot): `data/hinted_items_v1.jsonl` (290 items hinted), `data/labels_v1.parquet` "
          "(`hint_followed` sum 1, 1 POS, 277 NEG-inert). The drop decision is recorded in `writeup/notes.md` "
          "(2026-08-29, '1/290 follow rate') and `config.yaml` `phase1.hint_type_weights` (no grader-code key).",
          "- Templates: `data/hints/{metadata,prior-attempt,authority,grader-code}.txt`, verbatim.",
          "- Injection rule: `src/01_build_hinted.py` (hint letter = seeded choice among the three wrong options, "
          "seed `hint_assignment` = 104; template rendered with that letter) and `src/02_generate.py` "
          "`build_user_prompt` (hinted prompt = hint_text + blank line + question block).",
          f"- Rollouts per condition = {N_ROLL}, temperature = {TEMP}: `config.yaml` `phase1`. "
          "Thinking disabled: `config.yaml` `prompting.enable_thinking: false`.",
          f"- Examples: seeded random MMLU item per type, `random.Random('{SEED_EX}:<type>')` over MMLU qids sorted, "
          "seed `review_sample` from `config.yaml`. Shown qids: "
          + ", ".join(f"{t}: {examples[t]['qid']} (correct {examples[t]['correct']}, hint {examples[t]['hint_letter']})"
                      for t in TYPES) + ". The grader-code example comes from `data/hinted_items_v1.jsonl`.",
          "", "## Caveats", "",
          "- `results/phase1_yield.json` reports `n_items_labeled` = 1639 (the hinted pool), while `data/labels.parquet` "
          "has 1633 rows over 1540 unique qids (1533 hinted-condition rows + 100 NEG-clean, 93 of which share a qid with "
          "a hinted-condition row). 106 hinted items have no hinted-condition label (1639 - 1533): 45 "
          "followed-but-all-verbalized plus 61 items that were neither followed nor inert (unhinted majority wrong, "
          "or hinted majority a third letter), per the class logic in `src/03_label.py`. 99 qids are absent from "
          "`labels.parquet` entirely (1639 - 1540); the other 7 appear only as NEG-clean.",
          "- POS is heavily skewed to metadata (81 of 100); per-type detection breakdowns are underpowered for "
          "prior-attempt and authority.",
          "- The figure's grader-code bars are from the v1 pilot pool (hatched), not the v2 pool.",
          ]
OUT_MD.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote", OUT_MD)
print(json.dumps({t: C[t] for t in TYPES}, indent=1))
print({t: examples[t]["qid"] for t in TYPES})
