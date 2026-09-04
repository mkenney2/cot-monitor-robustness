# Hint-types figure: provenance note

Figure: `figures/hint_types.png` (dpi 200), produced by `src/24_hint_types_figure.py`, which reads only the artifacts below.

## Counts shown, per hint type

| type | items hinted | hint followed | POS (silent) | NEG-inert | POS yield | analysis set POS / NEG-inert / NEG-clean |
|---|---|---|---|---|---|---|
| metadata | 819 | 111 | 81 | 671 | 9.9% | 81 / 81 / 51 |
| prior-attempt | 492 | 21 | 11 | 459 | 2.2% | 11 / 11 / 25 |
| authority | 328 | 13 | 8 | 303 | 2.4% | 8 / 8 / 24 |
| grader-code (v1 pilot only) | 290 | 1 | 1 | 277 | 0.3% | 0 / 0 / 0 (dropped; 0 items in v2 pool) |

v2 totals (three types): 1639 hinted items, 145 followed, 100 POS, 1433 NEG-inert; analysis set 100 / 100 / 100. Followed-but-all-verbalized (excluded): 45.

## Artifact per number

- Items hinted per type (v2): `data/hinted_items.jsonl`, count of `hint_type` (819 / 492 / 328; 0 grader-code). Matches the per-type denominators in `writeup/notes.md` (v2 outcome).
- Hint followed per type (v2): `results/phase1_yield.json` -> `followed_by_hint_type` (111 / 21 / 13). Cross-checked against `data/labels.parquet` `hint_followed` sums, which give 81 / 11 / 8 because that table only holds items that became POS; the difference (30 / 10 / 5 = 45) is `followed_but_all_verbalized`.
- POS per type (v2): `results/phase1_yield.json` -> `pos_by_hint_type` (81 / 11 / 8), asserted equal to `data/labels.parquet` rows with `label == 'POS'` per `hint_type`.
- NEG-inert per type (v2): `data/labels.parquet` rows with `label == 'NEG-inert'` per `hint_type` (671 / 459 / 303).
- Analysis set per type: `results/scores_clean.parquet` rows with `arm == 'probe'` (300 rows: 100 POS / 100 NEG-inert / 100 NEG-clean). Its `hint_type` column is null, so `hint_type` is taken by joining to `data/labels.parquet` on (`qid`, `label`) (qid alone is not unique: an item can be both a hinted-condition POS/NEG-inert row and an unhinted NEG-clean row). All 300 rows join. All other arms score the same 300 (qid, label) pairs.
- grader-code (v1 pilot): `data/hinted_items_v1.jsonl` (290 items hinted), `data/labels_v1.parquet` (`hint_followed` sum 1, 1 POS, 277 NEG-inert). The drop decision is recorded in `writeup/notes.md` (2026-08-29, '1/290 follow rate') and `config.yaml` `phase1.hint_type_weights` (no grader-code key).
- Templates: `data/hints/{metadata,prior-attempt,authority,grader-code}.txt`, verbatim.
- Injection rule: `src/01_build_hinted.py` (hint letter = seeded choice among the three wrong options, seed `hint_assignment` = 104; template rendered with that letter) and `src/02_generate.py` `build_user_prompt` (hinted prompt = hint_text + blank line + question block).
- Rollouts per condition = 10, temperature = 0.7: `config.yaml` `phase1`. Thinking disabled: `config.yaml` `prompting.enable_thinking: false`.
- Examples: seeded random MMLU item per type, `random.Random('106:<type>')` over MMLU qids sorted, seed `review_sample` from `config.yaml`. Shown qids: metadata: mmlu_professional_law_1400 (correct B, hint D), prior-attempt: mmlu_college_chemistry_55 (correct C, hint D), authority: mmlu_college_physics_29 (correct D, hint C), grader-code: mmlu_professional_law_506 (correct C, hint D). The grader-code example comes from `data/hinted_items_v1.jsonl`.

## Caveats

- `results/phase1_yield.json` reports `n_items_labeled` = 1639 (the hinted pool), while `data/labels.parquet` has 1633 rows over 1540 unique qids (1533 hinted-condition rows + 100 NEG-clean, 93 of which share a qid with a hinted-condition row). 106 hinted items have no hinted-condition label (1639 - 1533): 45 followed-but-all-verbalized plus 61 items that were neither followed nor inert (unhinted majority wrong, or hinted majority a third letter), per the class logic in `src/03_label.py`. 99 qids are absent from `labels.parquet` entirely (1639 - 1540); the other 7 appear only as NEG-clean.
- POS is heavily skewed to metadata (81 of 100); per-type detection breakdowns are underpowered for prior-attempt and authority.
- The figure's grader-code bars are from the v1 pilot pool (hatched), not the v2 pool.
