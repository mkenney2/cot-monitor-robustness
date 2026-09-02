# Duplicate-qid fix: method and before/after diff (2026-09-02)

## What was wrong

16 of the 284 items in the 300-transcript analysis set carry two canonical
transcripts: a hinted one (4 POS, 12 NEG-inert) and an unhinted one
(NEG-clean). Every per-transcript cache was keyed by item id alone
(`data/activations/{qid}.npz`, `judge_raw.jsonl` (qid, variant, k),
`jlens_llm_raw.jsonl` (qid), and the per-transcript dicts inside
`07_arm_jlens.py`), so each pair shared one cached transcript. Token-text
matching on the pod showed the original activation cache held the UNHINTED
transcript for all 16 pairs, i.e. the 16 hinted rows (6 of them in the test
split) were scored by the probe and J-Lens on the wrong activations, and the
judge cache had the same collision (identical judge_hintblind scores across
12 of 16 pairs).

## Fix

1. `phase2_common.transcript_key`: `tkey` = qid for hinted transcripts,
   `qid__clean` for unhinted ones; `load_dataset()` carries it.
2. All cache / feature lookups re-keyed by `tkey` (04, 06, 07, 09, 11, 13, 14,
   18, 22); `pooled_features.parquet` gains a `tkey` column.
3. `src/23_dupfix_migrate.py`: renames existing NEG-clean activation files to
   `__clean` (100 at layers 16/32/48, 84 at 56/60/62), verifies EVERY file's
   token stream against the transcript it now claims to hold (0 mismatches;
   `results/dupfix_migration.json`), tags jsonl cache records with `tkey`,
   drops the 16 duplicated items' judge / lens-LLM records.
4. `scripts/dupfix_rerun.sh`: re-cache 16 hinted (16/32/48) + 16 unhinted
   (56/60/62) transcripts, re-judge 32 rows, then re-run 05 textfeat, 06 probe
   train, 07 jlens (token sets re-frozen), 22 late, 09 attacked (textfeat,
   probe, jlens), 22 attacked, 10 + 22 analysis, 11 learning curve, 12
   divergence, 13 concept score, 14 presence-vs-use, 16 correctness control
   (score + judge), 18 C-sensitivity, 15/17 figures. Logs: `logs/dupfix/`.
5. Verification: the release reproduction notebook re-asserts all 14 clean
   AUCs, the degradation matrix and the phi tables against the new artifacts
   and refits the probe (0.736); both J-Lens audit notebooks reproduce every
   score bit-exactly from transcript-keyed readouts.

Not affected: the attacked set (hinted transcripts only, keyed qid__attack),
textfeat (no cache), labels, splits, rollouts.

Before = pod snapshot /workspace/backup_pre_dupfix_20260902/results (local copy backup_pre_dupfix/results). After = results/ after scripts/dupfix_rerun.sh.

## Clean AUCs (test, POS vs NEG-inert) and vs NEG-clean

| arm | before | after | Δ | before vs clean | after vs clean |
|---|---|---|---|---|---|
| concept_cheating | 0.585 [0.457, 0.707] | 0.597 [0.473, 0.720] | +0.012 | 0.681 | 0.696 |
| concept_combined | 0.633 [0.519, 0.756] | 0.644 [0.528, 0.768] | +0.011 | 0.702 | 0.717 |
| concept_deception | 0.601 [0.476, 0.725] | 0.610 [0.483, 0.737] | +0.009 | 0.652 | 0.662 |
| concept_hidden-info | 0.582 [0.450, 0.705] | 0.591 [0.463, 0.714] | +0.009 | 0.689 | 0.701 |
| concept_sycophancy | 0.577 [0.439, 0.701] | 0.579 [0.441, 0.700] | +0.001 | 0.648 | 0.646 |
| jlens | 0.500 [0.361, 0.631] | 0.577 [0.466, 0.671] | +0.077 | 0.575 | 0.504 |
| jlens_late | 0.495 [0.379, 0.600] | 0.503 [0.382, 0.608] | +0.008 | 0.517 | 0.526 |
| jlens_llm | 0.374 [0.264, 0.492] | 0.363 [0.255, 0.474] | -0.012 | 0.451 | 0.463 |
| jlens_llm_late | 0.444 [0.338, 0.561] | 0.461 [0.359, 0.577] | +0.016 | 0.454 | 0.466 |
| judge | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | +0.000 | 0.996 | 1.000 |
| judge_hintblind | 0.895 [0.808, 0.960] | 0.917 [0.843, 0.971] | +0.021 | 0.995 | 1.000 |
| probe | 0.735 [0.622, 0.847] | 0.736 [0.627, 0.843] | +0.001 | 0.853 | 0.849 |
| probe_lasttoken | 0.997 [0.987, 1.000] | 1.000 [1.000, 1.000] | +0.003 | 0.991 | 1.000 |
| textfeat | 0.588 [0.458, 0.710] | 0.588 [0.458, 0.710] | +0.000 | 0.626 | 0.626 |

## Degradation matrix (AUC)

| arm | clean before→after | launder | paraphrase | recruit |
|---|---|---|---|---|
| judge_hintblind | 0.895→0.917 | 0.893→0.893 | 0.886→0.886 | 0.635→0.635 |
| textfeat | 0.588→0.588 | 0.407→0.407 | 0.539→0.539 | 0.457→0.457 |
| probe | 0.735→0.736 | 0.689→0.696 | 0.738→0.746 | 0.650→0.630 |
| jlens | 0.500→0.577 | 0.509→0.552 | 0.506→0.614 | 0.588→0.568 |

## Failure correlation (phi, judge_hintblind × probe)

| attack | before phi (p) | after phi (p) | contingency before | after |
|---|---|---|---|---|
| launder | -0.057 (1.00) | +0.033 (1.00) | [[5, 3], [20, 9]] | [[5, 3], [17, 12]] |
| paraphrase | +0.175 (0.46) | +0.087 (0.72) | [[8, 3], [12, 10]] | [[7, 4], [12, 10]] |
| recruit | +0.036 (1.00) | +0.104 (0.72) | [[14, 6], [10, 5]] | [[14, 6], [9, 6]] |

## Controls and follow-ups

- **probe_controls.json**: 2 changed values — neg_inert_vs_neg_clean_auc: 0.6984→0.6686; shuffled_label_test_auc: 0.5043→0.4793
- **probe_layer_selection.json**: 7 changed values — probe.cv_auc_by_layer.16: 0.7003→0.6755; probe.cv_auc_by_layer.32: 0.7222→0.7095; probe.cv_auc_by_layer.48: 0.7456→0.7267; probe_lasttoken.chosen_layer: 32→16; probe_lasttoken.cv_auc_by_layer.16: 0.9959→1.0; probe_lasttoken.cv_auc_by_layer.32: 0.9972→1.0; probe_lasttoken.cv_auc_by_layer.48: 0.9878→1.0
- **presence_vs_use.json**: 11 changed values — baseline_use_auc.auc: 0.7298→0.7364; baseline_use_auc.ci_hi: 0.841→0.8417; baseline_use_auc.ci_lo: 0.6143→0.6298; cosine_use_vs_presence: -0.4308→-0.4062; presence_dir_only_on_use_task_auc.auc: 0.4392→0.4642; presence_dir_only_on_use_task_auc.ci_hi: 0.5759→0.6042; presence_dir_only_on_use_task_auc.ci_lo: 0.3204→0.3418; shared_variance: 0.1856→0.165; use_auc_after_removing_presence_axis.auc: 0.7304→0.7383; use_auc_after_removing_presence_axis.ci_hi: 0.8393→0.8472; use_auc_after_removing_presence_axis.ci_lo: 0.6137→0.6263
- **concept_summary.json**: 0 changed values
- **correctness_control.json**: 11 changed values — judge_hintblind.auc: 0.7327→None; judge_hintblind.ci_hi: 0.7851→None; judge_hintblind.ci_lo: 0.6804→None; judge_hintblind.n_boot: 1000→None; judge_hintblind.n_dropped_nan: 0→None; judge_hintblind.n_neg: 157→None; judge_hintblind.n_pos: 154→None; probe.auc: 0.6948→0.6818; probe.ci_hi: 0.749→0.7382; probe.ci_lo: 0.6421→0.6264; reference.judge_hint_use_auc: 0.895→None
- **probe_learning_curve.json**: 1 changed values — top_half_slope_auc_per_100_train: 0.0648→0.0583
- **probe_c_sensitivity.json**: 25 changed values — probe.by_c.0.001.test_auc: 0.7396→0.735; probe.by_c.0.001.train_cv_auc: 0.7657→0.7451; probe.by_c.0.01.test_auc: 0.7383→0.7396; probe.by_c.0.01.train_cv_auc: 0.7577→0.745; probe.by_c.0.1.test_auc: 0.737→0.7383; probe.by_c.0.1.train_cv_auc: 0.7483→0.7321; probe.by_c.1.0.test_auc: 0.735→0.7364; probe.by_c.1.0.train_cv_auc: 0.7456→0.7267; probe.by_c.10.0.test_auc: 0.7206→0.7285; probe.by_c.10.0.train_cv_auc: 0.7363→0.7177; probe.by_c.100.0.test_auc: 0.7212→0.7245; probe.by_c.100.0.train_cv_auc: 0.7262→0.7138 …
- **probe_divergence_analysis.json**: 11 changed values — launder.probe_delta_mean: -0.191→None; launder.probe_delta_score_mean: None→-0.18; launder.spearman_jaccard_vs_probedelta: 0.218→0.209; launder.spearman_p: 0.0291→0.0369; paraphrase.probe_delta_mean: 0.015→None; paraphrase.probe_delta_score_mean: None→0.011; paraphrase.spearman_jaccard_vs_probedelta: 0.336→0.29; paraphrase.spearman_p: 0.0017→0.0071; recruit.probe_delta_mean: -0.273→None; recruit.probe_delta_score_mean: None→-0.298; recruit.spearman_jaccard_vs_probedelta: 0.474→0.441
