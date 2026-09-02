# Probe regularization (C) sensitivity — post-hoc appendix check

**Question.** The pre-registered probe used sklearn's default `C=1.0`
(L2 logistic regression), untuned. Would any reasonable regularization
strength have changed the story?

**Method.** `src/18_probe_c_sensitivity.py`, first run on the pod 2026-09-01,
re-run 2026-09-02 after the duplicate-qid fix (numbers below are the post-fix
ones; pre-fix test AUCs were 0.740/0.738/0.737/0.735/0.721/0.721). Identical
protocol to the pre-registered fit (same splits, StratifiedKFold(5, seed 109),
StandardScaler → class-balanced logistic regression, frozen layer choice from
`results/probe_layer_selection.json`); only C varies over
{0.001, 0.01, 0.1, 1, 10, 100}. **Post-hoc**: test AUCs across the grid are
sensitivity-only and were not used to select a new C (test was already
touched). Artifact: `results/probe_c_sensitivity.json`. n_train = 122,
n_test = 78 (POS vs NEG-inert, all 300 transcripts with activations).

**Result — meanpool layer 48 (primary probe):**

| C | train-CV AUC | test AUC |
|---|---|---|
| 0.001 | 0.745 | 0.735 |
| 0.01 | 0.745 | **0.740** |
| 0.1 | 0.732 | 0.738 |
| 1.0 (default, published) | 0.727 | **0.736** |
| 10 | 0.718 | 0.728 |
| 100 | 0.714 | 0.725 |

**Conclusion.** Test AUC moves less than ±0.01 across five orders of
magnitude of regularization; the published 0.736 sits within half a point of
the post-hoc best (0.740 at C=0.01). The untuned default is not
understating the activation channel; one limitations line suffices. The mild
monotone preference for stronger regularization in CV is consistent with the
heavily overparameterized regime (~5k features, 122 training rows) and
suggests no headroom was left on the table via C.
