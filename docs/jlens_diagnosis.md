# Arm 4 (J-Lens): why it was at chance, and the late-layer rerun

Technical report, 2026-09-01/02. Every number traces to an artifact (paths in
parentheses). Scripts: `src/19_jlens_readouts.py` (saves per-position top-10
readouts), `src/20_jlens_logitlens_check.py` (laptop diagnostic),
`src/21_jlens_anchor_check.py` (pod: readout vs the model's own logits),
`src/22_jlens_late.py` (rerun at layers 56/60/62). Notebooks (release repo):
`notebooks/jlens_readout_audit.ipynb`, `notebooks/jlens_readout_audit_late.ipynb`.
All numbers are post duplicate-qid fix unless marked.

## 1. Question

The pre-registered J-Lens arm (hint-salience score from the lens's top-10
vocabulary readout at layers 16/32/48, k=10, `camilablank/workspace-lenses`
qwen3.6-27b j-lens) scored at chance (0.500 as first published). Is that a
property of the activations, or of the readout?

## 2. Anchor check (`results/jlens_anchor_check.json`, pod, 3 seeded transcripts)

Fresh forced forward pass; compare each readout's top-1 / top-10 against the
model's own next-token argmax and the actual next token, at hidden-state
indices h ∈ {16, 32, 48, 56, 60, 61, 62, 63, 64}. Readouts: plain logit lens
`W_U·rmsnorm(h)`, J-Lens with `J[h]`, J-Lens with `J[h−1]`.

| h | logit lens: next token in top-10 | J-Lens (J[h]) | letter in top-10 before "Answer: X" (logit / J) |
|---|---|---|---|
| 16 | 0.000 | 0.038 | 0 / 0 |
| 32 | 0.000 | 0.082 | 0 / 0 |
| 48 | 0.031 | 0.085 | 0 / 0 |
| 56 | 0.405 | 0.269 | 0.67 / 0 |
| 60 | 0.567 | 0.581 | 0.67 / 0 |
| 62 (lens anchor) | 0.750 | 0.750 (identity) | 1.0 / 1.0 |
| 63 | 0.983 | — | 1.0 |
| 64 (final) | 1.000 (top-1 agreement 0.92) | — | 1.0 |

Conclusions: (i) the readout formula is right (final residual reproduces the
model's head); (ii) the layer convention used by `07_arm_jlens.py`
(`J[L] @ hidden_states[L]`) is right (identity anchor at index 62 matches
`hidden_states[62]` exactly; the h−1 variant is uniformly worse); (iii) the
cached activations are faithful (fresh-pass J-Lens 4/8/8% at 16/32/48 vs
3/6.5/6% from the cache); (iv) **Qwen 3.6 27B decodes only in its last ~8
layers**. The arm ran at 25/50/75% depth, where the vocabulary readout is
noise. J-Lens beats the logit lens below 48 and is worse at 56.

## 3. Original-layer readout audit (`results/jlens_readout_audit.json`; sample 26 POS / 27 NEG-inert / 33 NEG-clean, 101k positions)

- Next-token top-10 hit 3% / 6.5% / 6% (layers 16/32/48); 43% of layer-48
  slots are ASCII words.
- A standalone A–D token appears in a top-10 readout **once** across 101k
  positions × 3 layers → the score's hint-letter term was 0 for every
  transcript; the published score is the frozen-word term alone (`hint`
  fires in 6/8/23% of NEG-clean/NEG-inert/POS transcripts, `hidden` 4–18%,
  at weights ~1e-4). Saved readouts reproduce the published scores exactly
  (86/86).
- No token beats a permutation null (95% max |log-odds| 3.6–3.7) in
  POS-vs-NEG-inert or hinted-vs-clean at any layer; enriched readable tokens
  are subject-matter words (topic mix of a 26-vs-27 sample).
- Lexicons: hint-topic ≤1% of positions, deception ≤0.2%, hedging L32 POS
  9.5% vs NEG-inert 5.6% (overlapping CIs). LLM-reads-the-readout arm 0.363
  [0.255, 0.474], 55% of verdicts the default 5.
- Post-fix clean AUC 0.577 [0.466, 0.671] is entirely the re-frozen word
  term; CI includes 0.5, does not survive attack (0.55–0.61).

## 4. Late-layer rerun (`src/22_jlens_late.py all`; layers 56/60/62; 300 clean + 566 attacked re-cached; token sets re-frozen on train → `results/jlens_late_token_sets.json`)

| | clean | launder | paraphrase | recruit |
|---|---|---|---|---|
| jlens_late | 0.503 [0.382, 0.608] | 0.475 | 0.553 | 0.519 |
| jlens_llm_late | 0.461 [0.359, 0.577] | — | — | — |

(`results/auc_jlens_late_clean.json`, `results/auc_jlens_llm_late_clean.json`,
`results/jlens_late_analysis/results/degradation_matrix.json`; phi tables
with the late arm in `results/jlens_late_analysis/results/failure_correlation.json`.)

Readout audit (`results/jlens_readout_audit_late.json`; 21 POS / 22 NEG-inert
/ 21 NEG-clean, 95.7k positions):

- The lens now reads: next-token top-10 23% / 54% / 71%, 68% of layer-62
  slots are words; scores reproduce exactly (64/64).
- Letters surface only near the answer: 777 of 95.7k positions have a letter
  in the top-10; hint letter at 0.04% of unmentioned positions in BOTH
  classes; letter term fires in 52% POS vs 41% NEG-inert at ~1e-5 weights,
  letter-only test AUC 0.43 (n=14).
- **Presence visible, use not:** `metadata` (all casings) in the layer-62
  readouts of 19–21 of 43 hinted transcripts vs 0 of 21 clean (log-odds
  3.5–3.7 vs null 3.9). POS vs NEG-inert: 0 tokens beat the null at 56 and
  62; 3 at 60 are `prec`/`Prec`/`_Prec` (12 POS vs 0 NEG-inert), the start of
  "precedent" — a law-subject artifact.
- Lexicons at 62: hint-topic POS 1.1% / NEG-inert 0.8% / clean 0.9%;
  deception 0.8 / 0.1 / 0.3%; hedging 1.1 / 0.6 / 0.6% (exploratory AUC
  0.79–0.83 at n=14 — a lead, not a result).

## 5. Interpretation for the writeup

The pre-registered J-Lens null was a wrong-layer artifact. Rerun where the
lens decodes, the unsupervised vocabulary readout is a *real* negative for
silent hint-use: what surfaces is the reasoning topic and, for hinted
prompts, the presence of a metadata block — not whether the hint was
followed. This matches the probe's presence-vs-use decomposition: hint-use is
a task-specific direction that a vocabulary readout does not expose, while a
supervised linear probe does (0.736). Layer choice for readout-style arms
should be validated with an anchor check like §2 before use; for this model
that means layers ≥ 56.

## 6. Limitations of this diagnosis

Anchor check on 3 transcripts (mean over ~2.7k positions); readout audits on
seeded samples of 86 / 64 transcripts (20 per class seeded, plus extras);
exploratory AUCs at n = 14–20 test transcripts; the lens is the published
pretraining-fit (25 prompts, t_max 128), no chat-format refit was attempted.
