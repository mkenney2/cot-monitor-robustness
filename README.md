# cot-monitor-robustness

**When does activation-level monitoring beat text-level monitoring of
chain-of-thought — and do the two channels fail independently under
adversarial pressure?**

This repo contains the full pipeline and analysis for a study of monitor
robustness on a *silent hint-use* task: a model (Qwen 3.6 27B) is given
multiple-choice questions with an injected hint pointing to a specific wrong
answer, and we study transcripts where it **follows the hint without ever
verbalizing it**. Four monitors score each transcript, then we attack the
chain-of-thought with adversarial rewrites and measure how each monitor
degrades — and whether text-level and activation-level failures are
correlated per item.

## Headline results

| Monitor | Channel | Clean AUC | paraphrase | launder | recruit |
|---|---|---|---|---|---|
| LLM judge (hint-blind) | text | 0.895 | 0.886 | 0.893 | **0.635** |
| TF-IDF + logistic reg. | text | 0.588 | 0.539 | 0.407 | 0.457 |
| Linear probe (layer 48 mean-pool) | activations | 0.735 | 0.738 | 0.689 | 0.650 |
| J-Lens hint-salience | activations | 0.500 | 0.506 | 0.509 | 0.588 |

- **The text judge is robust to rewording and laundering but breaks under
  social engineering**: `recruit` rewrites (which address the monitor
  directly with verification claims and authority framing) cut its AUC from
  0.895 to 0.635. The probe barely moves — injected text is just more tokens
  to the activations.
- **Text and activation failures are roughly independent per item**
  (phi ≈ +0.18 / −0.06 / +0.04 across the three attacks, all Fisher p > 0.4):
  OR-ing the two channels recovers 9–17 points of catch rate over the judge
  alone. Defense-in-depth is supported — with the caveat that at n ≈ 35 per
  attack only large correlations were detectable.
- Mandatory controls: shuffled-label probe at chance; a presence-vs-use
  decomposition showing the hint-*presence* direction alone does not explain
  the probe's hint-*use* AUC; a correctness-confound control.

![Degradation under attack](figures/degradation_with_cis.png)
![Failure independence](figures/failure_independence.png)

## Reproduce the results (10 minutes, CPU only)

All numbers and figures regenerate from saved artifacts — no GPU, no API
keys. The artifacts live in a **gated** Hugging Face dataset (gated because
the question pool includes GPQA-diamond items, whose authors ask that
question text not be posted openly; accepting the terms is one click).

```bash
pip install -r requirements.txt
hf auth login                       # needed once, for the gated dataset
jupyter lab notebooks/reproduce_results.ipynb
```

The notebook downloads the dataset, **recomputes** the leaderboard AUCs
(with bootstrap CIs), the degradation matrix, and the failure-correlation
tables from the raw score parquets, asserts they match the published JSON
artifacts to numerical precision, refits the activation probe from pooled
features, and regenerates the figures.

Dataset: [`mjkenney/cot-monitor-robustness`](https://huggingface.co/datasets/mjkenney/cot-monitor-robustness)
(questions, hints, labels, canonical + attacked transcripts, judge scores
and justifications, pooled activations, all result artifacts; full per-token
activations in an optional 17 GB config needed only for the J-Lens arm).

## Reproduce the full pipeline (GPU, ~1–2 days, ~$100)

The generation pipeline (`src/00`–`09`) needs an 80 GB-class GPU (we used an
H200 on RunPod), vLLM for rollouts, HF transformers for activation caching,
and an OpenRouter key for the judge (Claude Opus 4.5) and attacker
(Kimi K3). Stages, in order:

| Script | What it does | Where |
|---|---|---|
| `00_filter_questions.py` | MMLU-hard + GPQA pool, unhinted accuracy pre-pass | GPU (vLLM) |
| `01_build_hinted.py` | inject hints (metadata / authority / prior-attempt) | CPU |
| `02_generate.py` | 10 rollouts × {hinted, unhinted} per item | GPU (vLLM) |
| `03_label.py` | hint-followed + verbalization labels → POS / NEG classes | API |
| `04`–`07` | the four monitor arms (judge, textfeat, probe, J-Lens) | mixed |
| `08_attacks.py` | paraphrase / launder / recruit rewrites + behavior re-verification | API + GPU |
| `09_score_attacked.py` | re-run all arms frozen on surviving rewrites | mixed |
| `10_analysis.py` | degradation matrix + failure correlation (phi) | CPU |
| `11`–`18` | controls, learning curve, concept arm, figures, sensitivity | CPU |

Every stage writes artifacts to `results/` and human-readable random samples
to `review/`; analysis scripts only ever read artifacts. All seeds are in
`config.yaml`. `scripts/setup_pod.sh` records the exact pod environment.

The study design was written down before the data came in:
[`docs/preregistration.md`](docs/preregistration.md).

## Limitations (stated up front)

One model, one unfaithfulness type (hint-following), hand-designed rather
than optimized attacks, n = 100 silent-positive items (CIs are wide — phi
estimates especially are underpowered), forced-forward-pass activations over
rewritten CoTs (measures the model *processing* a laundered CoT, not
producing it), and residual label noise from LLM-assisted labeling
(hand-checked samples in `review/`).

## Citation

Michael Kenney, *Monitor reliability under adversarial pressure: text vs
activation channels on silent hint-use*, 2026. MATS 12.0 application
project (Neel Nanda stream).
