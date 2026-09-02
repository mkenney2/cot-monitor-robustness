#!/usr/bin/env bash
# Duplicate-qid fix: migrate caches, fill the missing transcripts, re-run every
# stage downstream of the activation/judge caches. Pod-side. Logs to logs/dupfix/.
set -uo pipefail
cd /workspace/mats-application
source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf_cache
source /workspace/secrets.env
mkdir -p logs/dupfix
START=${1:-}   # optional: resume from this stage name (stages before it are skipped)
run() {  # run <name> <cmd...>; abort the chain on failure
  local name=$1; shift
  if [ -n "$START" ] && [[ "$name" < "$START" ]]; then echo "skip $name"; return 0; fi
  echo "===== $name: $* ($(date -u +%H:%M:%S))"
  "$@" > "logs/dupfix/$name.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "FAILED $name (rc=$rc) — see logs/dupfix/$name.log"; tail -20 "logs/dupfix/$name.log"; exit $rc; fi
  echo "ok $name ($(date -u +%H:%M:%S))"
}
run 00_migrate      python src/23_dupfix_migrate.py
run 01_cache        python src/06_arm_probe.py cache            # 16 hinted transcripts @16/32/48
run 02_cache_late   python src/22_jlens_late.py cache           # 16 unhinted transcripts @56/60/62
run 03_judge        python src/04_arm_judge.py                  # both variants, ~32 rows x 2 x 3 calls
run 04_textfeat     python src/05_arm_textfeat.py
run 05_probe_train  python src/06_arm_probe.py train
rm -f results/jlens_token_sets.json results/jlens_late_token_sets.json   # re-freeze on the corrected train split
run 06_jlens        python src/07_arm_jlens.py score
run 07_jlens_late   python src/22_jlens_late.py score
run 08_attacked     python src/09_score_attacked.py textfeat probe jlens
run 09_attacked_late python src/22_jlens_late.py score_attacked
run 10_analysis     python src/10_analysis.py
run 11_analysis_late python src/22_jlens_late.py analysis
run 12_learning     python src/11_probe_learning_curve.py
run 13_divergence   python src/12_probe_divergence.py
run 14_concept      python src/13_arm_concept.py score
run 15_presence     python src/14_presence_vs_use.py
run 16_correctness  python src/16_correctness_control.py score
run 17_c_sens       python src/18_probe_c_sensitivity.py
run 18_figures      python src/15_figures.py
run 19_narrative    python src/17_narrative_figures.py
echo "DUPFIX_ALL_DONE $(date -u)"
