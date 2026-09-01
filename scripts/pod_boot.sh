#!/usr/bin/env bash
# Run after every pod restart (the container disk resets; /workspace persists):
#   bash /workspace/mats-application/scripts/pod_boot.sh
set -euo pipefail

# git deploy key: stored on the volume (whose FS ignores chmod), copied to the
# container disk with sane perms so ssh accepts it.
install -d -m 700 /root/.ssh
install -m 600 /workspace/.ssh/github_deploy /root/.ssh/github_deploy

grep -q HF_HOME /root/.bashrc || echo 'export HF_HOME=/workspace/hf_cache' >> /root/.bashrc
# API keys live in /workspace/secrets.env (never in the repo)
grep -q secrets.env /root/.bashrc || echo '[ -f /workspace/secrets.env ] && source /workspace/secrets.env' >> /root/.bashrc

echo "boot bootstrap done. Start services with:"
echo "  cd /workspace/mats-application && source /workspace/venv/bin/activate"
echo "  HF_HOME=/workspace/hf_cache nohup vllm serve Qwen/Qwen3.6-27B --port 8000 --dtype bfloat16 > logs/vllm.log 2>&1 &"
