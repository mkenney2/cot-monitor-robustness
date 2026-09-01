#!/usr/bin/env bash
# One-shot RunPod bootstrap. Run from the repo root on the pod:
#   bash scripts/setup_pod.sh
# Assumes the RunPod PyTorch template (torch + CUDA preinstalled) and the
# persistent volume mounted at /workspace.

set -euo pipefail

echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Venv lives on the persistent volume so it survives pod restarts; system
# python is PEP-668 externally managed. All sessions must `source` this.
VENV=/workspace/venv
echo "=== venv ($VENV) ==="
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"

# Model weights (~55 GB) must go to the network volume — the container disk is
# only ~30 GB. Persist for all future shells too.
export HF_HOME=/workspace/hf_cache
mkdir -p "$HF_HOME"
grep -q HF_HOME /root/.bashrc || echo 'export HF_HOME=/workspace/hf_cache' >> /root/.bashrc

echo "=== deps ==="
pip install -q --upgrade pip
pip install -q -r requirements.txt
# Pinned for the RunPod 570.x driver (CUDA <= 12.8, i.e. major version 12):
# - latest vllm (0.28+) pins a cu130 torch that cannot init on this driver;
#   vllm 0.26.0 is the newest release pinning torch 2.11, which has cu128 wheels.
# - vllm's own PyPI wheel is ALSO built for CUDA 13; the +cu129 GitHub release
#   asset runs on 12.8 drivers via same-major minor-version compatibility.
pip install -q 'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128' \
    'torchaudio==2.11.0+cu128' --extra-index-url https://download.pytorch.org/whl/cu128
pip install -q 'https://github.com/vllm-project/vllm/releases/download/v0.26.0/vllm-0.26.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl'
# vllm's dep resolution pulls a CUDA-13-linked torchcodec whose import dies with
# an unguarded OSError inside vllm. The cu128 build fails with RuntimeError,
# which vllm catches and shims (we never decode video) — install it + ffmpeg.
apt-get update -qq && apt-get install -y -qq ffmpeg
pip install -q --no-deps --force-reinstall 'torchcodec==0.9.1+cu128' \
    --index-url https://download.pytorch.org/whl/cu128
pip install -q transformers accelerate jupyterlab

echo "=== HF auth check ==="
python - <<'EOF'
from huggingface_hub import whoami
try:
    print("logged in as:", whoami()["name"])
except Exception:
    # Fine as long as the policy model repo is public and the question pool
    # (data/base_questions_raw.jsonl) was synced from the local machine.
    print("WARNING: no HF login on this pod — ok for public model repos")
EOF

echo "=== verify policy model id ==="
python - <<'EOF'
import sys
import yaml
from huggingface_hub import model_info

model = yaml.safe_load(open("config.yaml"))["models"]["policy"]
try:
    info = model_info(model)
    print(f"OK: {model} exists ({info.safetensors.total / 1e9:.1f}B params)"
          if info.safetensors else f"OK: {model} exists")
except Exception as e:
    sys.exit(f"models.policy='{model}' not found on HF: {e}\n"
             "Fix config.yaml before starting the server.")
EOF

mkdir -p logs data/rollouts data/activations results figures review

cat <<'EOF'

Setup complete. Next steps (in every new shell: source /workspace/venv/bin/activate):

  # start vLLM (background, logged):
  MODEL=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['models']['policy'])")
  nohup vllm serve "$MODEL" --port 8000 --dtype bfloat16 > logs/vllm.log 2>&1 &
  tail -f logs/vllm.log            # wait for 'Application startup complete'

  # smoke-test the endpoint:
  curl -s localhost:8000/v1/models | head -c 400

  # start JupyterLab (persistent kernel for analysis):
  nohup jupyter lab --ip=0.0.0.0 --port 8888 --allow-root --no-browser > logs/jupyter.log 2>&1 &
  grep -o 'token=[a-f0-9]*' logs/jupyter.log | head -1

  # then run the pipeline:
  python src/00_filter_questions.py prepass
EOF
