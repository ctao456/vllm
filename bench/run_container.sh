#!/usr/bin/env bash
# Launch the vllm-xpu-kernel-0.1.9 container detached, with all mounts needed
# for the FP8-KV vs TurboQuant benchmark sweep, then install the editable
# vLLM (TurboQuant branch) inside it.
#
# Usage:
#   bench/run_container.sh up        # launch + install (idempotent-ish)
#   bench/run_container.sh install   # (re)run editable install inside container
#   bench/run_container.sh sh        # exec an interactive shell
#   bench/run_container.sh down      # stop & remove container
set -euo pipefail

IMAGE="vllm-xpu-kernel-0.1.9:latest"
NAME="tq-bench"
REPO="/home/intel/ctao/turboquant/vllm"          # editable vLLM source (TQ branch)
MODELS="/home/intel/models"                       # HF_HOME + results live here
HF_TOKEN_FILE="/home/intel/ctao/hf_token.json"

HF_TOKEN="$(python3 -c "import json,sys; print(json.load(open('$HF_TOKEN_FILE'))['hf_token'])")"

up() {
  if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "Container '$NAME' already exists; starting it."
    docker start "$NAME" >/dev/null
  else
    echo "Launching container '$NAME' from $IMAGE ..."
    docker run -d --name "$NAME" \
      --privileged --network=host --ipc=host \
      --device /dev/dri:/dev/dri \
      -v /dev/dri/by-path:/dev/dri/by-path \
      -v "$REPO":"$REPO" \
      -v "$MODELS":"$MODELS" \
      -e HF_HOME="$MODELS/hf-home" \
      -e HF_TOKEN="$HF_TOKEN" \
      -e http_proxy="http://proxy-dmz.intel.com:912" \
      -e https_proxy="http://proxy-dmz.intel.com:912" \
      -e no_proxy="10.0.0.0/8,intel.com,.intel.com,127.0.0.1,localhost" \
      -e VLLM_NO_USAGE_STATS=1 -e VLLM_DO_NOT_TRACK=1 \
      -w "$REPO" \
      --entrypoint sleep "$IMAGE" infinity
  fi
  echo "Container is up. Run 'bench/run_container.sh install' next."
}

install() {
  echo "Installing editable vLLM (TurboQuant branch) inside container ..."
  # --no-deps: the image already ships the full vLLM 0.22.1 dependency set
  # (torch 2.11.0+xpu, vllm-xpu-kernels 0.1.9, triton-xpu 3.7.x). We only need
  # to register THIS source tree as the active vllm so the TurboQuant backend is
  # used. Installing deps would upgrade torch/kernels off the image's tested
  # stack and pull CUDA "triton" (which shadows triton-xpu). Keep image versions.
  docker exec "$NAME" bash -lc '
    set -e
    git config --global --add safe.directory "'"$REPO"'" || true
    cd "'"$REPO"'"
    VLLM_TARGET_DEVICE=xpu pip install --no-build-isolation --no-deps -e . -v 2>&1 | tail -15
    echo "---- versions ----"
    pip list 2>/dev/null | grep -Ei "^(torch|vllm|triton|vllm-xpu-kernels|transformers|numpy) " || true
    python -c "import triton, triton.backends; print(\"triton\", triton.__version__, \"backends OK\")"
  '
}

sh()   { docker exec -it "$NAME" bash -l; }
down() { docker rm -f "$NAME" 2>/dev/null || true; echo "removed $NAME"; }

case "${1:-up}" in
  up) up ;;
  install) install ;;
  sh) sh ;;
  down) down ;;
  *) echo "usage: $0 {up|install|sh|down}"; exit 1 ;;
esac
